"""Student rollout의 env 축 미니배치와 JEPA 조건을 만드는 순수 함수."""

from __future__ import annotations

import torch


def proprio_windows(policy: torch.Tensor, resets: torch.Tensor, prefix: torch.Tensor,
                    prefix_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[T,N,D] 관측과 rollout 시작 prefix에서 오래된→최신 [T,N,K,D] 창을 만든다."""
    if policy.ndim != 3 or resets.shape != policy.shape[:2] or prefix.ndim != 3 or \
            prefix.shape[1:] != policy.shape[1:] or prefix_valid.shape != prefix.shape[:2]:
        raise ValueError("policy [T,N,D], resets [T,N], prefix [K-1,N,D], valid [K-1,N]")
    history = prefix
    valid = prefix_valid.bool()
    windows, masks = [], []
    for t in range(policy.shape[0]):
        if resets[t].any():
            history = torch.where(resets[t][None, :, None], torch.zeros_like(history), history)
            valid = valid & ~resets[t][None, :]
        window = torch.cat((history, policy[t:t + 1]), dim=0)
        mask = torch.cat((valid, torch.ones_like(resets[t:t + 1])), dim=0)
        windows.append(window.transpose(0, 1))
        masks.append(mask.transpose(0, 1))
        history, valid = window[1:], mask[1:]
    return torch.stack(windows), torch.stack(masks)


class ProprioHistory:
    """rollout/추론의 env별 최근 K-1 관측을 보관한다."""

    def __init__(self, window: int, num_envs: int, proprio_dim: int, device: torch.device):
        if window < 1:
            raise ValueError("proprio_window는 1 이상이어야 한다")
        self.policy = torch.zeros(window - 1, num_envs, proprio_dim, device=device)
        self.valid = torch.zeros(window - 1, num_envs, dtype=torch.bool, device=device)

    def append(self, policy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if policy.shape != self.policy.shape[1:]:
            raise ValueError("현재 policy 관측 shape가 history와 다르다")
        window = torch.cat((self.policy, policy[None]), dim=0)
        valid = torch.cat((self.valid, torch.ones_like(policy[None, :, 0], dtype=torch.bool)), dim=0)
        self.policy, self.valid = window[1:], valid[1:]
        return window.transpose(0, 1), valid.transpose(0, 1)

    def reset(self, done: torch.Tensor) -> None:
        if done.shape != self.valid.shape[1:]:
            raise ValueError("done shape가 history env 축과 다르다")
        self.policy = torch.where(done[None, :, None], torch.zeros_like(self.policy), self.policy)
        self.valid = self.valid & ~done[None, :]


def initial_episode_lengths(lengths: torch.Tensor, max_episode_length: int,
                            enabled: bool) -> torch.Tensor:
    """rsl-rl init_at_random_ep_len와 동일한 [0,max) 정수 초기 길이."""
    if lengths.ndim != 1 or lengths.dtype != torch.long or max_episode_length < 1:
        raise ValueError("episode lengths는 [N] long이고 max_episode_length는 양수여야 한다")
    return torch.randint_like(lengths, high=max_episode_length) if enabled else lengths.clone()


def env_batch_indices(num_envs: int, num_mini_batches: int, device: torch.device,
                      shuffle: bool = False) -> list[torch.Tensor]:
    if num_mini_batches < 1 or num_envs < num_mini_batches or num_envs % num_mini_batches:
        raise ValueError("num_envs는 num_mini_batches로 나누어떨어져야 한다")
    indices = (torch.randperm(num_envs, device=device) if shuffle and num_mini_batches > 1 else
               torch.arange(num_envs, device=device))
    return list(indices.chunk(num_mini_batches))


def select_env_batch(batch: dict[str, torch.Tensor], resets: torch.Tensor,
                     h0: torch.Tensor, h0_ema: torch.Tensor, env_ids: torch.Tensor,
                     mask: torch.Tensor | None = None
                     ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """시간축은 통째로 두고 env 축만 선택한다. JEPA mask도 같은 env 순서를 따른다."""
    selected = {key: value.index_select(1, env_ids) for key, value in batch.items()}
    return (selected, resets.index_select(1, env_ids),
            h0.index_select(0, env_ids), h0_ema.index_select(0, env_ids),
            None if mask is None else mask.index_select(1, env_ids))


def realized_se2_displacement(start: torch.Tensor, end: torch.Tensor) -> torch.Tensor:
    """[... , x_w, y_w, yaw_w] 두 pose 사이의 시작 몸통 좌표계 SE(2) 변위."""
    if start.shape != end.shape or start.shape[-1] != 3:
        raise ValueError("start/end는 같은 [...,3] shape여야 한다")
    delta = end[..., :2] - start[..., :2]
    c, s = start[..., 2].cos(), start[..., 2].sin()
    yaw_delta = end[..., 2] - start[..., 2]
    return torch.stack((c * delta[..., 0] + s * delta[..., 1],
                        -s * delta[..., 0] + c * delta[..., 1],
                        torch.atan2(yaw_delta.sin(), yaw_delta.cos())), dim=-1)


def jepa_pair_tensors(context: torch.Tensor, target_context: torch.Tensor,
                      command: torch.Tensor, pose: torch.Tensor | None, horizon_steps: int,
                      target_mode: str, condition_source: str
                      ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """future는 시작 t, present_from_past는 도착 t를 기준으로 같은 (t-Δ,t) 쌍을 표기한다."""
    if target_mode not in ("future", "present_from_past"):
        raise ValueError("지원하지 않는 JEPA 목표 시점")
    if condition_source not in ("command", "realized_displacement"):
        raise ValueError("지원하지 않는 JEPA 조건")
    if horizon_steps < 1 or horizon_steps >= context.shape[0]:
        raise ValueError("JEPA horizon이 rollout 범위를 벗어난다")
    if context.shape != target_context.shape or command.shape[:2] != context.shape[:2]:
        raise ValueError("JEPA context/command/pose shape가 맞지 않는다")
    if condition_source == "realized_displacement" and (pose is None or pose.shape != command.shape):
        raise ValueError("실현 변위 조건에는 command와 같은 [T,N,3] pose가 필요하다")
    condition = (command[:-horizon_steps] if condition_source == "command" else
                 realized_se2_displacement(pose[:-horizon_steps], pose[horizon_steps:]))
    return context[:-horizon_steps], target_context[horizon_steps:], condition
