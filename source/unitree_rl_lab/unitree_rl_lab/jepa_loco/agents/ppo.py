"""PPO + latent 통계 로깅 (CLAUDE.md §5, §8) + 패딩 없는 순환 미니배치.

rsl_rl 5.0.1 의 ``recurrent_mini_batch_generator`` 는 trajectory 마다 rollout 길이로 패딩한 복사본을 만든다.
trajectory 수 = env 수 + 에피소드 종료 수라 depth(스텝당 28–57 KB)에서는 넘어짐이 많을수록 메모리가 커져
OOM 이 났다(2026-10-01: 전체 패딩 48.9 GB @iter 22, 미니배치별 패딩 17.5 GB @iter 30 — 512 env 에 약 6,500 trajectory).
→ 패딩 없이 env 시퀀스를 그대로 쓰고 에피소드 경계에서 상태를 초기화한다.
"""

from __future__ import annotations

from collections.abc import Generator

import torch

from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict


class ResetMaskRolloutStorage(RolloutStorage):
    """trajectory 패딩 없이 env 별 rollout 시퀀스 [T, B] 를 그대로 미니배치로 준다.

    - observations: 저장 버퍼의 env 구간 view (복사 없음) + ``traj_start`` [T, B, 1]
      (t ≥ 1 에서 직전 스텝이 done → 이 스텝에서 상태 0. t = 0 은 저장된 h0 를 그대로 쓴다)
    - hidden_states: rollout 첫 스텝에 저장된 상태 [layers, B, H]
    - masks: 전부 True → 비순환 critic 의 unpad 는 항등
    메모리는 넘어짐 횟수와 무관하다. 순환 모델은 ``traj_start`` 를 읽어 경계에서 상태를 초기화해야 한다.
    """

    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[RolloutStorage.Batch, None, None]:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        T = self.observations.batch_size[0]
        mini_batch_size = self.num_envs // num_mini_batches
        traj_start = torch.zeros_like(self.dones, dtype=torch.bool)
        traj_start[1:] = self.dones[:-1].bool()
        masks = torch.ones(T, mini_batch_size, dtype=torch.bool, device=self.device)

        def h0(saved, start, stop):
            if saved is None:
                return None
            hs = [h[0, :, start:stop].contiguous() for h in saved]  # [layers, B, H]
            return hs[0] if len(hs) == 1 else tuple(hs)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start, stop = i * mini_batch_size, (i + 1) * mini_batch_size
                view = self.observations[:, start:stop]
                obs = TensorDict({k: view[k] for k in view.keys()}, batch_size=[T, stop - start], device=self.device)
                obs["traj_start"] = traj_start[:, start:stop]
                yield RolloutStorage.Batch(
                    observations=obs,  # type: ignore
                    actions=self.actions[:, start:stop],
                    values=self.values[:, start:stop],
                    advantages=self.advantages[:, start:stop],
                    returns=self.returns[:, start:stop],
                    old_actions_log_prob=self.actions_log_prob[:, start:stop],
                    old_distribution_params=tuple(p[:, start:stop] for p in self.distribution_params),  # type: ignore
                    hidden_states=(
                        h0(self.saved_hidden_state_a, start, stop),
                        h0(self.saved_hidden_state_c, start, stop),
                    ),  # type: ignore
                    masks=masks,
                )


class LatentLoggingPPO(PPO):
    collapse_warn_std: float = 1e-3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage.__class__ = ResetMaskRolloutStorage

    def update(self) -> dict[str, float]:
        loss = super().update()
        z = getattr(self.actor, "last_z", None)
        if z is not None and z.shape[0] > 1:
            std = z.float().std(dim=0)
            loss["latent_std_mean"] = std.mean().item()
            loss["latent_std_min"] = std.min().item()
            loss["latent_dead_frac"] = (std < self.collapse_warn_std).float().mean().item()
            if loss["latent_dead_frac"] > 0.5:
                print(f"[WARN] latent 붕괴 의심: 차원의 {loss['latent_dead_frac']:.0%} 가 std < {self.collapse_warn_std}")
        return loss
