"""PPO + latent 통계 로깅 (CLAUDE.md §5, §8) + 미니배치 단위 trajectory 패딩.

rsl_rl 5.0.1 의 ``recurrent_mini_batch_generator`` 는 rollout 전체를 trajectory 로 잘라 **한 번에** 패딩한다.
depth 관측은 스텝당 57 KB 이고 trajectory 수는 env 수 + 에피소드 종료 수라, 2048 env 에서 종료가 늘면
48 GB 단일 할당으로 OOM 이 났다(2026-10-01, iter 22). 미니배치(env 구간)마다 따로 패딩하면 같은 배치를
최대 메모리 약 1/num_mini_batches 로 만든다. 패딩 길이는 항상 rollout 길이라 결과가 원본과 동일하다.
"""

from __future__ import annotations

from collections.abc import Generator

import torch

from rsl_rl.algorithms import PPO
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import split_and_pad_trajectories


class SlicedPadRolloutStorage(RolloutStorage):
    def recurrent_mini_batch_generator(
        self, num_mini_batches: int, num_epochs: int = 8
    ) -> Generator[RolloutStorage.Batch, None, None]:
        if self.training_type != "rl":
            raise ValueError("This function is only available for reinforcement learning training.")
        mini_batch_size = self.num_envs // num_mini_batches
        dones = self.dones.squeeze(-1)
        last_was_done = torch.zeros_like(dones, dtype=torch.bool)
        last_was_done[1:] = dones[:-1]
        last_was_done[0] = True
        starts = last_was_done.permute(1, 0)  # [N, T]

        def hidden(saved, start, stop):
            if saved is None:
                return None
            hs = [h.permute(2, 0, 1, 3)[start:stop][starts[start:stop]].transpose(1, 0).contiguous() for h in saved]
            return hs[0] if len(hs) == 1 else hs

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start, stop = i * mini_batch_size, (i + 1) * mini_batch_size
                padded, masks = split_and_pad_trajectories(self.observations[:, start:stop], self.dones[:, start:stop])
                yield RolloutStorage.Batch(
                    observations=padded,  # type: ignore
                    actions=self.actions[:, start:stop],
                    values=self.values[:, start:stop],
                    advantages=self.advantages[:, start:stop],
                    returns=self.returns[:, start:stop],
                    old_actions_log_prob=self.actions_log_prob[:, start:stop],
                    old_distribution_params=tuple(p[:, start:stop] for p in self.distribution_params),  # type: ignore
                    hidden_states=(
                        hidden(self.saved_hidden_state_a, start, stop),
                        hidden(self.saved_hidden_state_c, start, stop),
                    ),  # type: ignore
                    masks=masks,
                )
                del padded


class LatentLoggingPPO(PPO):
    collapse_warn_std: float = 1e-3

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.storage.__class__ = SlicedPadRolloutStorage

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
