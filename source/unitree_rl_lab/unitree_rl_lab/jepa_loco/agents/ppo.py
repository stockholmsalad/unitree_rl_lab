"""PPO + latent 통계 로깅 (CLAUDE.md §5, §8: latent 표준편차 상시 기록)."""

from __future__ import annotations

import torch

from rsl_rl.algorithms import PPO


class LatentLoggingPPO(PPO):
    collapse_warn_std: float = 1e-3

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
