"""보상 항목별 스텝 극값을 PPO에 전달하는 환경."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv


def reward_step_extrema(reward_manager, step_dt: float) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """IsaacLab 0.54.4의 _step_reward는 dt를 나눈 값이므로 다시 곱한다."""
    values = reward_manager._step_reward * step_dt
    return (
        {name: values[:, i].amin() for i, name in enumerate(reward_manager._term_names)},
        {name: values[:, i].amax() for i, name in enumerate(reward_manager._term_names)},
    )


class JepaDiagnosticEnv(ManagerBasedRLEnv):
    def step(self, action):
        obs, reward, terminated, truncated, extras = super().step(action)
        lo, hi = reward_step_extrema(self.reward_manager, self.step_dt)
        extras["reward_step_min"] = lo
        extras["reward_step_max"] = hi
        return obs, reward, terminated, truncated, extras
