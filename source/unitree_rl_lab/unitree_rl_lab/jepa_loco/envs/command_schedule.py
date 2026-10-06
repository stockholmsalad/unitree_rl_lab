"""학습 iteration에만 의존하는 Go2 속도 명령 범위 스케줄."""

from __future__ import annotations

from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


def scheduled_ranges(start, end, step: int, ramp_steps: int) -> tuple[tuple[float, float], ...]:
    if ramp_steps <= 0:
        raise ValueError("ramp_steps는 양수여야 한다")
    alpha = min(max(step / ramp_steps, 0.0), 1.0)
    return tuple(tuple(a + alpha * (b - a) for a, b in zip(s, e)) for s, e in zip(start, end))


class ScheduledVelocityCommand(UniformVelocityCommand):
    """명령을 재샘플하기 직전에 전역 제어 스텝으로 범위를 갱신한다."""

    def _resample_command(self, env_ids):
        cfg = self.cfg
        initial = (cfg.initial_ranges.lin_vel_x, cfg.initial_ranges.lin_vel_y, cfg.initial_ranges.ang_vel_z)
        final = (cfg.final_ranges.lin_vel_x, cfg.final_ranges.lin_vel_y, cfg.final_ranges.ang_vel_z)
        x, y, yaw = scheduled_ranges(
            initial, final, self._env.common_step_counter, cfg.ramp_iterations * cfg.steps_per_iteration
        )
        cfg.ranges.lin_vel_x, cfg.ranges.lin_vel_y, cfg.ranges.ang_vel_z = x, y, yaw
        super()._resample_command(env_ids)


@configclass
class ScheduledVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = ScheduledVelocityCommand
    initial_ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
    )
    final_ranges: UniformVelocityCommandCfg.Ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.3, 2.0), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
    )
    ramp_iterations: int = 500
    steps_per_iteration: int = 100
