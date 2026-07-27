from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_lin_vel_xy",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.lin_vel_x = torch.clamp(
                torch.tensor(ranges.lin_vel_x, device=env.device) + delta_command,
                limit_ranges.lin_vel_x[0],
                limit_ranges.lin_vel_x[1],
            ).tolist()
            ranges.lin_vel_y = torch.clamp(
                torch.tensor(ranges.lin_vel_y, device=env.device) + delta_command,
                limit_ranges.lin_vel_y[0],
                limit_ranges.lin_vel_y[1],
            ).tolist()

    return torch.tensor(ranges.lin_vel_x[1], device=env.device)


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "track_ang_vel_z",
) -> torch.Tensor:
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges

    reward_term = env.reward_manager.get_term_cfg(reward_term_name)
    reward = torch.mean(env.reward_manager._episode_sums[reward_term_name][env_ids]) / env.max_episode_length_s

    if env.common_step_counter % env.max_episode_length == 0:
        if reward > reward_term.weight * 0.8:
            delta_command = torch.tensor([-0.1, 0.1], device=env.device)
            ranges.ang_vel_z = torch.clamp(
                torch.tensor(ranges.ang_vel_z, device=env.device) + delta_command,
                limit_ranges.ang_vel_z[0],
                limit_ranges.ang_vel_z[1],
            ).tolist()

    return torch.tensor(ranges.ang_vel_z[1], device=env.device)


def scripted_terrain_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg=None,
    steps_per_iter: int = 32,
    schedule: tuple[tuple[int, float], ...] = (
        (0, 0.10),
        (500, 0.40),
        (2000, 0.55),
        (4000, 0.70),
        (8000, 0.90),
        (12000, 1.00),
    ),
) -> torch.Tensor:
    """개방루프(open-loop) 지형 커리큘럼 — 난이도 상한을 '정책 성능'이 아니라 '학습 iteration'으로 올린다.

    표준 ``terrain_levels_vel`` 은 로봇이 걸은 거리로 레벨을 올리고 내린다(성능 의존). 그래서
    표현 헤드(A/B)나 seed 가 달라져 정책이 달라지면 **훈련 지형 분포까지 달라진다** = 내생적 교란.
    실측(2026-07): 동일 조건에서 Head A 3 seed 는 level 5.9~6.2 로 수렴한 반면, Head B 는 seed 별로
    0(실패)·5.0(정상)·4.8(지연) 로 요동쳤다. 이 상태에서 A/B 를 비교하면 표현 차이가 아니라
    훈련 분포 차이를 재게 된다.

    이 함수는 난이도 상한 C(t) 를 iteration 의 함수로 고정하고, reset 되는 env 를 [0, C(t)] 에
    균등 배정한다 → **모든 run 이 동일한 지형 분포**를 본다. 진행형 커리큘럼(쉬운→어려운)은
    유지하되 진행 속도가 정책과 무관해진다.

    Args:
        env_ids: 이번 스텝에 reset 되는 env (커리큘럼 매니저가 전달).
        steps_per_iter: iteration 당 policy step 수. ``num_steps_per_env`` 와 반드시 일치.
        schedule: (iteration, ceiling_fraction) 점들의 선형보간. fraction 1.0 = ``max_terrain_level``.
                  기본값은 baseline(Head A) 실측 등반 궤적에 맞춤.

    Returns:
        전체 env 의 평균 지형 레벨 (로깅용, 표준 term 과 동일한 반환 규약).
    """
    terrain = env.scene.terrain
    # 격자형(curriculum) 지형이 아니면 no-op (plane 등)
    if terrain.terrain_origins is None:
        return torch.zeros((), device=env.device)

    it = env.common_step_counter / steps_per_iter
    xs = [s[0] for s in schedule]
    ys = [s[1] for s in schedule]
    if it <= xs[0]:
        frac = ys[0]
    elif it >= xs[-1]:
        frac = ys[-1]
    else:
        frac = ys[-1]
        for i in range(1, len(xs)):
            if it <= xs[i]:
                t = (it - xs[i - 1]) / (xs[i] - xs[i - 1])
                frac = ys[i - 1] + t * (ys[i] - ys[i - 1])
                break

    # ceiling: [0, ceiling) 에서 균등 샘플. max_terrain_level = num_rows (유효 레벨 0..num_rows-1).
    ceiling = max(1, int(round(frac * terrain.max_terrain_level)))
    n = len(env_ids)
    terrain.terrain_levels[env_ids] = torch.randint(0, ceiling, (n,), device=terrain.device)
    terrain.env_origins[env_ids] = terrain.terrain_origins[
        terrain.terrain_levels[env_ids], terrain.terrain_types[env_ids]
    ]
    return torch.mean(terrain.terrain_levels.float())
