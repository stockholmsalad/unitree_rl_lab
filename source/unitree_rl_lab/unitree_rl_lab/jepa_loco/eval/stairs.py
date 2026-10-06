"""오르는 계단 평가의 기하와 판정 규칙."""

from __future__ import annotations

import torch
from isaaclab.utils import configclass


@configclass
class StairEvalCfg:
    low_step_height_m: float = 0.09
    high_step_height_m: float = 0.19
    envs_per_height: int = 16
    forward_command_mps: float = 0.6
    episode_steps: int = 1000
    hold_steps: int = 25
    top_height_fraction: float = 0.9
    top_position_margin_m: float = 0.1
    spawn_lateral_range_m: float = 0.25
    spawn_forward_m: float = 1.0
    seed: int = 42


def stair_geometry(size_xy: tuple[float, float], border_width: float, platform_width: float,
                   step_width: float) -> tuple[int, float]:
    """IsaacLab inverted_pyramid_stairs_terrain과 같은 단 수와 상단 경계 좌표."""
    if step_width <= 0:
        raise ValueError("step_width는 양수여야 한다")
    n_x = (size_xy[0] - 2 * border_width - platform_width) // (2 * step_width) + 1
    n_y = (size_xy[1] - 2 * border_width - platform_width) // (2 * step_width) + 1
    count = int(min(n_x, n_y))
    if count < 1:
        raise ValueError("계단을 하나 이상 생성할 공간이 필요하다")
    top_edge = min(size_xy) / 2 - border_width
    return count, top_edge


def stair_progress(root_xyz: torch.Tensor, origin_xyz: torch.Tensor, start_z: torch.Tensor,
                   step_heights: torch.Tensor, stair_count: int, top_edge: float,
                   height_fraction: float, position_margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    """상단 진입 여부 [N]와 몸통 상승량을 계단 단 수로 환산한 [N] 값을 반환한다."""
    gain = root_xyz[:, 2] - start_z
    forward = root_xyz[:, 0] - origin_xyz[:, 0]
    reached_top = (gain >= height_fraction * stair_count * step_heights) & (forward >= top_edge - position_margin)
    return reached_top, gain / step_heights
