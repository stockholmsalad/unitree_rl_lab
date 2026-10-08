"""오르는 계단 평가의 기하와 판정 규칙."""

from __future__ import annotations

import torch
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass


@configclass
class StairEvalCfg:
    low_step_height_m: float = 0.09
    high_step_height_m: float = 0.19
    low_gap_width_m: float = 0.10
    high_gap_width_m: float = 0.15
    envs_per_height: int = 16
    forward_command_mps: float = 0.6
    episode_steps: int = 1000
    hold_steps: int = 25
    top_height_fraction: float = 0.9
    top_position_margin_m: float = 0.1
    spawn_lateral_range_m: float = 0.25
    spawn_forward_m: float = 1.0
    gap_spawn_forward_m: float = 0.0
    gap_probe_ray_height_m: float = 20.0
    gap_probe_inner_inset_m: float = 0.05
    gap_probe_outer_outset_m: float = 0.05
    gap_fall_drop_m: float = 0.15
    gap_failure_margin_m: float = 0.10
    seed: int = 42
    trace_interval_steps: int = 100
    settle_steps: int = 50
    contact_probe_margin_m: float = 0.4
    contact_probe_foot_x_margin_m: float = 0.02
    contact_probe_force_threshold_n: float = 1.0
    contact_probe_steps: int = 4
    terrain_height_tolerance_m: float = 1.0e-4
    viewer_eye: tuple[float, float, float] = (-2.0, 2.5, 1.2)
    viewer_lookat: tuple[float, float, float] = (1.2, 0.0, 0.3)
    viewer_env_index: int = 0


def stair_geometry(size_xy: tuple[float, float], border_width: float, platform_width: float,
                   step_width: float) -> tuple[int, float, float]:
    """IsaacLab inverted_pyramid_stairs_terrain과 같은 단 수, 첫 단, 상단 경계 좌표."""
    if step_width <= 0:
        raise ValueError("step_width는 양수여야 한다")
    n_x = (size_xy[0] - 2 * border_width - platform_width) // (2 * step_width) + 1
    n_y = (size_xy[1] - 2 * border_width - platform_width) // (2 * step_width) + 1
    count = int(min(n_x, n_y))
    if count < 1:
        raise ValueError("계단을 하나 이상 생성할 공간이 필요하다")
    top_edge = min(size_xy) / 2 - border_width
    first_riser = top_edge - count * step_width
    return count, first_riser, top_edge


def stair_progress(root_xyz: torch.Tensor, origin_xyz: torch.Tensor, start_z: torch.Tensor,
                   step_heights: torch.Tensor, stair_count: int, top_edge: float,
                   height_fraction: float, position_margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    """상단 진입 여부 [N]와 몸통 상승량을 계단 단 수로 환산한 [N] 값을 반환한다."""
    gain = root_xyz[:, 2] - start_z
    forward = root_xyz[:, 0] - origin_xyz[:, 0]
    reached_top = (gain >= height_fraction * stair_count * step_heights) & (forward >= top_edge - position_margin)
    return reached_top, gain / step_heights


def fixed_height_stair_cfg(template, height_m: float, proportion: float):
    """학습용 평지 워밍업 함수를 제거한 평가 전용 고정 높이 계단."""
    if height_m <= 0 or proportion <= 0:
        raise ValueError("평가 계단 높이와 비율은 양수여야 한다")
    return terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        proportion=proportion, step_height_range=(height_m, height_m),
        step_width=template.step_width, platform_width=template.platform_width,
        border_width=template.border_width, holes=template.holes,
    )


def validate_stair_origins(origin_z: torch.Tensor, heights: torch.Tensor,
                           stair_count: int, tolerance_m: float) -> None:
    """IsaacLab 역피라미드 계단의 중앙 시작면이 실제 요청 높이인지 검사."""
    if origin_z.shape != heights.shape or origin_z.ndim != 1:
        raise ValueError("origin_z와 heights는 같은 [N] shape이어야 한다")
    if stair_count < 1 or tolerance_m < 0:
        raise ValueError("단 수와 허용오차가 올바르지 않다")
    expected_z = -(stair_count + 1) * heights
    if not torch.allclose(origin_z, expected_z, atol=tolerance_m, rtol=0):
        bad = (~torch.isclose(origin_z, expected_z, atol=tolerance_m, rtol=0)).nonzero(as_tuple=True)[0]
        raise ValueError(f"평가 지형에 요청 높이와 다른 열이 있다: {bad[:10].tolist()}")


def unmet_success_conditions(success: bool, height_ever: bool,
                             position_ever: bool, reached_ever: bool) -> list[str]:
    """단독 조건 충족, 동시 충족, 유지 시간 부족을 구분한다."""
    if success:
        return []
    missing = []
    if not height_ever:
        missing.append("body_rise")
    if not position_ever:
        missing.append("forward_x")
    if height_ever and position_ever:
        missing.append("hold_steps" if reached_ever else "simultaneous_conditions")
    return missing
