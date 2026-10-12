"""오르는 계단 평가의 기하와 판정 규칙."""

from __future__ import annotations

import torch
import isaaclab.terrains as terrain_gen
from isaaclab.utils import configclass


@configclass
class StairEvalCfg:
    down_step_heights_m: tuple[float, ...] = (0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15)
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
    center_spawn_forward_m: float = 0.0
    boundary_stop_forward_m: float = 3.5
    boundary_edge_margin_m: float = 0.5
    down_stair_ray_height_m: float = 20.0
    rough_max_heights_m: tuple[float, ...] = (0.04, 0.06, 0.08)
    rough_training_max_height_m: float = 0.06
    rough_success_forward_m: float = 3.0
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


@configclass
class StudentFixedEvalCfg:
    """Phase 3 student와 teacher 상한의 고정 평가 샘플 예산."""

    step_heights_m: tuple[float, ...] = (0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15)
    rough_max_heights_m: tuple[float, ...] = (0.04, 0.06, 0.08)
    envs_per_height: int = 128
    seed: int = 43
    spawn_forward_m: float = 0.0
    viewer_env_index: int = 0
    latent_noise_std: float = 0.0
    latent_noise_mode: str = "step"


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


def down_stair_geometry(size_xy: tuple[float, float], border_width: float,
                        platform_width: float, step_width: float) -> tuple[int, float, float]:
    """중앙 꼭대기에서 +x로 내려갈 때의 단 수·첫 하강 경계·마지막 단 바깥 경계."""
    return stair_geometry(size_xy, border_width, platform_width, step_width)


def down_stair_progress(root_xyz: torch.Tensor, origin_xyz: torch.Tensor, start_z: torch.Tensor,
                        step_heights: torch.Tensor, stair_count: int, outer_edge: float,
                        height_fraction: float, position_margin: float) -> tuple[torch.Tensor, torch.Tensor]:
    """출발 높이 기준 몸통 하강 단 수와 마지막 단 도달 여부."""
    drop = start_z - root_xyz[:, 2]
    forward = root_xyz[:, 0] - origin_xyz[:, 0]
    reached = (drop >= height_fraction * stair_count * step_heights) & (forward >= outer_edge - position_margin)
    return reached, drop / step_heights


def down_stair_failure_reason(success: bool, failed: bool, timed_out: bool,
                              max_forward_m: float, first_edge_m: float) -> str:
    if success:
        return "success"
    if failed:
        return "fall_termination"
    if timed_out and max_forward_m < first_edge_m:
        return "top_stall"
    return "other"


def update_swing_foot_peak(peak: torch.Tensor, foot_z: torch.Tensor, contact_force_w: torch.Tensor,
                           air_time: torch.Tensor, near_step: torch.Tensor,
                           contact_threshold: float, min_air_time: float) -> torch.Tensor:
    """첫 단 근처에서 실제 스윙 중인 발의 원점 기준 최고 z를 env별로 갱신한다."""
    if (peak.ndim != 1 or foot_z.ndim != 2 or foot_z.shape[0] != peak.shape[0]
            or contact_force_w.shape != (*foot_z.shape, 3) or air_time.shape != foot_z.shape
            or near_step.shape != peak.shape):
        raise ValueError("peak/near [N], foot_z/air_time [N,F], force [N,F,3] 필요")
    swing = ((contact_force_w.norm(dim=-1) < contact_threshold)
             & (air_time > min_air_time) & near_step[:, None])
    candidates = torch.where(swing, foot_z, -torch.inf).amax(dim=1)
    return torch.maximum(peak, candidates)


def fixed_height_stair_cfg(template, height_m: float, proportion: float):
    """학습용 평지 워밍업 함수를 제거한 평가 전용 고정 높이 계단."""
    if height_m <= 0 or proportion <= 0:
        raise ValueError("평가 계단 높이와 비율은 양수여야 한다")
    return terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
        proportion=proportion, step_height_range=(height_m, height_m),
        step_width=template.step_width, platform_width=template.platform_width,
        border_width=template.border_width, holes=template.holes,
    )


def fixed_height_down_stair_cfg(template, height_m: float, proportion: float):
    """학습용 EasyDownCfg를 버리고 IsaacLab 원본 피라미드 계단을 생성."""
    if height_m <= 0 or proportion <= 0:
        raise ValueError("평가 계단 높이와 비율은 양수여야 한다")
    return terrain_gen.MeshPyramidStairsTerrainCfg(
        proportion=proportion, step_height_range=(height_m, height_m),
        step_width=template.step_width, platform_width=template.platform_width,
        border_width=template.border_width, holes=template.holes,
    )


def validate_down_stair_origins(origin_z: torch.Tensor, heights: torch.Tensor,
                                stair_count: int, tolerance_m: float) -> None:
    if origin_z.shape != heights.shape or origin_z.ndim != 1 or stair_count < 1 or tolerance_m < 0:
        raise ValueError("내려가는 계단 원점 검사 입력이 올바르지 않다")
    expected = (stair_count + 1) * heights
    if not torch.isfinite(origin_z).all() or not torch.allclose(origin_z, expected, atol=tolerance_m, rtol=0):
        raise ValueError("내려가는 계단의 중앙 플랫폼 높이가 요청 높이와 다르다")


def validate_down_stair_ray_heights(center_z: torch.Tensor, last_step_z: torch.Tensor,
                                    heights: torch.Tensor, stair_count: int, tolerance_m: float) -> None:
    """실제 USD ray가 중앙 꼭대기와 바깥 마지막 단의 높이 차를 측정했는지 확인."""
    if center_z.shape != heights.shape or last_step_z.shape != heights.shape or center_z.ndim != 1:
        raise ValueError("내려가는 계단 ray 높이는 같은 [N] shape이어야 한다")
    if not torch.isfinite(center_z).all() or not torch.isfinite(last_step_z).all():
        raise ValueError("내려가는 계단 ray가 지면을 찾지 못했다")
    if not torch.allclose(center_z - last_step_z, stair_count * heights, atol=tolerance_m, rtol=0):
        raise ValueError("내려가는 계단의 실제 USD 높이 차가 요청한 단 수×높이와 다르다")


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


def boundary_stop_x(start_x: torch.Tensor, forward_m: float, tile_length_m: float,
                    edge_margin_m: float) -> torch.Tensor:
    """출발점 기준 정지 거리와 타일 끝 안전 여유 중 먼저 도달하는 x."""
    if forward_m <= 0 or tile_length_m <= 0 or not 0 < edge_margin_m < tile_length_m / 2:
        raise ValueError("경계 거리와 타일 끝 여유가 올바르지 않다")
    limit = torch.minimum(start_x + forward_m, torch.full_like(start_x, tile_length_m / 2 - edge_margin_m))
    if (limit <= start_x).any():
        raise ValueError("출발점이 평가 경계 정지 위치를 넘었다")
    return limit


def evaluation_stop_step(still_active: torch.Tensor, newly_successful: torch.Tensor,
                         boundary_hit: torch.Tensor, flat: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """성공 우선으로 이번 스텝의 성공·경계 종료와 다음 스텝 활성 마스크를 계산."""
    if still_active.shape != newly_successful.shape or still_active.shape != boundary_hit.shape:
        raise ValueError("평가 정지 마스크는 같은 shape이어야 한다")
    success = still_active & (newly_successful | (boundary_hit & flat))
    boundary = still_active & boundary_hit & ~success
    return success, boundary, still_active & ~success & ~boundary
