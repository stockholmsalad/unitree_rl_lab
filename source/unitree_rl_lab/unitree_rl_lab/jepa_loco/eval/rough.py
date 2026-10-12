"""고정 비정형 지형 평가의 원본 지형·성공 판정·메시 검사."""

from __future__ import annotations

import torch
import isaaclab.terrains as terrain_gen


def fixed_rough_cfg(template, max_height_m: float, proportion: float):
    """학습 rough의 샘플링 설정을 유지하고 높이 상한만 고정한다."""
    if max_height_m < template.noise_range[0] or proportion <= 0:
        raise ValueError("rough 높이 상한과 비율이 올바르지 않다")
    return terrain_gen.HfRandomUniformTerrainCfg(
        proportion=proportion, noise_range=(template.noise_range[0], max_height_m),
        noise_step=template.noise_step, border_width=template.border_width,
        horizontal_scale=template.horizontal_scale, vertical_scale=template.vertical_scale,
        slope_threshold=template.slope_threshold, downsampled_scale=template.downsampled_scale,
    )


def validate_rough_mesh_bounds(z_min: torch.Tensor, z_max: torch.Tensor,
                               min_height_m: float, max_heights_m: torch.Tensor,
                               tolerance_m: float) -> None:
    """높이 필드와 평평한 border를 포함한 실제 타일 z 범위를 검사한다."""
    if z_min.ndim != 1 or z_min.shape != z_max.shape or z_min.shape != max_heights_m.shape:
        raise ValueError("rough 타일 z 범위는 같은 [N] shape이어야 한다")
    if tolerance_m < 0 or min_height_m <= 0 or not torch.isfinite(z_min).all() or not torch.isfinite(z_max).all():
        raise ValueError("rough 타일 z 범위가 유효하지 않다")
    if (z_min < -tolerance_m).any() or (z_max > max_heights_m + tolerance_m).any():
        raise ValueError("rough 메시 높이가 설정된 noise_range를 벗어났다")
    if (z_max < min_height_m - tolerance_m).any():
        raise ValueError("rough 메시가 평지이거나 최소 noise 높이에 못 미친다")


def rough_success_step(forward_from_spawn: torch.Tensor, active: torch.Tensor,
                       consecutive: torch.Tensor, distance_m: float,
                       hold_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    if forward_from_spawn.shape != active.shape or forward_from_spawn.shape != consecutive.shape:
        raise ValueError("rough 성공 입력은 같은 [N] shape이어야 한다")
    if distance_m <= 0 or hold_steps < 1:
        raise ValueError("rough 성공 거리와 유지 스텝이 올바르지 않다")
    next_count = torch.where(active & (forward_from_spawn >= distance_m), consecutive + 1, 0)
    return next_count, next_count >= hold_steps


def rough_velocity_tracking_error(body_xy: torch.Tensor, command_xy: torch.Tensor) -> torch.Tensor:
    if body_xy.ndim != 2 or body_xy.shape != command_xy.shape or body_xy.shape[1] != 2:
        raise ValueError("몸통 속도와 명령 속도는 같은 [N,2] shape이어야 한다")
    return torch.linalg.vector_norm(body_xy - command_xy, dim=-1)
