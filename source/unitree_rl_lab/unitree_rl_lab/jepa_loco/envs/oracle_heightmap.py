"""특권 wide heightscan에서 명령 외삽 pose 주변의 future patch를 읽는다."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from unitree_rl_lab.jepa_loco.models.future_targets import extrapolate_command_pose


def sanitize_height_scan(values: torch.Tensor, clip_range: tuple[float, float]) -> torch.Tensor:
    """보간 전에 heightscan을 관측 clip 범위의 유한값으로 만든다."""
    lo, hi = clip_range
    if not (math.isfinite(lo) and math.isfinite(hi) and lo < hi):
        raise ValueError("heightscan clip_range는 유한한 (lo < hi)여야 한다")
    return torch.nan_to_num(values, nan=lo, posinf=hi, neginf=lo).clamp(lo, hi)


def sample_command_future_patch(wide_height: torch.Tensor, command: torch.Tensor, horizon_s: float,
                                wide_size: tuple[float, float], patch_size: tuple[float, float],
                                resolution: float) -> tuple[torch.Tensor, torch.Tensor]:
    """wide [N,R] → future patch [N,P], patch 전체가 wide 안에 있는지 [N].

    wide grid는 RayCaster ``ordering='yx'`` (x 바깥, y 안쪽), 출력 patch는
    기본 ``ordering='xy'`` (y 바깥, x 안쪽)로 만들어 current scanner와 순서를 맞춘다.
    """
    if wide_height.ndim != 2 or command.shape != (wide_height.shape[0], 3):
        raise ValueError("wide_height는 [N,R], command는 [N,3]이어야 한다")
    if resolution <= 0 or min(*wide_size, *patch_size) <= 0:
        raise ValueError("grid size와 resolution은 양수여야 한다")
    n_x = round(wide_size[0] / resolution) + 1
    n_y = round(wide_size[1] / resolution) + 1
    if wide_height.shape[1] != n_x * n_y:
        raise ValueError(f"wide grid 크기 {wide_height.shape[1]} != {n_x}×{n_y}")

    x = torch.arange(-patch_size[0] / 2, patch_size[0] / 2 + 1.0e-9,
                     resolution, device=wide_height.device, dtype=wide_height.dtype)
    y = torch.arange(-patch_size[1] / 2, patch_size[1] / 2 + 1.0e-9,
                     resolution, device=wide_height.device, dtype=wide_height.dtype)
    local_x, local_y = torch.meshgrid(x, y, indexing="xy")
    local_x, local_y = local_x.flatten(), local_y.flatten()

    zero_pose = torch.zeros_like(command)
    offset = extrapolate_command_pose(zero_pose, command, horizon_s)
    cos_yaw, sin_yaw = offset[:, 2].cos()[:, None], offset[:, 2].sin()[:, None]
    sample_x = offset[:, 0, None] + cos_yaw * local_x - sin_yaw * local_y
    sample_y = offset[:, 1, None] + sin_yaw * local_x + cos_yaw * local_y
    visible = ((sample_x.abs() <= wide_size[0] / 2) &
               (sample_y.abs() <= wide_size[1] / 2)).all(dim=1)

    # grid_sample의 마지막 좌표는 입력 W=y, H=x 순서다.
    grid = torch.stack((2 * sample_y / wide_size[1], 2 * sample_x / wide_size[0]), dim=-1)
    grid = grid[:, None, :, :]
    image = wide_height.reshape(wide_height.shape[0], 1, n_x, n_y)
    patch = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
    return patch[:, 0, 0, :], visible


def future_height_scan(env, wide_sensor_cfg, current_sensor_cfg, command_name: str,
                       horizon_s: float, offset: float = 0.5,
                       clip_range: tuple[float, float] = (-1.0, 1.0)) -> torch.Tensor:
    """IsaacLab 관측 항: 현재 월드 지형을 명령 외삽 pose 기준으로 읽는다."""
    wide = env.scene.sensors[wide_sensor_cfg.name]
    current = env.scene.sensors[current_sensor_cfg.name]
    if clip_range != env.cfg.observations.terrain_future.scan.clip:
        raise ValueError("terrain_future의 보간 전 clip_range와 관측 clip이 다르다")
    wide_pattern = wide.cfg.pattern_cfg
    current_pattern = current.cfg.pattern_cfg
    if wide_pattern.resolution != current_pattern.resolution or wide_pattern.ordering != "yx":
        raise ValueError("두 scanner의 해상도는 같고 wide scanner는 ordering='yx'여야 한다")
    values = wide.data.pos_w[:, 2, None] - wide.data.ray_hits_w[..., 2] - offset
    # Gap의 미충돌 광선은 hit_z=inf다. 보간 전에 유한한 값으로 바꿔야 grid_sample에서 NaN이 생기지 않는다.
    values = sanitize_height_scan(values, clip_range)
    patch, _ = sample_command_future_patch(values, env.command_manager.get_command(command_name),
                                            horizon_s, wide_pattern.size, current_pattern.size,
                                            wide_pattern.resolution)
    return patch
