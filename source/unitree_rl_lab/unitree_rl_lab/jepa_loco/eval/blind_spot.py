"""Teacher 높이맵 점의 D435i 현재 프레임 가시성 계산."""

from __future__ import annotations

import math

import numpy as np


def height_patch_xy(size_xy: tuple[float, float], resolution: float) -> np.ndarray:
    """IsaacLab GridPatternCfg(ordering='xy')의 187점과 같은 순서로 [P,2]."""
    if resolution <= 0 or min(size_xy) <= 0:
        raise ValueError("높이맵 크기와 해상도는 양수여야 한다")
    x = np.arange(-size_xy[0] / 2, size_xy[0] / 2 + resolution / 2, resolution)
    y = np.arange(-size_xy[1] / 2, size_xy[1] / 2 + resolution / 2, resolution)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return np.column_stack((xx.ravel(), yy.ravel()))


def project_base_points(points_b: np.ndarray, camera_origin_b: tuple[float, float, float],
                        pitch_down_deg: float, intrinsic: tuple[float, float, float, float],
                        image_wh: tuple[int, int], near_m: float, far_m: float
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """몸통 좌표 점을 카메라 +X 전방 영상으로 투영하고 FOV·거리 유효성 반환."""
    if points_b.ndim != 2 or points_b.shape[1] != 3:
        raise ValueError("points_b는 [P,3]이어야 한다")
    if near_m <= 0 or far_m <= near_m:
        raise ValueError("카메라 유효 거리 범위가 올바르지 않다")
    dx, dy, dz = (points_b - np.asarray(camera_origin_b)).T
    theta = math.radians(pitch_down_deg)
    axial = math.cos(theta) * dx - math.sin(theta) * dz
    up = math.sin(theta) * dx + math.cos(theta) * dz
    safe_axial = np.where(axial > 0, axial, 1.0)
    fx, fy, cx, cy = intrinsic
    u = cx - fx * dy / safe_axial
    v = cy - fy * up / safe_axial
    width, height = image_wh
    valid = ((axial >= near_m) & (axial <= far_m) &
             (u >= 0) & (u < width) & (v >= 0) & (v < height))
    return u, v, axial, valid


def visible_in_depth(points_b: np.ndarray, depth_m: np.ndarray,
                     camera_origin_b: tuple[float, float, float], pitch_down_deg: float,
                     intrinsic: tuple[float, float, float, float], near_m: float,
                     far_m: float, occlusion_tolerance_m: float) -> tuple[np.ndarray, np.ndarray]:
    """유효 FOV와 렌더 depth 일치 여부를 각각 [P] mask로 반환."""
    if depth_m.ndim != 2 or occlusion_tolerance_m < 0:
        raise ValueError("depth_m은 [H,W], 가림 허용오차는 0 이상이어야 한다")
    u, v, axial, geometric = project_base_points(
        points_b, camera_origin_b, pitch_down_deg, intrinsic,
        (depth_m.shape[1], depth_m.shape[0]), near_m, far_m,
    )
    pixel_u = np.clip(np.rint(u).astype(int), 0, depth_m.shape[1] - 1)
    pixel_v = np.clip(np.rint(v).astype(int), 0, depth_m.shape[0] - 1)
    sampled = depth_m[pixel_v, pixel_u]
    visible = geometric & np.isfinite(sampled) & (sampled >= near_m) & (
        np.abs(sampled - axial) <= occlusion_tolerance_m
    )
    return geometric, visible
