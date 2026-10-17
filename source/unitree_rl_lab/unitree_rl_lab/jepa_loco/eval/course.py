"""평가 전용 직선 복합 지형과 판정의 순수 계산."""

from __future__ import annotations

import math

import numpy as np
import trimesh
from isaaclab.terrains.height_field.utils import convert_height_field_to_mesh
from isaaclab.terrains.height_field.hf_terrains import random_uniform_terrain
from isaaclab.terrains.height_field.hf_terrains_cfg import HfRandomUniformTerrainCfg
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.utils import configclass


SECTION_NAMES = ("start_flat", "ramp", "upper_flat", "stairs_up", "top_flat",
                 "stairs_down", "lower_flat", "rough", "finish_flat")


@configclass
class CourseEvalCfg:
    step_heights_m: tuple[float, ...] = (0.10, 0.12, 0.14)
    envs_per_height: int = 128
    seed: int = 43
    start_flat_m: float = 2.0
    ramp_m: float = 2.0
    upper_flat_m: float = 1.5
    stair_count: int = 4
    step_width_m: float = 0.3
    top_flat_m: float = 1.5
    lower_flat_m: float = 1.5
    rough_m: float = 3.0
    finish_flat_m: float = 1.5
    tile_margin_m: float = 0.6
    width_m: float = 4.0
    generator_border_width_m: float = 1.0
    spawn_from_tile_start_m: float = 1.0
    spawn_lateral_range_m: float = 0.25
    slope_deg: float = 10.0
    rough_noise_range_m: tuple[float, float] = (0.01, 0.06)
    rough_noise_step_m: float = 0.01
    rough_border_width_m: float = 0.25
    rough_downsampled_scale_m: float | None = None
    horizontal_scale_m: float = 0.1
    vertical_scale_m: float = 0.005
    slope_threshold: float = 0.75
    forward_command_mps: float = 0.6
    heading_hold: bool = True
    heading_gain: float = 1.0
    heading_max_yaw_rate: float = 0.3
    episode_steps: int = 1750
    hold_steps: int = 25
    finish_inset_m: float = 0.5
    off_course_margin_m: float = 0.3
    stationary_command_threshold_mps: float = 0.3
    stationary_body_threshold_mps: float = 0.1
    profile_tolerance_m: float = 0.015
    profile_ray_height_m: float = 20.0
    profile_sample_spacing_m: float = 0.1
    viewer_env_index: int = 0
    viewer_eye: tuple[float, float, float] = (-2.0, 3.0, 1.8)
    viewer_lookat: tuple[float, float, float] = (1.0, 0.0, 0.3)


def section_edges(cfg: CourseEvalCfg) -> tuple[float, ...]:
    lengths = (cfg.start_flat_m, cfg.ramp_m, cfg.upper_flat_m,
               cfg.stair_count * cfg.step_width_m, cfg.top_flat_m,
               cfg.stair_count * cfg.step_width_m, cfg.lower_flat_m,
               cfg.rough_m, cfg.finish_flat_m)
    return tuple(np.cumsum(lengths).tolist())


def section_pass_edges(cfg: CourseEvalCfg) -> tuple[float, ...]:
    """마지막 구간은 전체 타일 끝 대신 도착 평지의 성공 지점에서 통과한다."""
    edges = section_edges(cfg)
    return (*edges[:-1], edges[7] + cfg.finish_inset_m)


def section_index(x_from_tile_start_m: float, cfg: CourseEvalCfg) -> int:
    return min(int(np.searchsorted(section_edges(cfg), x_from_tile_start_m, side="right")), len(SECTION_NAMES) - 1)


def designed_height(x: float | np.ndarray, height: float, cfg: CourseEvalCfg) -> np.ndarray:
    """계단 경계에서는 오른쪽 디딤면 높이를 사용한다."""
    x = np.asarray(x, dtype=float)
    e = section_edges(cfg)
    ramp_height = cfg.ramp_m * math.tan(math.radians(cfg.slope_deg))
    up = np.clip(np.floor((x - e[2]) / cfg.step_width_m + 1e-9) + 1, 0, cfg.stair_count)
    down = np.clip(np.floor((x - e[4]) / cfg.step_width_m + 1e-9) + 1, 0, cfg.stair_count)
    return np.where(x < e[0], 0.0,
                    np.where(x < e[1], (x - e[0]) * math.tan(math.radians(cfg.slope_deg)),
                             np.where(x < e[2], ramp_height,
                                      np.where(x < e[3], ramp_height + up * height,
                                               np.where(x < e[4], ramp_height + cfg.stair_count * height,
                                                        np.where(x < e[5], ramp_height + (cfg.stair_count - down) * height,
                                                                 ramp_height))))))


def heading_yaw_rate(yaw: float, cfg: CourseEvalCfg) -> float:
    if not cfg.heading_hold:
        return 0.0
    wrapped = math.atan2(math.sin(yaw), math.cos(yaw))
    return float(np.clip(-cfg.heading_gain * wrapped, -cfg.heading_max_yaw_rate, cfg.heading_max_yaw_rate))


def course_outcome(success: bool, off_course: bool, done: bool, time_out: bool, horizon: bool) -> str | None:
    if off_course:
        return "off_course"
    if done:
        return "stall" if time_out else "fall"
    if success:
        return "success"
    if horizon:
        return "stall"
    return None


def course_terrain(_difficulty: float, cfg: CourseTerrainCfg) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """단일 heightfield에 경사·수직 계단·학습 규칙의 rough를 조립한다."""
    c = cfg.course
    nx = round(cfg.size[0] / c.horizontal_scale_m) + 1
    ny = round(cfg.size[1] / c.horizontal_scale_m) + 1
    x = np.arange(nx) * c.horizontal_scale_m
    base = designed_height(x, cfg.step_height_m, c)
    field = np.broadcast_to(np.rint(base[:, None] / c.vertical_scale_m), (nx, ny)).copy().astype(np.int16)
    rough_start, rough_end = section_edges(c)[6:8]
    lo = round(rough_start / c.horizontal_scale_m)
    hi = round(rough_end / c.horizontal_scale_m)
    # IsaacLab height_field_to_mesh와 같은 border_pixels=int(width/scale)+1.
    pad = int(c.rough_border_width_m / c.horizontal_scale_m) + 1
    inner_nx, inner_ny = hi - lo + 1 - 2 * pad, ny - 2 * pad
    if inner_nx < 4 or inner_ny < 4:
        raise ValueError("rough 내부는 보간을 위해 각 축에 4개 이상 격자가 필요하다")
    rough_cfg = HfRandomUniformTerrainCfg(
        noise_range=c.rough_noise_range_m, noise_step=c.rough_noise_step_m,
        border_width=0.0, downsampled_scale=c.rough_downsampled_scale_m,
        size=(inner_nx * c.horizontal_scale_m, inner_ny * c.horizontal_scale_m),
        horizontal_scale=c.horizontal_scale_m, vertical_scale=c.vertical_scale_m,
    )
    rough = random_uniform_terrain.__wrapped__(0.0, rough_cfg)
    field[lo + pad:hi - pad + 1, pad:ny - pad] += rough
    vertices, faces = convert_height_field_to_mesh(
        field, c.horizontal_scale_m, c.vertical_scale_m, c.slope_threshold,
    )
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    # TerrainGenerator가 중심으로 옮기므로 x/y는 타일 중앙, z는 출발 평지 0.
    return [mesh], np.array([cfg.size[0] / 2, cfg.size[1] / 2, 0.0])


@configclass
class CourseTerrainCfg(SubTerrainBaseCfg):
    function = course_terrain
    course: CourseEvalCfg = CourseEvalCfg()
    step_height_m: float = 0.10
