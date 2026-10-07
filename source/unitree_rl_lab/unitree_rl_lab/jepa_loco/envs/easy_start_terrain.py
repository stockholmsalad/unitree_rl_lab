"""Oracle 장애물 열의 처음 두 행을 평지로 만드는 학습 전용 지형."""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.trimesh import mesh_terrains
from isaaclab.utils import configclass


def easy_start_level(difficulty: float, num_rows: int) -> int:
    if num_rows < 2 or not 0 <= difficulty <= 1:
        raise ValueError("행 수와 난이도 범위가 올바르지 않다")
    return min(int(difficulty * num_rows), num_rows - 1)


def obstacle_fraction(level: int, num_rows: int, warmup_rows: int) -> float | None:
    if not 0 <= level < num_rows or not 0 <= warmup_rows < num_rows - 1:
        raise ValueError("행/워밍업 범위가 올바르지 않다")
    if level < warmup_rows:
        return None
    return (level - warmup_rows) / (num_rows - warmup_rows - 1)


def obstacle_value(level: int, num_rows: int, warmup_rows: int, value_range: tuple[float, float]) -> float | None:
    fraction = obstacle_fraction(level, num_rows, warmup_rows)
    return None if fraction is None else value_range[0] + fraction * (value_range[1] - value_range[0])


def _easy_mesh(difficulty, cfg, original):
    level = easy_start_level(difficulty, cfg.num_rows)
    fraction = obstacle_fraction(level, cfg.num_rows, cfg.warmup_rows)
    if fraction is None:
        return mesh_terrains.flat_terrain(difficulty, terrain_gen.MeshPlaneTerrainCfg(size=cfg.size, proportion=cfg.proportion))
    # IsaacLab 지형 함수는 difficulty로 범위를 보간한다. 행별 값을 정확히 고정한다.
    return original(fraction, cfg)


def easy_up(difficulty, cfg):
    return _easy_mesh(difficulty, cfg, mesh_terrains.inverted_pyramid_stairs_terrain)


def easy_down(difficulty, cfg):
    return _easy_mesh(difficulty, cfg, mesh_terrains.pyramid_stairs_terrain)


def easy_gap(difficulty, cfg):
    return _easy_mesh(difficulty, cfg, mesh_terrains.gap_terrain)


@configclass
class EasyStartTerrainCfg:
    num_rows: int = 10
    warmup_rows: int = 2
    step_height_range: tuple[float, float] = (0.05, 0.20)
    gap_width_range: tuple[float, float] = (0.10, 0.30)


@configclass
class EasyUpCfg(terrain_gen.MeshInvertedPyramidStairsTerrainCfg):
    function = easy_up
    num_rows: int = 10
    warmup_rows: int = 2


@configclass
class EasyDownCfg(terrain_gen.MeshPyramidStairsTerrainCfg):
    function = easy_down
    num_rows: int = 10
    warmup_rows: int = 2


@configclass
class EasyGapCfg(terrain_gen.MeshGapTerrainCfg):
    function = easy_gap
    num_rows: int = 10
    warmup_rows: int = 2


def make_easy_start_terrain(base, settings: EasyStartTerrainCfg):
    if settings.num_rows < 3 or not 0 <= settings.warmup_rows < settings.num_rows - 1:
        raise ValueError("워밍업 뒤에 최소 두 개 장애물 행이 필요하다")
    if settings.step_height_range[0] <= 0 or settings.step_height_range[1] < settings.step_height_range[0]:
        raise ValueError("계단 높이 범위가 올바르지 않다")
    if settings.gap_width_range[0] <= 0 or settings.gap_width_range[1] < settings.gap_width_range[0]:
        raise ValueError("gap 폭 범위가 올바르지 않다")
    result = base.copy()
    result.num_rows = settings.num_rows
    up, down, gap = (result.sub_terrains[name] for name in ("stairs_up", "stairs_down", "gap"))
    common = {"num_rows": settings.num_rows, "warmup_rows": settings.warmup_rows}
    result.sub_terrains["stairs_up"] = EasyUpCfg(
        proportion=up.proportion, step_height_range=settings.step_height_range,
        step_width=up.step_width, platform_width=up.platform_width, border_width=up.border_width,
        holes=up.holes, **common,
    )
    result.sub_terrains["stairs_down"] = EasyDownCfg(
        proportion=down.proportion, step_height_range=settings.step_height_range,
        step_width=down.step_width, platform_width=down.platform_width, border_width=down.border_width,
        holes=down.holes, **common,
    )
    result.sub_terrains["gap"] = EasyGapCfg(
        proportion=gap.proportion, gap_width_range=settings.gap_width_range,
        platform_width=gap.platform_width, **common,
    )
    return result
