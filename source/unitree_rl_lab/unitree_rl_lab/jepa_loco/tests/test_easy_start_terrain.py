"""학습 지형 난이도와 행별 평지 워밍업 검증."""

import pytest
import numpy as np

from unitree_rl_lab.jepa_loco.envs.easy_start_terrain import (
    EasyStartTerrainCfg, easy_start_level, make_easy_start_terrain, obstacle_value,
)
from unitree_rl_lab.jepa_loco.envs.depth_env_cfg import JEPA_TERRAINS_CFG


def test_warmup_levels_and_exact_endpoints():
    settings = EasyStartTerrainCfg()
    expected = [None, None, 0.05, 0.20]
    for level, value in zip((0, 1, 2, 9), expected):
        assert easy_start_level((level + 0.5) / settings.num_rows, settings.num_rows) == level
        height = obstacle_value(level, settings.num_rows, settings.warmup_rows, settings.step_height_range)
        if value is None:
            assert height is None
        else:
            assert height == pytest.approx(value)
    terrain = make_easy_start_terrain(JEPA_TERRAINS_CFG, settings)
    assert terrain.num_rows == 10
    assert [term.proportion for term in terrain.sub_terrains.values()] == [
        term.proportion for term in JEPA_TERRAINS_CFG.sub_terrains.values()
    ]
    assert terrain.sub_terrains["stairs_up"].step_height_range == (0.05, 0.20)
    assert terrain.sub_terrains["stairs_down"].step_height_range == (0.05, 0.20)
    assert terrain.sub_terrains["gap"].gap_width_range == (0.10, 0.30)
    for name in ("stairs_up", "stairs_down", "gap"):
        sub = terrain.sub_terrains[name]
        assert sub.warmup_rows == 2
        assert sub.num_rows == 10


def test_generated_warmup_mesh_is_flat_and_stair_height_scales():
    terrain = make_easy_start_terrain(JEPA_TERRAINS_CFG, EasyStartTerrainCfg())
    up = terrain.sub_terrains["stairs_up"]
    up.size = terrain.size
    spans = []
    for level in (0, 1, 2, 9):
        meshes, _ = up.function((level + 0.5) / terrain.num_rows, up)
        z = np.concatenate([mesh.vertices[:, 2] for mesh in meshes])
        spans.append(float(z.max() - z.min()))
    assert spans[0] == pytest.approx(0)
    assert spans[1] == pytest.approx(0)
    assert spans[2] > 0
    assert spans[3] / spans[2] == pytest.approx(0.20 / 0.05)


def test_bad_warmup_rejected():
    with pytest.raises(ValueError):
        make_easy_start_terrain(JEPA_TERRAINS_CFG, EasyStartTerrainCfg(warmup_rows=9))
