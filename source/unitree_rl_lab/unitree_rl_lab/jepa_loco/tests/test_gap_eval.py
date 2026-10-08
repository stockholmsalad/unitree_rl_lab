"""고정 gap 지형의 함수·기하·ray 검증과 성공 상태 테스트."""

import pytest
import torch
import trimesh
import isaaclab.terrains as terrain_gen
from isaaclab.terrains.trimesh import mesh_terrains

from unitree_rl_lab.jepa_loco.envs.easy_start_terrain import EasyGapCfg
from unitree_rl_lab.jepa_loco.eval.gap import (
    fixed_width_gap_cfg, gap_failure_reason, gap_geometry, gap_success_step,
    validate_gap_origins, validate_gap_tiles,
)


def test_gap_geometry_and_original_function_from_easy_template():
    assert gap_geometry((8.0, 8.0), 3.0, 0.10) == pytest.approx((1.5, 1.6))
    assert gap_geometry((8.0, 8.0), 3.0, 0.30) == pytest.approx((1.5, 1.8))
    with pytest.raises(ValueError):
        gap_geometry((3.0, 3.0), 3.0, 0.10)
    template = EasyGapCfg(proportion=0.2, gap_width_range=(0.1, 0.3),
                          platform_width=3.0, num_rows=10, warmup_rows=2)
    fixed = fixed_width_gap_cfg(template, 0.15, 0.5)
    assert type(fixed) is terrain_gen.MeshGapTerrainCfg
    assert fixed.function is mesh_terrains.gap_terrain
    assert fixed.function is not template.function
    assert fixed.gap_width_range == (0.15, 0.15)
    assert fixed.platform_width == template.platform_width


def test_gap_ray_validation_rejects_flat_tile():
    cfg = terrain_gen.MeshGapTerrainCfg(proportion=1.0, size=(8.0, 8.0),
                                        gap_width_range=(0.2, 0.2), platform_width=3.0)
    meshes, origin = mesh_terrains.gap_terrain(0.5, cfg)
    mesh = trimesh.util.concatenate(meshes)
    origins = torch.from_numpy(origin).to(torch.float32).unsqueeze(0)
    rows = validate_gap_tiles(mesh, origins, torch.tensor([0.2]), cfg.size, cfg.platform_width,
                              ray_height_m=20.0, inner_inset_m=0.05, outer_outset_m=0.05,
                              tolerance_m=1.0e-4)
    assert len(rows) == 1
    assert rows[0]["inner_ground_hit"] and rows[0]["gap_miss"] and rows[0]["outer_ground_hit"]
    flat = trimesh.creation.box((8.0, 8.0, 1.0))
    flat.apply_translation((4.0, 4.0, -0.5))
    with pytest.raises(ValueError, match="실제 ray 기하"):
        validate_gap_tiles(flat, origins, torch.tensor([0.2]), cfg.size, cfg.platform_width,
                           ray_height_m=20.0, inner_inset_m=0.05, outer_outset_m=0.05,
                           tolerance_m=1.0e-4)
    validate_gap_origins(torch.tensor([0.0]), 1.0e-4)
    with pytest.raises(ValueError, match="origin z"):
        validate_gap_origins(torch.tensor([0.2]), 1.0e-4)


def test_gap_hold_requires_live_crossing_and_success_latches_after_done():
    outer = torch.tensor([1.7, 1.7])
    count = torch.zeros(2, dtype=torch.long)
    success = torch.zeros(2, dtype=torch.bool)
    for step in range(25):
        count, success = gap_success_step(torch.tensor([1.81, 1.81]), outer,
                                          torch.tensor([True, step < 24]), count, success,
                                          margin_m=0.10, hold_steps=25)
    assert count.shape == success.shape == (2,)
    assert count.tolist() == [25, 0]
    assert success.tolist() == [True, False]
    count, success = gap_success_step(torch.tensor([0.0, 0.0]), outer,
                                      torch.tensor([False, False]), count, success,
                                      margin_m=0.10, hold_steps=25)
    assert success.tolist() == [True, False]


def test_gap_failure_reasons():
    args = (1.5, 1.7, 0.15, 0.10)
    assert gap_failure_reason(False, True, 1.6, 1.6, 0.2, *args) == "gap_fall_termination"
    assert gap_failure_reason(False, False, 1.4, 1.4, 0.0, *args) == "gap_front_stall"
    assert gap_failure_reason(False, True, 1.4, 1.4, 0.0, *args) == "other"
    assert gap_failure_reason(False, True, 1.8, 1.8, 0.0, *args) == "other"
    assert gap_failure_reason(True, True, 2.0, 2.0, 0.2, *args) == "success"
