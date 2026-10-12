"""고정 비정형 지형의 높이 단계·메시·성공 판정 테스트."""

import pytest
import torch
import isaaclab.terrains as terrain_gen

from unitree_rl_lab.jepa_loco.eval.rough import (
    fixed_rough_cfg, rough_success_step, rough_velocity_tracking_error, validate_rough_mesh_bounds,
)


def test_fixed_rough_height_keeps_training_sampling():
    template = terrain_gen.HfRandomUniformTerrainCfg(
        proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01,
        border_width=0.25, horizontal_scale=0.1, vertical_scale=0.005,
    )
    cfg = fixed_rough_cfg(template, 0.08, 1.0)
    assert cfg.function.__name__ == "random_uniform_terrain"
    assert cfg.noise_range == (0.01, 0.08)
    assert cfg.noise_step == template.noise_step
    assert cfg.border_width == template.border_width
    assert cfg.horizontal_scale == template.horizontal_scale
    assert cfg.vertical_scale == template.vertical_scale
    assert cfg.downsampled_scale is None
    cfg.size = (8.0, 8.0)
    meshes, _ = cfg.function(0.9, cfg)
    assert 0.01 <= meshes[0].bounds[1, 2] <= 0.08 + 1e-6


def test_rough_mesh_bounds_and_hold_success():
    validate_rough_mesh_bounds(torch.tensor([0.0, 0.0]), torch.tensor([0.04, 0.08]),
                               0.01, torch.tensor([0.04, 0.08]), 1e-4)
    with pytest.raises(ValueError, match="noise_range"):
        validate_rough_mesh_bounds(torch.tensor([0.0]), torch.tensor([0.09]),
                                   0.01, torch.tensor([0.08]), 1e-4)
    count, success = rough_success_step(torch.tensor([3.0, 3.4, 3.5]),
                                        torch.tensor([True, False, True]),
                                        torch.tensor([24, 24, 0]), 3.0, 25)
    assert count.tolist() == [25, 0, 1]
    assert success.tolist() == [True, False, False]


def test_rough_body_frame_tracking_error_shape():
    err = rough_velocity_tracking_error(torch.tensor([[0.6, 0.0], [0.3, 0.4]]),
                                        torch.tensor([[0.6, 0.0], [0.0, 0.0]]))
    assert err.shape == (2,)
    assert err.tolist() == pytest.approx([0.0, 0.5])
