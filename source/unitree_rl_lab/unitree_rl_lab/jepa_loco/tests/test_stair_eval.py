"""고정 계단 평가의 shape와 성공 판정 테스트."""

import pytest
import torch

from unitree_rl_lab.jepa_loco.eval.stairs import (
    fixed_height_stair_cfg, stair_geometry, stair_progress,
    unmet_success_conditions, validate_stair_origins,
)
from unitree_rl_lab.jepa_loco.envs.easy_start_terrain import EasyUpCfg


def test_stair_geometry_matches_generator():
    count, first_riser, edge = stair_geometry((8.0, 8.0), 1.0, 3.0, 0.3)
    assert count == 6
    assert first_riser == pytest.approx(1.2)
    assert edge == 3.0
    with pytest.raises(ValueError):
        stair_geometry((8.0, 8.0), 1.0, 3.0, 0.0)


def test_stair_progress_shape_and_gate():
    root = torch.tensor([[3.0, 0.0, 0.54], [3.0, 0.0, 0.54], [0.0, 3.0, 0.54], [0.0, 0.0, 0.54]])
    origin = torch.zeros_like(root)
    start_z = torch.zeros(4)
    heights = torch.tensor([0.09, 0.19, 0.09, 0.09])
    reached, progress = stair_progress(root, origin, start_z, heights, 6, 3.0, 0.9, 0.1)
    assert reached.shape == (4,) and progress.shape == (4,)
    assert reached.tolist() == [True, False, False, False]
    assert progress[0].item() == pytest.approx(6.0)


def test_fixed_eval_stair_drops_training_warmup_function():
    template = EasyUpCfg(
        proportion=0.25, step_height_range=(0.05, 0.20), step_width=0.3,
        platform_width=3.0, border_width=1.0, num_rows=10, warmup_rows=2,
    )
    fixed = fixed_height_stair_cfg(template, 0.09, 0.5)
    assert fixed.step_height_range == (0.09, 0.09)
    assert fixed.step_width == template.step_width
    assert fixed.function is not template.function
    assert not hasattr(fixed, "warmup_rows")
    fixed.size = (8.0, 8.0)
    for difficulty in (0.05, 0.95):
        _, origin = fixed.function(difficulty, fixed)
        assert origin[2] == pytest.approx(-7 * 0.09)


def test_terrain_origin_guard_rejects_flat_columns():
    heights = torch.tensor([0.05, 0.07])
    validate_stair_origins(torch.tensor([-0.35, -0.49]), heights, 6, 1.0e-4)
    with pytest.raises(ValueError, match="요청 높이와 다른 열"):
        validate_stair_origins(torch.tensor([-0.35, 0.0]), heights, 6, 1.0e-4)


def test_unmet_conditions_separate_simultaneous_and_hold():
    assert unmet_success_conditions(False, False, True, False) == ["body_rise"]
    assert unmet_success_conditions(False, True, False, False) == ["forward_x"]
    assert unmet_success_conditions(False, True, True, False) == ["simultaneous_conditions"]
    assert unmet_success_conditions(False, True, True, True) == ["hold_steps"]
    assert unmet_success_conditions(True, True, True, True) == []
