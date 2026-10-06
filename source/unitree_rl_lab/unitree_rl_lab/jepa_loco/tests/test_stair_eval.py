"""고정 계단 평가의 shape와 성공 판정 테스트."""

import pytest
import torch

from unitree_rl_lab.jepa_loco.eval.stairs import stair_geometry, stair_progress


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
