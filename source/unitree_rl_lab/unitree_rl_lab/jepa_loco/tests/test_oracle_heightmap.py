"""wide heightscan의 future pose 샘플링 순서·기하·가시성."""

import pytest
import torch

from unitree_rl_lab.jepa_loco.envs.oracle_heightmap import sample_command_future_patch


def _linear_wide(size=(4.0, 3.0), resolution=0.1):
    x = torch.arange(-size[0] / 2, size[0] / 2 + 1.0e-9, resolution)
    y = torch.arange(-size[1] / 2, size[1] / 2 + 1.0e-9, resolution)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return (2 * xx + 3 * yy).reshape(1, -1)


def test_future_patch_shape_order_and_translation():
    wide = _linear_wide().repeat(2, 1)
    command = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    patch, visible = sample_command_future_patch(wide, command, 1.0, (4.0, 3.0), (1.6, 1.0), 0.1)
    assert patch.shape == (2, 187) and visible.tolist() == [True, True]
    x = torch.arange(-0.8, 0.8 + 1.0e-9, 0.1)
    y = torch.arange(-0.5, 0.5 + 1.0e-9, 0.1)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    expected = (2 * xx + 3 * yy).flatten()
    assert torch.allclose(patch[0], expected, atol=1e-5)
    assert torch.allclose(patch[1], expected + 1.0, atol=1e-5)


def test_future_patch_visibility_and_shape_validation():
    wide = _linear_wide()
    far_command = torch.tensor([[2.0, 0.0, 0.0]])
    patch, visible = sample_command_future_patch(wide, far_command, 1.0, (4.0, 3.0), (1.6, 1.0), 0.1)
    assert patch.shape == (1, 187) and not visible.item()
    with pytest.raises(ValueError):
        sample_command_future_patch(wide[:, :10], far_command, 1.0, (4.0, 3.0), (1.6, 1.0), 0.1)


def test_future_patch_rotates_with_extrapolated_yaw():
    wide = _linear_wide()
    command = torch.tensor([[0.0, 0.0, torch.pi / 2]])
    patch, visible = sample_command_future_patch(wide, command, 1.0, (4.0, 3.0), (1.6, 1.0), 0.1)
    x = torch.arange(-0.8, 0.8 + 1.0e-9, 0.1)
    y = torch.arange(-0.5, 0.5 + 1.0e-9, 0.1)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    expected = (-2 * yy + 3 * xx).flatten()
    assert visible.item()
    assert torch.allclose(patch[0], expected, atol=1e-5)
