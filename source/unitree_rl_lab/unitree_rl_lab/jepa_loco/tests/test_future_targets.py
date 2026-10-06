"""명령 외삽·정합 mask·latent 손실의 shape와 경계 조건."""

import math

import pytest
import torch

from unitree_rl_lab.jepa_loco.models.future_targets import (
    extrapolate_command_pose,
    future_alignment_mask,
    masked_latent_mse,
)


def test_extrapolate_straight_turn_and_wrap():
    poses = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 0.0, math.pi - 0.1]])
    cmds = torch.tensor([[0.6, 0.0, 0.0], [1.0, 0.0, math.pi / 2], [0.0, 0.0, 0.2]])
    result = extrapolate_command_pose(poses, cmds, 1.0)
    assert result.shape == (3, 3)
    assert torch.allclose(result[0], torch.tensor([0.6, 0.0, 0.0]), atol=1e-6)
    assert torch.allclose(result[1], torch.tensor([1 + 2 / math.pi, 2 + 2 / math.pi, math.pi / 2]), atol=1e-6)
    assert result[2, 2].item() == pytest.approx(-math.pi + 0.1, abs=1e-6)


def test_extrapolate_horizon_shape_and_gradient():
    pose = torch.zeros(2, 3)
    command = torch.tensor([[0.2, 0.1, 0.0], [0.6, 0.0, 0.3]], requires_grad=True)
    future = extrapolate_command_pose(pose, command, torch.tensor([0.5, 1.0]))
    assert future.shape == (2, 3)
    future.sum().backward()
    assert torch.isfinite(command.grad).all()
    with pytest.raises(ValueError):
        extrapolate_command_pose(pose, command, torch.ones(3))


def test_alignment_mask_rejects_pose_command_episode_and_visibility():
    expected = torch.zeros(5, 3)
    actual = expected.clone()
    actual[1, 0] = 0.3
    actual[2, 2] = 0.3
    commands = torch.zeros(5, 3, 3)
    commands[3, 2, 0] = 0.3
    same_episode = torch.tensor([1, 1, 1, 1, 0], dtype=torch.bool)
    mask = future_alignment_mask(actual, expected, commands, position_tolerance_m=0.2,
                                 yaw_tolerance_rad=0.2, command_tolerance=0.2,
                                 same_episode=same_episode)
    assert mask.shape == (5,) and mask.tolist() == [True, False, False, False, False]
    visibility = torch.tensor([False, True, True, True, True])
    assert not future_alignment_mask(actual, expected, commands, position_tolerance_m=0.2,
                                     yaw_tolerance_rad=0.2, command_tolerance=0.2,
                                     same_episode=same_episode, visible=visibility).any()


def test_masked_mse_normalizes_by_valid_count_and_stops_target_gradient():
    pred = torch.tensor([[1.0, 2.0], [5.0, 6.0]], requires_grad=True)
    target = torch.zeros_like(pred, requires_grad=True)
    mask = torch.tensor([True, False])
    loss = masked_latent_mse(pred, target, mask)
    assert loss.item() == pytest.approx(2.5)
    loss.backward()
    assert pred.grad[0].abs().sum() > 0 and pred.grad[1].abs().sum() == 0
    assert target.grad is None
    pred.grad.zero_()
    empty = masked_latent_mse(pred, target, torch.zeros_like(mask))
    assert empty.item() == 0
    empty.backward()
    assert pred.grad.abs().sum() == 0
