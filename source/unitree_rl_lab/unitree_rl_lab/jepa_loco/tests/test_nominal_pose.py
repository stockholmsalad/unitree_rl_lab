"""Oracle 기본 자세 보상 두 항의 순수 함수 테스트."""

import torch

from unitree_rl_lab.jepa_loco.envs.rewards import nominal_pose_penalty_values


def penalty(q, default, command, velocity, group):
    return nominal_pose_penalty_values(
        q, default, command, velocity, group=group,
        stand_still_scale=5.0, velocity_threshold=0.3,
        normalization="command_xy_sq_floor", normalization_floor=1.0,
    )


def test_zero_error_shape_and_finiteness():
    q = torch.zeros(3, 8)
    command = torch.tensor([[0.6, 0.0, 0.0]]).repeat(3, 1)
    velocity = torch.zeros(3, 3)
    for group in ("hip", "thigh_calf"):
        value = penalty(q, q.clone(), command, velocity, group)
        assert value.shape == (3,)
        assert torch.equal(value, torch.zeros(3))
        assert torch.isfinite(value).all()


def test_thigh_calf_command_normalization_and_hip_only_error():
    q = torch.zeros(3, 8)
    default = q.clone()
    q[:, 0] = 0.2
    command = torch.tensor([[0.6, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    velocity = torch.zeros(3, 3)
    value = penalty(q, default, command, velocity, "thigh_calf")
    assert torch.allclose(value, torch.tensor([0.04, 0.01, 0.2]))
    # Hip deviation is selected separately; thigh/calf input at default stays zero.
    assert torch.equal(penalty(default, default, command, velocity, "thigh_calf"), torch.zeros(3))
    hip = penalty(q[:, :1], default[:, :1], command, velocity, "hip")
    assert torch.allclose(hip, torch.tensor([0.2, 0.2, 1.0]))


def test_original_stillness_rule_uses_full_command_and_body_xy_speed():
    q = torch.tensor([[0.2], [0.2], [0.2]])
    default = torch.zeros_like(q)
    command = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [0.0, 0.0, 0.0]])
    velocity = torch.tensor([[0.3, 0.0, 0.0], [0.0, 0.0, 0.0], [0.31, 0.0, 0.0]])
    hip = penalty(q, default, command, velocity, "hip")
    thigh = penalty(q, default, command, velocity, "thigh_calf")
    assert torch.allclose(hip, torch.tensor([1.0, 0.2, 0.2]))
    assert torch.allclose(thigh, torch.tensor([0.2, 0.04, 0.04]))
