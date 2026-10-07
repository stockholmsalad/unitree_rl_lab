"""지형 기준 스윙 발 높이 보상의 순수 텐서 테스트."""

import torch

from unitree_rl_lab.jepa_loco.envs.rewards import swing_foot_clearance_values


def inputs():
    foot = torch.zeros(2, 4, 3)
    foot[:, :, 0] = torch.tensor([-0.3, 0.3, -0.3, 0.3])
    foot[:, :, 1] = torch.tensor([-0.2, -0.2, 0.2, 0.2])
    foot[:, :, 2] = 0.122
    ray = foot.clone()
    ray[:, :, 2] = 0.0
    force = torch.zeros_like(foot)
    air = torch.full((2, 4), 0.1)
    command = torch.tensor([[0.6, 0.0, 0.0]]).repeat(2, 1)
    body_vel = torch.tensor([[0.2, 0.0, 0.0]]).repeat(2, 1)
    return foot, ray, force, air, command, body_vel


def evaluate(values):
    return swing_foot_clearance_values(
        *values, target_clearance=0.1, foot_radius=0.022, radius=0.1,
        contact_threshold=1.0, min_air_time=0.05, v_gate=0.2,
        command_threshold=0.1,
    )


def test_shape_saturation_stance_and_below_ground():
    values = inputs()
    reward, clearance, mask = evaluate(values)
    assert reward.shape == (2,) and clearance.shape == mask.shape == (2, 4)
    assert torch.allclose(reward, torch.ones(2))
    values[0][:, 0, 2] += 0.1  # target 두 배도 발별 상한 1
    assert torch.allclose(evaluate(values)[0], torch.ones(2))
    values[2][:, 0, 2] = 2.0  # 접촉 중인 발은 제외
    assert torch.allclose(evaluate(values)[0], torch.full((2,), 0.75))
    values[0][:, 1, 2] = 0.0  # clearance <= 0
    assert torch.allclose(evaluate(values)[0], torch.full((2,), 0.5))


def test_terrain_relative_height_and_motion_gate():
    values = inputs()
    baseline = evaluate(values)[0]
    values[0][:, :, 2] += 0.15
    values[1][:, :, 2] += 0.15
    assert torch.allclose(evaluate(values)[0], baseline)
    values[4][:, 0] = 0.05
    assert torch.equal(evaluate(values)[0], torch.zeros(2))
    values[4][:, 0] = 0.6
    values[5][:, :2] = 0.0
    assert torch.equal(evaluate(values)[0], torch.zeros(2))


def test_missing_ray_is_masked_and_result_is_bounded():
    values = inputs()
    values[1][:, 0] = float("inf")
    reward, clearance, mask = evaluate(values)
    assert not mask[:, 0].any()
    assert torch.isfinite(reward).all() and torch.isfinite(clearance).all()
    assert torch.allclose(reward, torch.full((2,), 0.75))
    assert ((reward >= 0) & (reward <= 1)).all()
