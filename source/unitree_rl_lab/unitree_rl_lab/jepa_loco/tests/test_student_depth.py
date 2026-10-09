"""Teacher 동결·student shape·손실·JEPA mask 테스트."""

import torch
from tensordict import TensorDict

from unitree_rl_lab.jepa_loco.models.oracle_actor import OracleTerrainActor
from unitree_rl_lab.jepa_loco.models.student_depth import FrozenCurrentTeacher, StudentDepthModel
from unitree_rl_lab.jepa_loco.models.student_losses import (
    distillation_losses, jepa_valid_mask, masked_future_mse, normalized_latent_mse,
)
from unitree_rl_lab.jepa_loco.envs.oracle_env_cfg import OracleCurrentEasyStartEnvCfg
from unitree_rl_lab.jepa_loco.envs.student_env_cfg import StudentDepthEasyStartEnvCfg
from unitree_rl_lab.jepa_loco.agents.student_cfg import StudentTrainCfg


def _teacher():
    sample = TensorDict({"policy": torch.zeros(1, 45), "terrain_current": torch.zeros(1, 187)},
                        batch_size=[1])
    source = OracleTerrainActor(
        sample, {"actor": ["policy", "terrain_current"]}, "actor", 12,
        hidden_dims=(512, 256, 128), obs_normalization=True,
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    )
    return FrozenCurrentTeacher(source.state_dict())


def test_teacher_frozen_and_unchanged_after_student_update():
    teacher = _teacher()
    before = {k: v.clone() for k, v in teacher.state_dict().items()}
    assert all(not p.requires_grad for p in teacher.parameters())
    assert not teacher.training and not teacher.actor.training
    student = StudentDepthModel(method="no_memory")
    optimizer = torch.optim.Adam(student.parameters(), lr=1.0e-3)
    context, _ = student.encode_step(torch.rand(2, 2, 64, 112), torch.ones(2, dtype=torch.bool), None)
    z = student.terrain_latent(context)
    action = teacher.action_from_latent(torch.zeros(2, 45), z)
    action.square().mean().backward()
    assert student.terrain_head[0].weight.grad is not None
    optimizer.step()
    assert all(torch.equal(v, before[k]) for k, v in teacher.state_dict().items())


def test_losses_normalization_and_zero_case():
    z = torch.tensor([[[2.0, 4.0]]])
    mean = torch.tensor([1.0, 2.0])
    std = torch.tensor([1.0, 2.0])
    assert normalized_latent_mse(z, z, mean, std).item() == 0
    target = torch.tensor([[[3.0, 6.0]]])
    assert normalized_latent_mse(z, target, mean, std).item() == 1.0
    action = torch.zeros(1, 1, 12)
    latent_loss, action_loss = distillation_losses(z, z, action, action, mean, std)
    assert latent_loss.item() == action_loss.item() == 0
    assert masked_future_mse(z, z, torch.ones(1, 1, dtype=torch.bool)).item() == 0


def test_jepa_mask_rejects_resets_and_command_change():
    done = torch.zeros(5, 2, dtype=torch.bool)
    command = torch.zeros(5, 2, 3)
    done[1, 0] = True
    command[2, 1, 0] = 0.6
    mask = jepa_valid_mask(done, command, horizon_steps=2, command_tolerance=0.01)
    assert mask.shape == (3, 2)
    assert not mask[0, 0] and not mask[0, 1]
    assert mask[2, 0] and not mask[2, 1]


def test_no_memory_current_frame_independent_of_history():
    student = StudentDepthModel(method="no_memory")
    current = torch.rand(2, 2, 64, 112)
    previous_a = torch.rand_like(current)
    previous_b = torch.rand_like(current)
    sequence_a = torch.stack((previous_a, current))
    sequence_b = torch.stack((previous_b, current))
    fresh = torch.ones(2, 2, dtype=torch.bool)
    reset = torch.zeros_like(fresh)
    out_a, _ = student.encode_sequence(sequence_a, fresh, reset)
    out_b, _ = student.encode_sequence(sequence_b, fresh, reset)
    assert out_a.shape == (2, 2, 128)
    torch.testing.assert_close(out_a[-1], out_b[-1])


def test_gru_sequence_and_ema_shape():
    student = StudentDepthModel(method="gru_jepa")
    depth = torch.rand(3, 2, 2, 64, 112)
    fresh = torch.tensor([[True, True], [False, True], [True, False]])
    reset = torch.tensor([[False, False], [False, False], [True, False]])
    out, state = student.encode_sequence(depth, fresh, reset)
    target, _ = student.encode_sequence(depth, fresh, reset, target=True)
    assert out.shape == target.shape == (3, 2, 128)
    assert state.shape == (2, 128)
    assert student.terrain_latent(out).shape == (3, 2, 32)
    assert student.predict(out, torch.zeros(3, 2, 3)).shape == out.shape


def test_student_env_matches_current_teacher_except_camera_and_observations():
    teacher = OracleCurrentEasyStartEnvCfg()
    student = StudentDepthEasyStartEnvCfg()
    for name in ("rewards", "commands", "curriculum", "terminations", "events", "progress",
                 "easy_start"):
        assert getattr(student, name).to_dict() == getattr(teacher, name).to_dict()
    assert student.scene.terrain.terrain_generator.to_dict() == teacher.scene.terrain.terrain_generator.to_dict()
    assert student.observations.policy.to_dict() == teacher.observations.policy.to_dict()
    assert student.observations.terrain_current.to_dict() == teacher.observations.terrain_current.to_dict()
    assert student.observations.critic.to_dict() == teacher.observations.critic.to_dict()
    assert student.observations.depth is not None and student.observations.depth_fresh is not None


def test_student_horizon_and_dagger_schedule():
    cfg = StudentTrainCfg(beta_initial=0.5, beta_final=0.0, beta_decay_iterations=10)
    assert cfg.validate_training(control_dt=0.02, depth_period_steps=5) == 25
    assert cfg.dagger_beta(0) == 0.5
    assert cfg.dagger_beta(5) == 0.25
    assert cfg.dagger_beta(10) == 0.0
