"""Teacher 동결·student shape·손실·JEPA mask 테스트."""

import torch
from tensordict import TensorDict

from unitree_rl_lab.jepa_loco.models.oracle_actor import OracleTerrainActor
from unitree_rl_lab.jepa_loco.models.student_depth import FrozenCurrentTeacher, StudentDepthModel
from unitree_rl_lab.jepa_loco.models.student_batches import (
    env_batch_indices, jepa_pair_tensors, realized_se2_displacement, select_env_batch,
)
from unitree_rl_lab.jepa_loco.models.student_losses import (
    distillation_losses, jepa_fresh_pair_mask, jepa_valid_mask, masked_future_mse, normalized_jepa_with_copy,
    normalized_latent_mse,
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


def test_normalized_jepa_scale_and_copy_ratio():
    source = torch.tensor([[[0.0, 1.0]], [[2.0, 3.0]], [[4.0, 5.0]]])
    target = torch.tensor([[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]])
    pred = source + 0.5
    mask = torch.ones(3, 1, dtype=torch.bool)
    loss, copy, variance = normalized_jepa_with_copy(pred, source, target, mask, 1.0e-6)
    scaled, scaled_copy, scaled_variance = normalized_jepa_with_copy(
        pred * 7, source * 7, target * 7, mask, 1.0e-6,
    )
    torch.testing.assert_close(loss, scaled)
    torch.testing.assert_close(copy, scaled_copy)
    torch.testing.assert_close(scaled_variance, variance * 49)
    copy_pred, copy_reference, _ = normalized_jepa_with_copy(source, source, target, mask, 1.0e-6)
    torch.testing.assert_close(copy_pred / copy_reference, torch.ones(()))
    assert loss < copy


def test_frame_target_uses_only_fresh_future_in_same_episode():
    model = StudentDepthModel(method="gru_jepa", image_hw=(8, 8), cnn_channels=(4,),
                              cnn_kernels=(3,), cnn_strides=(1,), feature_dim=8, context_dim=8)
    depth = torch.rand(5, 2, 2, 8, 8)
    fresh = torch.tensor([[True, True], [False, False], [True, False],
                          [True, True], [False, True]])
    done = torch.zeros(5, 2, dtype=torch.bool)
    done[1, 1] = True
    command = torch.zeros(5, 2, 3)
    horizon = 2
    mask = jepa_fresh_pair_mask(jepa_valid_mask(done, command, horizon, 0.01), fresh, horizon)
    assert mask.shape == (3, 2)
    assert mask[0, 0] and not mask[0, 1]
    assert mask[1, 0] and not mask[1, 1]
    assert not mask[2, 0] and mask[2, 1]
    valid = torch.zeros_like(fresh)
    valid[horizon:] = mask
    target = model.encode_frame_targets(depth, valid, batch_size=2)
    assert target.shape == (5, 2, 8)
    torch.testing.assert_close(target[2, 0], model.ema_cnn(depth[2, 0:1])[0])
    assert torch.count_nonzero(target[2, 1]) == 0
    assert torch.count_nonzero(target[3, 1]) == 0
    assert torch.count_nonzero(target[4, 0]) == 0


def test_frame_target_projection_when_feature_and_context_dims_differ():
    model = StudentDepthModel(method="gru_jepa", image_hw=(8, 8), cnn_channels=(4,),
                              cnn_kernels=(3,), cnn_strides=(1,), feature_dim=6, context_dim=8)
    depth = torch.rand(2, 1, 2, 8, 8)
    fresh = torch.ones(2, 1, dtype=torch.bool)
    reset = torch.zeros_like(fresh)
    context, _ = model.encode_sequence(depth, fresh, reset)
    target = model.encode_frame_targets(depth, fresh, batch_size=1)
    assert context.shape == target.shape == (2, 1, 8)


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


def test_no_memory_rollout_matches_sequence_with_fresh_and_reset():
    student = StudentDepthModel(method="no_memory", image_hw=(8, 8), cnn_channels=(4,),
                                cnn_kernels=(3,), cnn_strides=(1,), feature_dim=8, context_dim=8)
    depth = torch.rand(5, 2, 2, 8, 8)
    fresh = torch.tensor([[False, True], [True, False], [False, False],
                          [False, True], [True, False]])
    resets = torch.tensor([[False, False], [False, False], [False, True],
                           [True, False], [False, False]])
    seen_batch_sizes = []
    hook = student.cnn.register_forward_hook(lambda _module, inputs, _output: seen_batch_sizes.append(inputs[0].shape[0]))
    sequence, _ = student.encode_sequence(depth, fresh, resets)
    hook.remove()
    assert sum(seen_batch_sizes) == int(fresh.sum())
    state = student.initial_state(2, depth.device)
    rollout = []
    for t in range(5):
        state = torch.where(resets[t, :, None], torch.zeros_like(state), state)
        value, state = student.encode_step(depth[t], fresh[t], state)
        rollout.append(value)
    torch.testing.assert_close(sequence, torch.stack(rollout))
    assert torch.count_nonzero(sequence[0, 0]) == 0
    assert torch.count_nonzero(sequence[2, 1]) == 0
    torch.testing.assert_close(sequence[1, 0], sequence[2, 0])


def test_env_minibatch_slices_states_resets_and_mask():
    batch = {"depth": torch.arange(4 * 8 * 2).reshape(4, 8, 2),
             "command": torch.arange(4 * 8 * 3).reshape(4, 8, 3)}
    resets = torch.arange(4 * 8).reshape(4, 8) % 3 == 0
    mask = torch.arange(3 * 8).reshape(3, 8) % 2 == 0
    h0 = torch.arange(8 * 5).reshape(8, 5)
    h0_ema = -h0
    shuffled = torch.cat(env_batch_indices(8, 4, h0.device, shuffle=True))
    torch.testing.assert_close(shuffled.sort().values, torch.arange(8))
    for ids in env_batch_indices(8, 4, h0.device):
        mini, r, h, e, m = select_env_batch(batch, resets, h0, h0_ema, ids, mask)
        for key in batch:
            torch.testing.assert_close(mini[key], batch[key][:, ids])
        torch.testing.assert_close(r, resets[:, ids])
        torch.testing.assert_close(h, h0[ids])
        torch.testing.assert_close(e, h0_ema[ids])
        torch.testing.assert_close(m, mask[:, ids])


def test_single_minibatch_single_epoch_matches_full_loss_and_gradient():
    torch.manual_seed(7)
    model = StudentDepthModel(method="gru_jepa", image_hw=(8, 8), cnn_channels=(4,),
                              cnn_kernels=(3,), cnn_strides=(1,), feature_dim=8, context_dim=8,
                              terrain_dim=3, terrain_hidden_dim=8)
    depth = torch.rand(3, 2, 2, 8, 8)
    batch = {"depth": depth, "fresh": torch.tensor([[True, True], [False, True], [True, False]]),
             "teacher_z": torch.rand(3, 2, 3), "command": torch.rand(3, 2, 3),
             "pose": torch.zeros(3, 2, 3), "teacher_action": torch.rand(3, 2, 2)}
    resets = torch.tensor([[False, False], [False, False], [True, False]])
    h0 = model.initial_state(2, depth.device)
    mask = torch.tensor([[True, True], [False, True]])
    fixed_head = torch.rand(3, 2)

    def loss(data, reset, initial, initial_ema, valid):
        context, _ = model.encode_sequence(data["depth"], data["fresh"], reset, initial)
        z = model.terrain_latent(context)
        latent, action = distillation_losses(z, data["teacher_z"], z @ fixed_head,
                                              data["teacher_action"], torch.zeros(3), torch.ones(3))
        with torch.no_grad():
            target, _ = model.encode_sequence(data["depth"], data["fresh"], reset, initial_ema, target=True)
        source, target, condition = jepa_pair_tensors(context, target, data["command"], None, 1,
                                                       "future", "command")
        jepa = masked_future_mse(model.predict(source, condition), target, valid)
        return latent + action + 0.1 * jepa

    whole = loss(batch, resets, h0, h0, mask)
    whole.backward()
    gradients = [p.grad.clone() for p in model.parameters() if p.grad is not None]
    model.zero_grad(set_to_none=True)
    ids = env_batch_indices(2, 1, depth.device)[0]
    mini, mini_reset, mini_h0, mini_h0_ema, mini_mask = select_env_batch(batch, resets, h0, h0, ids, mask)
    single = loss(mini, mini_reset, mini_h0, mini_h0_ema, mini_mask)
    single.backward()
    torch.testing.assert_close(whole, single)
    for before, after in zip(gradients, (p.grad for p in model.parameters() if p.grad is not None)):
        torch.testing.assert_close(before, after)


def test_realized_se2_displacement_rotates_and_wraps_yaw():
    start = torch.tensor([[1.0, 2.0, torch.pi / 2], [0.0, 0.0, 3.13]])
    end = torch.tensor([[1.0, 3.0, torch.pi / 2], [1.0, 0.0, -3.13]])
    delta = realized_se2_displacement(start, end)
    torch.testing.assert_close(delta[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6, rtol=0)
    assert 0 < delta[1, 2] < 0.03


def test_jepa_target_modes_and_condition_sources():
    context = torch.arange(6.).reshape(3, 2, 1).expand(3, 2, 4)
    command = torch.ones(3, 2, 3)
    pose = torch.zeros_like(command)
    pose[1:, :, 0] = 0.2
    for mode in ("future", "present_from_past"):
        source, target, condition = jepa_pair_tensors(context, context + 10, command, pose, 1,
                                                       mode, "command")
        assert source.shape == target.shape == (2, 2, 4)
        torch.testing.assert_close(source, context[:-1])
        torch.testing.assert_close(target, context[1:] + 10)
        torch.testing.assert_close(condition, command[:-1])
        _, _, displacement = jepa_pair_tensors(context, context, command, pose, 1,
                                                 mode, "realized_displacement")
        assert displacement.shape == (2, 2, 3)
        torch.testing.assert_close(displacement[0, :, 0], torch.full((2,), 0.2))


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


def test_student_update_defaults_and_invalid_batch_count():
    cfg = StudentTrainCfg(num_envs=64)
    assert (cfg.num_mini_batches, cfg.num_learning_epochs) == (4, 2)
    assert cfg.dagger_beta(0) == 1.0
    assert cfg.dagger_beta(250) == 0.5
    assert cfg.dagger_beta(500) == 0.0
    assert (cfg.jepa_target_mode, cfg.condition_source) == ("future", "command")
    assert cfg.jepa_target == "context" and cfg.jepa_variance_floor > 0
    try:
        StudentTrainCfg(num_envs=65).validate_training(0.02, 5)
    except ValueError:
        pass
    else:
        raise AssertionError("나누어떨어지지 않는 env 미니배치를 허용했다")
