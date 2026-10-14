"""선택적 고유수용감각 입력의 회귀·shape·기울기 테스트."""

from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from unitree_rl_lab.jepa_loco.models.student_depth import StudentDepthModel, StudentInferencePolicy
from unitree_rl_lab.jepa_loco.models.student_losses import distillation_losses
from unitree_rl_lab.jepa_loco.tests.test_student_depth import _teacher


def _model(method: str, enabled: bool) -> StudentDepthModel:
    return StudentDepthModel(method=method, image_hw=(8, 8), cnn_channels=(4,),
                             cnn_kernels=(3,), cnn_strides=(1,), feature_dim=128,
                             proprio_input=enabled)


@pytest.mark.parametrize("method", ("no_memory", "gru", "gru_jepa", "gru_copy"))
def test_disabled_matches_legacy_path_bitwise(method):
    torch.manual_seed(31)
    model = _model(method, False)
    assert not any("proprio_encoder" in key for key in model.state_dict())
    depth = torch.rand(3, 2, 2, 8, 8)
    fresh = torch.tensor([[True, True], [False, True], [True, False]])
    resets = torch.zeros_like(fresh)
    proprio = torch.rand(3, 2, 45)
    normalizer = _teacher().actor.obs_normalizer
    actual, _ = model.encode_sequence(depth, fresh, resets, proprio=proprio,
                                       proprio_normalizer=normalizer)
    state = model.initial_state(2, depth.device)
    old_steps = []
    for t in range(3):
        state = torch.where(resets[t, :, None], torch.zeros_like(state), state)
        if method == "no_memory":
            if fresh[t].any():
                updated = model.frame_projection(model.cnn(depth[t][fresh[t]].float()))
                state = state.index_copy(0, fresh[t].nonzero(as_tuple=True)[0], updated)
            old_steps.append(state)
        else:
            features = torch.zeros(2, 128)
            if fresh[t].any():
                features[fresh[t]] = model.frame_projection(model.cnn(depth[t][fresh[t]].float()))
            value, state = model.backbone.gated_step(features, state, fresh[t])
            old_steps.append(value)
    old = torch.stack(old_steps)
    assert torch.equal(actual, old)
    actual_z, old_z = model.terrain_latent(actual), model.terrain_latent(old)
    assert torch.equal(actual_z, old_z)
    command = torch.rand(3, 2, 3)
    manual_prediction = old if method == "gru_copy" else model.predictor(torch.cat((old, command), dim=-1))
    assert torch.equal(model.predict(actual, command, proprio, normalizer), manual_prediction)
    target = torch.rand_like(actual_z)
    action = torch.rand(3, 2, 12)
    reference_action = torch.zeros_like(action)
    mean = torch.zeros(32)
    std = torch.ones(32)
    actual_losses = distillation_losses(actual_z, target, action, reference_action, mean, std)
    old_losses = distillation_losses(old_z, target, action, reference_action, mean, std)
    assert all(torch.equal(a, b) for a, b in zip(actual_losses, old_losses))


@pytest.mark.parametrize("method", ("no_memory", "gru", "gru_jepa", "gru_copy"))
def test_proprio_shapes_and_hold_between_depth_frames(method):
    model = _model(method, True)
    normalizer = _teacher().actor.obs_normalizer
    depth = torch.rand(2, 2, 8, 8)
    policy = torch.rand(2, 45)
    assert model.encode_proprio(policy, normalizer).shape == (2, 64)
    if method == "no_memory":
        assert model.frame_projection[0].in_features == 192
    else:
        assert model.backbone.cell.input_size == 192
    assert model.predictor[0].in_features == 195
    state, _ = model.encode_step(depth, torch.ones(2, dtype=torch.bool), None,
                                 proprio=policy, proprio_normalizer=normalizer)
    changed, _ = model.encode_step(depth, torch.zeros(2, dtype=torch.bool), state,
                                   proprio=policy + 10, proprio_normalizer=normalizer)
    assert torch.equal(state, changed)
    if method in ("gru_jepa", "gru_copy"):
        assert model.predict(state, torch.zeros(2, 3), policy, normalizer).shape == (2, 128)


def test_proprio_gradient_teacher_frozen_and_ema_update():
    teacher = _teacher()
    model = _model("gru_jepa", True)
    depth = torch.rand(2, 2, 8, 8)
    policy = torch.rand(2, 45)
    before = next(model.ema_proprio_encoder.parameters()).clone()
    context, _ = model.encode_step(depth, torch.ones(2, dtype=torch.bool), None,
                                   proprio=policy, proprio_normalizer=teacher.actor.obs_normalizer)
    model.terrain_latent(context).square().mean().backward()
    assert next(model.proprio_encoder.parameters()).grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.actor.obs_normalizer.parameters())
    assert all(p.grad is None for p in model.ema_proprio_encoder.parameters())
    with torch.no_grad():
        next(model.proprio_encoder.parameters()).add_(1)
    model.update_ema(0.5)
    torch.testing.assert_close(next(model.ema_proprio_encoder.parameters()),
                               before + 0.5)


def test_ema_target_uses_its_own_proprio_encoder_and_actual_target_policy():
    model = _model("gru", True)
    normalizer = _teacher().actor.obs_normalizer
    with torch.no_grad():
        for param in model.ema_proprio_encoder.parameters():
            param.zero_()
    depth = torch.rand(2, 1, 2, 8, 8)
    fresh = torch.ones(2, 1, dtype=torch.bool)
    resets = torch.zeros_like(fresh)
    policy = torch.rand(2, 1, 45)
    result, _ = model.encode_sequence(depth, fresh, resets, target=True,
                                       proprio=policy, proprio_normalizer=normalizer)
    with torch.no_grad():
        model.ema_proprio_encoder[-2].bias.fill_(1)
    changed, _ = model.encode_sequence(depth, fresh, resets, target=True,
                                        proprio=policy, proprio_normalizer=normalizer)
    assert not torch.equal(result, changed)


def test_frame_embedding_target_does_not_use_proprio_encoder():
    model = _model("gru_jepa", True)
    depth = torch.rand(2, 1, 2, 8, 8)
    fresh = torch.ones(2, 1, dtype=torch.bool)
    before = model.encode_frame_targets(depth, fresh, batch_size=1)
    with torch.no_grad():
        for param in model.ema_proprio_encoder.parameters():
            param.add_(10)
    after = model.encode_frame_targets(depth, fresh, batch_size=1)
    assert torch.equal(before, after)


@pytest.mark.parametrize("method", ("no_memory", "gru", "gru_jepa", "gru_copy"))
def test_proprio_rollout_matches_sequence_with_episode_reset(method):
    model = _model(method, True)
    normalizer = _teacher().actor.obs_normalizer
    depth = torch.rand(5, 2, 2, 8, 8)
    policy = torch.rand(5, 2, 45)
    fresh = torch.tensor([[False, True], [True, False], [False, False],
                          [True, True], [False, False]])
    resets = torch.tensor([[False, False], [False, False], [False, True],
                           [True, False], [False, False]])
    sequence, _ = model.encode_sequence(depth, fresh, resets, proprio=policy,
                                        proprio_normalizer=normalizer)
    state = model.initial_state(2, depth.device)
    rollout = []
    for t in range(5):
        state = torch.where(resets[t, :, None], torch.zeros_like(state), state)
        value, state = model.encode_step(depth[t], fresh[t], state, proprio=policy[t],
                                         proprio_normalizer=normalizer)
        rollout.append(value)
    assert torch.equal(sequence, torch.stack(rollout))
    assert torch.count_nonzero(sequence[2, 1]) == 0
    assert torch.equal(sequence[3], sequence[4])


def test_legacy_checkpoint_missing_option_loads_disabled(tmp_path):
    teacher = _teacher()
    model = StudentDepthModel(method="gru")
    cfg = {"method": "gru", "image_hw": (64, 112), "feature_dim": 128,
           "context_dim": 128, "terrain_dim": 32, "terrain_hidden_dim": 128,
           "predictor_hidden_dim": 128, "cnn_channels": (16, 32, 64),
           "cnn_kernels": (5, 3, 3), "cnn_strides": (2, 2, 2)}
    path = tmp_path / "legacy.pt"
    torch.save({"student_cfg": cfg, "model_state_dict": model.state_dict(),
                "teacher_state_dict": teacher.actor.state_dict()}, path)
    policy = StudentInferencePolicy(str(path), torch.device("cpu"))
    assert not policy.model.proprio_input
    assert not any("proprio_encoder" in key for key in policy.model.state_dict())
    obs = TensorDict({"depth": torch.rand(1, 2, 64, 112),
                      "depth_fresh": torch.ones(1, 1), "policy": torch.rand(1, 45)}, batch_size=[1])
    assert policy(obs).shape == (1, 12)


def test_existing_depth_only_checkpoint_still_loads_if_available():
    path = Path("logs/student/gru_t17000_s43/model_3000.pt")
    if not path.exists():
        pytest.skip("로컬 17000 depth-only checkpoint 없음")
    policy = StudentInferencePolicy(str(path), torch.device("cpu"))
    assert policy.method == "gru" and not policy.model.proprio_input


def test_proprio_checkpoint_inference_uses_policy_observation(tmp_path):
    teacher = _teacher()
    model = StudentDepthModel(method="gru", proprio_input=True)
    cfg = {"method": "gru", "image_hw": (64, 112), "feature_dim": 128,
           "context_dim": 128, "terrain_dim": 32, "terrain_hidden_dim": 128,
           "predictor_hidden_dim": 128, "cnn_channels": (16, 32, 64),
           "cnn_kernels": (5, 3, 3), "cnn_strides": (2, 2, 2),
           "proprio_input": True, "proprio_dim": 45, "proprio_hidden_dims": (64, 64)}
    path = tmp_path / "proprio.pt"
    torch.save({"student_cfg": cfg, "model_state_dict": model.state_dict(),
                "teacher_state_dict": teacher.actor.state_dict()}, path)
    policy = StudentInferencePolicy(str(path), torch.device("cpu"))
    obs = TensorDict({"depth": torch.rand(1, 2, 64, 112),
                      "depth_fresh": torch.ones(1, 1), "policy": torch.rand(1, 45)}, batch_size=[1])
    assert policy(obs).shape == (1, 12)
    policy.reset(torch.ones(1, dtype=torch.bool))
    assert torch.count_nonzero(policy.state) == 0
