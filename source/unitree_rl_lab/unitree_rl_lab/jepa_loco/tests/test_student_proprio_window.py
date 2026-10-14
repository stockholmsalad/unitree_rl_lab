"""5-step proprio 요약, rollout prefix, reset, EMA, checkpoint 회귀."""

import pytest
import torch
from tensordict import TensorDict

from unitree_rl_lab.jepa_loco.agents.student_cfg import StudentTrainCfg
from unitree_rl_lab.jepa_loco.models.student_batches import ProprioHistory, proprio_windows, select_env_batch
from unitree_rl_lab.jepa_loco.models.student_depth import StudentDepthModel, StudentInferencePolicy
from unitree_rl_lab.jepa_loco.tests.test_student_depth import _teacher


def _small(method="gru", window=5):
    return StudentDepthModel(method=method, image_hw=(8, 8), cnn_channels=(4,),
                             cnn_kernels=(3,), cnn_strides=(1,), feature_dim=128,
                             proprio_input=True, proprio_window=window)


@pytest.mark.parametrize("method", ("no_memory", "gru", "gru_jepa", "gru_copy"))
def test_window_one_is_bitwise_a_path(method):
    torch.manual_seed(17)
    old = StudentDepthModel(method=method, image_hw=(8, 8), cnn_channels=(4,),
                            cnn_kernels=(3,), cnn_strides=(1,), feature_dim=128,
                            proprio_input=True)
    torch.manual_seed(17)
    new = _small(method, window=1)
    assert list(old.state_dict()) == list(new.state_dict())
    assert all(torch.equal(old.state_dict()[key], new.state_dict()[key]) for key in old.state_dict())
    assert not any("proprio_summary" in key for key in new.state_dict())
    normalizer = _teacher().actor.obs_normalizer
    depth = torch.rand(2, 2, 8, 8)
    policy = torch.rand(2, 45)
    fresh = torch.tensor([True, False])
    a, _ = old.encode_step(depth, fresh, None, proprio=policy, proprio_normalizer=normalizer)
    b, _ = new.encode_step(depth, fresh, None, proprio=policy, proprio_normalizer=normalizer)
    assert torch.equal(a, b)
    command = torch.rand(2, 3)
    assert torch.equal(old.predict(a, command, policy, normalizer),
                       new.predict(b, command, policy, normalizer))


def test_window_shapes_and_order_and_padding():
    model = _small()
    normalizer = _teacher().actor.obs_normalizer
    history = ProprioHistory(5, 2, 45, torch.device("cpu"))
    for value in (1.0, 2.0, 3.0, 4.0):
        history.append(torch.full((2, 45), value))
    history.reset(torch.tensor([True, False]))
    window, valid = history.append(torch.full((2, 45), 5.0))
    assert window.shape == (2, 5, 45)
    assert valid.shape == (2, 5)
    assert window[0, :, 0].tolist() == [0, 0, 0, 0, 5]
    assert valid[0].tolist() == [False, False, False, False, True]
    assert window[1, :, 0].tolist() == [1, 2, 3, 4, 5]
    assert valid[1].tolist() == [True] * 5
    embeddings = model.proprio_encoder(normalizer(window))
    assert embeddings.shape == (2, 5, 64)
    assert model.proprio_summary[0].in_features == 5 * 64 + 5
    assert model.encode_proprio(window, normalizer, valid=valid).shape == (2, 64)
    changed_padding = window.clone()
    changed_padding[0, :4] = 99
    assert torch.equal(model.encode_proprio(window, normalizer, valid=valid)[0],
                       model.encode_proprio(changed_padding, normalizer, valid=valid)[0])
    raw = torch.stack((torch.full((2, 45), 6.0), torch.full((2, 45), 7.0)))
    resets = torch.tensor([[False, False], [True, False]])
    built, masks = proprio_windows(raw, resets, history.policy, history.valid)
    assert built[0, 1, :, 0].tolist() == [2, 3, 4, 5, 6]
    assert built[1, 0, :, 0].tolist() == [0, 0, 0, 0, 7]
    assert masks[1, 0].tolist() == [False, False, False, False, True]


def test_prefix_is_sliced_with_env_minibatch():
    prefix = torch.arange(4 * 4 * 45).reshape(4, 4, 45).float()
    valid = torch.arange(4 * 4).reshape(4, 4) % 2 == 0
    batch = {"policy": torch.zeros(3, 4, 45), "proprio_prefix": prefix,
             "proprio_prefix_valid": valid}
    ids = torch.tensor([3, 1])
    h0 = torch.zeros(4, 128)
    mini, _, _, _, _ = select_env_batch(batch, torch.zeros(3, 4, dtype=torch.bool),
                                        h0, h0, ids)
    assert torch.equal(mini["proprio_prefix"], prefix[:, ids])
    assert torch.equal(mini["proprio_prefix_valid"], valid[:, ids])


def test_predictor_uses_source_window():
    model = _small("gru_jepa")
    normalizer = _teacher().actor.obs_normalizer
    history = ProprioHistory(5, 1, 45, torch.device("cpu"))
    a = b = None
    for t in range(6):
        window, valid = history.append(torch.full((1, 45), float(t)))
        if t == 4:
            a = model.predict(torch.zeros(1, 128), torch.zeros(1, 3), window,
                              normalizer, valid)
        if t == 5:
            b = model.predict(torch.zeros(1, 128), torch.zeros(1, 3), window,
                              normalizer, valid)
    assert a.shape == b.shape == (1, 128)
    assert not torch.equal(a, b)


@pytest.mark.parametrize("method", ("no_memory", "gru", "gru_jepa", "gru_copy"))
@pytest.mark.parametrize("target", (False, True))
def test_two_rollouts_step_matches_sequence_with_prefix_and_reset(method, target):
    model = _small(method)
    normalizer = _teacher().actor.obs_normalizer
    depth = torch.rand(10, 2, 2, 8, 8)
    policy = torch.rand(10, 2, 45)
    fresh = torch.tensor([[True, True], [False, False], [False, False], [False, False], [False, False],
                          [True, True], [False, False], [False, False], [False, False], [False, False]])
    resets = torch.zeros(10, 2, dtype=torch.bool)
    resets[3, 0] = True
    resets[6, 1] = True
    history = ProprioHistory(5, 2, 45, depth.device)
    step_state = model.initial_state(2, depth.device)
    seq_state = model.initial_state(2, depth.device)
    for start, stop in ((0, 5), (5, 10)):
        prefix, prefix_valid = history.policy.clone(), history.valid.clone()
        step_out = []
        for t in range(start, stop):
            step_state = torch.where(resets[t, :, None], torch.zeros_like(step_state), step_state)
            history.reset(resets[t])
            window, valid = history.append(policy[t])
            value, step_state = model.encode_step(depth[t], fresh[t], step_state,
                                                  target=target,
                                                  proprio=window, proprio_normalizer=normalizer,
                                                  proprio_valid=valid)
            step_out.append(value)
        seq_out, seq_state = model.encode_sequence(
            depth[start:stop], fresh[start:stop], resets[start:stop], seq_state,
            target=target,
            proprio=policy[start:stop], proprio_normalizer=normalizer,
            proprio_prefix=prefix, proprio_prefix_valid=prefix_valid,
        )
        assert torch.equal(seq_out, torch.stack(step_out))
        assert torch.equal(seq_state, step_state)
    assert torch.equal(step_out[0][0], step_out[1][0])  # env 1은 이 사이 reset됨


def test_interframe_proprio_changes_next_fresh_output():
    model = _small()
    normalizer = _teacher().actor.obs_normalizer
    depth = torch.rand(1, 2, 8, 8)
    def run(middle_value):
        history = ProprioHistory(5, 1, 45, depth.device)
        state = model.initial_state(1, depth.device)
        for t in range(6):
            raw = torch.full((1, 45), middle_value if 1 <= t <= 4 else 0.0)
            window, valid = history.append(raw)
            next_state, state = model.encode_step(depth, torch.tensor([t in (0, 5)]), state,
                                                  proprio=window, proprio_normalizer=normalizer,
                                                  proprio_valid=valid)
            if t == 4:
                held = next_state.clone()
        return held, state
    held_a, next_a = run(1.0)
    held_b, next_b = run(9.0)
    assert torch.equal(held_a, held_b)
    assert not torch.equal(next_a, next_b)


def test_summary_and_step_mlp_gradient_and_ema():
    model = _small("gru_jepa")
    teacher = _teacher()
    history = ProprioHistory(5, 2, 45, torch.device("cpu"))
    window, valid = history.append(torch.rand(2, 45))
    before = next(model.ema_proprio_summary.parameters()).clone()
    context, _ = model.encode_step(torch.rand(2, 2, 8, 8), torch.ones(2, dtype=torch.bool), None,
                                   proprio=window, proprio_normalizer=teacher.actor.obs_normalizer,
                                   proprio_valid=valid)
    model.terrain_latent(context).square().mean().backward()
    assert next(model.proprio_encoder.parameters()).grad.abs().sum() > 0
    assert next(model.proprio_summary.parameters()).grad.abs().sum() > 0
    assert all(p.grad is None for p in model.ema_proprio_encoder.parameters())
    assert all(p.grad is None for p in model.ema_proprio_summary.parameters())
    assert all(p.grad is None for p in teacher.actor.obs_normalizer.parameters())
    with torch.no_grad():
        next(model.proprio_summary.parameters()).add_(1)
    model.update_ema(0.5)
    torch.testing.assert_close(next(model.ema_proprio_summary.parameters()), before + 0.5)


def test_inference_buffer_reset_and_old_a_checkpoint(tmp_path):
    teacher = _teacher()
    model = _small()
    cfg = {"method": "gru", "image_hw": (8, 8), "feature_dim": 128,
           "context_dim": 128, "terrain_dim": 32, "terrain_hidden_dim": 128,
           "predictor_hidden_dim": 128, "cnn_channels": (4,),
           "cnn_kernels": (3,), "cnn_strides": (1,), "proprio_input": True,
           "proprio_dim": 45, "proprio_hidden_dims": (64, 64), "proprio_window": 5}
    path = tmp_path / "window.pt"
    torch.save({"student_cfg": cfg, "model_state_dict": model.state_dict(),
                "teacher_state_dict": teacher.actor.state_dict()}, path)
    policy = StudentInferencePolicy(str(path), torch.device("cpu"))
    obs = TensorDict({"depth": torch.rand(2, 2, 8, 8),
                      "depth_fresh": torch.ones(2, 1), "policy": torch.rand(2, 45)}, batch_size=[2])
    for _ in range(3):
        assert policy(obs).shape == (2, 12)
    assert policy.proprio_history.valid.sum(dim=0).tolist() == [3, 3]
    policy.reset(torch.tensor([True, False]))
    assert policy.proprio_history.valid.sum(dim=0).tolist() == [0, 3]
    policy(obs)
    assert policy.proprio_history.valid.sum(dim=0).tolist() == [1, 4]
    old = _small(window=1)
    cfg.pop("proprio_window")
    torch.save({"student_cfg": cfg, "model_state_dict": old.state_dict(),
                "teacher_state_dict": teacher.actor.state_dict()}, path)
    legacy = StudentInferencePolicy(str(path), torch.device("cpu"))
    assert legacy.model.proprio_window == 1 and legacy.proprio_history is None
    assert legacy(obs).shape == (2, 12)


def test_window_cfg_validation_and_warning():
    with pytest.raises(ValueError):
        StudentTrainCfg(proprio_window=0).validate_training(0.02, 5)
    with pytest.warns(UserWarning, match="depth_period_steps"):
        assert StudentTrainCfg(proprio_input=True, proprio_window=1).validate_training(0.02, 5) == 25
