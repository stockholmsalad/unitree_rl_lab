"""FLA Mamba-1 상태·순환 수식·student 연결 회귀 테스트."""

import pytest
import torch
import torch.nn.functional as F

from unitree_rl_lab.jepa_loco.agents.student_cfg import StudentTrainCfg
from unitree_rl_lab.jepa_loco.models.backbone import GRUBackbone, MambaBackbone
from unitree_rl_lab.jepa_loco.models.student_depth import StudentDepthModel


def _reference_step(model: MambaBackbone, x: torch.Tensor, state: torch.Tensor):
    """FLA slow_forward를 호출하지 않는 Mamba-1 1토큰 순환식."""
    mixer = model.mixer
    width, kernel, rank = model.intermediate_size, model.conv_kernel, model.state_size
    _, conv_flat, ssm_flat = torch.split(state, (model.out_dim, width * kernel, width * rank), -1)
    conv = conv_flat.reshape(-1, width, kernel)
    ssm = ssm_flat.reshape(-1, width, rank)
    u = model.input_projection(x)
    hidden, gate = mixer.in_proj(model.pre_norm(u)).chunk(2, dim=-1)
    conv = torch.cat((conv[:, :, 1:], hidden.unsqueeze(-1)), dim=-1)
    conv_out = (conv * mixer.conv1d.weight[:, 0, :]).sum(-1) + mixer.conv1d.bias
    activated = F.silu(conv_out)
    dt_raw, b, c = torch.split(mixer.x_proj(activated), (mixer.dt_rank, rank, rank), -1)
    dt = F.softplus(mixer.dt_proj(dt_raw))
    a = -mixer.A_log.float().exp()
    ssm = torch.exp(dt[:, :, None] * a[None]) * ssm + dt[:, :, None] * b[:, None, :] * activated[:, :, None]
    scan = (ssm * c[:, None, :]).sum(-1) + activated * mixer.D
    mixed = mixer.out_proj(scan * F.silu(gate))
    z = model.post_norm(u + mixed)
    return z, torch.cat((z, conv.flatten(1), ssm.flatten(1)), -1)


def test_mamba_shapes_and_reference():
    torch.manual_seed(15)
    for in_dim in (128, 192):
        model = MambaBackbone(in_dim)
        state = model.initial_state(3, torch.device("cuda"))
        assert state.shape == (3, 5248)
        model = model.cuda()
        ref = state.clone()
        for _ in range(5):
            x = torch.randn(3, in_dim, device="cuda")
            output, state = model.step(x, state)
            expected, ref = _reference_step(model, x, ref)
            torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)
            torch.testing.assert_close(state, ref, rtol=1e-5, atol=1e-5)
        assert output.shape == (3, 128)
    assert sum(p.numel() for p in MambaBackbone(128).mixer.parameters()) == 116480
    assert sum(p.numel() for p in GRUBackbone(128).parameters()) == 99072


def test_mamba_gating_reset_and_autograd():
    torch.manual_seed(16)
    model = MambaBackbone(128).cuda()
    inputs = torch.randn(10, 2, 128, device="cuda", requires_grad=True)
    state = model.initial_state(2, inputs.device)
    previous = state
    for step in range(10):
        if step == 5:
            state = torch.where(torch.tensor([[False], [True]], device="cuda"), torch.zeros_like(state), state)
        update = torch.tensor([step != 4, step % 3 == 0], device="cuda")
        output, state = model.gated_step(inputs[step], state, update)
        if step == 4:
            torch.testing.assert_close(state[0], previous[0], rtol=0, atol=0)
        previous = state
    output.sum().backward()
    assert inputs.grad[0, 0].abs().sum() > 0
    for name, parameter in model.mixer.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
    assert model.output(state).shape == (2, 128)


def test_mamba_student_sequence_ema_and_gru_default():
    torch.manual_seed(17)
    old = StudentDepthModel(method="gru", image_hw=(8, 8), cnn_channels=(4,))
    torch.manual_seed(17)
    explicit = StudentDepthModel(method="gru", backbone="gru", image_hw=(8, 8), cnn_channels=(4,))
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    legacy_depth = torch.randn(3, 2, 8, 8)
    legacy_fresh = torch.tensor([True, False, True])
    old_context, _ = old.encode_step(legacy_depth, legacy_fresh, None)
    new_context, _ = explicit.encode_step(legacy_depth, legacy_fresh, None)
    torch.testing.assert_close(old_context, new_context, rtol=0, atol=0)
    torch.testing.assert_close(old.terrain_latent(old_context).square().mean(),
                               explicit.terrain_latent(new_context).square().mean(), rtol=0, atol=0)
    model = StudentDepthModel(method="gru", backbone="mamba", image_hw=(8, 8), cnn_channels=(4,)).cuda()
    assert model.ema_backbone.state_dim == model.backbone.state_dim
    assert not any(p.requires_grad for p in model.ema_backbone.parameters())
    with torch.no_grad():
        online = next(model.backbone.parameters())
        target = next(model.ema_backbone.parameters())
        before = target.clone()
        online.add_(1)
        model.update_ema(0.5)
        torch.testing.assert_close(target, before + 0.5)
    depth = torch.randn(7, 2, 2, 8, 8, device="cuda")
    fresh = torch.tensor([[1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 1]],
                         device="cuda", dtype=torch.bool)
    reset = torch.zeros_like(fresh)
    reset[3, 0] = True
    h0 = model.initial_state(2, depth.device)
    with torch.no_grad():
        collected = []
        state = h0
        for t in range(len(depth)):
            state = torch.where(reset[t, :, None], torch.zeros_like(state), state)
            out, state = model.encode_step(depth[t], fresh[t], state)
            collected.append(out)
        first, middle = model.encode_sequence(depth[:4], fresh[:4], reset[:4], h0)
        second, _ = model.encode_sequence(depth[4:], fresh[4:], reset[4:], middle)
    torch.testing.assert_close(torch.stack(collected), torch.cat((first, second)), rtol=1e-5, atol=1e-5)


def test_mamba_config_and_proprio():
    assert StudentTrainCfg().backbone == "gru"
    with pytest.raises(ValueError):
        StudentTrainCfg(method="no_memory", backbone="mamba").validate_training(0.02, 5)
    with pytest.raises(ValueError):
        StudentDepthModel(method="no_memory", backbone="mamba")
    model = StudentDepthModel(method="gru", backbone="mamba", proprio_input=True)
    assert model.backbone.in_dim == 192
    assert model.backbone.input_projection.in_features == 192
