"""DepthCNN / backbone / DepthRecurrentActor shape·의미 테스트."""

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.utils import split_and_pad_trajectories

from unitree_rl_lab.jepa_loco.models.backbone import GRUBackbone, make_backbone
from unitree_rl_lab.jepa_loco.models.depth_actor import DepthRecurrentActor
from unitree_rl_lab.jepa_loco.models.depth_cnn import DepthCNN

N, H, W = 6, 64, 112


def _obs(n=N, fresh=None, lead=()):
    shape = (*lead, n)
    return TensorDict(
        {
            "policy": torch.randn(*shape, 45),
            "depth": torch.rand(*shape, 2, H, W),
            "depth_fresh": (torch.ones(*shape, 1) if fresh is None else fresh),
        },
        batch_size=list(shape),
    )


def _actor():
    torch.manual_seed(0)
    return DepthRecurrentActor(
        _obs(),
        {"actor": ["policy", "depth", "depth_fresh"]},
        "actor",
        12,
        hidden_dims=[64],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )


def test_cnn_shape():
    assert DepthCNN()(torch.rand(3, 2, H, W)).shape == (3, 128)


def test_backbone_gating_keeps_state():
    bb = GRUBackbone(8, 16)
    h = torch.randn(4, 16)
    upd = torch.tensor([True, False, True, False])
    z, h2 = bb.gated_step(torch.randn(4, 8), h, upd)
    assert z.shape == (4, 16)
    assert torch.equal(h2[~upd], h[~upd]) and not torch.allclose(h2[upd], h[upd])
    with pytest.raises(ValueError):
        make_backbone("transformer", 8, 16)


def test_actor_rollout_shapes_and_reset():
    a = _actor()
    assert a.obs_groups == ["policy"] and a.obs_dim == 45
    out = a(_obs(), stochastic_output=True)
    assert out.shape == (N, 12)
    hs = a.get_hidden_state()
    assert hs.shape == (1, N, 128)
    dones = torch.zeros(N)
    dones[2] = 1
    a.reset(dones)
    h = a.get_hidden_state()[0]
    assert (h[2] == 0).all() and (h[0] != 0).any()


def test_stale_frames_reuse_latent():
    a = _actor()
    a(_obs())
    z0 = a.last_z.clone()
    a(_obs(fresh=torch.zeros(N, 1)))  # 새 프레임 없음 → z 재사용(depth 가 바뀌어도)
    assert torch.equal(a.last_z, z0)


def test_rollout_matches_batched_update():
    """rollout 에서 스텝별로 만든 latent 와 update 의 trajectory 재계산이 같아야 PPO 비율이 맞는다."""
    a = _actor()
    T = 12
    fresh = torch.zeros(T, N, 1)
    fresh[::3] = 1  # 3스텝마다 새 프레임
    seq = _obs(fresh=fresh, lead=(T,))
    dones = torch.zeros(T, N)
    dones[4, 1] = 1  # env 1 은 중간에 에피소드 종료
    dones[7, 3] = 1

    a.reset()
    lat_roll, hidden = [], []
    for t in range(T):
        hidden.append(a.get_hidden_state())
        lat_roll.append(a.get_latent(seq[t]))
        a.reset(dones[t])
    lat_roll = torch.stack(lat_roll)

    # 저장된 hidden: 첫 스텝 None → 0
    hidden[0] = torch.zeros(1, N, 128)
    saved = torch.stack(hidden)  # [T, 1, N, H]
    padded, masks = split_and_pad_trajectories(seq, dones[..., None])
    last_was_done = torch.zeros(T, N, dtype=torch.bool)
    last_was_done[1:] = dones[:-1].bool()
    last_was_done[0] = True
    h0 = saved.permute(2, 0, 1, 3)[last_was_done.T].transpose(1, 0)
    lat_batch = a.get_latent(padded, masks=masks, hidden_state=h0)
    assert lat_batch.shape == lat_roll.shape
    assert torch.allclose(lat_batch, lat_roll, atol=1e-5)
