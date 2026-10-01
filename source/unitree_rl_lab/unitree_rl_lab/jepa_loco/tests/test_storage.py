"""패딩 없는 순환 미니배치: update 에서 재계산한 latent 가 rollout 의 latent 와 같아야 PPO 비율이 맞는다."""

import torch
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage

from unitree_rl_lab.jepa_loco.agents.ppo import ResetMaskRolloutStorage
from unitree_rl_lab.jepa_loco.models.depth_actor import DepthRecurrentActor

T, N, H, W = 12, 8, 64, 112


def _obs(lead):
    return TensorDict(
        {
            "policy": torch.randn(*lead, 45),
            "depth": torch.rand(*lead, 2, H, W).half(),
            "depth_fresh": (torch.rand(*lead, 1) < 0.3).float(),
        },
        batch_size=list(lead),
    )


def test_reset_mask_generator_reproduces_rollout_latents():
    torch.manual_seed(0)
    actor = DepthRecurrentActor(
        _obs((N,)), {"actor": ["policy", "depth", "depth_fresh"]}, "actor", 12, hidden_dims=[64],
        distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0},
    )
    st = ResetMaskRolloutStorage("rl", N, T, _obs((N,)), [12], "cpu")
    assert isinstance(st, RolloutStorage)
    seq = _obs((T, N))
    seq["depth_fresh"][0] = 1.0
    dones = (torch.rand(T, N, 1) < 0.25).float()

    # 이전 rollout 에서 이어진 0 아닌 상태로 시작
    actor.reset()
    warm = _obs((N,))
    warm["depth_fresh"] = torch.ones(N, 1)
    actor.get_latent(warm)
    assert actor.get_hidden_state().abs().sum() > 0
    saved, lat_roll = [], []
    for t in range(T):
        saved.append(actor.get_hidden_state().clone())
        lat_roll.append(actor.get_latent(seq[t]))
        actor.reset(dones[t, :, 0])
    lat_roll = torch.stack(lat_roll)

    st.observations = seq
    st.dones = dones
    for name in ("values", "advantages", "returns", "actions_log_prob"):
        setattr(st, name, torch.zeros(T, N, 1))
    st.actions = torch.zeros(T, N, 12)
    st.distribution_params = (torch.zeros(T, N, 12), torch.ones(T, N, 12))
    st.saved_hidden_state_a = [torch.stack(saved)]  # [T, 1, N, H]
    st.saved_hidden_state_c = None

    batches = list(st.recurrent_mini_batch_generator(2, num_epochs=1))
    assert len(batches) == 2
    for i, b in enumerate(batches):
        sl = slice(i * N // 2, (i + 1) * N // 2)
        assert b.masks.all() and b.hidden_states[1] is None
        assert b.observations["traj_start"][0].sum() == 0  # t=0 은 저장된 h0 사용
        lat = actor.get_latent(b.observations, masks=b.masks, hidden_state=b.hidden_states[0])
        assert torch.allclose(lat, lat_roll[:, sl], atol=1e-5)
