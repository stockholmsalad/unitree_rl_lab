"""미니배치 단위 패딩 생성기가 rsl_rl 원본과 같은 배치를 내는지."""

import torch
from tensordict import TensorDict

from rsl_rl.storage import RolloutStorage

from unitree_rl_lab.jepa_loco.agents.ppo import SlicedPadRolloutStorage


def _filled(cls, T=10, N=8, A=3, H=5):
    torch.manual_seed(0)
    obs = TensorDict({"policy": torch.zeros(N, 4), "depth": torch.zeros(N, 2, 3, 4)}, batch_size=[N])
    st = cls("rl", N, T, obs, [A], "cpu")
    st.observations = TensorDict(
        {"policy": torch.randn(T, N, 4), "depth": torch.randn(T, N, 2, 3, 4)}, batch_size=[T, N]
    )
    st.dones = (torch.rand(T, N, 1) < 0.2).float()
    for name in ("values", "advantages", "returns", "actions_log_prob"):
        setattr(st, name, torch.randn(T, N, 1))
    st.actions = torch.randn(T, N, A)
    st.distribution_params = (torch.randn(T, N, A), torch.rand(T, N, A))
    st.saved_hidden_state_a = [torch.randn(T, 1, N, H)]
    st.saved_hidden_state_c = None
    return st


def test_sliced_generator_matches_original():
    ref = list(_filled(RolloutStorage).recurrent_mini_batch_generator(4, num_epochs=2))
    new = list(_filled(SlicedPadRolloutStorage).recurrent_mini_batch_generator(4, num_epochs=2))
    assert len(ref) == len(new) == 8
    for a, b in zip(ref, new):
        assert torch.equal(a.masks, b.masks)
        for k in ("policy", "depth"):
            assert torch.equal(a.observations[k], b.observations[k])
        assert torch.equal(a.hidden_states[0], b.hidden_states[0]) and b.hidden_states[1] is None
        assert torch.equal(a.actions, b.actions) and torch.equal(a.returns, b.returns)
        assert all(torch.equal(x, y) for x, y in zip(a.old_distribution_params, b.old_distribution_params))
