"""스케줄 검사·resume 스텝 복원·기억 없는 blind MLP 정책 — Isaac 앱 불필요."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel

from unitree_rl_lab.jepa_loco.agents.checks import check_command_schedule, restore_global_step
from unitree_rl_lab.jepa_loco.models.depth_actor import DepthRecurrentActor

DIST = {"class_name": "GaussianDistribution", "init_std": 1.0}


def test_schedule_check():
    check_command_schedule(SimpleNamespace(steps_per_iteration=100), 100)
    check_command_schedule(SimpleNamespace(), 24)  # 스케줄 없는 명령은 통과
    check_command_schedule(None, 24)
    with pytest.raises(ValueError, match="steps_per_iteration=100"):
        check_command_schedule(SimpleNamespace(steps_per_iteration=100), 24)


def test_restore_global_step():
    env = SimpleNamespace(common_step_counter=0)
    assert restore_global_step(env, 1200, 100) == 120000 and env.common_step_counter == 120000
    assert restore_global_step(env, 0, 100) == 0
    with pytest.raises(ValueError):
        restore_global_step(env, -1, 100)
    with pytest.raises(ValueError):
        restore_global_step(env, 10, 0)


def _obs(n=5):
    return TensorDict(
        {
            "policy": torch.randn(n, 45),
            "depth": torch.rand(n, 2, 64, 112),
            "depth_fresh": torch.ones(n, 1),
        },
        batch_size=[n],
    )


def test_blind_mlp_shape_and_matches_depth_head():
    """blind MLP = depth 정책 MLP head 에서 입력의 z_t(128) 만 뺀 구조여야 한다."""
    obs = _obs()
    blind = MLPModel(obs, {"actor": ["policy"]}, "actor", 12, hidden_dims=[512, 256, 128], activation="elu",
                     obs_normalization=True, distribution_cfg=dict(DIST))
    depth = DepthRecurrentActor(obs, {"actor": ["policy", "depth", "depth_fresh"]}, "actor", 12,
                                hidden_dims=[512, 256, 128], obs_normalization=True, distribution_cfg=dict(DIST))
    assert not blind.is_recurrent and blind.get_hidden_state() is None
    assert blind(obs, stochastic_output=True).shape == (5, 12)
    lin_b = [m for m in blind.mlp if isinstance(m, torch.nn.Linear)]
    lin_d = [m for m in depth.mlp if isinstance(m, torch.nn.Linear)]
    assert [m.in_features for m in lin_b] == [45, 512, 256, 128]
    assert [m.in_features for m in lin_d] == [45 + 128, 512, 256, 128]
    assert [m.out_features for m in lin_b] == [m.out_features for m in lin_d]
    # depth 관측이 있어도 blind 는 policy 그룹만 쓴다
    obs2 = obs.clone()
    obs2["depth"] = torch.rand_like(obs2["depth"])
    torch.testing.assert_close(blind(obs), blind(obs2))
