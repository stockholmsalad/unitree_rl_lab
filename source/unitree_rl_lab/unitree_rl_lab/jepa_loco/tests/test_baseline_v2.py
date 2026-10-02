"""재학습 설정, blind 순환 정책, 진단 로깅의 단위 테스트."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from unitree_rl_lab.jepa_loco.envs.command_schedule import scheduled_ranges
from unitree_rl_lab.jepa_loco.envs.diagnostic_env import reward_step_extrema
from unitree_rl_lab.jepa_loco.models.proprio_actor import ProprioRecurrentActor


def test_command_schedule_endpoints_and_midpoint():
    start = ((-0.1, 0.1),) * 3
    end = ((-0.3, 1.0), (-0.4, 0.4), (-0.8, 0.8))
    assert scheduled_ranges(start, end, 0, 50000) == start
    assert scheduled_ranges(start, end, 50000, 50000) == end
    assert scheduled_ranges(start, end, 60000, 50000) == end
    mid = scheduled_ranges(start, end, 25000, 50000)
    assert mid[0] == pytest.approx((-0.2, 0.55))
    assert mid[2] == pytest.approx((-0.45, 0.45))
    with pytest.raises(ValueError):
        scheduled_ranges(start, end, 0, 0)


def test_reward_step_extrema_shape_and_dt():
    manager = SimpleNamespace(_term_names=["tracking", "joint_acc"],
                              _step_reward=torch.tensor([[1.0, -5.0], [2.0, -3.0]]))
    lo, hi = reward_step_extrema(manager, 0.02)
    assert set(lo) == set(hi) == {"tracking", "joint_acc"}
    assert lo["joint_acc"].item() == pytest.approx(-0.1)
    assert hi["tracking"].item() == pytest.approx(0.04)


def test_blind_actor_rollout_matches_update_across_reset():
    torch.manual_seed(2)
    T, N = 6, 4
    obs = TensorDict({"policy": torch.randn(T, N, 45)}, batch_size=[T, N])
    actor = ProprioRecurrentActor(obs[0], {"actor": ["policy"]}, "actor", 12,
                                  hidden_dims=[64], distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0})
    hidden = torch.zeros(1, N, 128)
    zs = []
    for t in range(T):
        zs.append(actor.get_latent(obs[t]))
        if t == 2:
            actor.reset(torch.tensor([0, 1, 0, 0]))
    reset = torch.zeros(T, N, 1)
    reset[3, 1] = 1
    obs["traj_start"] = reset
    batch = actor.get_latent(obs, masks=torch.ones(T, N, dtype=torch.bool), hidden_state=hidden)
    assert batch.shape == (T, N, 173)
    assert torch.allclose(batch, torch.stack(zs), atol=1e-5)


def test_baseline_cfg_camera_and_termination():
    from unitree_rl_lab.jepa_loco.envs.blind_env_cfg import JepaBlindEnvCfg
    from unitree_rl_lab.jepa_loco.envs.depth_env_cfg import JepaDepthEnvCfg
    from unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg import BlindGRUPPORunnerCfg, DepthGRUPPORunnerCfg

    depth, blind = JepaDepthEnvCfg(), JepaBlindEnvCfg()
    assert depth.rewards.termination.weight == blind.rewards.termination.weight == -200
    assert depth.scene.d435i is not None and blind.scene.d435i is None
    assert blind.observations.depth is None and blind.observations.depth_fresh is None
    assert depth.commands.base_velocity.final_ranges == blind.commands.base_velocity.final_ranges
    assert DepthGRUPPORunnerCfg().algorithm.entropy_coef == BlindGRUPPORunnerCfg().algorithm.entropy_coef == 0.01


def test_termination_reward_excludes_timeout():
    from isaaclab.envs.mdp.rewards import is_terminated

    manager = SimpleNamespace(terminated=torch.tensor([False, True, False, True]))
    assert torch.equal(is_terminated(SimpleNamespace(termination_manager=manager)),
                       torch.tensor([0.0, 1.0, 0.0, 1.0]))
