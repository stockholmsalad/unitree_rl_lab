"""현재 heightscan oracle의 관측·latent·actor 연결."""

import torch
from tensordict import TensorDict

from unitree_rl_lab.jepa_loco.envs.oracle_env_cfg import (
    OracleCurrentEnvCfg,
    OracleCurrentFutureEnvCfg,
    OracleWideEnvCfg,
)
from unitree_rl_lab.jepa_loco.models.oracle_actor import OracleTerrainActor
from unitree_rl_lab.jepa_loco.envs.path_curriculum import terrain_levels_path
from unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg import (
    OracleCurrentPPORunnerCfg,
    OracleWidePPORunnerCfg,
    OracleCurrentFuturePPORunnerCfg,
)


def test_oracle_cfg_has_heightscan_and_no_camera():
    cfg = OracleCurrentEnvCfg()
    assert cfg.scene.d435i is None
    assert cfg.observations.depth is None and cfg.observations.depth_fresh is None
    assert cfg.observations.terrain_current.scan.func is not None
    assert cfg.observations.critic.height_scan.func is not None


def test_oracle_actor_shapes_and_terrain_gradient():
    obs = TensorDict({"policy": torch.randn(4, 45), "terrain_current": torch.randn(4, 187)}, batch_size=[4])
    actor = OracleTerrainActor(obs, {"actor": ["policy", "terrain_current"]}, "actor", 12,
                               hidden_dims=[64], terrain_latent_dim=32, terrain_hidden_dim=48,
                               distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0})
    assert actor(obs).shape == (4, 12)
    assert actor.get_terrain_latent(obs["terrain_current"]).shape == (4, 32)
    assert actor.get_latent(obs).shape == (4, 77)
    actor(obs).sum().backward()
    assert actor.terrain_encoder[0].weight.grad.abs().sum() > 0


def test_wide_and_future_cfg_share_scan_grid_and_camera_free():
    wide = OracleWideEnvCfg()
    future = OracleCurrentFutureEnvCfg()
    assert wide.scene.d435i is None and future.scene.d435i is None
    assert wide.scene.wide_height_scanner.pattern_cfg.ordering == "yx"
    assert future.scene.wide_height_scanner.pattern_cfg.resolution == future.scene.height_scanner.pattern_cfg.resolution
    assert future.observations.terrain_future.scan.params["horizon_s"] == future.future_horizon_s


def test_three_oracles_share_settings_except_current_clearance_reward():
    current, wide, future = OracleCurrentEnvCfg(), OracleWideEnvCfg(), OracleCurrentFutureEnvCfg()
    for cfg in (current, wide, future):
        assert cfg.curriculum.terrain_levels.func is terrain_levels_path
        assert cfg.curriculum.terrain_levels.params == current.curriculum.terrain_levels.params
        assert cfg.scene.terrain.terrain_generator.to_dict() == current.scene.terrain.terrain_generator.to_dict()
        assert cfg.commands.base_velocity.to_dict() == current.commands.base_velocity.to_dict()
        assert cfg.progress.to_dict() == current.progress.to_dict()
        assert cfg.commands.base_velocity.final_ranges.lin_vel_x == (-0.3, 2.0)
        assert cfg.commands.base_velocity.final_ranges.ang_vel_z == (-1.0, 1.0)
    assert current.rewards.swing_foot_clearance.weight == current.swing_clearance.weight
    assert current.rewards.joint_pos is None
    assert current.rewards.nominal_hip.weight == current.nominal_pose.hip_weight
    assert current.rewards.nominal_thigh_calf.weight == current.nominal_pose.thigh_calf_weight
    assert not hasattr(wide.rewards, "swing_foot_clearance")
    assert not hasattr(future.rewards, "swing_foot_clearance")
    assert wide.rewards.to_dict() == future.rewards.to_dict()
    for name, term in wide.rewards.to_dict().items():
        if name == "joint_pos":
            continue
        assert current.rewards.to_dict()[name] == term
    runners = (OracleCurrentPPORunnerCfg(), OracleWidePPORunnerCfg(), OracleCurrentFuturePPORunnerCfg())
    for runner in runners[1:]:
        assert runner.num_steps_per_env == runners[0].num_steps_per_env
        assert runner.max_iterations == runners[0].max_iterations
        assert runner.algorithm.to_dict() == runners[0].algorithm.to_dict()


def test_future_oracle_shares_encoder_weights_and_matches_wide_capacity():
    obs = TensorDict({"policy": torch.randn(4, 45), "terrain_current": torch.randn(4, 187),
                      "terrain_future": torch.randn(4, 187)}, batch_size=[4])
    future = OracleTerrainActor(obs, {"actor": ["policy", "terrain_current", "terrain_future"]},
                                "actor", 12, hidden_dims=[64], future_group="terrain_future",
                                terrain_latent_dim=32)
    assert future.get_latent(obs).shape == (4, 109)
    assert future(obs).shape == (4, 12)
    wide_obs = TensorDict({"policy": torch.randn(4, 45), "terrain_wide": torch.randn(4, 1271)}, batch_size=[4])
    wide = OracleTerrainActor(wide_obs, {"actor": ["policy", "terrain_wide"]}, "actor", 12,
                              hidden_dims=[64], terrain_group="terrain_wide", terrain_latent_dim=64)
    assert wide.get_latent(wide_obs).shape == (4, 109)
