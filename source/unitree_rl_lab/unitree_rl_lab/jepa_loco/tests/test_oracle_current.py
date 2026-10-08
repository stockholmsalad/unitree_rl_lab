"""현재 heightscan oracle의 관측·latent·actor 연결."""

from itertools import product

import torch
from tensordict import TensorDict

from unitree_rl_lab.jepa_loco.envs.oracle_env_cfg import (
    OracleCurrentEnvCfg,
    OracleCurrentFutureEnvCfg,
    OracleCurrentEasyStartEnvCfg,
    OracleCurrentFutureEasyStartEnvCfg,
    OracleCurrentFutureEasyStartEnvCfg_PLAY,
    OracleWideEnvCfg,
)
from unitree_rl_lab.jepa_loco.envs.oracle_heightmap import sample_command_future_patch
from unitree_rl_lab.jepa_loco.models.oracle_actor import OracleTerrainActor
from unitree_rl_lab.jepa_loco.envs.path_curriculum import terrain_levels_path
from unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg import (
    OracleCurrentPPORunnerCfg,
    OracleWidePPORunnerCfg,
    OracleCurrentFuturePPORunnerCfg,
    OracleCurrentFutureEasyStartPPORunnerCfg,
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


def test_easystart_current_future_shares_all_non_actor_settings():
    current, future = OracleCurrentEasyStartEnvCfg(), OracleCurrentFutureEasyStartEnvCfg()
    for name in ("rewards", "curriculum", "commands", "terminations", "events", "progress",
                 "swing_clearance", "nominal_pose"):
        assert getattr(current, name).to_dict() == getattr(future, name).to_dict(), name
    assert current.scene.terrain.terrain_generator.to_dict() == future.scene.terrain.terrain_generator.to_dict()
    assert current.observations.critic.to_dict() == future.observations.critic.to_dict()
    assert current.observations.policy.to_dict() == future.observations.policy.to_dict()
    assert current.episode_length_s == future.episode_length_s
    assert current.sim.to_dict() == future.sim.to_dict()
    assert current.decimation == future.decimation
    assert future.rewards.joint_pos is None
    assert future.rewards.swing_foot_clearance.weight == 1.0
    assert future.rewards.nominal_hip.weight == -0.7
    assert future.rewards.nominal_thigh_calf.weight == -0.15
    assert future.progress.warmup_levels == 2
    assert future.scene.d435i is None
    assert future.scene.wide_height_scanner.update_period == future.sim.dt

    current_runner, future_runner = OracleCurrentPPORunnerCfg(), OracleCurrentFutureEasyStartPPORunnerCfg()
    for name in ("algorithm", "critic"):
        assert getattr(current_runner, name).to_dict() == getattr(future_runner, name).to_dict(), name
    assert current_runner.num_steps_per_env == future_runner.num_steps_per_env == 100
    assert current_runner.max_iterations == future_runner.max_iterations
    assert current_runner.save_interval == future_runner.save_interval
    assert current_runner.obs_groups["critic"] == future_runner.obs_groups["critic"]
    assert current_runner.actor.terrain_latent_dim == future_runner.actor.terrain_latent_dim == 32
    assert current_runner.actor.terrain_hidden_dim == future_runner.actor.terrain_hidden_dim == 128
    assert future_runner.actor.future_group == "terrain_future"
    current_runner_dict, future_runner_dict = current_runner.to_dict(), future_runner.to_dict()
    for key in ("actor", "obs_groups", "experiment_name"):
        current_runner_dict.pop(key)
        future_runner_dict.pop(key)
    assert current_runner_dict == future_runner_dict


def test_easystart_current_future_encoder_capacity_and_play_cfg():
    current, future = OracleCurrentPPORunnerCfg(), OracleCurrentFutureEasyStartPPORunnerCfg()
    obs = TensorDict({"policy": torch.randn(3, 45), "terrain_current": torch.randn(3, 187),
                      "terrain_future": torch.randn(3, 187)}, batch_size=[3])
    current_actor = OracleTerrainActor(obs, current.obs_groups, "actor", 12,
                                       terrain_latent_dim=current.actor.terrain_latent_dim,
                                       terrain_hidden_dim=current.actor.terrain_hidden_dim)
    future_actor = OracleTerrainActor(obs, future.obs_groups, "actor", 12,
                                      future_group=future.actor.future_group,
                                      terrain_latent_dim=future.actor.terrain_latent_dim,
                                      terrain_hidden_dim=future.actor.terrain_hidden_dim)
    assert current_actor.get_terrain_latent(obs["terrain_current"]).shape == (3, 32)
    assert future_actor.get_terrain_latent(obs["terrain_future"]).shape == (3, 32)
    assert current_actor.last_z is None and future_actor.last_z is None
    assert current_actor.get_latent(obs).shape == (3, 77)
    assert future_actor.get_latent(obs).shape == (3, 109)
    assert future_actor.last_z.shape == (3, 64)
    assert future_actor.terrain_encoder[0].in_features == 187
    assert future_actor.terrain_encoder[0].out_features == 128
    play = OracleCurrentFutureEasyStartEnvCfg_PLAY()
    assert play.scene.num_envs == 32
    assert play.commands.base_velocity.ranges == play.commands.base_velocity.final_ranges
    assert play.scene.terrain.max_init_terrain_level == play.easy_start.num_rows - 1
    assert play.events.push_robot is None


def test_future_patch_visible_at_final_command_extremes():
    cfg = OracleCurrentFutureEasyStartEnvCfg()
    ranges = cfg.commands.base_velocity.final_ranges
    wide = cfg.scene.wide_height_scanner.pattern_cfg
    current = cfg.scene.height_scanner.pattern_cfg
    # 회전 중 패치의 가장 먼 점은 yaw-rate 끝값 사이에서 나타날 수도 있다.
    yaw_rates = torch.linspace(*ranges.ang_vel_z, 201).tolist()
    commands = torch.tensor(list(product(ranges.lin_vel_x, ranges.lin_vel_y, yaw_rates)))
    n_wide = (round(wide.size[0] / wide.resolution) + 1) * (round(wide.size[1] / wide.resolution) + 1)
    patch, visible = sample_command_future_patch(torch.zeros(len(commands), n_wide), commands,
                                                  cfg.future_horizon_s, wide.size, current.size, wide.resolution)
    assert patch.shape == (len(commands), 187)
    assert visible.all()
