"""BlindMLP task 설정 — configclass 를 쓰므로 Isaac 앱을 먼저 띄운 상태에서 실행 (scripts/jepa_loco/run_tests_in_app.py)."""


def test_blind_mlp_runner_cfg_matches_depth_except_actor():
    from unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg import (
        BlindGRUPPORunnerCfg,
        BlindMLPPPORunnerCfg,
        DepthGRUPPORunnerCfg,
    )

    depth, mlp, gru = DepthGRUPPORunnerCfg(), BlindMLPPPORunnerCfg(), BlindGRUPPORunnerCfg()
    assert mlp.actor.class_name == "MLPModel"
    assert mlp.obs_groups == {"actor": ["policy"], "critic": ["critic"]}
    assert mlp.actor.hidden_dims == depth.actor.hidden_dims
    assert mlp.actor.obs_normalization == depth.actor.obs_normalization
    assert mlp.num_steps_per_env == depth.num_steps_per_env
    assert mlp.algorithm.to_dict() == depth.algorithm.to_dict()
    assert mlp.critic.to_dict() == depth.critic.to_dict()
    assert mlp.experiment_name not in (depth.experiment_name, gru.experiment_name)
    # 기존 BlindGRU 는 그대로
    assert gru.actor.class_name.endswith("proprio_actor:ProprioRecurrentActor")


def test_command_schedule_consistent_with_rollout():
    from unitree_rl_lab.jepa_loco.agents.checks import check_command_schedule
    from unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg import (
        BlindGRUPPORunnerCfg,
        BlindMLPPPORunnerCfg,
        DepthGRUPPORunnerCfg,
    )
    from unitree_rl_lab.jepa_loco.envs.blind_env_cfg import JepaBlindEnvCfg
    from unitree_rl_lab.jepa_loco.envs.depth_env_cfg import JepaDepthEnvCfg

    for env_cfg, runner in (
        (JepaDepthEnvCfg(), DepthGRUPPORunnerCfg()),
        (JepaBlindEnvCfg(), BlindGRUPPORunnerCfg()),
        (JepaBlindEnvCfg(), BlindMLPPPORunnerCfg()),
    ):
        check_command_schedule(env_cfg.commands.base_velocity, runner.num_steps_per_env)
