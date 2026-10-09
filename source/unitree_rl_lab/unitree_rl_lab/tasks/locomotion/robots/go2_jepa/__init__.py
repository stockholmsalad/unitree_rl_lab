import gymnasium as gym

gym.register(
    id="Unitree-Go2-JepaLoco-DepthGRU",
    entry_point="unitree_rl_lab.jepa_loco.envs.diagnostic_env:JepaDiagnosticEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.depth_env_cfg:JepaDepthEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.depth_env_cfg:JepaDepthEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg:DepthGRUPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-JepaLoco-BlindGRU",
    entry_point="unitree_rl_lab.jepa_loco.envs.diagnostic_env:JepaDiagnosticEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.blind_env_cfg:JepaBlindEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.blind_env_cfg:JepaBlindEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg:BlindGRUPPORunnerCfg",
    },
)

# 기억 없는 proprio-only 대조군 (depth 경로만 뺀 첫 비교군). BlindGRU 는 "proprio 기억이 있는 강한 대조군".
gym.register(
    id="Unitree-Go2-JepaLoco-BlindMLP",
    entry_point="unitree_rl_lab.jepa_loco.envs.diagnostic_env:JepaDiagnosticEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.blind_env_cfg:JepaBlindEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.blind_env_cfg:JepaBlindEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg:BlindMLPPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-Go2-JepaLoco-OracleCurrent",
    entry_point="unitree_rl_lab.jepa_loco.envs.oracle_curriculum_env:OracleCurriculumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.oracle_env_cfg:OracleCurrentEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.oracle_env_cfg:OracleCurrentEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg:OracleCurrentPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-Go2-JepaLoco-OracleCurrent-EasyStart",
    entry_point="unitree_rl_lab.jepa_loco.envs.oracle_curriculum_env:OracleCurriculumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.oracle_env_cfg:OracleCurrentEasyStartEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.oracle_env_cfg:OracleCurrentEasyStartEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg:OracleCurrentPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-Go2-JepaLoco-OracleWide",
    entry_point="unitree_rl_lab.jepa_loco.envs.oracle_curriculum_env:OracleCurriculumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.oracle_env_cfg:OracleWideEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.oracle_env_cfg:OracleWideEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg:OracleWidePPORunnerCfg",
    },
)


gym.register(
    id="Unitree-Go2-JepaLoco-OracleCurrentFuture",
    entry_point="unitree_rl_lab.jepa_loco.envs.oracle_curriculum_env:OracleCurriculumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.oracle_env_cfg:OracleCurrentFutureEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.oracle_env_cfg:OracleCurrentFutureEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg:OracleCurrentFuturePPORunnerCfg",
    },
)


gym.register(
    id="Unitree-Go2-JepaLoco-OracleCurrentFuture-EasyStart",
    entry_point="unitree_rl_lab.jepa_loco.envs.oracle_curriculum_env:OracleCurriculumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.oracle_env_cfg:OracleCurrentFutureEasyStartEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.oracle_env_cfg:OracleCurrentFutureEasyStartEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg:OracleCurrentFutureEasyStartPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-Go2-JepaLoco-StudentDepth-EasyStart",
    entry_point="unitree_rl_lab.jepa_loco.envs.oracle_curriculum_env:OracleCurriculumEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.student_env_cfg:StudentDepthEasyStartEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.student_env_cfg:StudentDepthEasyStartEnvCfg_PLAY",
    },
)
