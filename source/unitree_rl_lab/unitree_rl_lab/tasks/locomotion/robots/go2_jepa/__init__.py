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
