import gymnasium as gym

gym.register(
    id="Unitree-Go2-JepaLoco-DepthGRU",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.depth_env_cfg:JepaDepthEnvCfg",
        "play_env_cfg_entry_point": "unitree_rl_lab.jepa_loco.envs.depth_env_cfg:JepaDepthEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg:DepthGRUPPORunnerCfg",
    },
)
