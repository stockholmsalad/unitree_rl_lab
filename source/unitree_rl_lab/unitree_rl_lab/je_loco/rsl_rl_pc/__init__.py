# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""JE-Loco point cloud locomotion — **rsl_rl 5.0.1 검증 PPO** 기반.

배경(2026-07-09): 커스텀 PPO 루프가 걷기 학습에 실패(붕괴)했고, working 계단 실험은
rsl_rl OnPolicyRunner + recurrent(GRU) ActorCritic + PointNet 으로 잘 됐다. 그 구조를
rsl_rl 5.0.1 의 새 Model API(RNNModel 상속 + get_latent 오버라이드로 PointNet 주입)로 이식.

구성: RayCaster 격자(12×8=96점) → base-frame point cloud(288) → PointCloudRNNModel
(pointcloud→PointNet(64) ⊕ proprio → GRU(256) → MLP → action). rsl_rl 이 PPO/GAE/BPTT 처리.
"""

import gymnasium as gym

# lazy 등록(문자열 entry point) — isaaclab import 없이 등록되어 앱 기동 전 task 목록에 뜬다.
gym.register(
    id="Unitree-Go2-JELoco-PC",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:JELocoPCEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:JELocoPCPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:JELocoPCPPORunnerCfg",
    },
)

# Stage 2 — 계단 정밀 foothold (계단 지형 + foothold 보상 2항). 정책 입력은 Stage 1 과 동일.
gym.register(
    id="Unitree-Go2-JELoco-Foothold",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:JELocoPCFootholdEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:JELocoPCFootholdPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:JELocoPCPPORunnerCfg",
    },
)

# Phase 3b — DAgger 증류. Foothold env + teacher 관측 그룹(232). student=pc+proprio.
# 비교 축 = student 인코더 초기화(jepa/recon/scratch). teacher 체크포인트는
# train_pc.py --teacher_checkpoint 로 지정.
gym.register(
    id="Unitree-Go2-JELoco-Distill",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:JELocoDistillEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:JELocoDistillPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agent_cfg:JELocoDistillRunnerCfg",
    },
)

# Phase 1 Teacher — privileged heightmap actor, 순정 MLP PPO (pc·GRU 없음). Phase 2 데이터 공장.
gym.register(
    id="Unitree-Go2-JELoco-Teacher",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:JELocoTeacherEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.env_cfg:JELocoTeacherPlayEnvCfg",
        "rsl_rl_cfg_entry_point": "unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg:BasePPORunnerCfg",
    },
)
