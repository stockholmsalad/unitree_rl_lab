# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause

"""JE-Loco point cloud 정책 재생(play) — 학습된 체크포인트 시각화.

scripts/rsl_rl/play.py 기반 + (1) je_loco.rsl_rl_pc task 등록, (2) cudnn 비활성(Blackwell GRU).
plain OnPolicyRunner 로 로드하되, 모델은 agent_cfg(repr_head="jepa")로 동일 재생성되어
jepa 헤드까지 포함해 정확히 로드됨(추론엔 pc_encoder+rnn+mlp 만 사용, 헤드는 미사용).

사용:
  python scripts/je_loco/play_pc.py --task Unitree-Go2-JELoco-PC --num_envs 32 --real-time
  # 특정 체크포인트: --checkpoint logs/rsl_rl/je_loco_pc/<run>/model_800.pt
  # GUI 없이 영상만: --headless --video --enable_cameras
"""

import argparse
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Play a JE-Loco point cloud RL checkpoint.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during play.")
parser.add_argument("--video_length", type=int, default=400, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Unitree-Go2-JELoco-PC", help="Name of the task.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--terrain_level",
    type=int,
    default=-1,
    help="지형 난이도 상한(0~9). 로봇은 [0, N] 에 흩어져 스폰된다. -1 = env cfg 기본값 사용.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlVecEnvWrapper,
    handle_deprecated_rsl_rl_cfg,
)
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
import unitree_rl_lab.je_loco.rsl_rl_pc  # noqa: F401  ← JE-Loco PC task 등록
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

# RTX 50-series(Blackwell) + torch2.7 는 GRU cudnn 커널 부재 → native 폴백.
torch.backends.cudnn.enabled = False


def main():
    """Play with the JE-Loco PC agent."""
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    # ── 지형 커리큘럼 해제 (2026-08-19 버그 수정) ────────────────────────────
    # scripted_terrain_levels 는 난이도 상한을 iteration(= common_step_counter/32)으로 정하는데,
    # play 는 step_counter 가 0 부터라 frac=0.10 → ceiling=1 → **매 reset 마다 전 로봇이 레벨 0**.
    # 즉 play_env_cfg 의 max_init_terrain_level 이 통째로 무시돼, 지금까지 육안 확인을 전부
    # 평지에서 해왔다. 커리큘럼을 끄면 max_init_terrain_level 이 지배한다.
    # (eval_foothold.py 는 원래 이 처리를 하고 있었다 — play 만 빠져 있었음.)
    if getattr(env_cfg, "curriculum", None) is not None and hasattr(env_cfg.curriculum, "terrain_levels"):
        env_cfg.curriculum.terrain_levels = None
    if args_cli.terrain_level >= 0:
        env_cfg.scene.terrain.max_init_terrain_level = args_cli.terrain_level
    print(f"[play] 지형 커리큘럼 OFF, 스폰 레벨 = 0~{env_cfg.scene.terrain.max_init_terrain_level}")

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    if args_cli.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=os.path.join(log_dir, "videos", "play"),
            step_trigger=lambda step: step == 0,
            video_length=args_cli.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # 증류 체크포인트는 키가 다르다(student/teacher_state_dict). PPO 의 load 는
    # loaded_dict["actor_state_dict"] 를 직접 인덱싱하므로 그대로 쓰면 KeyError 이고,
    # Distill 태스크의 agent cfg 도 actor/critic 이 아니라 student/teacher 라 runner 자체가 다르다.
    # → 알고리즘에 맞는 runner 를 골라야 한다. (2026-08-19: 이 분기 없이 재생하면 정책이 로드되지
    #    않아 로봇이 기본 자세로 가만히 서 있는 것처럼 보인다 — 학습 실패로 오인하기 쉬움.)
    if getattr(agent_cfg.algorithm, "class_name", "") == "Distillation":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        # student·teacher·optimizer 전부 들어 있음 → load_cfg=None 이면 알아서 다 적재.
        # get_policy() 가 student 를 반환하므로 재생되는 건 배포 대상 정책이 맞다.
        runner.load(resume_path, load_cfg={"student": True, "teacher": True,
                                           "optimizer": False, "iteration": False}, strict=False)
    else:
        # plain OnPolicyRunner 로 충분(추론만). 모델은 cfg 로 동일 재생성되어 헤드 포함 정확히 로드.
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        # 표현 헤드 무관하게 policy backbone 만 로드(config repr_head ≠ 체크포인트 헤드여도 재생 가능)
        runner.load(resume_path,
                    load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
                    strict=False)

    policy = runner.get_inference_policy(device=env.unwrapped.device)

    dt = env.unwrapped.step_dt
    obs = env.get_observations()
    timestep = 0
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _ = env.step(actions)
        if args_cli.video:
            timestep += 1
            if timestep == args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
