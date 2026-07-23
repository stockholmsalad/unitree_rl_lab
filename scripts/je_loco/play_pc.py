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

from rsl_rl.runners import OnPolicyRunner

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
