# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""학습 전 라이브 시각화 — 정책 없이 GUI 로 로봇 + D435i 전방 프러스텀 point cloud 마커 확인.

zero action 이 기본 자세(default joint)를 유지시켜 로봇이 **서 있고**, pc_scanner debug_vis 로
전방 프러스텀 hit 점이 마커(구슬)로 표시된다. 지형(계단/박스/슬로프) 위에서 카메라가 앞을
어떻게 보는지 GUI 로 직접 확인. (체크포인트/정책 불필요 → 학습 전에 사용)

사용:  python scripts/je_loco/viz_pointcloud.py --num_envs 16
       (GUI 창이 열림. 창 닫으면 종료. 마우스로 시점 이동해서 마커 관찰)
"""

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Live viz of JE-Loco D435i frustum point cloud (no policy).")
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)          # --headless 안 주면 GUI
simulation_app = app_launcher.app

import torch

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
import unitree_rl_lab.je_loco.rsl_rl_pc  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

torch.backends.cudnn.enabled = False


def main():
    # play cfg 사용(pc_scanner debug_vis=True, rough+계단 지형, max_init_level=5)
    env_cfg = parse_env_cfg("Unitree-Go2-JELoco-PC", num_envs=args_cli.num_envs,
                            entry_point_key="play_env_cfg_entry_point")
    env = gym.make("Unitree-Go2-JELoco-PC", cfg=env_cfg)
    uenv = env.unwrapped

    obs, _ = env.reset()
    n_act = uenv.action_manager.total_action_dim
    act = torch.zeros((uenv.num_envs, n_act), device=uenv.device)   # 0 → 기본 자세 유지(서 있음)
    dt = uenv.step_dt

    print("\n[viz] 로봇이 기본 자세로 서 있고, 앞쪽에 D435i 프러스텀 마커(구슬)가 표시됩니다.")
    print("[viz] 마우스로 시점 이동해 계단/박스/슬로프 위 프러스텀을 관찰하세요. 창 닫으면 종료.\n")

    while simulation_app.is_running():
        t0 = time.time()
        with torch.inference_mode():
            obs, *_ = env.step(act)
        sleep = dt - (time.time() - t0)
        if sleep > 0:
            time.sleep(sleep)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
