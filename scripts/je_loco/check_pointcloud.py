# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""point cloud 입력 검증 — D435i 전방 프러스텀이 여러 지형을 제대로 잡고 obs 로 흐르는지 확인.

확인 항목:
  (1) ray hit 유효율(지형 맞히는지)         (2) obs["pointcloud"] 차원이 모델 입력과 일치
  (3) 프러스텀 기하(전방 사다리꼴)           (4) 지형별 반응(평지=평면 / 박스·슬로프=굴곡)
출력: 콘솔 + pc_check_stats.txt(로그에 안 묻히게) + <out>(지형별 프러스텀 산점도 격자)

사용:  python scripts/je_loco/check_pointcloud.py --num_envs 9 --headless
       라이브 마커로 보려면 --headless 빼고(GUI) 실행 → 로봇 앞 프러스텀 마커
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Verify JE-Loco point cloud input.")
parser.add_argument("--num_envs", type=int, default=9)
parser.add_argument("--steps", type=int, default=5, help="reset 후 스텝(로봇 upright 유지 위해 소수)")
parser.add_argument("--out", type=str, default="pc_frustum_terrains.png")
parser.add_argument("--train", action="store_true", help="학습 cfg(env_cfg_entry_point) 로 확인(스폰 난이도)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import unitree_rl_lab.tasks  # noqa: F401
import unitree_rl_lab.je_loco.rsl_rl_pc  # noqa: F401
from isaaclab.managers import SceneEntityCfg
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.je_loco.rsl_rl_pc.mdp_pc import raycaster_pointcloud

torch.backends.cudnn.enabled = False
LINES = []
def log(s=""):
    print(s); LINES.append(str(s))


def main():
    ep_key = "env_cfg_entry_point" if args_cli.train else "play_env_cfg_entry_point"
    env_cfg = parse_env_cfg("Unitree-Go2-JELoco-PC", num_envs=args_cli.num_envs, entry_point_key=ep_key)
    env = gym.make("Unitree-Go2-JELoco-PC", cfg=env_cfg)
    uenv = env.unwrapped

    obs, _ = env.reset()
    n_act = uenv.action_manager.total_action_dim
    act = torch.zeros((uenv.num_envs, n_act), device=uenv.device)
    for _ in range(args_cli.steps):
        obs, *_ = env.step(act)

    N = uenv.num_envs
    sensor = uenv.scene.sensors["pc_scanner"]
    hits_w = sensor.data.ray_hits_w                 # (N, P, 3)
    P = hits_w.shape[1]
    finite = torch.isfinite(hits_w).all(dim=-1)     # (N, P)
    robot = uenv.scene["robot"]; base_pos = robot.data.root_pos_w

    pc = raycaster_pointcloud(uenv, SceneEntityCfg("pc_scanner")).reshape(N, -1, 3)   # base frame

    # obs 파이프라인 확인
    pol_obs = uenv.observation_manager.compute()
    pc_obs = pol_obs["pointcloud"]
    try:
        lvl = uenv.scene.terrain.terrain_levels.detach().cpu().numpy()
    except Exception:
        lvl = np.full(N, -1)

    log("=" * 64)
    log("[point cloud 입력 검증]")
    log(f"  센서 ray 수/env: {P}   (16×12=192 이면 정상)")
    log(f"  ray hit 유효율(지형 맞힘): {finite.float().mean().item()*100:.1f}%   "
        f"(전방 프러스텀은 상단 일부가 하늘/원거리→무효일 수 있음)")
    log(f"  obs['pointcloud'] shape: {tuple(pc_obs.shape)}   "
        f"(= (envs, {P}×3={P*3}) 여야 모델 입력과 일치)")
    log(f"  전방성 검증 base-frame x(전방) 범위: [{pc[...,0].min():.2f}, {pc[...,0].max():.2f}] m   "
        f"(전부 >0 이면 '앞만 봄'=D435i 정합)")
    log(f"  좌우 y 범위: [{pc[...,1].min():.2f}, {pc[...,1].max():.2f}]   "
        f"z(높이) 범위: [{pc[...,2].min():.2f}, {pc[...,2].max():.2f}]")
    log("-" * 64)
    log("  env별 지형반응 (level=난이도, relief=프러스텀 내 높이편차):")
    reliefs = []
    for i in range(N):
        v = finite[i]
        zr = (pc[i, v, 2].max() - pc[i, v, 2].min()).item() if v.any() else 0.0
        reliefs.append(zr)
        log(f"    env{i}: level={int(lvl[i]):2d}  valid={int(v.sum())}/{P}  "
            f"relief={zr:.3f} m  {'(평지)' if zr<0.05 else '(굴곡/경사)'}")
    log("=" * 64)

    with open("pc_check_stats.txt", "w") as f:
        f.write("\n".join(LINES))
    log("[saved] pc_check_stats.txt")

    # 지형별 프러스텀 산점도 격자
    cols = min(3, N); rows = math.ceil(N / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.6 * rows), squeeze=False)
    for i in range(N):
        ax = axes[i // cols][i % cols]
        p = pc[i].detach().cpu().numpy(); v = finite[i].detach().cpu().numpy()
        ax.scatter(p[v, 0], p[v, 1], c=p[v, 2], cmap="viridis", s=22)
        ax.set_title(f"env{i}  lvl{int(lvl[i])}  relief={reliefs[i]:.2f}m", fontsize=9)
        ax.set_xlabel("x fwd [m]"); ax.set_ylabel("y left [m]"); ax.axis("equal")
    for j in range(N, rows * cols):
        axes[j // cols][j % cols].axis("off")
    fig.suptitle("D435i 전방 프러스텀 point cloud — 지형별 (색=높이 z)", fontsize=12)
    fig.tight_layout(); fig.savefig(args_cli.out, dpi=120)
    log(f"[saved] {args_cli.out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
