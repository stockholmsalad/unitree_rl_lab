# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""단계 1 핵심 검증 — D435i **TiledCamera** depth 렌더링 처리량 벤치마크.

수천 병렬 환경에서 image-like depth 렌더가 학습 처리량(FPS)을 죽이는지 측정한다.
depth on/off 로 env-steps/s 를 비교해 로그로 남긴다. 이 결과가 이후 설계를 좌우한다
(depth CNN 확정 vs point cloud 변환 필요 여부 — CLAUDE.md 단계 1).

**중요**: Isaac Lab 의 SimulationContext 는 프로세스당 싱글턴이라 한 프로세스에서 환경을
여러 번 만들면 깨진다(재귀 에러). 따라서 이 스크립트는 **한 번에 한 설정(num_envs, depth)만**
측정하고, 스윕은 셸 드라이버(benchmark_depth_sweep.sh)가 프로세스를 반복 기동해서 한다.
결과는 같은 JSON 파일에 누적(append)된다.

사용:
  # 단일 설정 (depth ON — 카메라 필요)
  python scripts/je_loco/benchmark_depth.py --num_envs 2048 --steps 200 --headless --enable_cameras
  # depth OFF 기준선
  python scripts/je_loco/benchmark_depth.py --num_envs 2048 --steps 200 --headless --no_depth
  # 스윕(권장): 프로세스 반복 기동
  bash scripts/je_loco/benchmark_depth_sweep.sh 256 1024 2048 4096
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="JE-Loco depth 렌더링 처리량 벤치마크 (단일 설정)")
parser.add_argument("--num_envs", type=int, default=2048, help="환경 수.")
parser.add_argument("--steps", type=int, default=200, help="측정 env.step 횟수(워밍업 제외).")
parser.add_argument("--warmup", type=int, default=20, help="워밍업 스텝(측정 제외).")
parser.add_argument("--no_depth", action="store_true", help="depth 센서/관측 끄기(기준선).")
parser.add_argument("--out", type=str, default="logs/je_loco/benchmark_depth.json",
                    help="결과 누적 JSON 경로.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# depth 렌더에는 카메라 필요
if not args_cli.no_depth:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱 기동 후 import ----
import json
import os
import time

import gymnasium as gym
import torch

import unitree_rl_lab.je_loco.envs  # noqa: F401  (gym 등록)
from unitree_rl_lab.je_loco.envs.je_loco_env_cfg import JELocoEnvCfg


def _cuda_sync(device) -> None:
    # u.device 는 "cuda:0" 같은 문자열일 수 있다 → torch.device 로 정규화
    dev = torch.device(device) if isinstance(device, str) else device
    if dev.type == "cuda":
        torch.cuda.synchronize()


def run_one(num_envs: int, enable_depth: bool, steps: int, warmup: int) -> dict:
    # NOTE: enable_depth 는 __post_init__(카메라 제거)보다 먼저 정해져야 하므로 **생성자 인자**로 넘긴다.
    #       construct 후에 cfg.enable_depth 를 바꾸면 post_init 이 이미 끝나 카메라가 안 지워진다.
    cfg = JELocoEnvCfg(enable_depth=enable_depth)
    cfg.scene.num_envs = num_envs
    if args_cli.device is not None:
        cfg.sim.device = args_cli.device

    env = gym.make("Unitree-Go2-JELoco", cfg=cfg)
    u = env.unwrapped
    device = u.device
    action_dim = u.action_manager.total_action_dim

    env.reset()
    act = torch.zeros((num_envs, action_dim), device=device)

    for _ in range(warmup):
        env.step(act)
    _cuda_sync(device)

    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(act)
    _cuda_sync(device)
    dt = time.perf_counter() - t0

    # 디바이스 전체 사용 메모리 (RTX 렌더러는 torch allocator 밖 → mem_get_info 로 전체 포착)
    if torch.cuda.is_available():
        free_b, total_b = torch.cuda.mem_get_info()
        used_gb = (total_b - free_b) / 2**30
    else:
        used_gb = 0.0
    decimation = u.cfg.decimation
    policy_steps_per_s = steps / dt
    env_steps_per_s = policy_steps_per_s * num_envs
    sim_steps_per_s = env_steps_per_s * decimation

    result = {
        "num_envs": num_envs,
        "depth": enable_depth,
        "steps": steps,
        "wall_s": round(dt, 3),
        "policy_steps_per_s": round(policy_steps_per_s, 1),
        "env_steps_per_s": round(env_steps_per_s, 0),
        "sim_steps_per_s": round(sim_steps_per_s, 0),
        "device_used_gb": round(used_gb, 2),
    }
    env.close()
    return result


def main():
    enable_depth = not args_cli.no_depth
    print(f"\n[bench] num_envs={args_cli.num_envs} depth={'ON' if enable_depth else 'OFF'} "
          f"steps={args_cli.steps} ...", flush=True)
    result = run_one(args_cli.num_envs, enable_depth, args_cli.steps, args_cli.warmup)
    print(f"[bench] → env_steps/s={result['env_steps_per_s']:.0f}  "
          f"policy_steps/s={result['policy_steps_per_s']:.1f}  "
          f"gpu_used={result['device_used_gb']}GB  wall={result['wall_s']}s", flush=True)

    # 같은 JSON 에 누적 (스윕 드라이버가 여러 프로세스로 채운다)
    os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)
    existing = []
    if os.path.exists(args_cli.out):
        try:
            with open(args_cli.out) as f:
                existing = json.load(f).get("results", [])
        except Exception:  # noqa: BLE001
            existing = []
    existing.append(result)
    with open(args_cli.out, "w") as f:
        json.dump({"results": existing}, f, indent=2)
    print(f"[bench] 결과 누적 저장({len(existing)}개): {args_cli.out}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
