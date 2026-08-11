# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""1단계(Stage 2) — foothold 가 시각-필수 과제인지 측정.

질문: 마스킹 학습 정책이 **계단**에서, depth 결손 시 발을 더 잘못 딛는가(edge 접촉↑)?
- YES → foothold 는 시각-필수 → Stage 2(예측 표현이 이길 축) 유효
- NO  → foothold 도 proprioception 으로 처리 → 이 방향도 포화(방향 재검토)

계측: 발밑 dense RayCaster(foot_scan, eval 전용)로 각 발 주변 지형 높이범위(max−min)를 구해,
접지 순간 그 범위가 edge_threshold 초과면 "edge 접촉"(계단코/단차를 밟음). clean vs 결손 비교.
지형은 계단만(pyramid_stairs)으로 강제. 정책·인코더·마스킹은 그대로(입력 안 바꿈).

사용법:
  python -u scripts/je_loco/eval_foothold.py --checkpoint <ckpt> \
    --dropout_levels 0,1.0 --degradation dropout --terrain_level 6 --headless
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Foothold 시각-필수성 측정 (계단 edge 접촉률).")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1300)
parser.add_argument("--warmup", type=int, default=120)
parser.add_argument("--dropout_levels", type=str, default="0.0,1.0")
parser.add_argument("--degradation", type=str, default="dropout", choices=["dropout", "hole", "occlusion"])
parser.add_argument("--terrain_level", type=int, default=6, help="계단 난이도(0~9). 높을수록 단높이 큼")
parser.add_argument("--keep_terrain", action="store_true",
                    help="지형 강제 안 함 → env 자체 지형(학습분포) 사용. 순수계단 강제 시 정책이 얼어붙는 문제 회피")
parser.add_argument("--edge_threshold", type=float, default=0.04, help="발 주변 높이차[m] 초과 시 edge 접촉")
parser.add_argument("--foot_radius", type=float, default=0.10, help="발 주변 edge 탐색 반경[m]")
parser.add_argument("--eval_seed", type=int, default=42)
parser.add_argument("--task", type=str, default="Unitree-Go2-JELoco-PC")
import cli_args  # noqa: E402
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab.terrains as terrain_gen  # noqa: E402
from isaaclab.sensors import RayCasterCfg, patterns  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import get_checkpoint_path  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from importlib.metadata import version  # noqa: E402

import unitree_rl_lab.je_loco.rsl_rl_pc  # noqa: F401, E402
from unitree_rl_lab.je_loco.rsl_rl_pc.mdp_pc import DEGRADATIONS  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

torch.backends.cudnn.enabled = False


def edge_contact(uenv, foot_ids, contact_ids, radius, thr):
    """접지한 발 중 주변 지형 높이범위>thr 인 비율. 반환 (edge접촉수, 총접지수)."""
    fs = uenv.scene.sensors["foot_scan"]
    ray_xy = fs.data.ray_hits_w[..., :2]                    # (N, R, 2)
    ray_z = fs.data.ray_hits_w[..., 2]                      # (N, R)
    robot = uenv.scene["robot"]
    foot_xy = robot.data.body_pos_w[:, foot_ids, :2]        # (N, F, 2)
    d2 = torch.sum((foot_xy[:, :, None, :] - ray_xy[:, None, :, :]) ** 2, dim=-1)  # (N,F,R)
    within = d2 <= radius * radius
    has = within.any(dim=-1)
    h = ray_z[:, None, :].expand(-1, foot_xy.shape[1], -1)
    hmax = torch.where(within, h, torch.full_like(h, -1e9)).amax(-1)
    hmin = torch.where(within, h, torch.full_like(h, 1e9)).amin(-1)
    rng = torch.where(has, (hmax - hmin).clamp(min=0.0), torch.zeros_like(hmax))   # (N,F)
    forces = uenv.scene.sensors["contact_forces"].data.net_forces_w[:, contact_ids, :]
    in_contact = torch.linalg.norm(forces, dim=-1) > 1.0    # (N, F)
    edge = (in_contact & (rng > thr))
    return int(edge.sum()), int(in_contact.sum())


def run_level(env, uenv, policy, robot, foot_ids, contact_ids, dev, level, degrade_fn):
    obs = env.get_observations()
    falls = timeouts = 0
    edge_c = cont_c = 0
    cur_len = torch.zeros(uenv.num_envs, device=dev); ep_lens = []
    for t in range(args_cli.warmup + args_cli.steps):
        with torch.inference_mode():
            obs["pointcloud"] = degrade_fn(obs["pointcloud"], level)
            obs, _, dones, extras = env.step(policy(obs))
        if t < args_cli.warmup:
            continue
        e, c = edge_contact(uenv, foot_ids, contact_ids, args_cli.foot_radius, args_cli.edge_threshold)
        edge_c += e; cont_c += c
        cur_len += 1
        d = dones.bool()
        if d.any():
            to = extras.get("time_outs", torch.zeros_like(d)).bool().to(dev)
            timeouts += int((d & to).sum()); falls += int((d & ~to).sum())
            ep_lens += cur_len[d].tolist(); cur_len[d] = 0
    n = timeouts + falls
    spd = robot.data.root_lin_vel_b[:, 0].mean().item()   # 전진속도(회피/정지 진단)
    lvl = uenv.scene.terrain.terrain_levels.float().mean().item()
    return {
        "edge_rate": edge_c / max(1, cont_c),        # ★ 발 착지 중 edge 밟은 비율
        "contacts": cont_c,                          # 총 접지수(분모) — 0 이면 edge_rate 무의미
        "speed": spd,                                # 전진속도 — 낮으면 계단 회피/정지 의심
        "terrain": lvl,
        "survival": 1.0 - falls / max(1, n),
        "ep_frac": (sum(ep_lens) / max(1, len(ep_lens))) / float(uenv.max_episode_length),
    }


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed = args_cli.eval_seed
    if not args_cli.keep_terrain:
        # 지형 = 계단만 강제 (주의: 순수 계단은 정책이 얼어붙을 수 있음 → --keep_terrain 권장)
        env_cfg.curriculum.terrain_levels = None
        env_cfg.scene.terrain.terrain_generator.sub_terrains = {
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.5, step_height_range=(0.05, 0.18), step_width=0.30,
                platform_width=3.0, border_width=1.0, holes=False),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.5, step_height_range=(0.05, 0.18), step_width=0.30,
                platform_width=3.0, border_width=1.0, holes=False),
        }
    else:
        # 학습 분포 그대로(혼합 계단) — 정책이 실제로 걷는 지형에서 foothold 측정.
        # 커리큘럼만 정지(레벨 고정), sub_terrains 는 env 자체(Foothold=계단 0.75) 유지.
        env_cfg.curriculum.terrain_levels = None
    # 발밑 dense 스캔 (보상/평가 privileged. Foothold env 는 이미 있음 → 동일 config 로 덮어써도 무방)
    env_cfg.scene.foot_scan = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.04, size=[1.0, 0.7], ordering="yx"),
        debug_vis=False, mesh_prim_paths=["/World/ground"],
    )

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    if args_cli.terrain_level >= 0 and not args_cli.keep_terrain:
        terr = uenv.scene.terrain
        lv = min(args_cli.terrain_level, terr.max_terrain_level - 1)
        terr.terrain_levels[:] = lv
        terr.env_origins[:] = terr.terrain_origins[terr.terrain_levels, terr.terrain_types]
        uenv.reset()
        print(f"[foothold] 계단 지형 level={lv} 강제")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint),
                load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
                strict=False)
    policy = runner.get_inference_policy(device=uenv.device)
    robot = uenv.scene["robot"]
    foot_ids = robot.find_bodies(".*_foot")[0]
    contact_ids = uenv.scene.sensors["contact_forces"].find_bodies(".*_foot")[0]

    deg = args_cli.degradation
    degrade_fn = DEGRADATIONS[deg]
    levels = [float(x) for x in args_cli.dropout_levels.split(",")]
    print(f"[foothold] {deg} 스윕 {levels} — edge_thr={args_cli.edge_threshold} r={args_cli.foot_radius}")
    print(f"\n{'level':>6} | {'edge_rate':>9} | {'contacts':>8} | {'speed':>6} | {'terrain':>7} | {'surv':>5} | {'ep_frac':>7}")
    print("-" * 70)
    for lv in levels:
        r = run_level(env, uenv, policy, robot, foot_ids, contact_ids, uenv.device, lv, degrade_fn)
        print(f"{lv:>6.2f} | {r['edge_rate']:>9.4f} | {r['contacts']:>8d} | {r['speed']:>6.2f} | "
              f"{r['terrain']:>7.2f} | {r['survival']:>5.2f} | {r['ep_frac']:>7.3f}")
    env.close(); simulation_app.close()


if __name__ == "__main__":
    main()
