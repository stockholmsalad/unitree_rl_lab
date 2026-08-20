# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""정책 정량 평가 + depth 결손 스윕 (게이트 2).

학습된 체크포인트를 결정론 평가. --dropout_levels 로 여러 결손 레벨을 한 번에 스윕 →
레벨별 지표(추종오차·성공률·terrain) 표 + CSV(저하 곡선용) 저장. Exp1(clean 학습분을
eval 때만 결손 주입 = zero-shot 강건성).

사용:
  # clean (결손 0)
  python scripts/je_loco/eval_pc.py --num_envs 256 --headless --load_run <run> --checkpoint model_24000.pt
  # dropout 스윕
  python scripts/je_loco/eval_pc.py --num_envs 256 --headless --load_run <run> --checkpoint model_24000.pt \
      --dropout_levels 0,0.2,0.4,0.6,0.8,1.0
"""

import argparse
import os
from importlib.metadata import version

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Evaluate JE-Loco PC policy (+ depth dropout sweep).")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1300, help="레벨당 측정 스텝(에피소드 1000 완주 위해 >1000)")
parser.add_argument("--warmup", type=int, default=120, help="레벨 전환 후 측정 전 안정화 스텝")
parser.add_argument("--dropout_levels", type=str, default="0.0", help="쉼표구분 결손 레벨 (예: 0,0.2,0.4,0.6,0.8,1.0)")
parser.add_argument("--degradation", type=str, default="dropout", choices=["dropout", "hole", "occlusion"],
                    help="결손 종류: dropout(i.i.d 점) · hole(블록) · occlusion(하단 대역). valid 채널 0=마스킹")
parser.add_argument("--fix_terrain", action="store_true", default=True,
                    help="eval 중 terrain 커리큘럼 정지 → 지형 분포 고정(공정 비교, 논문용 기본 on)")
parser.add_argument("--no_fix_terrain", dest="fix_terrain", action="store_false",
                    help="terrain 커리큘럼 켠 채 평가(기존 Exp1 방식)")
parser.add_argument("--terrain_level", type=int, default=-1,
                    help="모든 env 를 이 지형 레벨에 강제 spawn (0~num_rows-1). -1=기본(max_init 분포). 어려운 지형 평가용")
parser.add_argument("--eval_seed", type=int, default=42,
                    help="지형 배정 결정론화 → A·B 가 동일 지형을 밟게(공정 비교). 헤드 무관 동일 seed 사용")
parser.add_argument("--task", type=str, default="Unitree-Go2-JELoco-PC")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
import unitree_rl_lab.je_loco.rsl_rl_pc  # noqa: F401
from unitree_rl_lab.je_loco.rsl_rl_pc.mdp_pc import DEGRADATIONS
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

torch.backends.cudnn.enabled = False


def run_level(env, uenv, policy, robot, dev, level, degrade_fn):
    """결손 level 로 warmup 후 steps 측정. 지표 dict 반환."""
    err_xy = err_yaw = spd = 0.0
    m = 0
    ep_lens, falls, timeouts = [], 0, 0
    cur_len = torch.zeros(uenv.num_envs, device=dev)
    obs = env.get_observations()
    total = args_cli.warmup + args_cli.steps
    for t in range(total):
        with torch.inference_mode():
            obs["pointcloud"] = degrade_fn(obs["pointcloud"], level)   # 정책이 보는 점군에 결손
            obs, _, dones, extras = env.step(policy(obs))
        if t < args_cli.warmup:
            continue
        cmd = uenv.command_manager.get_command("base_velocity")
        v = robot.data.root_lin_vel_b
        w = robot.data.root_ang_vel_b
        err_xy += torch.norm(cmd[:, :2] - v[:, :2], dim=1).mean().item()
        err_yaw += (cmd[:, 2] - w[:, 2]).abs().mean().item()
        spd += v[:, 0].mean().item()
        m += 1
        cur_len += 1
        d = dones.bool()
        if d.any():
            to = extras.get("time_outs", torch.zeros_like(d)).bool().to(dev)
            timeouts += int((d & to).sum())
            falls += int((d & ~to).sum())
            ep_lens += cur_len[d].tolist()
            cur_len[d] = 0
    n_done = timeouts + falls
    lvl = uenv.scene.terrain.terrain_levels.float().mean().item() if hasattr(uenv.scene.terrain, "terrain_levels") else float("nan")
    return {
        "err_xy": err_xy / m, "err_yaw": err_yaw / m, "speed": spd / m,
        "ep_frac": (sum(ep_lens) / max(1, len(ep_lens))) / float(uenv.max_episode_length),
        "fall_rate": (falls / n_done) if n_done else 0.0,
        "terrain": lvl, "n_ep": len(ep_lens),
    }


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed = args_cli.eval_seed   # A·B 동일 지형 배정 → 공정 비교
    print(f"[eval] env seed = {args_cli.eval_seed} (A·B 동일 지형)")
    if args_cli.fix_terrain and getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
        env_cfg.curriculum.terrain_levels = None   # eval 중 지형 승강 정지 → max_init_terrain_level 분포로 고정
        print("[eval] terrain 커리큘럼 정지 → 지형 분포 고정 (모든 dropout 레벨에서 동일 분포)")
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    ckpt = args_cli.checkpoint
    if ckpt and ("/" in ckpt or os.path.isfile(ckpt)):
        resume_path = retrieve_file_path(ckpt)
    else:
        run = args_cli.load_run if args_cli.load_run else agent_cfg.load_run
        resume_path = get_checkpoint_path(log_root, run, ckpt if ckpt else agent_cfg.load_checkpoint)
    print(f"[eval] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    # 어려운 지형 평가: 모든 env 를 지정 레벨에 강제 배정(terrain_types 는 유지 → 모든 지형종류 최고난이도).
    if args_cli.terrain_level >= 0:
        terr = uenv.scene.terrain
        import torch as _t
        lv = min(args_cli.terrain_level, terr.max_terrain_level - 1)
        terr.terrain_levels[:] = lv
        terr.env_origins[:] = terr.terrain_origins[terr.terrain_levels, terr.terrain_types]
        uenv.reset()   # 새 origin 에 리스폰
        print(f"[eval] terrain_level 강제 = {lv} (max {terr.max_terrain_level-1}) — 어려운 지형")
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    # 증류 체크포인트는 키가 student/teacher_state_dict 라 OnPolicyRunner 로는 못 읽는다.
    # (2026-08-20: play_pc.py 만 고쳐져 있었고 eval 은 누락 — 그대로 돌리면 정책이 로드되지 않아
    #  모든 조건이 똑같이 무너지는 '가짜 무승부' 곡선이 나온다.)
    if getattr(agent_cfg.algorithm, "class_name", "") == "Distillation":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path, load_cfg={"student": True, "teacher": True,
                                           "optimizer": False, "iteration": False}, strict=False)
    else:
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        runner.load(resume_path,
                    load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
                    strict=False)
    policy = runner.get_inference_policy(device=uenv.device)
    robot = uenv.scene["robot"]

    deg = args_cli.degradation
    degrade_fn = DEGRADATIONS[deg]                          # 마스킹: valid 채널 0 (좌표 무주입)
    levels = [float(x) for x in args_cli.dropout_levels.split(",")]
    print(f"[eval] {deg} 스윕 (마스킹): {levels}  "
          f"({uenv.num_envs} envs × {args_cli.steps} steps/level)")
    rows = []
    for lv in levels:
        r = run_level(env, uenv, policy, robot, uenv.device, lv, degrade_fn)
        rows.append((lv, r))
        print(f"  {deg} {lv:.2f}: err_xy {r['err_xy']:.3f}  성공률 {100*(1-r['fall_rate']):.1f}%  "
              f"완주 {100*r['ep_frac']:.0f}%  속도 {r['speed']:.2f}  terrain {r['terrain']:.2f}")

    # 표 + CSV
    tag = os.path.basename(os.path.dirname(resume_path)) + "_" + os.path.basename(resume_path).replace(".pt", "")
    print("\n" + "=" * 74)
    print(f"[{deg} 저하 곡선]  {tag}")
    print(f"{'dropout':>8} | {'err_xy':>7} | {'err_yaw':>7} | {'speed':>6} | {'완주%':>6} | {'성공%':>6} | {'terrain':>7}")
    print("-" * 74)
    for lv, r in rows:
        print(f"{lv:>8.2f} | {r['err_xy']:>7.3f} | {r['err_yaw']:>7.3f} | {r['speed']:>6.2f} | "
              f"{100*r['ep_frac']:>6.1f} | {100*(1-r['fall_rate']):>6.1f} | {r['terrain']:>7.2f}")
    print("=" * 74)
    csv = f"{deg}_curve_{tag}.csv"
    with open(csv, "w") as f:
        f.write(f"{deg},err_xy,err_yaw,speed,ep_frac,success_rate,terrain\n")
        for lv, r in rows:
            f.write(f"{lv},{r['err_xy']:.4f},{r['err_yaw']:.4f},{r['speed']:.4f},"
                    f"{r['ep_frac']:.4f},{1-r['fall_rate']:.4f},{r['terrain']:.4f}\n")
    print(f"[saved] {csv}  ← 저하 곡선 (A/B 각각 저장 후 함께 플롯)")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
