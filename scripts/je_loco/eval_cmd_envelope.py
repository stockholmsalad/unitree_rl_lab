# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""명령 봉투(command envelope) 측정 — "이 정책은 어느 속도·회전까지 실제로 걷는가".

왜 필요한가 (2026-08-31):
  현재 학생 환경은 lin_vel_x=(0.3,0.6), ang_vel_z=(0,0) — 느리고 직진만 한다.
  예측(JEPA)의 가치는 속도에 비례한다: latency 10스텝이 만드는 위치 오차가
  0.4m/s 에서는 8cm 지만 1.5m/s 에서는 30cm 다. 그래서 속도·회전 범위를 넓히려 한다.
  그런데 **증류는 teacher 를 못 넘는다.** teacher 가 레벨 5 지형에서 1.5m/s 로 못 걸으면
  증류 타깃이 쓰레기가 되고 학습 4일을 날린다. 넓히기 **전에** 봉투를 먼저 잰다.

무엇을 하는가:
  명령을 (v, w) 로 **고정**하고(범위 lo=hi) 격자를 훑으며 성공률·추종오차를 잰다.
  · 커리큘럼 2종(terrain_levels, lin_vel_cmd_levels) 정지 → 명령·지형이 측정 중 안 변함
  · heading_command 해제 · rel_standing_envs=0 → 모든 env 가 지정 명령을 실제로 추종
  · --terrain_level 로 난이도 고정

사용:
  # teacher 봉투 (레벨 5 지형)
  python scripts/je_loco/eval_cmd_envelope.py --headless --num_envs 256 \
      --task Unitree-Go2-JELoco-Teacher --terrain_level 5 \
      --speeds 0.5,0.8,1.1,1.4 --yaws 0,0.3 \
      --load_run <teacher_run> --checkpoint model_19999.pt

  # 현재 학생과 비교 (학생은 0.3~0.6 으로 학습됐으니 그 밖에서 무너지는 게 정상)
  python scripts/je_loco/eval_cmd_envelope.py --headless --task Unitree-Go2-JELoco-Distill ...
"""

import argparse
import os
from importlib.metadata import version

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="Measure command envelope (speed x yaw) of a policy.")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=1500, help="셀당 측정 스텝(에피소드 1000 완주 위해 >1000)")
parser.add_argument("--warmup", type=int, default=150, help="명령 전환 후 측정 전 안정화 스텝")
parser.add_argument("--speeds", type=str, default="0.5,0.8,1.1,1.4", help="쉼표구분 lin_vel_x 고정값")
parser.add_argument("--yaws", type=str, default="0,0.3", help="쉼표구분 ang_vel_z 고정값(rad/s)")
parser.add_argument("--terrain_level", type=int, default=5,
                    help="모든 env 를 이 지형 레벨에 강제 spawn. -1=기본 분포")
parser.add_argument("--eval_seed", type=int, default=42, help="지형 배정 결정론화")
parser.add_argument("--task", type=str, default="Unitree-Go2-JELoco-Teacher")
parser.add_argument("--out", type=str, default="", help="CSV 경로(기본: cmd_envelope_<tag>.csv)")
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
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

torch.backends.cudnn.enabled = False


def pin_command(uenv, v: float, w: float) -> None:
    """명령을 (v, 0, w) 로 고정하고 즉시 전 env 재샘플.

    범위를 lo=hi 로 좁히는 방식이라 이후 주기적 재샘플에도 같은 값이 나온다.
    heading_command 가 켜져 있으면 ang_vel_z 가 heading 오차에서 계산되어 w 가 무시되므로 끈다.
    rel_standing_envs>0 이면 일부 env 가 정지 명령을 받아 성공률이 부풀려지므로 0 으로 둔다.
    """
    term = uenv.command_manager.get_term("base_velocity")
    cfg = term.cfg
    cfg.ranges.lin_vel_x = (v, v)
    cfg.ranges.lin_vel_y = (0.0, 0.0)
    cfg.ranges.ang_vel_z = (w, w)
    for name in ("limit_ranges",):                      # 커리큘럼용 상한도 같이 맞춰둔다
        if hasattr(cfg, name):
            lr = getattr(cfg, name)
            lr.lin_vel_x, lr.lin_vel_y, lr.ang_vel_z = (v, v), (0.0, 0.0), (w, w)
    if getattr(cfg, "heading_command", False):
        cfg.heading_command = False
    if hasattr(cfg, "rel_standing_envs"):
        cfg.rel_standing_envs = 0.0
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[:] = False
    term._resample_command(torch.arange(uenv.num_envs, device=uenv.device))


def run_cell(env, uenv, policy, robot, dev, v, w):
    """명령 (v, w) 고정 상태에서 warmup 후 steps 측정."""
    pin_command(uenv, v, w)
    err_xy = err_yaw = spd = yaw_rate = 0.0
    m = 0
    ep_lens, falls, timeouts = [], 0, 0
    cur_len = torch.zeros(uenv.num_envs, device=dev)
    obs = env.get_observations()
    for t in range(args_cli.warmup + args_cli.steps):
        with torch.inference_mode():
            obs, _, dones, extras = env.step(policy(obs))
        if t < args_cli.warmup:
            continue
        cmd = uenv.command_manager.get_command("base_velocity")
        lv, av = robot.data.root_lin_vel_b, robot.data.root_ang_vel_b
        err_xy += torch.norm(cmd[:, :2] - lv[:, :2], dim=1).mean().item()
        err_yaw += (cmd[:, 2] - av[:, 2]).abs().mean().item()
        spd += lv[:, 0].mean().item()
        yaw_rate += av[:, 2].mean().item()
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
    return {
        "err_xy": err_xy / m, "err_yaw": err_yaw / m,
        "speed": spd / m, "yaw_rate": yaw_rate / m,
        "ep_frac": (sum(ep_lens) / max(1, len(ep_lens))) / float(uenv.max_episode_length),
        "success": 1.0 - ((falls / n_done) if n_done else 0.0),
        "n_ep": len(ep_lens),
    }


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed = args_cli.eval_seed
    # 커리큘럼 전면 정지 — 측정 중 명령 범위나 지형 난이도가 움직이면 봉투가 아니라 궤적을 재게 된다.
    for term in ("terrain_levels", "lin_vel_cmd_levels"):
        if getattr(env_cfg.curriculum, term, None) is not None:
            setattr(env_cfg.curriculum, term, None)
            print(f"[envelope] curriculum.{term} 정지")

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    ckpt = args_cli.checkpoint
    if ckpt and ("/" in ckpt or os.path.isfile(ckpt)):
        resume_path = retrieve_file_path(ckpt)
    else:
        run = args_cli.load_run if args_cli.load_run else agent_cfg.load_run
        resume_path = get_checkpoint_path(log_root, run, ckpt if ckpt else agent_cfg.load_checkpoint)
    print(f"[envelope] checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    if args_cli.terrain_level >= 0:
        terr = uenv.scene.terrain
        lv = min(args_cli.terrain_level, terr.max_terrain_level - 1)
        terr.terrain_levels[:] = lv
        terr.env_origins[:] = terr.terrain_origins[terr.terrain_levels, terr.terrain_types]
        uenv.reset()
        print(f"[envelope] terrain_level 강제 = {lv} (max {terr.max_terrain_level-1})")

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    # 증류 체크포인트는 키가 student/teacher_state_dict 라 OnPolicyRunner 로는 못 읽는다.
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

    speeds = [float(x) for x in args_cli.speeds.split(",")]
    yaws = [float(x) for x in args_cli.yaws.split(",")]
    print(f"[envelope] {len(speeds)}속도 × {len(yaws)}회전 = {len(speeds)*len(yaws)}셀  "
          f"({uenv.num_envs} envs × {args_cli.steps} steps/셀)")

    rows = []
    for w in yaws:
        for v in speeds:
            r = run_cell(env, uenv, policy, robot, uenv.device, v, w)
            rows.append((v, w, r))
            print(f"  v={v:.2f} w={w:+.2f} → 성공 {100*r['success']:5.1f}%  "
                  f"실제속도 {r['speed']:.2f}  err_xy {r['err_xy']:.3f}  완주 {100*r['ep_frac']:.0f}%")

    tag = os.path.basename(os.path.dirname(resume_path)) + "_" + os.path.basename(resume_path).replace(".pt", "")
    print("\n" + "=" * 78)
    print(f"[명령 봉투]  {tag}   terrain_level={args_cli.terrain_level}")
    print(f"{'v_cmd':>6} | {'w_cmd':>6} | {'성공%':>6} | {'실제v':>6} | {'실제w':>6} | "
          f"{'err_xy':>7} | {'완주%':>6}")
    print("-" * 78)
    for v, w, r in rows:
        print(f"{v:>6.2f} | {w:>+6.2f} | {100*r['success']:>6.1f} | {r['speed']:>6.2f} | "
              f"{r['yaw_rate']:>+6.2f} | {r['err_xy']:>7.3f} | {100*r['ep_frac']:>6.1f}")
    print("=" * 78)
    # 봉투 상한: 성공률 80% 를 유지하는 최대 속도(회전별). 새 학생 환경 범위는 여기서 정한다.
    for w in yaws:
        ok = [v for v, ww, r in rows if ww == w and r["success"] >= 0.80]
        print(f"  w={w:+.2f}: 성공률 80% 유지 최대속도 = {max(ok):.2f}" if ok
              else f"  w={w:+.2f}: 성공률 80% 를 만족하는 속도 없음")

    out = args_cli.out or f"cmd_envelope_{tag}.csv"
    with open(out, "w") as f:
        f.write("v_cmd,w_cmd,success_rate,speed,yaw_rate,err_xy,err_yaw,ep_frac,n_ep\n")
        for v, w, r in rows:
            f.write(f"{v},{w},{r['success']:.4f},{r['speed']:.4f},{r['yaw_rate']:.4f},"
                    f"{r['err_xy']:.4f},{r['err_yaw']:.4f},{r['ep_frac']:.4f},{r['n_ep']}\n")
    print(f"[saved] {out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
