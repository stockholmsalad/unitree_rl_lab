"""Phase 2 환경의 depth 관측 점검: 렌더링 고유 지연 d0, fresh 플래그 타이밍, 2채널 영상.

d0 측정: 로봇을 제자리에 세운 뒤 렌더 직후 스텝에 base 를 0.15 m 들어 올린다(순간이동).
이후 렌더 스텝마다 카메라 원시 출력(노이즈·추가지연 이전)의 중앙 depth 를 보고, 순간이동이
처음 반영된 렌더가 몇 번째 렌더인지 센다. d0 [제어 스텝] = (반영 렌더 − 첫 렌더) × depth_period_steps.

    python scripts/jepa_loco/check_depth_obs.py --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--out_dir", type=str, default="results/jepa_loco/depth_obs_check")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
import os  # noqa: E402

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

TASK = "Unitree-Go2-JepaLoco-DepthGRU"


def main():
    env_cfg = parse_env_cfg(TASK, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.events.push_robot = None
    env_cfg.episode_length_s = 1000.0
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    period = env.cfg.depth_period_steps
    cam = env.scene["d435i"]
    robot = env.scene["robot"]
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    obs, _ = env.reset()

    # 1) fresh 플래그 타이밍 + 관측 영상
    fresh_log, frames = [], []
    for _ in range(40):
        obs, *_ = env.step(zero)
        fresh_log.append(obs["depth_fresh"][:, 0].cpu())
        frames.append(obs["depth"][0].cpu())
    fresh = torch.stack(fresh_log)  # [T, N]
    frames_all = obs["depth"].cpu()
    terrain = env.scene.terrain
    pose_info = [
        f"env{n} g_z={robot.data.projected_gravity_b[n, 2]:.2f} lv{int(terrain.terrain_levels[n])} t{int(terrain.terrain_types[n])}"
        for n in range(env.num_envs)
    ]
    gaps = []
    for n in range(env.num_envs):
        idx = fresh[:, n].nonzero().squeeze(-1).tolist()
        gaps += [b - a for a, b in zip(idx, idx[1:])]

    # 2) d0: 렌더 직후에 순간이동
    def render_id():
        return env._sim_step_counter // env.cfg.sim.render_interval

    rid0 = render_id()
    while render_id() == rid0:
        env.step(zero)
    center = lambda: cam.data.output["distance_to_image_plane"][:, 32, 56, 0].clone()  # noqa: E731
    before = center()
    root = robot.data.root_state_w.clone()
    root[:, 2] += 0.15
    robot.write_root_state_to_sim(root)
    teleport_rid = render_id()
    trace = []
    for _ in range(4 * period):
        env.step(zero)
        if render_id() != (trace[-1][0] if trace else teleport_rid):
            trace.append((render_id(), (center() - before).abs().mean().item()))
    changed = [r for r, dlt in trace if dlt > 0.02]
    d0_renders = (changed[0] - teleport_rid - 1) if changed else None

    report = {
        "depth_period_steps": period,
        "render_interval_physics_steps": env.cfg.sim.render_interval,
        "fresh_rate": fresh.mean().item(),
        "fresh_gap_hist": {g: gaps.count(g) for g in sorted(set(gaps))},
        "obs_valid_frac": frames_all[:, 1].mean().item(),
        "pose_info": pose_info,
        "teleport_trace(render_id, mean|Δcenter depth|)": trace,
        "d0_renders": d0_renders,
        "d0_control_steps": None if d0_renders is None else d0_renders * period,
    }
    os.makedirs(args_cli.out_dir, exist_ok=True)
    with open(os.path.join(args_cli.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    # 마지막 관측(순간이동 전 40스텝째)의 env 별 영상 + 자세·지형 — 이상 영상이 넘어짐/지형 탓인지 구분
    last, info = frames_all, pose_info
    k = min(8, env.num_envs)
    fig, axes = plt.subplots(2, k, figsize=(2.6 * k, 3.6))
    for n in range(k):
        axes[0, n].imshow(last[n, 0], cmap="viridis", vmin=0, vmax=1)
        axes[0, n].set_title(info[n], fontsize=6)
        axes[1, n].imshow(last[n, 1], cmap="gray", vmin=0, vmax=1)
    for ax in axes.flat:
        ax.axis("off")
    fig.savefig(os.path.join(args_cli.out_dir, "depth_obs.png"), dpi=110)
    print("REPORT", json.dumps(report), flush=True)  # close() 가 버퍼를 버리므로 flush
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
