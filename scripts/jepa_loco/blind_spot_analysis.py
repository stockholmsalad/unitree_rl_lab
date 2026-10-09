"""Teacher 187점 높이맵이 현재 D435i 프레임에 보이는 비율을 측정한다.

평지는 순수 기하, 9/13 cm 계단은 실제 렌더 depth와 RayCaster hit로 가림을 검사한다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", default="results/jepa_loco/blind_spot")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from unitree_rl_lab.jepa_loco.eval.blind_spot import (  # noqa: E402
    height_patch_xy, project_base_points, visible_in_depth,
)
from unitree_rl_lab.jepa_loco.eval.stairs import (  # noqa: E402
    StairEvalCfg, fixed_height_stair_cfg, stair_geometry, validate_stair_origins,
)
from unitree_rl_lab.jepa_loco.sensors.d435i_cfg import D435iParams, make_tiled_camera_cfg  # noqa: E402


@configclass
class BlindSpotAnalysisCfg:
    stair_heights_m: tuple[float, float] = (0.09, 0.13)
    approach_distances_m: tuple[float, ...] = (1.5, 1.0, 0.6, 0.3, 0.0)
    occlusion_tolerance_m: float = 0.06
    flat_base_height_m: float = 0.279
    render_settle_frames: int = 3
    pose_steps: int = 1
    seed: int = 43


def render_visibility(env, analysis: BlindSpotAnalysisCfg, camera: D435iParams,
                      robot, scanner, cam, origin: torch.Tensor,
                      grid_xy: np.ndarray) -> list[dict]:
    """현재 로봇 pose에서 실제 지형 hit와 렌더 depth를 대조한다."""
    scanner.update(env.physics_dt, force_recompute=True)
    for _ in range(analysis.render_settle_frames):
        env.sim.render()
    cam.update(env.physics_dt, force_recompute=True)
    hits = scanner.data.ray_hits_w.detach().cpu().numpy()
    raw = cam.data.output["distance_to_image_plane"].detach().cpu().numpy()[..., 0]
    base = robot.data.root_pos_w.detach().cpu().numpy()
    rows = []
    for idx in range(env.num_envs):
        world_hits = scanner.data.ray_hits_w[idx]
        base_points = math_utils.quat_apply_inverse(
            robot.data.root_quat_w[idx].expand(world_hits.shape[0], 4),
            world_hits - robot.data.root_pos_w[idx],
        ).detach().cpu().numpy()
        finite = np.isfinite(hits[idx]).all(axis=1)
        safe_points = np.where(finite[:, None], base_points, 0.0)
        geometric, visible = visible_in_depth(
            safe_points, raw[idx], camera.depth_origin_in_base,
            camera.pitch_down_deg, camera.intrinsics(camera.render_wh),
            camera.min_range, camera.clip_far, analysis.occlusion_tolerance_m,
        )
        geometric &= finite
        visible &= finite
        rows.append({"finite": finite, "geometric": geometric, "visible": visible,
                     "depth": raw[idx], "points_base": base_points,
                     "base_x_m": float(base[idx, 0] - origin[idx, 0].item())})
    return rows


def main() -> None:
    analysis = BlindSpotAnalysisCfg()
    camera = D435iParams()
    stair_eval = StairEvalCfg()
    task = "Unitree-Go2-JepaLoco-OracleCurrent-EasyStart"
    cfg = parse_env_cfg(task, device=args.device, num_envs=len(analysis.stair_heights_m),
                        use_fabric=True)
    cfg.seed = analysis.seed
    cfg.curriculum.terrain_levels = None
    cfg.events.push_robot = None
    cfg.sim.render_interval = 1
    generator = cfg.scene.terrain.terrain_generator.copy()
    generator.num_rows = 1
    generator.num_cols = len(analysis.stair_heights_m)
    generator.seed = analysis.seed
    template = generator.sub_terrains["stairs_up"]
    generator.sub_terrains = {
        f"stairs_{idx}": fixed_height_stair_cfg(template, height, 1 / len(analysis.stair_heights_m))
        for idx, height in enumerate(analysis.stair_heights_m)
    }
    cfg.scene.terrain.terrain_generator = generator
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.scene.d435i = make_tiled_camera_cfg(camera).replace(update_period=0.0)
    env = gym.make(task, cfg=cfg).unwrapped
    robot = env.scene["robot"]
    scanner = env.scene.sensors["height_scanner"]
    cam = env.scene.sensors["d435i"]
    origin = env.scene.env_origins.clone()
    obs, _ = env.reset()
    del obs
    count, first_riser, _ = stair_geometry(generator.size, template.border_width,
                                            template.platform_width, template.step_width)
    heights = torch.tensor(analysis.stair_heights_m, device=env.device)
    validate_stair_origins(origin[:, 2], heights, count, stair_eval.terrain_height_tolerance_m)
    patch = scanner.cfg.pattern_cfg
    grid_xy = height_patch_xy(tuple(patch.size), patch.resolution)
    if scanner.data.ray_hits_w.shape[1] != len(grid_xy):
        raise ValueError("height_scanner 점 순서/개수가 187점 격자와 맞지 않는다")

    flat_points = np.column_stack((grid_xy, np.full(len(grid_xy), -analysis.flat_base_height_m)))
    _, _, _, flat_visible = project_base_points(
        flat_points, camera.depth_origin_in_base, camera.pitch_down_deg,
        camera.intrinsics(camera.render_wh), camera.render_wh,
        camera.min_range, camera.clip_far,
    )
    rows = []
    near_maps = {}
    zero = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    start_state = robot.data.root_state_w.clone()
    for distance in analysis.approach_distances_m:
        state = start_state.clone()
        state[:, 0] = origin[:, 0] + first_riser - distance
        state[:, 1] = origin[:, 1]
        state[:, 7:] = 0
        robot.write_root_state_to_sim(state)
        for _ in range(analysis.pose_steps):
            env.step(zero)
        samples = render_visibility(env, analysis, camera, robot, scanner, cam, origin, grid_xy)
        for idx, sample in enumerate(samples):
            row = {
                "step_height_m": analysis.stair_heights_m[idx],
                "distance_to_first_riser_m": distance,
                "actual_base_x_m": sample["base_x_m"],
                "finite_hit_count": int(sample["finite"].sum()),
                "geometry_visible_count": int(sample["geometric"].sum()),
                "render_visible_count": int(sample["visible"].sum()),
                "render_visible_fraction": float(sample["visible"].mean()),
                "geometry_visible_mask": sample["geometric"].tolist(),
                "render_visible_mask": sample["visible"].tolist(),
            }
            rows.append(row)
            if distance == min(analysis.approach_distances_m):
                near_maps[analysis.stair_heights_m[idx]] = sample["visible"]

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = {"camera": camera.to_dict(), "analysis": analysis.to_dict(),
              "grid_size": list(patch.size), "grid_resolution_m": patch.resolution,
              "flat_visible_count": int(flat_visible.sum()),
              "flat_visible_fraction": float(flat_visible.mean()),
              "flat_visible_mask": flat_visible.tolist(),
              "stairs": rows}
    (out / "visibility.json").write_text(json.dumps(report, indent=2))
    (out / "visibility.csv").write_text(
        "step_height_m,distance_to_first_riser_m,render_visible_count,render_visible_fraction\n" +
        "\n".join(f'{r["step_height_m"]},{r["distance_to_first_riser_m"]},'
                  f'{r["render_visible_count"]},{r["render_visible_fraction"]:.6f}' for r in rows) + "\n"
    )
    fig, axes = plt.subplots(1, 1 + len(analysis.stair_heights_m), figsize=(12, 3.4), sharex=True, sharey=True)
    for ax, title, mask in zip(axes,
                               ["flat (geometry)"] + [f"{int(h*100)} cm, at riser" for h in analysis.stair_heights_m],
                               [flat_visible] + [near_maps[h] for h in analysis.stair_heights_m]):
        ax.scatter(grid_xy[~mask, 0], grid_xy[~mask, 1], s=14, c="#bdbdbd", label="hidden")
        ax.scatter(grid_xy[mask, 0], grid_xy[mask, 1], s=14, c="#1673a5", label="visible")
        ax.set_title(f"{title}: {mask.mean():.0%}")
        ax.set_xlabel("x from base (m)")
        ax.set_aspect("equal")
    axes[0].set_ylabel("y from base (m)")
    axes[-1].legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "visibility_maps.png", dpi=180)
    plt.close(fig)
    print("BLIND_SPOT_ANALYSIS", json.dumps({"flat_visible_fraction": report["flat_visible_fraction"],
                                            "rows": rows, "output_dir": str(out)}), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
