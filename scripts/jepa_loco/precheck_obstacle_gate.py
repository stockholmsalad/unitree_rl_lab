"""현재 지형에서 첫 장애물 통과 게이트와 reset 로그를 실제 로봇 pose로 검증한다."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402, F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402


@configclass
class GatePrecheckCfg:
    seed: int = 43
    num_envs: int = 3
    before_margin_m: float = 0.1
    on_gap_margin_m: float = 0.05
    beyond_margin_m: float = 0.05
    probe_lift_m: float = 0.5
    settle_steps: int = 20


def main():
    probe = GatePrecheckCfg()
    cfg = parse_env_cfg("Unitree-Go2-JepaLoco-OracleCurrent", device=args.device, num_envs=probe.num_envs,
                        entry_point_key="env_cfg_entry_point")
    cfg.seed = probe.seed
    cfg.curriculum.terrain_levels = None
    cfg.events.push_robot = None
    generator = cfg.scene.terrain.terrain_generator.copy()
    generator.num_cols = 3
    generator.seed = cfg.seed
    names = ("stairs_up", "stairs_down", "gap")
    generator.sub_terrains = {name: generator.sub_terrains[name].replace(proportion=1 / 3) for name in names}
    cfg.scene.terrain.terrain_generator = generator
    cfg.scene.terrain.max_init_terrain_level = 0
    env = gym.make("Unitree-Go2-JepaLoco-OracleCurrent", cfg=cfg)
    env.reset()
    task = env.unwrapped
    robot = task.scene["robot"]
    origins = task.scene.env_origins.clone()
    ids = torch.arange(task.num_envs, device=task.device)
    body_z = robot.data.root_pos_w[:, 2].clone()
    initial_pose = robot.data.root_pose_w.clone()
    types = task._terrain_type_per_env.tolist()
    type_names = [task._terrain_type_names[index] for index in types]
    edges = task._clearance_edge_per_env.tolist()
    riser = task._first_riser_m
    first_far = task._first_step_far_edge_m
    gap_inner = generator.sub_terrains["gap"].platform_width / 2
    positions = {
        "before": [riser - probe.before_margin_m if name != "gap" else gap_inner - probe.before_margin_m for name in type_names],
        "on_first_step": [(riser + first_far) / 2 if name != "gap" else gap_inner + probe.on_gap_margin_m for name in type_names],
        "beyond": [edge + probe.beyond_margin_m for edge in edges],
    }
    report = {"terrain_types": type_names, "first_riser_m": riser, "first_far_edge_m": first_far,
              "clearance_edges_m": edges, "cases": {}}
    for label, x_values in positions.items():
        task._clearance_consecutive[:] = 0
        task.episode_obstacle_cleared[:] = False
        task._max_axis_excursion_m[:] = 0
        task.episode_path_length_m[:] = 0
        pose = initial_pose.clone()
        pose[:, 0] = origins[:, 0] + torch.tensor(x_values, device=task.device)
        pose[:, 1] = origins[:, 1]
        pose[:, 2] = body_z + probe.probe_lift_m
        zero_velocity = torch.zeros(task.num_envs, 6, device=task.device)
        for _ in range(cfg.progress.clearance_hold_steps + 1):
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(zero_velocity)
            env.step(torch.zeros(task.num_envs, 12, device=task.device))
        cleared = task.episode_obstacle_cleared.clone()
        consecutive = task._clearance_consecutive.clone()
        task._reset_idx(ids)
        log = task.extras.get("log", {})
        rates = {name: float(log.get(f"Curriculum/obstacle_clear_rate/{name}", float("nan")))
                 for name in names}
        report["cases"][label] = {"x_m": x_values, "cleared": cleared.tolist(),
                                  "consecutive": consecutive.tolist(), "logged_rates": rates}

    # Compare zero-action settling on the center platform with settling on the first tread.
    task._reset_idx(ids)
    for _ in range(probe.settle_steps):
        env.step(torch.zeros(task.num_envs, 12, device=task.device))
    flat_standing_z = robot.data.root_pos_w[:, 2].clone()

    # Ray hits on the first tread provide an independent collision-mesh geometry probe.
    task._reset_idx(ids)
    pose = robot.data.root_pose_w.clone()
    pose[:, 0] = origins[:, 0] + torch.tensor(positions["on_first_step"], device=task.device)
    pose[:, 1] = origins[:, 1]
    pose[:, 2] = body_z + torch.tensor([0.08 if n == "stairs_up" else -0.08 if n == "stairs_down" else 0.0
                                      for n in type_names], device=task.device)
    robot.write_root_pose_to_sim(pose)
    robot.write_root_velocity_to_sim(torch.zeros(task.num_envs, 6, device=task.device))
    mesh_profile = {}
    for step in range(probe.settle_steps):
        env.step(torch.zeros(task.num_envs, 12, device=task.device))
        if step == 0:
            hits = task.scene.sensors["height_scanner"].data.ray_hits_w
            for env_id, name in enumerate(type_names):
                samples = {}
                for target_x in (riser - 0.1, (riser + first_far) / 2, first_far + 0.15):
                    target = origins[env_id, :2] + torch.tensor([target_x, 0.0], device=task.device)
                    distances = torch.linalg.vector_norm(hits[env_id, :, :2] - target, dim=-1)
                    distances = torch.where(torch.isfinite(distances), distances, torch.inf)
                    nearest = distances.argmin()
                    samples[f"x_{target_x:.2f}"] = {
                        "hit_z_m": float(hits[env_id, nearest, 2]),
                        "xy_error_m": float(distances[nearest]),
                    }
                mesh_profile[name] = samples
            tread_delta = [
                mesh_profile[name][f"x_{(riser + first_far) / 2:.2f}"]["hit_z_m"]
                - mesh_profile[name][f"x_{riser - 0.1:.2f}"]["hit_z_m"]
                if name != "gap" else 0.0 for name in type_names
            ]
            pose[:, 2] = body_z + torch.tensor(tread_delta, device=task.device)
            robot.write_root_pose_to_sim(pose)
            robot.write_root_velocity_to_sim(torch.zeros(task.num_envs, 6, device=task.device))
    hit = task.scene.sensors["height_scanner"].data.ray_hits_w
    foot_ids, foot_names = robot.find_bodies(".*_foot")
    contact = task.scene.sensors["contact_forces"]
    contact_ids, contact_names = contact.find_bodies(".*_foot")
    contact_by_name = dict(zip(contact_names, contact_ids))
    report["mesh_probe"] = {
        "ray_profile_at_first_step": mesh_profile,
        "first_tread_height_delta_m": tread_delta,
        "body_x_m": (robot.data.root_pos_w[:, 0] - origins[:, 0]).tolist(),
        "body_z_from_initial_m": (robot.data.root_pos_w[:, 2] - body_z).tolist(),
        "body_z_above_flat_standing_m": (robot.data.root_pos_w[:, 2] - flat_standing_z).tolist(),
        "ray_hit_z_min_m": hit[:, :, 2].amin(dim=1).tolist(),
        "ray_hit_z_max_m": hit[:, :, 2].amax(dim=1).tolist(),
        "base_contact": task.termination_manager.get_term("base_contact").tolist(),
        "bad_orientation": task.termination_manager.get_term("bad_orientation").tolist(),
        "feet": {
            name: {
                "x_from_origin_m": (robot.data.body_pos_w[:, foot_id, 0] - origins[:, 0]).tolist(),
                "z_m": robot.data.body_pos_w[:, foot_id, 2].tolist(),
                "contact_force_n": contact.data.net_forces_w[:, contact_by_name[name]].norm(dim=-1).tolist(),
            }
            for foot_id, name in zip(foot_ids, foot_names)
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print("GATE_PRECHECK", json.dumps(report), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
