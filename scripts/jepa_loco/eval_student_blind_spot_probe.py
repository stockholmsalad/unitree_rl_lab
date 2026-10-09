"""고정 계단에서 student latent와 teacher heightmap의 probe 데이터 한 seed를 수집한다."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--steps", type=int, default=None)
parser.add_argument("--envs_per_height", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402, F401
from unitree_rl_lab.jepa_loco.eval.stairs import (  # noqa: E402
    StairEvalCfg, fixed_height_stair_cfg, stair_geometry, validate_stair_origins,
)
from unitree_rl_lab.jepa_loco.models.student_depth import StudentInferencePolicy  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


@configclass
class ProbeEvalCfg:
    heights_m: tuple[float, float] = (0.09, 0.13)
    envs_per_height: int = 32
    steps: int = 1000
    sample_interval_steps: int = 5
    forward_command_mps: float = 0.6
    lateral_spawn_range_m: float = 0.25
    spawn_x_m: float = 0.0


def collect(checkpoint: str, cfg: ProbeEvalCfg, seed: int, device: str):
    task = "Unitree-Go2-JepaLoco-StudentDepth-EasyStart"
    env_cfg = parse_env_cfg(task, device=device, num_envs=2 * cfg.envs_per_height,
                            entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed = seed
    env_cfg.curriculum.terrain_levels = None
    generator = env_cfg.scene.terrain.terrain_generator.copy()
    generator.num_rows = 1
    generator.num_cols = 2 * cfg.envs_per_height
    generator.seed = seed
    template = generator.sub_terrains["stairs_up"]
    generator.sub_terrains = {
        "low": fixed_height_stair_cfg(template, cfg.heights_m[0], 0.5),
        "high": fixed_height_stair_cfg(template, cfg.heights_m[1], 0.5),
    }
    env_cfg.scene.terrain.terrain_generator = generator
    env_cfg.scene.terrain.max_init_terrain_level = 0
    command = env_cfg.commands.base_velocity
    fixed = command.Ranges(lin_vel_x=(cfg.forward_command_mps, cfg.forward_command_mps),
                           lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0))
    command.initial_ranges = fixed.copy()
    command.final_ranges = fixed.copy()
    command.ranges = fixed.copy()
    command.rel_standing_envs = 0.0
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (cfg.spawn_x_m, cfg.spawn_x_m),
        "y": (-cfg.lateral_spawn_range_m, cfg.lateral_spawn_range_m),
        "yaw": (0.0, 0.0),
    }
    env = gym.make(task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    policy = StudentInferencePolicy(checkpoint, torch.device(env.device))
    terrain = env.scene.terrain
    origin = env.scene.env_origins.clone()
    count, first_riser, _ = stair_geometry(generator.size, template.border_width,
                                            template.platform_width, template.step_width)
    height_idx = (terrain.terrain_types >= cfg.envs_per_height).long()
    requested_heights = torch.where(height_idx == 0, cfg.heights_m[0], cfg.heights_m[1])
    validate_stair_origins(origin[:, 2], requested_heights, count,
                           StairEvalCfg().terrain_height_tolerance_m)
    features, targets, distances, heights = [], [], [], []
    with torch.inference_mode():
        for step in range(cfg.steps):
            actions = policy(obs)
            if step % cfg.sample_interval_steps == 0:
                features.append(policy.last_z.cpu().clone())
                targets.append(obs["terrain_current"].cpu().clone())
                distances.append((first_riser - (env.scene["robot"].data.root_pos_w[:, 0] - origin[:, 0])).cpu().clone())
                heights.append(height_idx.cpu().clone())
            obs, _, terminated, truncated, _ = env.step(actions)
            policy.reset((terminated | truncated).bool())
    data = (torch.cat(features), torch.cat(targets), torch.cat(distances), torch.cat(heights))
    env.close()
    return data


def main() -> None:
    cfg = ProbeEvalCfg()
    if args.steps is not None:
        cfg.steps = args.steps
    if args.envs_per_height is not None:
        cfg.envs_per_height = args.envs_per_height
    z, heightmap, distance, terrain_height_index = collect(args.checkpoint, cfg, args.seed, args.device)
    if heightmap.shape[1] != 187:
        raise ValueError("teacher 높이맵 차원이 187이 아니다")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": z, "heightmap": heightmap, "distance": distance,
                "height_index": terrain_height_index, "seed": args.seed,
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "config": cfg.to_dict()}, output)
    print(f"STUDENT_PROBE_DATA seed={args.seed} samples={len(z)} output={output}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
