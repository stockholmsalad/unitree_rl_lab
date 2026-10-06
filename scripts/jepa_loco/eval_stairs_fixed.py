"""저난도/고난도 오르는 계단에서 동일 직진 명령으로 정책을 평가한다.

한 env는 한 계단 패치에서 한 에피소드만 기록한다. 메트릭은 실제 보행 성공의 대리 지표이며,
영상 판정과 함께 사용한다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=("Unitree-Go2-JepaLoco-DepthGRU", "Unitree-Go2-JepaLoco-BlindGRU"), required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--terrain", choices=("stairs", "flat"), default="stairs")
parser.add_argument("--spawn_forward_m", type=float, default=None)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
import isaaclab.terrains as terrain_gen  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402, F401
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry  # noqa: E402
from unitree_rl_lab.jepa_loco.eval.stairs import StairEvalCfg, stair_geometry, stair_progress  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def make_eval_env_cfg(task: str, cfg: StairEvalCfg):
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=2 * cfg.envs_per_height,
                            entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed = cfg.seed
    env_cfg.episode_length_s = cfg.episode_steps * env_cfg.sim.dt * env_cfg.decimation

    generator = env_cfg.scene.terrain.terrain_generator.copy()
    generator.num_rows = 1
    generator.num_cols = 2 * cfg.envs_per_height
    generator.seed = cfg.seed
    if args.terrain == "flat":
        generator.sub_terrains = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
    else:
        stair = generator.sub_terrains["stairs_up"]
        generator.sub_terrains = {
            "stairs_up_low": stair.replace(proportion=0.5, step_height_range=(cfg.low_step_height_m, cfg.low_step_height_m)),
            "stairs_up_high": stair.replace(proportion=0.5, step_height_range=(cfg.high_step_height_m, cfg.high_step_height_m)),
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
        "x": (cfg.spawn_forward_m, cfg.spawn_forward_m),
        "y": (-cfg.spawn_lateral_range_m, cfg.spawn_lateral_range_m),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    return env_cfg


def main():
    cfg = StairEvalCfg()
    if args.spawn_forward_m is not None:
        cfg.spawn_forward_m = args.spawn_forward_m
    env_cfg = make_eval_env_cfg(args.task, cfg)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    import importlib.metadata as metadata
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.seed = cfg.seed

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(Path(args.checkpoint).resolve()))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()
    robot = env.unwrapped.scene["robot"]
    foot_ids, foot_names = robot.find_bodies(".*_foot")
    front_foot_ids = [body_id for body_id, name in zip(foot_ids, foot_names) if name.startswith("F")]
    terrain = env.unwrapped.scene.terrain
    origin = env.unwrapped.scene.env_origins.clone()
    start_z = robot.data.root_pos_w[:, 2].clone()
    command_sample = env.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().tolist()

    if args.terrain == "stairs":
        stair_cfg = env_cfg.scene.terrain.terrain_generator.sub_terrains["stairs_up_low"]
        count, first_riser, top_edge = stair_geometry(env_cfg.scene.terrain.terrain_generator.size,
                                         stair_cfg.border_width, stair_cfg.platform_width, stair_cfg.step_width)
    else:
        count, first_riser, top_edge = 0, None, 0.0
    heights = torch.where(terrain.terrain_types < cfg.envs_per_height, cfg.low_step_height_m,
                          cfg.high_step_height_m).to(env.unwrapped.device)
    active = torch.ones(env.num_envs, device=env.unwrapped.device, dtype=torch.bool)
    success = torch.zeros_like(active)
    failed = torch.zeros_like(active)
    timed_out = torch.zeros_like(active)
    max_steps = torch.zeros(env.num_envs, device=env.unwrapped.device)
    max_outward = torch.zeros_like(max_steps)
    min_body_drop = torch.zeros_like(max_steps)
    consecutive = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.int)
    trace = []
    peak_front_foot_z = torch.full_like(max_steps, -float("inf"))
    peak_front_foot_x = torch.full_like(max_steps, -float("inf"))
    peak_front_foot_z_settled = torch.full_like(max_steps, -float("inf"))

    for step in range(cfg.episode_steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, rewards, dones, extras = env.step(actions)
            if getattr(policy, "is_recurrent", False):
                policy.reset(dones)
            just_done = active & dones.bool()
            time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()
            failed |= just_done & ~time_outs
            timed_out |= just_done & time_outs
            still_active = active & ~dones.bool()
            reached, progress = stair_progress(robot.data.root_pos_w, origin, start_z, heights, count,
                                                top_edge, cfg.top_height_fraction, cfg.top_position_margin_m)
            max_steps = torch.where(still_active, torch.maximum(max_steps, progress), max_steps)
            forward = robot.data.root_pos_w[:, 0] - origin[:, 0]
            max_outward = torch.where(still_active, torch.maximum(max_outward, forward), max_outward)
            front_z = (robot.data.body_pos_w[:, front_foot_ids, 2] - origin[:, None, 2]).amax(dim=1)
            front_x = (robot.data.body_pos_w[:, front_foot_ids, 0] - origin[:, None, 0]).amax(dim=1)
            peak_front_foot_z = torch.where(still_active, torch.maximum(peak_front_foot_z, front_z), peak_front_foot_z)
            peak_front_foot_x = torch.where(still_active, torch.maximum(peak_front_foot_x, front_x), peak_front_foot_x)
            if step >= cfg.settle_steps:
                peak_front_foot_z_settled = torch.where(still_active, torch.maximum(peak_front_foot_z_settled, front_z), peak_front_foot_z_settled)
            min_body_drop = torch.where(still_active, torch.minimum(min_body_drop, progress), min_body_drop)
            consecutive = torch.where(still_active & reached, consecutive + 1, torch.zeros_like(consecutive))
            success |= consecutive >= cfg.hold_steps
            active = still_active
            if (step % cfg.trace_interval_steps == 0 or step == cfg.episode_steps - 1) and active.any():
                reward_terms = env.unwrapped.reward_manager
                trace.append({"step": step + 1,
                              "forward_m_mean": forward[active].mean().item(),
                              "forward_m_min": forward[active].min().item(),
                              "forward_m_max": forward[active].max().item(),
                              "body_vx_mps_mean": robot.data.root_lin_vel_b[active, 0].mean().item(),
                              "body_z_gain_m_mean": (robot.data.root_pos_w[active, 2] - start_z[active]).mean().item(),
                              "front_foot_z_m_mean": front_z[active].mean().item(),
                              "front_foot_x_m_mean": front_x[active].mean().item(),
                              "action_abs_mean": actions[active].abs().mean().item(),
                              "reward_per_step_mean": rewards[active].mean().item(),
                              "reward_terms_per_s": {name: reward_terms._step_reward[active, i].mean().item()
                                                     for i, name in enumerate(reward_terms._term_names)},
                              "active_rate": active.float().mean().item()})

    report = {"task": args.task, "terrain": args.terrain, "checkpoint": str(Path(args.checkpoint).resolve()),
              "command_mps": cfg.forward_command_mps, "observed_command": command_sample,
              "spawn_forward_m": cfg.spawn_forward_m, "episode_steps": cfg.episode_steps,
              "stair_count": count, "first_riser_m": first_riser, "top_edge_m": top_edge, "success_rule":
              None if args.terrain == "flat" else
              f"body gain >= {cfg.top_height_fraction} * {count} * step height and forward x >= {top_edge-cfg.top_position_margin_m:.2f} m for {cfg.hold_steps} steps",
              "groups": {}, "trace": trace, "foot_names": foot_names}
    selections = (("flat", torch.ones_like(active)),) if args.terrain == "flat" else (
        ("low", heights == cfg.low_step_height_m), ("high", heights == cfg.high_step_height_m))
    for name, sel in selections:
        report["groups"][name] = {
            "step_height_m": None if name == "flat" else cfg.low_step_height_m if name == "low" else cfg.high_step_height_m,
            "n": int(sel.sum().item()), "success_rate": success[sel].float().mean().item(),
            "non_timeout_rate": failed[sel].float().mean().item(),
            "timeout_rate": timed_out[sel].float().mean().item(),
            "max_body_rise_steps_mean": max_steps[sel].mean().item(),
            "max_body_rise_steps_median": max_steps[sel].median().item(),
            "max_body_rise_steps_max": max_steps[sel].max().item(),
            "max_forward_m_mean": max_outward[sel].mean().item(),
            "max_forward_m_max": max_outward[sel].max().item(),
            "min_body_drop_steps_mean": min_body_drop[sel].mean().item(),
            "peak_front_foot_z_m_mean": peak_front_foot_z[sel].mean().item(),
            "peak_front_foot_z_settled_m_mean": peak_front_foot_z_settled[sel].mean().item(),
            "peak_front_foot_z_settled_m_max": peak_front_foot_z_settled[sel].max().item(),
            "peak_front_foot_x_m_mean": peak_front_foot_x[sel].mean().item(),
        }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print("STAIR_EVAL", json.dumps(report), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
