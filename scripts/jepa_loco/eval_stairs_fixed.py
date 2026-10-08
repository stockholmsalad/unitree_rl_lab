"""저난도/고난도 오르는 계단에서 동일 직진 명령으로 정책을 평가한다.

한 env는 한 계단 패치에서 한 에피소드만 기록한다. 메트릭은 실제 보행 성공의 대리 지표이며,
영상 판정과 함께 사용한다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", choices=(
    "Unitree-Go2-JepaLoco-DepthGRU",
    "Unitree-Go2-JepaLoco-BlindGRU",
    "Unitree-Go2-JepaLoco-OracleCurrent",
    "Unitree-Go2-JepaLoco-OracleCurrent-EasyStart",
    "Unitree-Go2-JepaLoco-OracleWide",
    "Unitree-Go2-JepaLoco-OracleCurrentFuture",
    "Unitree-Go2-JepaLoco-OracleCurrentFuture-EasyStart",
), required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--terrain", choices=("stairs", "flat"), default="stairs")
parser.add_argument("--spawn_forward_m", type=float, default=None)
parser.add_argument("--episode_steps", type=int, default=None)
parser.add_argument("--envs_per_height", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--low_step_height_m", type=float, default=None)
parser.add_argument("--high_step_height_m", type=float, default=None)
parser.add_argument("--video", action="store_true", help="고정 지형 평가 env 0의 영상을 저장한다.")
parser.add_argument("--probe_terrain", action="store_true", help="현재 heightscan을 평탄하게 바꿨을 때 행동 변화를 기록한다.")
parser.add_argument("--flatten_terrain_input", action="store_true", help="정책 입력 heightscan의 지형 높이 차이를 제거한다.")
parser.add_argument("--contact_diagnostics", action="store_true", help="첫 단 부근 종료 사유·접촉 body·스텝 보상을 기록한다.")
parser.add_argument("--audit_terrain_only", action="store_true", help="실제 생성 타일의 원점·메시 z 범위를 검증하고 종료한다.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
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
from unitree_rl_lab.jepa_loco.eval.stairs import (  # noqa: E402
    StairEvalCfg, fixed_height_stair_cfg, stair_geometry, stair_progress,
    unmet_success_conditions, validate_stair_origins,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def make_eval_env_cfg(task: str, cfg: StairEvalCfg):
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=2 * cfg.envs_per_height,
                            entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed = cfg.seed
    env_cfg.episode_length_s = cfg.episode_steps * env_cfg.sim.dt * env_cfg.decimation
    # 고정 평가에서는 학습용 승급·강등을 적용하지 않는다.
    env_cfg.curriculum.terrain_levels = None
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = cfg.viewer_env_index
    env_cfg.viewer.eye = cfg.viewer_eye
    env_cfg.viewer.lookat = cfg.viewer_lookat

    generator = env_cfg.scene.terrain.terrain_generator.copy()
    generator.num_rows = 1
    generator.num_cols = 2 * cfg.envs_per_height
    generator.seed = cfg.seed
    if args.terrain == "flat":
        generator.sub_terrains = {"flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0)}
    else:
        stair = generator.sub_terrains["stairs_up"]
        generator.sub_terrains = {
            "stairs_up_low": fixed_height_stair_cfg(stair, cfg.low_step_height_m, 0.5),
            "stairs_up_high": fixed_height_stair_cfg(stair, cfg.high_step_height_m, 0.5),
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
    if args.episode_steps is not None:
        cfg.episode_steps = args.episode_steps
    if args.envs_per_height is not None:
        cfg.envs_per_height = args.envs_per_height
    if args.seed is not None:
        cfg.seed = args.seed
    if args.low_step_height_m is not None:
        cfg.low_step_height_m = args.low_step_height_m
    if args.high_step_height_m is not None:
        cfg.high_step_height_m = args.high_step_height_m
    if cfg.low_step_height_m <= 0 or cfg.high_step_height_m <= cfg.low_step_height_m:
        raise ValueError("계단 높이는 0보다 크고 low < high여야 한다")
    env_cfg = make_eval_env_cfg(args.task, cfg)
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    import importlib.metadata as metadata
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
    agent_cfg.seed = cfg.seed

    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    if args.audit_terrain_only:
        if args.terrain != "stairs":
            raise ValueError("계단 메시 감사는 --terrain stairs에서만 실행한다")
        terrain = env.unwrapped.scene.terrain
        generator_cfg = env_cfg.scene.terrain.terrain_generator
        regenerated = terrain_gen.TerrainGenerator(generator_cfg.copy(), device=env.unwrapped.device)
        actual_origins = terrain.terrain_origins
        expected_origins = torch.as_tensor(
            regenerated.terrain_origins, device=actual_origins.device, dtype=actual_origins.dtype,
        )
        if not torch.allclose(actual_origins, expected_origins, atol=cfg.terrain_height_tolerance_m, rtol=0):
            raise ValueError("시뮬레이터에 올린 타일 원점과 재생성 메시 원점이 다르다")
        stair_cfg = generator_cfg.sub_terrains["stairs_up_low"]
        count, _, _ = stair_geometry(generator_cfg.size, stair_cfg.border_width,
                                     stair_cfg.platform_width, stair_cfg.step_width)
        heights = torch.where(terrain.terrain_types < cfg.envs_per_height,
                              cfg.low_step_height_m, cfg.high_step_height_m)
        validate_stair_origins(terrain.env_origins[:, 2], heights, count, cfg.terrain_height_tolerance_m)
        columns = []
        for col in range(generator_cfg.num_cols):
            mesh = regenerated.terrain_meshes[col]
            z_min, z_max = float(mesh.bounds[0, 2]), float(mesh.bounds[1, 2])
            if z_max - z_min <= cfg.terrain_height_tolerance_m:
                raise ValueError(f"계단 대신 평면 메시가 생성된 열: {col}")
            env_ids = (terrain.terrain_types == col).nonzero(as_tuple=True)[0].tolist()
            if env_ids != [col]:
                raise ValueError(f"열과 평가 env ID의 대응이 다르다: column={col}, env_ids={env_ids}")
            columns.append({"column": col, "terrain_origin_z": float(actual_origins[0, col, 2]),
                            "mesh_z_min": z_min, "mesh_z_max": z_max,
                            "mesh_z_span": z_max - z_min,
                            "env_ids": env_ids})
        audit = {"task": args.task, "seed": cfg.seed, "num_rows": generator_cfg.num_rows,
                 "num_cols": generator_cfg.num_cols, "low_step_height_m": cfg.low_step_height_m,
                 "high_step_height_m": cfg.high_step_height_m, "stair_count": count,
                 "configured_functions": {name: term.function.__name__
                                          for name, term in generator_cfg.sub_terrains.items()},
                 "columns": columns}
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(audit, indent=2))
        print(f"TERRAIN_AUDIT_OK columns={len(columns)} output={output}", flush=True)
        env.close()
        return
    if args.video:
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(Path(args.output).with_suffix("")) + "_video",
            step_trigger=lambda step: step == 0, video_length=cfg.episode_steps,
            disable_logger=True,
        )
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(Path(args.checkpoint).resolve()))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    obs = env.get_observations()
    initial_terrain_scan = {}
    for group_name in ("terrain_current", "terrain_wide", "terrain_future"):
        if group_name in obs:
            values = obs[group_name].float()
            initial_terrain_scan[group_name] = {
                "min": values.min().item(), "max": values.max().item(),
                "mean": values.mean().item(), "std": values.std(unbiased=False).item(),
                "fraction_at_clip_max": (values == 1.0).float().mean().item(),
            }
    initial_scan_geometry = {}
    if "height_scanner" in env.unwrapped.scene.sensors:
        sensor = env.unwrapped.scene.sensors["height_scanner"]
        initial_scan_geometry = {
            "sensor_z": sensor.data.pos_w[:, 2].detach().cpu().tolist(),
            "ray_hit_z_min": sensor.data.ray_hits_w[..., 2].amin(dim=1).detach().cpu().tolist(),
            "ray_hit_z_max": sensor.data.ray_hits_w[..., 2].amax(dim=1).detach().cpu().tolist(),
            "robot_base_z": env.unwrapped.scene["robot"].data.root_pos_w[:, 2].detach().cpu().tolist(),
            "terrain_origin_z": env.unwrapped.scene.env_origins[:, 2].detach().cpu().tolist(),
        }
    robot = env.unwrapped.scene["robot"]
    foot_ids, foot_names = robot.find_bodies(".*_foot")
    front_foot_ids = [body_id for body_id, name in zip(foot_ids, foot_names) if name.startswith("F")]
    terrain = env.unwrapped.scene.terrain
    origin = env.unwrapped.scene.env_origins.clone()
    start_z = robot.data.root_pos_w[:, 2].clone()
    # env별 기록용 출발 자세 (원점 기준 xy, yaw)
    import isaaclab.utils.math as math_utils
    start_xy = (robot.data.root_pos_w[:, :2] - origin[:, :2]).clone()
    start_yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)[2].clone()
    command_sample = env.unwrapped.command_manager.get_command("base_velocity")[0].detach().cpu().tolist()

    if args.terrain == "stairs":
        stair_cfg = env_cfg.scene.terrain.terrain_generator.sub_terrains["stairs_up_low"]
        count, first_riser, top_edge = stair_geometry(env_cfg.scene.terrain.terrain_generator.size,
                                         stair_cfg.border_width, stair_cfg.platform_width, stair_cfg.step_width)
    else:
        count, first_riser, top_edge = 0, None, 0.0
    heights = torch.where(terrain.terrain_types < cfg.envs_per_height, cfg.low_step_height_m,
                          cfg.high_step_height_m).to(env.unwrapped.device)
    if args.terrain == "stairs":
        validate_stair_origins(origin[:, 2], heights, count, cfg.terrain_height_tolerance_m)
    active = torch.ones(env.num_envs, device=env.unwrapped.device, dtype=torch.bool)
    success = torch.zeros_like(active)
    failed = torch.zeros_like(active)
    timed_out = torch.zeros_like(active)
    max_steps = torch.zeros(env.num_envs, device=env.unwrapped.device)
    max_outward = torch.zeros_like(max_steps)
    min_body_drop = torch.zeros_like(max_steps)
    consecutive = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.int)
    max_consecutive = torch.zeros_like(consecutive)
    height_ok_ever = torch.zeros_like(active)
    position_ok_ever = torch.zeros_like(active)
    reached_top_ever = torch.zeros_like(active)
    done_step = torch.full((env.num_envs,), -1, device=env.unwrapped.device, dtype=torch.long)
    done_reasons = [[] for _ in range(env.num_envs)]
    last_xy = start_xy.clone()
    last_yaw = start_yaw.clone()
    terminal_xy = start_xy.clone()
    terminal_yaw = start_yaw.clone()
    term_manager_all = env.unwrapped.termination_manager
    trace = []
    peak_front_foot_z = torch.full_like(max_steps, -float("inf"))
    peak_front_foot_x = torch.full_like(max_steps, -float("inf"))
    peak_front_foot_z_settled = torch.full_like(max_steps, -float("inf"))
    reward_terms = env.unwrapped.reward_manager
    reward_term_sums = torch.zeros((env.num_envs, len(reward_terms._term_names)), device=env.unwrapped.device)
    contact_diagnostics = {"near_step_terminations": [], "first_contact_traces": []}
    if args.contact_diagnostics and args.terrain == "stairs":
        contact_sensor = env.unwrapped.scene.sensors["contact_forces"]
        contact_ids, contact_names = contact_sensor.find_bodies(".*")
        front_contact_ids = [idx for idx, name in zip(contact_ids, contact_names) if name.startswith("F") and name.endswith("_foot")]
        term_manager = env.unwrapped.termination_manager
        contact_trace_started = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.bool)
        contact_trace_remaining = torch.zeros(env.num_envs, device=env.unwrapped.device, dtype=torch.long)
    original_reset_idx = env.unwrapped._reset_idx
    step_index = [0]

    def capture_reset(env_ids):
        ids = torch.as_tensor(env_ids, device=env.unwrapped.device, dtype=torch.long)
        terminal_xy[ids] = robot.data.root_pos_w[ids, :2] - origin[ids, :2]
        terminal_yaw[ids] = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[ids])[2]
        if args.contact_diagnostics and args.terrain == "stairs":
            root_x = (robot.data.root_pos_w[ids, 0] - origin[ids, 0]).abs()
            for env_id, distance in zip(ids.tolist(), root_x.tolist()):
                if abs(distance - first_riser) > cfg.contact_probe_margin_m:
                    continue
                force = contact_sensor.data.net_forces_w[env_id, contact_ids].norm(dim=-1)
                recent_force = contact_sensor.data.net_forces_w_history[env_id, :, contact_ids].norm(dim=-1).amax(dim=0)
                contact_diagnostics["near_step_terminations"].append({
                    "env_id": env_id, "step": int(step_index[0]), "height_m": heights[env_id].item(),
                    "body_x_m": distance, "termination_reasons": [
                        name for name in term_manager.active_terms if term_manager.get_term(name)[env_id].item()
                    ],
                    "contact_bodies": [name for name, value in zip(contact_names, force.tolist())
                                       if value > cfg.contact_probe_force_threshold_n],
                    "contact_bodies_recent": [name for name, value in zip(contact_names, recent_force.tolist())
                                              if value > cfg.contact_probe_force_threshold_n],
                    "reward_step": {name: (reward_terms._step_reward[env_id, i] * env.unwrapped.step_dt).item()
                                    for i, name in enumerate(reward_terms._term_names)},
                })
        return original_reset_idx(env_ids)

    env.unwrapped._reset_idx = capture_reset

    for step in range(cfg.episode_steps):
        if args.contact_diagnostics and args.terrain == "stairs":
            step_index[0] = step + 1
        with torch.inference_mode():
            action_obs = obs
            if args.flatten_terrain_input:
                if "terrain_current" not in obs:
                    raise ValueError("terrain_current가 없는 정책에는 heightscan 평탄화를 사용할 수 없다")
                action_obs = obs.clone()
                scan = action_obs["terrain_current"]
                action_obs["terrain_current"] = scan.amax(dim=1, keepdim=True).expand_as(scan)
            actions = policy(action_obs)
            obs, rewards, dones, extras = env.step(actions)
            reward_term_sums += torch.where(
                active[:, None], reward_terms._step_reward * env.unwrapped.step_dt, 0.0,
            )
            if getattr(policy, "is_recurrent", False):
                policy.reset(dones)
            just_done = active & dones.bool()
            time_outs = extras.get("time_outs", torch.zeros_like(dones)).bool()
            failed |= just_done & ~time_outs
            for env_id in torch.nonzero(just_done, as_tuple=True)[0].tolist():
                done_step[env_id] = step + 1
                done_reasons[env_id] = [name for name in term_manager_all.active_terms
                                        if term_manager_all.get_term(name)[env_id].item()]
            timed_out |= just_done & time_outs
            still_active = active & ~dones.bool()
            reached, progress = stair_progress(robot.data.root_pos_w, origin, start_z, heights, count,
                                                top_edge, cfg.top_height_fraction, cfg.top_position_margin_m)
            max_steps = torch.where(still_active, torch.maximum(max_steps, progress), max_steps)
            forward = robot.data.root_pos_w[:, 0] - origin[:, 0]
            if args.terrain == "stairs":
                height_ok_ever |= still_active & (progress >= cfg.top_height_fraction * count)
                position_ok_ever |= still_active & (forward >= top_edge - cfg.top_position_margin_m)
                reached_top_ever |= still_active & reached
            # done env는 _reset_idx 직전 캡처한 종료 위치를 사용한다.
            last_xy = torch.where(still_active[:, None], robot.data.root_pos_w[:, :2] - origin[:, :2], last_xy)
            last_yaw = torch.where(still_active, math_utils.euler_xyz_from_quat(robot.data.root_quat_w)[2], last_yaw)
            last_xy = torch.where(just_done[:, None], terminal_xy, last_xy)
            last_yaw = torch.where(just_done, terminal_yaw, last_yaw)
            max_outward = torch.where(still_active, torch.maximum(max_outward, forward), max_outward)
            front_z = (robot.data.body_pos_w[:, front_foot_ids, 2] - origin[:, None, 2]).amax(dim=1)
            front_x = (robot.data.body_pos_w[:, front_foot_ids, 0] - origin[:, None, 0]).amax(dim=1)
            peak_front_foot_z = torch.where(still_active, torch.maximum(peak_front_foot_z, front_z), peak_front_foot_z)
            peak_front_foot_x = torch.where(still_active, torch.maximum(peak_front_foot_x, front_x), peak_front_foot_x)
            if args.contact_diagnostics and args.terrain == "stairs":
                front_force = contact_sensor.data.net_forces_w[:, front_contact_ids].norm(dim=-1).amax(dim=1)
                start_trace = still_active & ~contact_trace_started & (
                    (front_x >= first_riser - cfg.contact_probe_foot_x_margin_m)
                    & (front_x <= first_riser + cfg.contact_probe_margin_m)
                ) & (front_force > cfg.contact_probe_force_threshold_n)
                contact_trace_started |= start_trace
                contact_trace_remaining = torch.where(start_trace, cfg.contact_probe_steps, contact_trace_remaining)
                for env_id in torch.nonzero(still_active & (contact_trace_remaining > 0), as_tuple=True)[0].tolist():
                    force = contact_sensor.data.net_forces_w[env_id, contact_ids].norm(dim=-1)
                    recent_force = contact_sensor.data.net_forces_w_history[env_id, :, contact_ids].norm(dim=-1).amax(dim=0)
                    contact_diagnostics["first_contact_traces"].append({
                        "env_id": env_id, "step": step + 1, "height_m": heights[env_id].item(),
                        "body_x_m": forward[env_id].item(), "front_foot_x_m": front_x[env_id].item(),
                        "contact_bodies": [name for name, value in zip(contact_names, force.tolist())
                                           if value > cfg.contact_probe_force_threshold_n],
                        "contact_bodies_recent": [name for name, value in zip(contact_names, recent_force.tolist())
                                                  if value > cfg.contact_probe_force_threshold_n],
                        "reward_step": {name: (reward_terms._step_reward[env_id, i] * env.unwrapped.step_dt).item()
                                        for i, name in enumerate(reward_terms._term_names)},
                    })
                contact_trace_remaining = torch.where(still_active, (contact_trace_remaining - 1).clamp(min=0), 0)
            if step >= cfg.settle_steps:
                peak_front_foot_z_settled = torch.where(still_active, torch.maximum(peak_front_foot_z_settled, front_z), peak_front_foot_z_settled)
            min_body_drop = torch.where(still_active, torch.minimum(min_body_drop, progress), min_body_drop)
            consecutive = torch.where(still_active & reached, consecutive + 1, torch.zeros_like(consecutive))
            max_consecutive = torch.maximum(max_consecutive, consecutive)
            success |= consecutive >= cfg.hold_steps
            active = still_active
            if (step % cfg.trace_interval_steps == 0 or step == cfg.episode_steps - 1) and active.any():
                reward_terms = env.unwrapped.reward_manager
                terrain_scan = obs["terrain_current"][active] if "terrain_current" in obs else None
                terrain_action_delta = None
                if args.probe_terrain and terrain_scan is not None:
                    probe_obs = obs.clone()
                    scan = probe_obs["terrain_current"]
                    probe_obs["terrain_current"] = scan.amax(dim=1, keepdim=True).expand_as(scan)
                    probe_actions = policy(probe_obs)
                    terrain_action_delta = (policy(obs)[active] - probe_actions[active]).abs().mean().item()
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
                              "terrain_scan_min": None if terrain_scan is None else terrain_scan.min().item(),
                              "terrain_scan_max": None if terrain_scan is None else terrain_scan.max().item(),
                              "terrain_scan_std": None if terrain_scan is None else terrain_scan.float().std(unbiased=False).item(),
                              "terrain_action_delta_abs_mean": terrain_action_delta,
                              "reward_terms_per_s": {name: reward_terms._step_reward[active, i].mean().item()
                                                     for i, name in enumerate(reward_terms._term_names)},
                              "active_rate": active.float().mean().item()})

    report = {"task": args.task, "terrain": args.terrain, "checkpoint": str(Path(args.checkpoint).resolve()),
              "flatten_terrain_input": args.flatten_terrain_input,
              "initial_terrain_scan": initial_terrain_scan, "initial_scan_geometry": initial_scan_geometry,
              "command_mps": cfg.forward_command_mps, "observed_command": command_sample,
              "spawn_forward_m": cfg.spawn_forward_m, "episode_steps": cfg.episode_steps,
              "stair_count": count, "first_riser_m": first_riser, "top_edge_m": top_edge, "success_rule":
              None if args.terrain == "flat" else
              f"body gain >= {cfg.top_height_fraction} * {count} * step height and forward x >= {top_edge-cfg.top_position_margin_m:.2f} m for {cfg.hold_steps} steps",
              "groups": {}, "trace": trace, "foot_names": foot_names,
              "contact_diagnostics": contact_diagnostics if args.contact_diagnostics else None}
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
            "reward_term_sum_mean": {
                term_name: reward_term_sums[sel, i].mean().item()
                for i, term_name in enumerate(reward_terms._term_names)
            },
        }
    def _unmet(env_id: int) -> list[str]:
        if args.terrain != "stairs" or success[env_id]:
            return []
        return unmet_success_conditions(
            False, bool(height_ok_ever[env_id]), bool(position_ok_ever[env_id]),
            bool(reached_top_ever[env_id]),
        )

    report["per_env"] = [{
        "env_id": i,
        "terrain_type": int(terrain.terrain_types[i].item()),
        "step_height_m": None if args.terrain == "flat" else heights[i].item(),
        "spawn_x_m": start_xy[i, 0].item(), "spawn_y_m": start_xy[i, 1].item(), "spawn_yaw_rad": start_yaw[i].item(),
        "success": bool(success[i].item()),
        "outcome": "success" if success[i] else ("non_timeout" if failed[i] else ("time_out" if timed_out[i] else "active_at_end")),
        "termination_reasons": done_reasons[i], "done_step": int(done_step[i].item()),
        "max_body_rise_steps": max_steps[i].item(), "max_forward_m": max_outward[i].item(),
        "last_x_m": last_xy[i, 0].item(), "last_y_m": last_xy[i, 1].item(), "last_yaw_rad": last_yaw[i].item(),
        "max_consecutive_reached": int(max_consecutive[i].item()),
        "unmet_success_conditions": _unmet(i),
    } for i in range(env.num_envs)]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print("STAIR_EVAL", json.dumps(report), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
