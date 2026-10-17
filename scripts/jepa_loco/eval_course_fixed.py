"""Teacher와 student가 공유하는 평가 전용 연속 코스 실행기."""

from __future__ import annotations

import json
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import load_cfg_from_registry
from unitree_rl_lab.jepa_loco.agents.rsl_rl_cfg import OracleCurrentPPORunnerCfg
from unitree_rl_lab.jepa_loco.eval.course import (
    CourseEvalCfg, CourseTerrainCfg, SECTION_NAMES, course_outcome, section_edges, section_pass_edges,
    designed_height,
)
from unitree_rl_lab.jepa_loco.models.student_depth import StudentInferencePolicy
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


def make_course_env_cfg(task: str, c: CourseEvalCfg, device: str):
    env_cfg = parse_env_cfg(task, device=device, num_envs=len(c.step_heights_m) * c.envs_per_height,
                            entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed = c.seed
    env_cfg.episode_length_s = c.episode_steps * env_cfg.sim.dt * env_cfg.decimation
    env_cfg.curriculum.terrain_levels = None
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = c.viewer_env_index
    env_cfg.viewer.eye = c.viewer_eye
    env_cfg.viewer.lookat = c.viewer_lookat
    generator = env_cfg.scene.terrain.terrain_generator.copy()
    generator.num_rows = 1
    generator.num_cols = len(c.step_heights_m) * c.envs_per_height
    generator.seed = c.seed
    generator.curriculum = True  # 열 순서대로 높이 그룹을 배치한다.
    generator.use_cache = False
    generator.size = (section_edges(c)[-1] + c.tile_margin_m, c.width_m)
    generator.border_width = c.generator_border_width_m
    generator.horizontal_scale = c.horizontal_scale_m
    generator.vertical_scale = c.vertical_scale_m
    generator.slope_threshold = c.slope_threshold
    generator.sub_terrains = {
        f"course_{i:02d}": CourseTerrainCfg(
            proportion=1.0 / len(c.step_heights_m), course=c.copy(), step_height_m=h,
        ) for i, h in enumerate(c.step_heights_m)
    }
    env_cfg.scene.terrain.terrain_generator = generator
    env_cfg.scene.terrain.max_init_terrain_level = 0
    command = env_cfg.commands.base_velocity
    fixed = command.Ranges(lin_vel_x=(c.forward_command_mps, c.forward_command_mps),
                           lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0))
    command.initial_ranges = fixed.copy()
    command.final_ranges = fixed.copy()
    command.ranges = fixed.copy()
    command.rel_standing_envs = 0.0
    command.resampling_time_range = (env_cfg.episode_length_s, env_cfg.episode_length_s)
    spawn_x = -generator.size[0] / 2 + c.spawn_from_tile_start_m
    env_cfg.events.reset_base.params["pose_range"] = {
        "x": (spawn_x, spawn_x), "y": (-c.spawn_lateral_range_m, c.spawn_lateral_range_m),
        "yaw": (0.0, 0.0),
    }
    env_cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
    return env_cfg


def audit_course(env, c: CourseEvalCfg, output: Path) -> list[dict]:
    """USD 충돌 메시를 수직 raycast하고 설계 단면과 비교한다."""
    from eval_stairs_fixed import imported_terrain_mesh

    terrain = env.unwrapped.scene.terrain
    n = env.unwrapped.num_envs
    columns = torch.arange(n, device=terrain.terrain_types.device)
    if not torch.equal(terrain.terrain_types, columns):
        raise ValueError("course env ID와 지형 열이 1:1로 대응하지 않는다")
    if not torch.allclose(terrain.env_origins, terrain.terrain_origins[0, columns], atol=c.profile_tolerance_m, rtol=0):
        raise ValueError("course env가 지정된 타일 원점에 배치되지 않았다")
    if not torch.allclose(terrain.env_origins[:, 2], torch.zeros(n, device=terrain.env_origins.device),
                          atol=c.profile_tolerance_m, rtol=0):
        raise ValueError("course 출발면 높이 원점은 0이어야 한다")
    mesh = imported_terrain_mesh(terrain)
    length = section_edges(c)[-1]
    # 경계의 수직면에서는 좌/우 어느 면을 ray가 만나는지 불안정하므로 내부만 검사한다.
    sample_x = np.arange(c.horizontal_scale_m / 2, length, c.profile_sample_spacing_m)
    edges = np.asarray(section_edges(c))
    risers = np.r_[edges[2] + np.arange(c.stair_count) * c.step_width_m,
                   edges[4] + np.arange(c.stair_count) * c.step_width_m]
    discontinuities = np.r_[edges, risers]
    clean = (np.abs(sample_x[:, None] - discontinuities[None, :]).min(axis=1)
             > c.horizontal_scale_m * 0.6)
    clean &= (sample_x < edges[6]) | (sample_x >= edges[7])
    if not clean.any():
        raise ValueError("course 단면 검사에 유효한 점이 없다")
    xs = sample_x[clean]
    origins = terrain.env_origins.detach().cpu().numpy()
    checks = []
    first_actual = None
    for env_id in range(n):
        h = c.step_heights_m[env_id // c.envs_per_height]
        ray_origins = np.repeat(origins[env_id:env_id + 1], len(xs), axis=0).astype(np.float64)
        ray_origins[:, 0] += xs - env.unwrapped.cfg.scene.terrain.terrain_generator.size[0] / 2
        ray_origins[:, 2] = c.profile_ray_height_m
        directions = np.tile(np.array([[0.0, 0.0, -1.0]]), (len(xs), 1))
        locations, ids, _ = mesh.ray.intersects_location(ray_origins, directions, multiple_hits=False)
        actual = np.full(len(xs), np.nan)
        actual[ids] = locations[:, 2]
        expected = designed_height(xs, h, c)
        error = np.abs(actual - expected)
        if not np.isfinite(actual).all() or np.max(error) > c.profile_tolerance_m:
            bad = int(np.nanargmax(error)) if np.isfinite(error).any() else -1
            raise ValueError(f"course USD 단면 불일치 env={env_id} x={xs[bad]:.2f} "
                             f"actual={actual[bad]:.3f} expected={expected[bad]:.3f}")
        if first_actual is None:
            first_actual = actual.copy()
        checks.append({"env_id": env_id, "step_height_m": h, "origin_z_m": float(origins[env_id, 2]),
                       "profile_max_error_m": float(error.max())})
    # 첫 타일의 촘촘한 단면 그림. 실제 ray가 있는 표본만 점으로 표시한다.
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(sample_x, designed_height(sample_x, c.step_heights_m[0], c), label="design")
    ax.scatter(xs, first_actual, s=7, label="USD ray")
    for edge in edges[:-1]:
        ax.axvline(edge, color="gray", linewidth=0.4)
    ax.set(xlabel="x from tile start (m)", ylabel="terrain z (m)", title="Continuous course profile")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output.with_name(output.stem + "_profile.png"), dpi=170)
    plt.close(fig)
    return checks


def run_course_eval(args, *, policy_factory=None, latent_stats=None):
    c = CourseEvalCfg()
    for name, arg in (("step_heights_m", "course_step_heights_m"), ("slope_deg", "course_slope_deg"),
                      ("heading_hold", "course_heading_hold"), ("envs_per_height", "envs_per_height"),
                      ("seed", "seed"), ("episode_steps", "episode_steps"),
                      ("viewer_env_index", "viewer_env_index")):
        value = getattr(args, arg, None)
        if value is not None:
            setattr(c, name, tuple(value) if name == "step_heights_m" else value)
    if any(h <= 0 for h in c.step_heights_m) or not c.step_heights_m:
        raise ValueError("course 높이 목록은 양수여야 한다")
    if c.width_m <= 2 * c.off_course_margin_m or c.spawn_from_tile_start_m >= c.start_flat_m:
        raise ValueError("course 폭 또는 출발 위치가 유효하지 않다")
    if not 0 <= c.viewer_env_index < c.envs_per_height * len(c.step_heights_m):
        raise ValueError("viewer_env_index가 env 범위를 벗어났다")
    if c.slope_threshold <= math.tan(math.radians(c.slope_deg)):
        raise ValueError("slope_threshold가 ramp 기울기보다 커야 한다")
    if c.slope_threshold >= min(c.step_heights_m) / c.horizontal_scale_m:
        raise ValueError("slope_threshold가 계단 수직화 기울기보다 작아야 한다")
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"기존 평가 JSON을 덮어쓰지 않는다: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    env_cfg = make_course_env_cfg(args.task, c, args.device)
    student_task = args.task == "Unitree-Go2-JepaLoco-StudentDepth-EasyStart"
    if student_task and any(getattr(args, k, None) is not None for k in
                            ("camera_pos", "camera_pitch_deg", "camera_intrinsics")):
        from eval_stairs_fixed import apply_camera_override
        apply_camera_override(env_cfg, args)
    if student_task:
        agent_cfg = OracleCurrentPPORunnerCfg()
    else:
        import importlib.metadata as metadata
        agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
        agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, metadata.version("rsl-rl-lib"))
        agent_cfg.seed = c.seed
    env = gym.make(args.task, cfg=env_cfg, render_mode="rgb_array" if args.video else None)
    try:
        checks = audit_course(env, c, output)
        if args.audit_terrain_only:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps({"terrain": "course", "course": c.to_dict(), "tiles": checks}, indent=2))
            print(f"COURSE_AUDIT_OK tiles={len(checks)} output={output}", flush=True)
            return
        video_wrapper = None
        if args.video:
            video_wrapper = gym.wrappers.RecordVideo(env, video_folder=str(output.with_suffix("")) + "_video",
                                                     step_trigger=lambda step: step == 0,
                                                     video_length=c.episode_steps, disable_logger=True)
            env = video_wrapper
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        if student_task:
            policy = (policy_factory(str(Path(args.checkpoint).resolve()), torch.device(env.unwrapped.device))
                      if policy_factory is not None else
                      StudentInferencePolicy(str(Path(args.checkpoint).resolve()), torch.device(env.unwrapped.device)))
        else:
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
            runner.load(str(Path(args.checkpoint).resolve()))
            policy = runner.get_inference_policy(device=env.unwrapped.device)
        obs = env.get_observations()
        robot = env.unwrapped.scene["robot"]
        origin = env.unwrapped.scene.env_origins
        n = env.num_envs
        device = env.unwrapped.device
        active = torch.ones(n, dtype=torch.bool, device=device)
        consecutive = torch.zeros(n, dtype=torch.long, device=device)
        max_yaw_rate = torch.zeros(n, device=device)
        start_y = robot.data.root_pos_w[:, 1].clone()
        tile_start = origin[:, 0] - env_cfg.scene.terrain.terrain_generator.size[0] / 2
        edges = section_edges(c)
        pass_edges = section_pass_edges(c)
        target_x = pass_edges[-1]
        end_steps = torch.full((n, len(SECTION_NAMES)), -1, dtype=torch.long, device=device)
        last_section = torch.zeros(n, dtype=torch.long, device=device)
        outcome = [None] * n
        failure_section = [-1] * n
        stats_sum = torch.zeros((n, len(SECTION_NAMES), 4), device=device)
        stats_count = torch.zeros((n, len(SECTION_NAMES)), device=device)
        max_lateral = torch.zeros(n, device=device)
        max_forward = torch.zeros(n, device=device)
        done_reasons = [[] for _ in range(n)]
        # 종료 직전 pose를 기록한다. env.step() 반환 시에는 해당 env가 이미 reset된다.
        terminal_x = torch.full((n,), float("nan"), device=device)
        original_reset = env.unwrapped._reset_idx
        def capture_reset(env_ids):
            if len(env_ids):
                terminal_x[env_ids] = robot.data.root_pos_w[env_ids, 0] - tile_start[env_ids]
            return original_reset(env_ids)
        env.unwrapped._reset_idx = capture_reset
        import isaaclab.utils.math as math_utils
        for step in range(c.episode_steps):
            yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w)[2]
            yaw_rate = torch.clamp(-c.heading_gain * torch.atan2(torch.sin(yaw), torch.cos(yaw)),
                                   -c.heading_max_yaw_rate, c.heading_max_yaw_rate) if c.heading_hold else torch.zeros_like(yaw)
            with torch.inference_mode():
                command = env.unwrapped.command_manager.get_command("base_velocity")
                command[:, 0] = c.forward_command_mps
                command[:, 1] = 0.0
                command[:, 2] = yaw_rate
                obs["policy"][:, 6:9] = command  # policy 관측 순서: 각속도3, 중력3, 명령3.
            max_yaw_rate = torch.where(active, torch.maximum(max_yaw_rate, yaw_rate.abs()), max_yaw_rate)
            if args.video:
                i = c.viewer_env_index
                pos = robot.data.root_pos_w[i].detach().cpu().numpy()
                env.unwrapped.sim.set_camera_view(
                    eye=(float(pos[0] - 2.5), float(pos[1] + 3.0), float(pos[2] + 1.6)),
                    target=(float(pos[0] + 0.7), float(pos[1]), float(pos[2])),
                )
            with torch.inference_mode():
                actions = policy(obs)
                x_before = robot.data.root_pos_w[:, 0] - tile_start
                sec = torch.as_tensor(np.searchsorted(edges, x_before.detach().cpu().numpy(), side="right"),
                                      device=device).clamp(max=len(SECTION_NAMES) - 1)
                body_xy = robot.data.root_lin_vel_b[:, :2]
                velocity_error = torch.linalg.vector_norm(body_xy - command[:, :2], dim=-1)
                stationary = (torch.linalg.vector_norm(command[:, :2], dim=-1) > c.stationary_command_threshold_mps) & (
                    torch.linalg.vector_norm(body_xy, dim=-1) < c.stationary_body_threshold_mps)
                lateral = (robot.data.root_pos_w[:, 1] - start_y).abs()
                max_lateral = torch.where(active, torch.maximum(max_lateral, lateral), max_lateral)
                for s in range(len(SECTION_NAMES)):
                    selected = active & (sec == s)
                    stats_sum[:, s, 0] += torch.where(selected, robot.data.root_lin_vel_w[:, 0], 0.0)
                    stats_sum[:, s, 1] += torch.where(selected, velocity_error, 0.0)
                    stats_sum[:, s, 2] += selected & stationary
                    stats_sum[:, s, 3] = torch.maximum(stats_sum[:, s, 3], torch.where(selected, lateral, 0.0))
                    stats_count[:, s] += selected.float()
                obs, _, dones, extras = env.step(actions)
                if getattr(policy, "is_recurrent", False):
                    policy.reset(dones)
                timeouts = extras.get("time_outs", torch.zeros_like(dones)).bool()
                live = active & ~dones.bool()
                x = robot.data.root_pos_w[:, 0] - tile_start
                x = torch.where(active & dones.bool(), terminal_x, x)
                max_forward = torch.where(active, torch.maximum(max_forward, x), max_forward)
                lateral_after = (robot.data.root_pos_w[:, 1] - start_y).abs()
                off_course = live & (lateral_after > c.width_m / 2 - c.off_course_margin_m)
                for s, edge in enumerate(pass_edges):
                    crossed = active & (end_steps[:, s] < 0) & (x >= edge)
                    end_steps[:, s] = torch.where(crossed, step + 1, end_steps[:, s])
                current_sec = torch.as_tensor(np.searchsorted(edges, x.detach().cpu().numpy(), side="right"),
                                              device=device).clamp(max=len(SECTION_NAMES) - 1)
                last_section = torch.where(active, torch.maximum(last_section, current_sec), last_section)
                reached = live & ~off_course & (x >= target_x)
                consecutive = torch.where(reached, consecutive + 1, torch.zeros_like(consecutive))
                success = reached & (consecutive >= c.hold_steps)
                stopping = active & (dones.bool() | off_course | success)
                for i in torch.nonzero(stopping, as_tuple=True)[0].tolist():
                    outcome[i] = course_outcome(bool(success[i]), bool(off_course[i]), bool(dones[i]),
                                                bool(timeouts[i]), False)
                    failure_section[i] = -1 if outcome[i] == "success" else int(current_sec[i])
                    if bool(dones[i]):
                        tm = env.unwrapped.termination_manager
                        done_reasons[i] = [name for name in tm.active_terms if bool(tm.get_term(name)[i])]
                active &= ~stopping
                if video_wrapper is not None and not bool(active[c.viewer_env_index]) and video_wrapper.recording:
                    video_wrapper.stop_recording()
                if not bool(active.any()):
                    break
        for i in torch.nonzero(active, as_tuple=True)[0].tolist():
            outcome[i] = "stall"
            failure_section[i] = int(last_section[i])
        per_env = []
        for i in range(n):
            count = stats_count[i].cpu().tolist()
            values = stats_sum[i].cpu().tolist()
            per_env.append({"env_id": i, "step_height_m": c.step_heights_m[i // c.envs_per_height],
                            "success": outcome[i] == "success", "outcome": outcome[i],
                            "failure_section": failure_section[i], "failure_section_name": (
                                SECTION_NAMES[failure_section[i]] if failure_section[i] >= 0 else None),
                            "last_section": int(last_section[i]), "last_section_name": SECTION_NAMES[int(last_section[i])],
                            "section_end_steps": {name: int(end_steps[i, s]) if end_steps[i, s] > 0 else None
                                                  for s, name in enumerate(SECTION_NAMES)},
                            "section_metrics": {name: {"samples": int(count[s]),
                                "mean_forward_speed_mps": values[s][0] / count[s] if count[s] else None,
                                "mean_velocity_error_mps": values[s][1] / count[s] if count[s] else None,
                                "stationary_rate": values[s][2] / count[s] if count[s] else None,
                                "max_lateral_m": values[s][3]}
                                for s, name in enumerate(SECTION_NAMES)},
                            "max_forward_m": float(max_forward[i]), "max_lateral_m": float(max_lateral[i]),
                            "max_abs_yaw_command_radps": float(max_yaw_rate[i]),
                            "termination_reasons": done_reasons[i]})
        groups = {}
        for height in c.step_heights_m:
            group = [row for row in per_env if row["step_height_m"] == height]
            groups[f"{height:.2f}"] = {"n": len(group), "step_height_m": height,
                "success_rate": sum(row["success"] for row in group) / len(group),
                "section_pass_rates": {name: sum(row["section_end_steps"][name] is not None for row in group) / len(group)
                                       for name in SECTION_NAMES},
                "failure_counts": {kind: sum(row["outcome"] == kind for row in group)
                                   for kind in ("fall", "stall", "off_course")},
                "failure_section_counts": {name: sum(row["failure_section_name"] == name for row in group)
                                           for name in SECTION_NAMES}}
        report = {"task": args.task, "terrain": "course", "checkpoint": str(Path(args.checkpoint).resolve()),
                  "seed": c.seed, "course": c.to_dict(), "section_names": SECTION_NAMES,
                  "section_end_x_m": pass_edges, "physical_section_end_x_m": edges,
                  "finish_target_x_m": target_x,
                  "success_rule": f"x from tile start >= {target_x:.2f} m for {c.hold_steps} consecutive active steps",
                  "failure_rule": "fall=non-timeout termination; stall=timeout/horizon; off_course=abs(y-y0)>width/2-margin",
                  "heading_rule": "yaw_rate=clip(-gain*wrap(yaw), +/-max)" if c.heading_hold else "yaw_rate=0",
                  "tiles": checks, "groups": groups, "per_env": per_env,
                  "policy_mode": getattr(policy, "policy_mode", "teacher"),
                  "student_method": getattr(policy, "method", None)}
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2))
        print(f"COURSE_EVAL_OK output={output} groups={groups}", flush=True)
    finally:
        env.close()
