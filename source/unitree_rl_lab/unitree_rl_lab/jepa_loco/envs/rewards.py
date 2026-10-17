"""Oracle용 지형 기준 스윙 발 높이 보상. 계산 본체는 Isaac에 의존하지 않는다."""

from __future__ import annotations

import torch


def track_lin_vel_progress_values(
    command_xy: torch.Tensor, body_velocity_xy: torch.Tensor, *,
    command_threshold: float, std_small: float, min_reward: float,
) -> torch.Tensor:
    """명령 방향의 몸통 좌표계 진행률. 작은 명령에서는 정지 추종 보상을 사용한다."""
    if command_xy.ndim != 2 or command_xy.shape[1] != 2 or body_velocity_xy.shape != command_xy.shape:
        raise ValueError("command_xy와 body_velocity_xy는 같은 [N,2] shape이어야 한다")
    if command_threshold <= 0 or std_small <= 0 or min_reward > 0:
        raise ValueError("command_threshold/std_small은 양수이고 min_reward는 0 이하여야 한다")
    speed = torch.linalg.vector_norm(command_xy, dim=1)
    projected_ratio = (command_xy * body_velocity_xy).sum(dim=1) / speed.square().clamp_min(
        command_threshold**2
    )
    progress = projected_ratio.clamp(min=min_reward, max=1.0)
    small_command = torch.exp(-((command_xy - body_velocity_xy).square().sum(dim=1)) / std_small**2)
    return torch.where(speed >= command_threshold, progress, small_command)


def track_lin_vel_progress(
    env, command_name: str, command_threshold: float = 0.1,
    std_small: float = 0.5, min_reward: float = -1.0,
) -> torch.Tensor:
    command_xy = env.command_manager.get_command(command_name)[:, :2]
    body_velocity_xy = env.scene["robot"].data.root_lin_vel_b[:, :2]
    return track_lin_vel_progress_values(
        command_xy, body_velocity_xy, command_threshold=command_threshold,
        std_small=std_small, min_reward=min_reward,
    )


def nominal_pose_penalty_values(
    joint_pos: torch.Tensor,
    default_joint_pos: torch.Tensor,
    command: torch.Tensor,
    body_vel_b: torch.Tensor,
    *,
    group: str,
    stand_still_scale: float,
    velocity_threshold: float,
    normalization: str,
    normalization_floor: float,
) -> torch.Tensor:
    """Hip L2 norm or thigh/calf squared error, with the original stillness rule."""
    n = joint_pos.shape[0]
    if (joint_pos.ndim != 2 or default_joint_pos.shape != joint_pos.shape
            or command.shape != (n, 3) or body_vel_b.shape != (n, 3)):
        raise ValueError("joint/default [N,J], command/body velocity [N,3] 필요")
    if stand_still_scale < 0 or velocity_threshold < 0 or normalization_floor <= 0:
        raise ValueError("stand_still_scale/velocity_threshold는 음수가 아니고 normalization_floor는 양수여야 한다")
    delta = joint_pos - default_joint_pos
    if group == "hip":
        penalty = torch.linalg.vector_norm(delta, dim=1)
    elif group == "thigh_calf":
        if normalization != "command_xy_sq_floor":
            raise ValueError(f"지원하지 않는 정규화 방식: {normalization}")
        command_xy_sq = (command[:, :2] ** 2).sum(dim=1)
        penalty = delta.square().sum(dim=1) / torch.maximum(
            command_xy_sq, torch.full_like(command_xy_sq, normalization_floor)
        )
    else:
        raise ValueError(f"지원하지 않는 관절 그룹: {group}")
    moving = (torch.linalg.vector_norm(command, dim=1) > 0) | (
        torch.linalg.vector_norm(body_vel_b[:, :2], dim=1) > velocity_threshold
    )
    return torch.where(moving, penalty, stand_still_scale * penalty)


def nominal_pose_hip(
    env, asset_cfg, command_name: str, stand_still_scale: float,
    velocity_threshold: float, normalization: str, normalization_floor: float,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    return nominal_pose_penalty_values(
        robot.data.joint_pos[:, ids], robot.data.default_joint_pos[:, ids],
        env.command_manager.get_command(command_name), robot.data.root_lin_vel_b,
        group="hip", stand_still_scale=stand_still_scale,
        velocity_threshold=velocity_threshold, normalization=normalization,
        normalization_floor=normalization_floor,
    )


def nominal_pose_thigh_calf(
    env, asset_cfg, command_name: str, stand_still_scale: float,
    velocity_threshold: float, normalization: str, normalization_floor: float,
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    ids = asset_cfg.joint_ids
    return nominal_pose_penalty_values(
        robot.data.joint_pos[:, ids], robot.data.default_joint_pos[:, ids],
        env.command_manager.get_command(command_name), robot.data.root_lin_vel_b,
        group="thigh_calf", stand_still_scale=stand_still_scale,
        velocity_threshold=velocity_threshold, normalization=normalization,
        normalization_floor=normalization_floor,
    )


def swing_foot_clearance_values(
    foot_pos_w: torch.Tensor,
    ray_hits_w: torch.Tensor,
    contact_force_w: torch.Tensor,
    air_time: torch.Tensor,
    command: torch.Tensor,
    body_vel_b: torch.Tensor,
    *,
    target_clearance: float,
    foot_radius: float,
    radius: float,
    contact_threshold: float,
    min_air_time: float,
    command_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return [N] reward, [N,4] terrain-relative clearance, [N,4] valid swing mask."""
    n, feet, xyz = foot_pos_w.shape
    if (feet != 4 or xyz != 3 or ray_hits_w.ndim != 3 or ray_hits_w.shape[0] != n
            or ray_hits_w.shape[-1] != 3 or contact_force_w.shape != (n, 4, 3)
            or air_time.shape != (n, 4) or command.shape != (n, 3)
            or body_vel_b.shape != (n, 3)):
        raise ValueError("발 [N,4,3], ray [N,R,3], force [N,4,3], air [N,4], command/velocity [N,3] 필요")
    if target_clearance <= 0 or radius <= 0 or foot_radius < 0 or command_threshold < 0:
        raise ValueError("target_clearance와 radius는 양수, foot_radius와 command_threshold는 0 이상이어야 한다")

    finite_hit = torch.isfinite(ray_hits_w).all(dim=-1)
    xy_distance_sq = ((ray_hits_w[:, None, :, :2] - foot_pos_w[:, :, None, :2]) ** 2).sum(dim=-1)
    nearby = finite_hit[:, None, :] & (xy_distance_sq <= radius**2)
    ground_z = ray_hits_w[:, None, :, 2].expand(-1, 4, -1)
    ground_z = torch.where(nearby, ground_z, -torch.inf).amax(dim=-1)
    valid = torch.isfinite(ground_z)
    clearance = torch.where(valid, foot_pos_w[:, :, 2] - foot_radius - ground_z, 0.0)
    swing = (torch.linalg.vector_norm(contact_force_w, dim=-1) < contact_threshold) & (air_time > min_air_time)
    valid_swing = valid & swing
    foot_score = torch.where(valid_swing, (clearance / target_clearance).clamp(0.0, 1.0), 0.0)
    command_xy = command[:, :2]
    command_sq = (command_xy**2).sum(dim=-1)
    command_active = command_sq > command_threshold**2
    # Both vectors are in the body frame. The inactive denominator is set to one
    # before division so a zero command cannot create NaN.
    progress = (body_vel_b[:, :2] * command_xy).sum(dim=-1) / torch.where(
        command_active, command_sq, torch.ones_like(command_sq),
    )
    gate = torch.where(command_active, progress.clamp(0.0, 1.0), 0.0)
    reward = gate * foot_score.mean(dim=-1)
    return reward, clearance, valid_swing


def swing_foot_clearance(
    env,
    asset_cfg,
    sensor_cfg,
    ray_cfg,
    command_name: str,
    target_clearance: float,
    foot_radius: float,
    radius: float,
    contact_threshold: float,
    min_air_time: float,
    command_threshold: float,
) -> torch.Tensor:
    """Isaac reward-manager adapter; cache matching foot IDs and episode diagnostics."""
    if not hasattr(env, "_swing_reward_foot_ids"):
        foot_ids, foot_names = env.scene[asset_cfg.name].find_bodies(".*_foot")
        contact_ids, contact_names = env.scene.sensors[sensor_cfg.name].find_bodies(".*_foot")
        by_name = dict(zip(contact_names, contact_ids))
        if len(foot_ids) != 4 or set(foot_names) != set(contact_names):
            raise ValueError("robot/contact sensor의 네 발 이름이 일치해야 한다")
        env._swing_reward_foot_ids = foot_ids
        env._swing_reward_contact_ids = [by_name[name] for name in foot_names]
        env._swing_clearance_sum = torch.zeros(env.num_envs, device=env.device)
        env._swing_clearance_count = torch.zeros(env.num_envs, device=env.device)
        env._swing_clearance_max = torch.full((env.num_envs,), -torch.inf, device=env.device)

    robot = env.scene[asset_cfg.name]
    contact = env.scene.sensors[sensor_cfg.name]
    ids = env._swing_reward_contact_ids
    value, clearance, valid_swing = swing_foot_clearance_values(
        robot.data.body_pos_w[:, env._swing_reward_foot_ids],
        env.scene.sensors[ray_cfg.name].data.ray_hits_w,
        contact.data.net_forces_w[:, ids],
        contact.data.current_air_time[:, ids],
        env.command_manager.get_command(command_name),
        robot.data.root_lin_vel_b,
        target_clearance=target_clearance,
        foot_radius=foot_radius,
        radius=radius,
        contact_threshold=contact_threshold,
        min_air_time=min_air_time,
        command_threshold=command_threshold,
    )
    env._swing_clearance_sum += torch.where(valid_swing, clearance, 0.0).sum(dim=1)
    env._swing_clearance_count += valid_swing.sum(dim=1)
    env._swing_clearance_max = torch.maximum(
        env._swing_clearance_max,
        torch.where(valid_swing, clearance, -torch.inf).amax(dim=1),
    )
    return value
