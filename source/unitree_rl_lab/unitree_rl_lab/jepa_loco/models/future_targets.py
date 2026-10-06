"""미래 pose 외삽, target 정합 mask, latent 회귀 손실.

명령은 시점 t의 몸통 좌표계 평면 twist이며 예측 지평 동안 일정하다고 가정한다.
허용 오차는 호출자가 configclass에서 전달한다.
"""

from __future__ import annotations

import torch


def extrapolate_command_pose(pose_xy_yaw: torch.Tensor, command_vx_vy_wz: torch.Tensor,
                             horizon_s: float | torch.Tensor) -> torch.Tensor:
    """SE(2) exponential로 명령을 적분한다. 입력·출력 shape [N, 3]."""
    if pose_xy_yaw.ndim != 2 or pose_xy_yaw.shape[-1] != 3 or command_vx_vy_wz.shape != pose_xy_yaw.shape:
        raise ValueError("pose와 command는 모두 [N, 3]이어야 한다")
    horizon = torch.as_tensor(horizon_s, dtype=pose_xy_yaw.dtype, device=pose_xy_yaw.device)
    if horizon.ndim > 1 or (horizon.ndim == 1 and horizon.shape[0] != pose_xy_yaw.shape[0]):
        raise ValueError("horizon은 scalar 또는 [N]이어야 한다")
    if torch.any(horizon < 0):
        raise ValueError("horizon은 음수가 될 수 없다")

    theta = command_vx_vy_wz[:, 2] * horizon
    sinc = torch.sinc(theta / torch.pi)
    # (1-cos(theta))/theta = theta/2 * sinc(theta/2)^2, theta=0에서도 안정적이다.
    cosc = (theta / 2) * torch.sinc(theta / (2 * torch.pi)).square()
    dx_body = horizon * (sinc * command_vx_vy_wz[:, 0] - cosc * command_vx_vy_wz[:, 1])
    dy_body = horizon * (cosc * command_vx_vy_wz[:, 0] + sinc * command_vx_vy_wz[:, 1])
    yaw = pose_xy_yaw[:, 2]
    dx_world = torch.cos(yaw) * dx_body - torch.sin(yaw) * dy_body
    dy_world = torch.sin(yaw) * dx_body + torch.cos(yaw) * dy_body
    future_yaw = torch.atan2(torch.sin(yaw + theta), torch.cos(yaw + theta))
    return torch.stack((pose_xy_yaw[:, 0] + dx_world, pose_xy_yaw[:, 1] + dy_world, future_yaw), dim=-1)


def future_alignment_mask(actual_pose: torch.Tensor, expected_pose: torch.Tensor,
                          command_history: torch.Tensor, *, position_tolerance_m: float,
                          yaw_tolerance_rad: float, command_tolerance: float,
                          same_episode: torch.Tensor, visible: torch.Tensor | None = None) -> torch.Tensor:
    """실제 미래와 명령 외삽 pose가 정합된 [N] 표본을 표시한다.

    command_history [N, T, 3]은 t부터 t+Δ까지의 명령이다. visible은 teacher
    영역이 현재 센서에서 관측 가능한지 따로 계산한 [N] mask다.
    """
    if actual_pose.ndim != 2 or actual_pose.shape[-1] != 3 or expected_pose.shape != actual_pose.shape:
        raise ValueError("actual_pose와 expected_pose는 [N, 3]이어야 한다")
    if command_history.ndim != 3 or command_history.shape[0] != actual_pose.shape[0] or command_history.shape[1] < 1 or command_history.shape[2] != 3:
        raise ValueError("command_history는 [N, T>=1, 3]이어야 한다")
    n = actual_pose.shape[0]
    if same_episode.shape != (n,) or (visible is not None and visible.shape != (n,)):
        raise ValueError("same_episode와 visible은 [N]이어야 한다")
    if min(position_tolerance_m, yaw_tolerance_rad, command_tolerance) < 0:
        raise ValueError("허용 오차는 음수가 될 수 없다")

    position_error = torch.linalg.vector_norm(actual_pose[:, :2] - expected_pose[:, :2], dim=-1)
    yaw_delta = actual_pose[:, 2] - expected_pose[:, 2]
    yaw_error = torch.abs(torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)))
    command_change = torch.linalg.vector_norm(command_history - command_history[:, :1], dim=-1).amax(dim=1)
    mask = ((position_error <= position_tolerance_m) & (yaw_error <= yaw_tolerance_rad)
            & (command_change <= command_tolerance) & same_episode.bool())
    return mask if visible is None else mask & visible.bool()


def masked_latent_mse(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """유효 표본 수로 정규화한 latent MSE. 빈 mask는 gradient가 0인 0을 반환한다."""
    if prediction.ndim != 2 or target.shape != prediction.shape or mask.shape != (prediction.shape[0],):
        raise ValueError("prediction/target은 [N,D], mask는 [N]이어야 한다")
    per_sample = (prediction - target.detach()).square().mean(dim=-1)
    weights = mask.to(dtype=per_sample.dtype)
    return (per_sample * weights).sum() / weights.sum().clamp(min=1)
