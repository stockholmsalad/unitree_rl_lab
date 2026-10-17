"""Student 학습 전용 depth/height 복원 decoder와 분산 정규화 손실."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class FutureDepthDecoder(nn.Module):
    """Context·조건 → 2×H×W (depth 값, validity logit)."""

    def __init__(self, input_dim: int, image_hw: tuple[int, int], hidden_dim: int = 64,
                 seed_channels: int = 4):
        super().__init__()
        height, width = image_hw
        if min(height, width, hidden_dim, seed_channels) < 1 or height % 8 or width % 8:
            raise ValueError("depth decoder 영상 크기는 8의 배수이고 층 크기는 양수여야 한다")
        self.seed_shape = (seed_channels, height // 8, width // 8)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ELU(),
            nn.Linear(hidden_dim, seed_channels * (height // 8) * (width // 8)), nn.ELU(),
        )
        self.up = nn.Sequential(
            nn.ConvTranspose2d(seed_channels, 8, 4, stride=2, padding=1), nn.ELU(),
            nn.ConvTranspose2d(8, 4, 4, stride=2, padding=1), nn.ELU(),
            nn.ConvTranspose2d(4, 2, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lead = x.shape[:-1]
        hidden = self.mlp(x.reshape(-1, x.shape[-1])).reshape(-1, *self.seed_shape)
        return self.up(hidden).reshape(*lead, 2, self.seed_shape[1] * 8, self.seed_shape[2] * 8)


class CurrentHeightDecoder(nn.Module):
    def __init__(self, input_dim: int, height_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ELU(),
                                 nn.Linear(hidden_dim, hidden_dim), nn.ELU(),
                                 nn.Linear(hidden_dim, height_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


def normalized_reconstruction_mse(
    prediction: torch.Tensor, target: torch.Tensor, valid: torch.Tensor, variance_floor: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """마지막 차원별 유효 표본 분산으로 MSE를 나눈다.

    Returns loss, constant-mean baseline, raw target variance, valid fraction.
    """
    if prediction.shape != target.shape or valid.shape != target.shape or prediction.ndim != 2:
        raise ValueError("prediction/target/valid는 같은 [samples, dimensions] shape이어야 한다")
    if variance_floor <= 0:
        raise ValueError("variance_floor는 양수여야 한다")
    valid = valid.bool()
    count = valid.sum(dim=0)
    if not valid.any():
        zero = prediction.sum() * 0
        return zero, zero.detach(), zero.detach(), zero.detach()
    detached = target.detach()
    mean = torch.where(valid, detached, 0).sum(dim=0) / count.clamp_min(1)
    variance = (torch.where(valid, detached - mean, 0).square().sum(dim=0) / count.clamp_min(1)).detach()
    scale = variance.clamp_min(variance_floor)
    squared = torch.where(valid, (prediction - detached).square(), 0)
    loss = (squared / scale).sum() / valid.sum()
    baseline = (torch.where(valid, detached - mean, 0).square() / scale).sum() / valid.sum()
    return loss, baseline, variance[count > 0].mean(), valid.float().mean()


def depth_reconstruction_loss(
    prediction: torch.Tensor, target: torch.Tensor, variance_floor: float,
    depth_weight: float, mask_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """미래 새 프레임의 유효 depth 픽셀 MSE와 전체 validity BCE."""
    if prediction.shape != target.shape or prediction.ndim != 4 or prediction.shape[1] != 2:
        raise ValueError("prediction/target은 같은 [samples,2,H,W] shape이어야 한다")
    if prediction.shape[0] == 0:
        zero = prediction.sum() * 0
        return zero, zero.detach(), zero.detach(), zero.detach()
    mask = target[:, 1].detach() > 0.5
    depth_loss, depth_baseline, target_var, valid_fraction = normalized_reconstruction_mse(
        prediction[:, 0].flatten(1), target[:, 0].float().flatten(1), mask.flatten(1), variance_floor,
    )
    mask_target = mask.float()
    mask_loss = F.binary_cross_entropy_with_logits(prediction[:, 1], mask_target)
    p = mask_target.mean().clamp(1.0e-6, 1 - 1.0e-6)
    mask_baseline = F.binary_cross_entropy_with_logits(torch.logit(p).expand_as(mask_target), mask_target)
    return (depth_weight * depth_loss + mask_weight * mask_loss,
            depth_weight * depth_baseline + mask_weight * mask_baseline,
            target_var, valid_fraction)


def height_points_visible(
    points_b: torch.Tensor, depth: torch.Tensor, camera_origin_b: tuple[float, float, float],
    pitch_down_deg: float, intrinsic: tuple[float, float, float, float],
    near_m: float, far_m: float, tolerance_m: float,
) -> torch.Tensor:
    """기존 blind_spot.visible_in_depth와 같은 투영·depth 일치 판정의 GPU 버전."""
    if points_b.ndim != 3 or points_b.shape[-1] != 3 or depth.ndim != 4 or depth.shape[1] != 2:
        raise ValueError("points_b [N,P,3], depth [N,2,H,W]이어야 한다")
    if points_b.shape[0] != depth.shape[0] or near_m <= 0 or far_m <= near_m or tolerance_m < 0:
        raise ValueError("카메라 범위 또는 batch shape가 올바르지 않다")
    offset = points_b - points_b.new_tensor(camera_origin_b)
    dx, dy, dz = offset.unbind(dim=-1)
    angle = math.radians(pitch_down_deg)
    axial = math.cos(angle) * dx - math.sin(angle) * dz
    up = math.sin(angle) * dx + math.cos(angle) * dz
    safe = axial.clamp_min(near_m)
    fx, fy, cx, cy = intrinsic
    u = cx - fx * dy / safe
    v = cy - fy * up / safe
    height, width = depth.shape[-2:]
    geometric = (torch.isfinite(axial) & (axial >= near_m) & (axial <= far_m) &
                 (u >= 0) & (u < width) & (v >= 0) & (v < height))
    pixel_u = u.nan_to_num().round().long().clamp(0, width - 1)
    pixel_v = v.nan_to_num().round().long().clamp(0, height - 1)
    batch = torch.arange(depth.shape[0], device=depth.device)[:, None]
    sampled_valid = depth[batch, 1, pixel_v, pixel_u] > 0.5
    sampled_m = near_m + depth[batch, 0, pixel_v, pixel_u].float() * (far_m - near_m)
    return geometric & sampled_valid & ((sampled_m - axial).abs() <= tolerance_m)
