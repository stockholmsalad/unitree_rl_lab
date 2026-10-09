"""Current teacher 증류와 보조 JEPA 손실의 순수 함수."""

from __future__ import annotations

import torch


def normalized_latent_mse(pred: torch.Tensor, target: torch.Tensor,
                          mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.shape[-1] != mean.numel() or mean.shape != std.shape:
        raise ValueError("latent·정규화 통계 shape가 맞지 않는다")
    if (std <= 0).any() or not torch.isfinite(std).all():
        raise ValueError("고정 teacher 표준편차는 유한한 양수여야 한다")
    return (((pred - mean) / std - (target.detach() - mean) / std) ** 2).mean()


def jepa_valid_mask(done: torch.Tensor, command: torch.Tensor,
                    horizon_steps: int, command_tolerance: float) -> torch.Tensor:
    """[T-h,N]: 같은 에피소드, 전체 구간 명령 유지인 시작점만 True."""
    if done.ndim != 2 or command.shape[:2] != done.shape or command.shape[-1] != 3:
        raise ValueError("done [T,N], command [T,N,3]이어야 한다")
    if horizon_steps < 1 or horizon_steps >= done.shape[0] or command_tolerance < 0:
        raise ValueError("JEPA horizon/tolerance가 올바르지 않다")
    masks = []
    for t in range(done.shape[0] - horizon_steps):
        same_episode = ~done[t:t + horizon_steps].any(dim=0)
        command_held = ((command[t:t + horizon_steps + 1] - command[t]).abs()
                        <= command_tolerance).all(dim=(0, 2))
        masks.append(same_episode & command_held)
    return torch.stack(masks)


def jepa_fresh_pair_mask(base_mask: torch.Tensor, fresh: torch.Tensor,
                         horizon_steps: int) -> torch.Tensor:
    """같은 에피소드·명령 mask 중 목표가 새 depth인 쌍만 남긴다.

    DepthFrame은 렌더마다 지연을 다시 뽑으므로 Δ가 depth 주기의 정수배여도
    시작과 목표가 동시에 fresh라는 보장은 없다. 시작 context는 hold 상태도 유효하다.
    """
    if fresh.ndim != 2 or base_mask.shape != (fresh.shape[0] - horizon_steps, fresh.shape[1]):
        raise ValueError("fresh/base_mask/horizon shape가 맞지 않는다")
    return base_mask & fresh[horizon_steps:]


def masked_future_mse(pred: torch.Tensor, target: torch.Tensor,
                      mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.ndim != 3 or mask.shape != pred.shape[:2]:
        raise ValueError("JEPA pred/target [T,N,D], mask [T,N]이어야 한다")
    per_sample = (pred - target.detach()).square().mean(dim=-1)
    return (per_sample * mask).sum() / mask.sum().clamp(min=1)


def normalized_jepa_with_copy(pred: torch.Tensor, source: torch.Tensor, target: torch.Tensor,
                              mask: torch.Tensor, variance_floor: float
                              ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """유효 target의 차원별 배치 분산으로 JEPA와 copy MSE를 함께 정규화한다."""
    if pred.shape != source.shape or pred.shape != target.shape or pred.ndim != 3 or mask.shape != pred.shape[:2]:
        raise ValueError("JEPA pred/source/target [T,N,D], mask [T,N]이어야 한다")
    if variance_floor <= 0:
        raise ValueError("variance_floor는 양수여야 한다")
    target = target.detach()
    if not mask.any():
        zero = pred.sum() * 0
        return zero, zero.detach(), zero.detach()
    target_valid = target[mask]
    variance = target_valid.var(dim=0, unbiased=False)
    scale = variance.clamp_min(variance_floor)
    loss = ((pred[mask] - target_valid).square().mean(dim=0) / scale).mean()
    copy_loss = ((source.detach()[mask] - target_valid).square().mean(dim=0) / scale).mean()
    return loss, copy_loss, variance.mean()


def distillation_losses(student_z: torch.Tensor, teacher_z: torch.Tensor,
                        student_action: torch.Tensor, teacher_action: torch.Tensor,
                        mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if student_action.shape != teacher_action.shape:
        raise ValueError("student/teacher 행동 shape가 달라서는 안 된다")
    latent = normalized_latent_mse(student_z, teacher_z, mean, std)
    action = (student_action - teacher_action.detach()).square().mean()
    return latent, action
