"""D435i depth 노이즈 모델 (CLAUDE.md §6). 순수 torch. 입력·출력은 미터 단위, 결측은 0.

수치 기본값은 초기값이다 — config 로 노출하고 실기 통계(`compare_depth_stats.py`)로 맞춘다.
"""

from __future__ import annotations

import torch

from .depth_proc import apply_left_band


def depth_noise(
    depth: torch.Tensor,
    fx: float,
    baseline: float = 0.050,
    gauss_coef: float = 0.01,
    edge_rel_thresh: float = 0.2,
    clip_far: float = 2.0,
    edge_drop_prob: float = 0.5,
    hole_prob: float = 0.5,
    max_holes: int = 3,
    hole_size_px: tuple[int, int] = (2, 8),
    left_band: bool = True,
) -> torch.Tensor:
    """depth [N, H, W] → 노이즈 적용 depth.

    - 거리 제곱 비례 가우시안: σ = gauss_coef · d² (D435 RMS ≤ 2% @ 2 m → 0.01)
    - 경계 결측: clip_far 로 자른 depth 에서 이웃과의 상대 차 |Δd|/d > edge_rel_thresh 인 픽셀을 edge_drop_prob 로 0.
      절대 차를 쓰면 먼 지면이 원근 때문에 경계로 오인된다(평지 2 m 이내 상대 차 ≤ 약 0.1, 계단 수직면 약 0.3).
    - 랜덤 hole: 프레임마다 hole_prob 로 1..max_holes 개 사각형(변 hole_size_px)을 0
    - 왼쪽 결측 띠: baseline · fx / Z (fx 는 렌더링 해상도 기준)
    """
    N, H, W = depth.shape
    valid = depth > 0
    out = apply_left_band(depth, fx, baseline) if left_band else depth.clone()

    out = out + torch.randn_like(out) * gauss_coef * out.square()

    # 경계: 상하좌우 이웃과의 최대 상대 depth 차 (clip_far 로 자른 값 기준)
    dc = depth.clamp(max=clip_far)
    grad = torch.zeros_like(depth)
    dx = (dc[:, :, 1:] - dc[:, :, :-1]).abs() / dc[:, :, 1:].minimum(dc[:, :, :-1]).clamp(min=1e-3)
    dy = (dc[:, 1:, :] - dc[:, :-1, :]).abs() / dc[:, 1:, :].minimum(dc[:, :-1, :]).clamp(min=1e-3)
    grad[:, :, 1:] = torch.maximum(grad[:, :, 1:], dx)
    grad[:, :, :-1] = torch.maximum(grad[:, :, :-1], dx)
    grad[:, 1:, :] = torch.maximum(grad[:, 1:, :], dy)
    grad[:, :-1, :] = torch.maximum(grad[:, :-1, :], dy)
    edge_drop = (grad > edge_rel_thresh) & (torch.rand_like(depth) < edge_drop_prob)

    holes = torch.zeros_like(valid)
    if max_holes > 0 and hole_prob > 0:
        k = torch.randint(1, max_holes + 1, (N,), device=depth.device)
        k = torch.where(torch.rand(N, device=depth.device) < hole_prob, k, torch.zeros_like(k))
        rows = torch.arange(H, device=depth.device)
        cols = torch.arange(W, device=depth.device)
        for j in range(max_holes):
            on = k > j
            if not on.any():
                break
            sh = torch.randint(hole_size_px[0], hole_size_px[1] + 1, (N,), device=depth.device)
            sw = torch.randint(hole_size_px[0], hole_size_px[1] + 1, (N,), device=depth.device)
            r0 = (torch.rand(N, device=depth.device) * (H - sh + 1)).long()
            c0 = (torch.rand(N, device=depth.device) * (W - sw + 1)).long()
            in_r = (rows[None] >= r0[:, None]) & (rows[None] < (r0 + sh)[:, None])
            in_c = (cols[None] >= c0[:, None]) & (cols[None] < (c0 + sw)[:, None])
            holes |= on[:, None, None] & in_r[:, :, None] & in_c[:, None, :]

    keep = valid & ~edge_drop & ~holes & (out > 0)
    return torch.where(keep, out, torch.zeros_like(out))
