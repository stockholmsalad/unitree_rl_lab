"""Depth 전처리 — 2채널 (정규화 depth, validity mask) 과 실기체 해상도 변환. 순수 torch.

시뮬레이터 출력과 실기체 입력이 **같은 함수**를 거쳐 정책/Target 인코더로 들어간다.
"""

from __future__ import annotations

import math

import torch


def to_two_channel(depth: torch.Tensor, near: float = 0.28, far: float = 2.0) -> torch.Tensor:
    """(..., H, W) depth[m] → (..., 2, H, W).

    - [near, far]: 유효, (d - near) / (far - near) ∈ [0, 1]
    - > far: 1.0 으로 clamp, 유효
    - 0 / NaN / inf / < near: depth 0, mask 0
    """
    valid = torch.isfinite(depth) & (depth >= near)
    d = torch.where(valid, depth, torch.zeros_like(depth))
    norm = ((d.clamp(max=far) - near) / (far - near)) * valid
    return torch.stack([norm, valid.to(depth.dtype)], dim=-3)


def apply_left_band(depth: torch.Tensor, fx: float, baseline: float = 0.050) -> torch.Tensor:
    """D435 왼쪽 결측 띠: 열 u 가 baseline × fx / Z(u, v) 보다 왼쪽이면 결측(0).

    Z 는 그 픽셀 자신의 depth 를 쓴다(띠 경계 근처의 장면 거리 근사). 이미 결측인 픽셀은 그대로 0.
    """
    u = torch.arange(depth.shape[-1], device=depth.device, dtype=depth.dtype)
    safe = torch.where(depth > 0, depth, torch.full_like(depth, math.inf))
    band = u < baseline * fx / safe
    return torch.where(band, torch.zeros_like(depth), depth)


def _footprint_index(src: int, dst: int) -> tuple[torch.Tensor, torch.Tensor]:
    """출력 픽셀 i 가 덮는 원본 구간 [floor(i·s), floor((i+1)·s)) 의 인덱스 (dst, K) 와 구간 내 여부."""
    edges = [math.floor(i * src / dst) for i in range(dst + 1)]
    k = max(edges[i + 1] - edges[i] for i in range(dst))
    start = torch.tensor(edges[:-1])
    offs = torch.arange(k)
    idx = start[:, None] + offs[None, :]
    inside = idx < torch.tensor(edges[1:])[:, None]
    return idx.clamp(max=src - 1), inside


def downsample_real(
    depth: torch.Tensor,
    out_hw: tuple[int, int],
    mode: str = "nearest",
    near: float = 0.28,
    min_valid_frac: float = 0.5,
) -> torch.Tensor:
    """실기체 depth (..., H, W)[m] → (..., h, w)[m]. 결측(0)과 유효를 섞지 않는다. bilinear 금지.

    - ``nearest``: footprint 중심에 가장 가까운 **유효** 픽셀. footprint 전부 결측이면 0.
      시뮬레이터의 저해상도 직접 렌더링(픽셀 중심 점샘플)과 의미가 같다.
    - ``median``: footprint 유효 픽셀의 중앙값. 유효 비율 < ``min_valid_frac`` 이면 0.
    """
    H, W = depth.shape[-2:]
    h, w = out_hw
    ir, inr = _footprint_index(H, h)
    ic, inc = _footprint_index(W, w)
    ir, inr, ic, inc = (t.to(depth.device) for t in (ir, inr, ic, inc))
    # (..., h, Kr, w, Kc) → (..., h, w, Kr*Kc)
    patch = depth[..., ir[:, :, None, None], ic[None, None, :, :]].movedim(-3, -2).flatten(-2)
    inside = (inr[:, None, :, None] & inc[None, :, None, :]).flatten(-2)  # (h, w, K)
    valid = inside & torch.isfinite(patch) & (patch >= near)

    if mode == "nearest":
        # footprint 중심까지의 원본 픽셀 거리 (픽셀 중심 관례)
        cr = (torch.arange(h, device=depth.device) + 0.5) * H / h - 0.5
        cc = (torch.arange(w, device=depth.device) + 0.5) * W / w - 0.5
        dr = ir.to(depth.dtype) - cr[:, None]
        dc = ic.to(depth.dtype) - cc[:, None]
        dist = (dr[:, None, :, None] ** 2 + dc[None, :, None, :] ** 2).flatten(-2)  # (h, w, K)
        dist = torch.where(valid, dist, torch.full_like(patch, math.inf))
        j = dist.argmin(dim=-1, keepdim=True)
        out = patch.gather(-1, j).squeeze(-1)
        return torch.where(valid.any(-1), out, torch.zeros_like(out))
    if mode == "median":
        med = torch.where(valid, patch, torch.full_like(patch, math.nan)).nanmedian(dim=-1).values
        frac = valid.sum(-1) / inside.sum(-1)
        return torch.where(frac >= min_valid_frac, med, torch.zeros_like(med))
    raise ValueError(f"mode={mode!r} (nearest|median)")
