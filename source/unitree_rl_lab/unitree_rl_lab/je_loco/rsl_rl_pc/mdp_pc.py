# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""point cloud 관측 — RayCaster 격자 hit 위치를 robot base frame 으로 변환 → (N, P*3).

working 계단 실험 방식(raycaster → base-frame points). depth 카메라 대신 raycaster 를 써서
가볍고 빠르며(RTX 렌더 불필요) sim/real 정합(실기는 LiDAR/depth→ROI→base frame 동일 표현).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors.ray_caster.patterns.patterns_cfg import PatternBaseCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def raycaster_pointcloud(
    env: "ManagerBasedRLEnv",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("pc_scanner"),
) -> torch.Tensor:
    """RayCaster hit 점들을 base frame 으로 변환해 (N, P*3) 로 flatten."""
    sensor = env.scene.sensors[sensor_cfg.name]
    robot = env.scene["robot"]

    hits_w = sensor.data.ray_hits_w.clone().float()                # (N, P, 3) world
    valid = torch.isfinite(hits_w).all(dim=-1)                     # (N, P) 유효 hit
    # 무효 hit 은 센서 위치로 채워 transform 을 유한하게 유지(어차피 valid=0 이라 max-pool 제외됨).
    sensor_pos_w = sensor.data.pos_w.unsqueeze(1).expand_as(hits_w)
    hits_w = torch.where(valid.unsqueeze(-1), hits_w, sensor_pos_w)

    n_envs, num_points, _ = hits_w.shape
    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w
    hits_rel = hits_w - base_pos_w.unsqueeze(1)
    hits_b = quat_apply_inverse(
        base_quat_w.unsqueeze(1).expand(-1, num_points, -1).reshape(n_envs * num_points, 4),
        hits_rel.reshape(n_envs * num_points, 3),
    ).reshape(n_envs, num_points, 3)
    # 각 점에 valid 채널 부착 → (N, P, 4) = [x,y,z,valid]. 인코더가 valid=0 을 max-pool 에서 제외.
    # 학습 시 무효 hit 도 valid=0 → eval 결손(valid=0)과 의미 일치(train/eval consistent).
    out = torch.cat([hits_b, valid.float().unsqueeze(-1)], dim=-1)  # (N, P, 4)
    return out.reshape(n_envs, num_points * 4)


# ─────────────────────────────────────────────────────────────────────────────
# D435i 전방 프러스텀 ray 패턴 (top-down 격자 대체 — 실기 카메라 시야 정합)
# ─────────────────────────────────────────────────────────────────────────────
def tilt_quat_y(deg: float) -> tuple[float, float, float, float]:
    """로컬 +X(전방) 광축을 Y축 기준 아래로 deg 숙이는 쿼터니언 (w,x,y,z). 양수=하향."""
    h = math.radians(deg) * 0.5
    return (math.cos(h), 0.0, math.sin(h), 0.0)


def frustum_camera_pattern(cfg: "FrustumPatternCfg", device: str):
    """D435i 전방 프러스텀 ray 방향 — 광축 로컬 +X, HFoV/VFoV 로 spread.

    로봇 카메라 프레임(x 전방, y 좌, z 상) 기준 방향벡터 (H*W, 3) 반환. offset.rot(틸트)이
    이후 RayCaster 에서 적용되어 실제 하향 시야가 됨. ray_starts=0(센서 원점).
    """
    hh = math.radians(cfg.hfov_deg) * 0.5
    vh = math.radians(cfg.vfov_deg) * 0.5
    yaw = torch.linspace(hh, -hh, cfg.width, device=device)      # 좌(+y) → 우(-y)
    pitch = torch.linspace(vh, -vh, cfg.height, device=device)   # 상(+z) → 하(-z)
    pit, ya = torch.meshgrid(pitch, yaw, indexing="ij")
    pit, ya = pit.reshape(-1), ya.reshape(-1)
    x = torch.cos(pit) * torch.cos(ya)
    y = torch.cos(pit) * torch.sin(ya)
    z = torch.sin(pit)
    dirs = torch.stack([x, y, z], dim=-1)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    return torch.zeros_like(dirs), dirs


@configclass
class FrustumPatternCfg(PatternBaseCfg):
    """D435i 전방 프러스텀 ray 패턴 cfg (표준 grid 패턴 대체)."""

    func: Callable = frustum_camera_pattern
    hfov_deg: float = 78.7   # D435i intrinsics (fx=390 → HFoV 78.7°)
    vfov_deg: float = 63.1
    width: int = 16          # 가로 ray (다운샘플; 실기 64)
    height: int = 12         # 세로 ray (실기 48). 16×12=192점


# ─────────────────────────────────────────────────────────────────────────────
# depth 결손(degradation) — 게이트 2. Exp1(eval-time) · Exp2(train-time aug) 공용.
# base-frame 무효 hit sentinel = pc_scanner mount offset (raycaster_pointcloud 의 무효 처리와 동일).
#
# 세 결손은 "같은 양(level)을 제거하되 공간 구조가 다르다"로 통제 비교:
#   dropout    = i.i.d. 무작위 점 (구조 없음)           ← 센서 랜덤 노이즈/저조도
#   hole       = 블록 단위 무작위 제거 (중간 클러스터)   ← 반사·투명 표면 구멍
#   occlusion  = 하단 대역 통째 가림 (최대 구조)         ← 다리/장애물이 근거리 시야 차단
# frustum 격자 = pitch 12행(상→하) × yaw 16열(좌→우), row-major (frustum_camera_pattern 순서와 일치).
# ─────────────────────────────────────────────────────────────────────────────
_FRUSTUM_H = 12   # FrustumPatternCfg.height (pitch 행)
_FRUSTUM_W = 16   # FrustumPatternCfg.width  (yaw 열)

# 결손 = valid 채널(각 점의 4번째)만 0 으로 → 인코더 max-pool 에서 제외. 좌표는 안 건드림.
# (측정 ④ 2026-07-31: sentinel 좌표 주입은 max-pool hijack 아티팩트를 만듦. mount↔origin 반전으로
#  방법론 실패 확정. 진짜 마스킹 = 어떤 좌표도 주입 안 함.)


def _drop_mask(pc_flat: torch.Tensor, drop: torch.Tensor) -> torch.Tensor:
    """(N, P*4) point cloud 에서 drop(N,P, bool) 위치의 valid 채널을 0 으로. 좌표 불변."""
    n, d = pc_flat.shape
    p = d // 4
    pc = pc_flat.reshape(n, p, 4).clone()
    pc[..., 3] = pc[..., 3] * (~drop).to(pc.dtype)     # 제거 점 valid→0
    return pc.reshape(n, d)


def dropout_pointcloud(pc_flat: torch.Tensor, level: float) -> torch.Tensor:
    """(N, P*4) 각 점을 확률 level 로 제거(valid→0). 실기 i.i.d depth dropout 모사."""
    if level <= 0.0:
        return pc_flat
    n, d = pc_flat.shape
    p = d // 4
    drop = torch.rand(n, p, device=pc_flat.device) < level
    return _drop_mask(pc_flat, drop)


def hole_pointcloud(pc_flat: torch.Tensor, level: float, bh: int = 3, bw: int = 4) -> torch.Tensor:
    """블록(bh×bw) 단위로 확률 level 제거(valid→0) → 공간적으로 연속된 구멍. 반사/투명 표면 모사.

    dropout 의 클러스터판: bh×bw 블록을 통째로 Bernoulli(level) 제거. 기대 제거 비율은 dropout 과
    동일(level)하되 결손이 뭉쳐 있어 국소 정보 전체 손실. (12,16) 격자 bh=3,bw=4 → 4×4=16 블록.
    """
    if level <= 0.0:
        return pc_flat
    n, d = pc_flat.shape
    h, w = _FRUSTUM_H, _FRUSTUM_W
    assert d == h * w * 4, f"pc dim {d} != {h}*{w}*4 — frustum 격자 크기 불일치"
    hb, wb = h // bh, w // bw
    block_drop = torch.rand(n, hb, wb, device=pc_flat.device) < level        # (n, hb, wb)
    mask = block_drop.repeat_interleave(bh, dim=1).repeat_interleave(bw, dim=2)  # (n, h, w)
    return _drop_mask(pc_flat, mask.reshape(n, h * w))


def occlude_pointcloud(pc_flat: torch.Tensor, level: float) -> torch.Tensor:
    """frustum 하단 level 비율의 행(근거리 발밑 시야)을 통째로 가림(valid→0). 다리/장애물 근거리 차단.

    level=0.5 → 아래 절반 행 제거. 최대 구조적 결손(연속 대역). 근거리 지형 정보가 가장 먼저 사라짐.
    """
    if level <= 0.0:
        return pc_flat
    n, d = pc_flat.shape
    h, w = _FRUSTUM_H, _FRUSTUM_W
    assert d == h * w * 4, f"pc dim {d} != {h}*{w}*4 — frustum 격자 크기 불일치"
    k = int(round(level * h))                                          # 가릴 하단 행 수
    if k <= 0:
        return pc_flat
    mask = torch.zeros(n, h, w, dtype=torch.bool, device=pc_flat.device)
    mask[:, h - k:, :] = True
    return _drop_mask(pc_flat, mask.reshape(n, h * w))


# eval_pc.py 에서 --degradation 스위치로 선택
DEGRADATIONS = {
    "dropout": dropout_pointcloud,
    "hole": hole_pointcloud,
    "occlusion": occlude_pointcloud,
}
