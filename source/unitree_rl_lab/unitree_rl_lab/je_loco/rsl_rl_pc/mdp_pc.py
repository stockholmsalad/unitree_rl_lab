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


# ─────────────────────────────────────────────────────────────────────────────
# 시간적 결손 (2026-08-20 추가) — 게이트 3
#
# 위 3종은 전부 **공간** 마스킹이고 매 프레임 독립이다. 그래서 공간 복원을 학습한 Head A 에
# 유리하게 짜여 있다. Head B(JEPA)가 학습한 건 **시간 구조**(z_e(t)→z_e(t+k))이므로,
# 원리상 우위가 나와야 하는 자리는 **관측이 시간적으로 끊기는** 고장이다.
#
#   freeze   = 확률적 프레임 드롭 (센서가 갱신 실패 → 직전 프레임 유지)
#   latency  = 관측이 d 스텝 지연 도착 (파이프라인 지연)
#   lowfps   = 결정론적 저프레임레이트 (D435i 30fps vs 제어 50Hz — 실기 상시 조건)
#
# 셋 다 level=0 에서 **정확히 항등**이라 게이트 1(대등성) 비교가 그대로 성립한다.
# 공간 결손과 달리 상태를 들고 있어야 하므로 함수가 아니라 객체다. eval 루프는
#   d.reset() → 매 스텝 d(pc, level) → env.step 후 d.notify_done(dones)
# 순서로 쓴다. notify_done 이 없으면 리셋된 env 가 **이전 에피소드 지형의 프레임**을 물고
# 있게 되어(순간이동) 결손이 아니라 시뮬 아티팩트를 측정하게 된다.
# ─────────────────────────────────────────────────────────────────────────────

LATENCY_MAX_STEPS = 10   # level 1.0 = 10 스텝 지연 (50Hz 기준 0.2s)
LOWFPS_MAX_STRIDE = 10   # level 1.0 = 10 스텝마다 1회 갱신 (50Hz → 5Hz)


class _StatelessDegradation:
    """공간 결손(프레임 독립) 래퍼 — eval 루프가 한 가지 인터페이스만 쓰게 한다."""

    temporal = False

    def __init__(self, fn):
        self._fn = fn

    def reset(self) -> None:
        pass

    def notify_done(self, dones) -> None:
        pass

    def __call__(self, pc_flat: torch.Tensor, level: float) -> torch.Tensor:
        return self._fn(pc_flat, level)


class _TemporalDegradation:
    """시간 결손 공통 뼈대: 프레임 버퍼 + 에피소드 경계 처리."""

    temporal = True

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._buf = None        # (N, D) 마지막으로 정책에 준 프레임
        self._force = None      # (N,) True = 다음 호출에서 무조건 최신 프레임 채택
        self._t = 0

    def _ensure(self, pc_flat: torch.Tensor) -> None:
        if self._buf is None or self._buf.shape != pc_flat.shape:
            self._buf = pc_flat.clone()
            self._force = torch.ones(pc_flat.shape[0], dtype=torch.bool, device=pc_flat.device)

    def notify_done(self, dones) -> None:
        """리셋된 env 는 이전 지형의 stale 프레임을 버리고 다음 관측을 즉시 채택."""
        if self._force is not None and dones is not None:
            self._force |= dones.reshape(-1).bool().to(self._force.device)

    def _commit(self, pc_flat: torch.Tensor, update: torch.Tensor) -> torch.Tensor:
        upd = update | self._force
        self._buf[upd] = pc_flat[upd]
        self._force = torch.zeros_like(self._force)
        self._t += 1
        return self._buf.clone()


class FreezeDegradation(_TemporalDegradation):
    """확률적 프레임 드롭: 매 스텝 확률 `level` 로 갱신 실패 → 직전 프레임 유지.

    level=0 항등(항상 갱신) · level=1.0 완전 정지(에피소드 첫 프레임에 고정).
    실기의 USB 대역폭 부족·간헐 드롭 모사. env 마다 독립.
    """

    def __call__(self, pc_flat: torch.Tensor, level: float) -> torch.Tensor:
        self._ensure(pc_flat)
        if level <= 0.0:
            return self._commit(pc_flat, torch.ones_like(self._force))
        upd = torch.rand(pc_flat.shape[0], device=pc_flat.device) >= level
        return self._commit(pc_flat, upd)


class LowFpsDegradation(_TemporalDegradation):
    """결정론적 저프레임레이트: stride 스텝마다 1회만 갱신.

    stride = 1 + round(level × (LOWFPS_MAX_STRIDE − 1)) → level=0 은 50Hz(항등),
    level=1.0 은 5Hz. **실기 D435i 30fps 는 제어 50Hz 대비 stride≈1.67 로 이미 이 축 위에 있다.**
    """

    def __call__(self, pc_flat: torch.Tensor, level: float) -> torch.Tensor:
        self._ensure(pc_flat)
        stride = 1 + int(round(max(0.0, level) * (LOWFPS_MAX_STRIDE - 1)))
        hit = (self._t % stride) == 0
        upd = torch.full_like(self._force, bool(hit))
        return self._commit(pc_flat, upd)


class LatencyDegradation(_TemporalDegradation):
    """관측 지연: d = round(level × LATENCY_MAX_STEPS) 스텝 전 프레임을 정책에 준다.

    level=0 항등. 에피소드 시작 직후(이력이 d 스텝에 못 미침)에는 지연시킬 과거가 없으므로
    현재 프레임을 준다 — 없는 데이터를 지어내지 않는다.
    """

    def reset(self) -> None:
        super().reset()
        self._hist = None      # (LATENCY_MAX_STEPS+1, N, D) 링버퍼
        self._age = None       # (N,) 리셋 이후 경과 스텝

    def __call__(self, pc_flat: torch.Tensor, level: float) -> torch.Tensor:
        n, d_dim = pc_flat.shape
        cap = LATENCY_MAX_STEPS + 1
        if self._hist is None or self._hist.shape[1:] != pc_flat.shape:
            self._hist = pc_flat.unsqueeze(0).repeat(cap, 1, 1)
            self._age = torch.zeros(n, dtype=torch.long, device=pc_flat.device)
            self._force = torch.zeros(n, dtype=torch.bool, device=pc_flat.device)
            self._t = 0
        # 에피소드 리셋 env: 이력 폐기(현재 프레임으로 채움) + 나이 0
        if self._force.any():
            self._hist[:, self._force] = pc_flat[self._force].unsqueeze(0)
            self._age[self._force] = 0
            self._force = torch.zeros_like(self._force)

        self._hist[self._t % cap] = pc_flat
        delay = int(round(max(0.0, level) * LATENCY_MAX_STEPS))
        self._t += 1
        self._age += 1
        if delay <= 0:
            return pc_flat
        past = self._hist[(self._t - 1 - delay) % cap]          # d 스텝 전 프레임
        ready = (self._age > delay).unsqueeze(-1)                # 이력이 충분한 env 만 지연 적용
        return torch.where(ready, past, pc_flat)


# ─────────────────────────────────────────────────────────────────────────────
# blind 대조군 (2026-08-31 추가) — "시각 입력이 도대체 얼마나 기여하는가"
#
# 동기: n=5 판정에서 freeze 1.0(카메라 영구 정지)인데도 레벨 5 지형 성공률 91% 가 나왔다.
# 반면 occlusion 0.4 는 32~51% 로 정책을 부순다. 이 둘이 동시에 참이려면 설명은 하나다 —
# occlusion 은 "공간 정보 상실"이 아니라 **분포 밖 입력**(발밑 행 전체 valid=0, 학습 중 없던
# 패턴; 극단적으로 dropout 1.0 은 유효점 0 → glob=0 상수 → 성공률 0.02%)을 만들고 있고,
# 정책 자체는 신선한 시각에 거의 의존하지 않는다.
#
# 그 가설을 직접 재려면 **분포 안에 있으면서 지형 정보만 없는** 입력이 필요하다.
# 그래서 마스킹(valid→0)을 일절 쓰지 않고 좌표를 **환경 평균 점군**으로 갈아끼운다:
#   · valid=1 유지 → max-pool 퇴화(glob=0) 아티팩트 원천 차단
#   · 좌표 스케일·격자 구조는 실제 관측 분포 그대로 → OOD 아님
#   · 자기 env 지형과의 상호정보량 ≈ 0 → 순수 "눈 감은" 조건
#
# 판정: blind 1.0 성공률이 무결손과 비슷하면 인코더 초기화 비교축 자체가 검정력이 없다
# (이 프로젝트의 모든 무효 결과를 한 번에 설명한다). 크게 떨어지면 시각은 중요하고
# 게이트 3 의 무효는 시간 결손 설계 문제로 좁혀진다.
# ─────────────────────────────────────────────────────────────────────────────


class BlindDegradation:
    """좌표를 지형 무관 기준 점군으로 치환. level = 치환할 점의 비율(0 항등, 1.0 완전 blind).

    기준 점군은 **첫 호출에서 유효점만의 env 평균**으로 한 번 계산하고 이후 고정한다
    (레벨 간 비교 가능성 확보 — reset 이 와도 유지).
    """

    temporal = False

    def __init__(self) -> None:
        self._ref = None       # (P, 3) 기준 좌표
        self.reset()

    def reset(self) -> None:
        pass                   # _ref 는 의도적으로 보존한다

    def notify_done(self, dones) -> None:
        pass

    def __call__(self, pc_flat: torch.Tensor, level: float) -> torch.Tensor:
        n, d = pc_flat.shape
        p = d // 4
        pc = pc_flat.reshape(n, p, 4)
        if self._ref is None:
            v = (pc[..., 3:4] > 0.5).to(pc.dtype)                   # (n, p, 1)
            self._ref = (pc[..., :3] * v).sum(0) / v.sum(0).clamp(min=1.0)   # (p, 3)
        if level <= 0.0:
            return pc_flat
        swap = torch.rand(n, p, device=pc.device) < level           # (n, p)
        out = pc.clone()
        out[..., :3] = torch.where(swap.unsqueeze(-1), self._ref.expand(n, p, 3), out[..., :3])
        out[..., 3] = torch.where(swap, torch.ones_like(out[..., 3]), out[..., 3])
        return out.reshape(n, d)


TRAIN_STALENESS_MAX = 25   # 학습 중 주입 지연 상한 (50Hz 기준 0.5s)


class TrainStaleness:
    """**학습 중** 관측 노후화 주입 — env 마다 지연 d_i 를 뽑아 에피소드 내내 고정.

    왜 필요한가 (2026-08-31): 관측이 항상 신선하면 ẑ_e(t+k) 는 z_e(t) 에서 사소하게 유도되는
    값이라 정책이 그냥 무시한다. 예측기가 **실제 할 일**을 가지려면 관측이 낡아야 한다.
    근거는 측정값이다 — 어려운 지형에서 freeze 1.0 은 −35pp 로, 지형 정보를 통째로 지운
    blind(−26pp)보다 9pp 더 해롭다. 노후화는 이 과제에서 가장 비싼 고장이다.

    **지연은 에피소드 내에서 상수다.** 그래야 저장된 관측열이 진짜 관측열의 시간 이동본이 되어
    (t, t+k) 쌍이 실제로 k 스텝 떨어진 채 남는다 — JEPA 보조손실의 의미가 보존된다.
    지연이 매 스텝 흔들리면 쌍 간격이 뭉개져 타깃이 오염된다.
    """

    def __init__(self, d_max: int = TRAIN_STALENESS_MAX) -> None:
        self.d_max = int(d_max)
        self.reset()

    def reset(self) -> None:
        self._hist = None      # (d_max+1, N, D) 링버퍼
        self._delay = None     # (N,) env 별 지연(에피소드 상수)
        self._age = None       # (N,) 리셋 이후 경과
        self._force = None     # (N,) True = 리셋됨 → 이력 폐기 + 지연 재추첨
        self._t = 0

    def notify_done(self, dones) -> None:
        if self._force is not None and dones is not None:
            self._force |= dones.reshape(-1).bool().to(self._force.device)

    def __call__(self, pc_flat: torch.Tensor) -> torch.Tensor:
        n = pc_flat.shape[0]
        cap = self.d_max + 1
        dev = pc_flat.device
        if self._hist is None or self._hist.shape[1:] != pc_flat.shape:
            self._hist = pc_flat.unsqueeze(0).repeat(cap, 1, 1)
            self._delay = torch.randint(0, self.d_max + 1, (n,), device=dev)
            self._age = torch.zeros(n, dtype=torch.long, device=dev)
            self._force = torch.zeros(n, dtype=torch.bool, device=dev)
            self._t = 0
        if self._force.any():
            f = self._force
            self._hist[:, f] = pc_flat[f].unsqueeze(0)     # 이전 지형 프레임 폐기
            self._age[f] = 0
            self._delay[f] = torch.randint(0, self.d_max + 1, (int(f.sum()),), device=dev)
            self._force = torch.zeros_like(self._force)

        self._hist[self._t % cap] = pc_flat
        self._t += 1
        self._age += 1
        idx = (self._t - 1 - self._delay) % cap                       # (N,) env 별 과거 슬롯
        past = self._hist[idx, torch.arange(n, device=dev)]           # (N, D)
        ready = (self._age > self._delay).unsqueeze(-1)               # 이력이 부족하면 현재 프레임
        return torch.where(ready, past, pc_flat)


TEMPORAL_DEGRADATIONS = {
    "freeze": FreezeDegradation,
    "latency": LatencyDegradation,
    "lowfps": LowFpsDegradation,
}

STATEFUL_DEGRADATIONS = {"blind": BlindDegradation}

ALL_DEGRADATIONS = (tuple(DEGRADATIONS) + tuple(TEMPORAL_DEGRADATIONS)
                    + tuple(STATEFUL_DEGRADATIONS))


def make_degradation(name: str):
    """이름 → 결손 객체(공간·시간 공통 인터페이스: reset / __call__ / notify_done)."""
    if name in TEMPORAL_DEGRADATIONS:
        return TEMPORAL_DEGRADATIONS[name]()
    if name in STATEFUL_DEGRADATIONS:
        return STATEFUL_DEGRADATIONS[name]()
    if name in DEGRADATIONS:
        return _StatelessDegradation(DEGRADATIONS[name])
    raise KeyError(f"알 수 없는 결손 '{name}' — 가능: {list(ALL_DEGRADATIONS)}")
