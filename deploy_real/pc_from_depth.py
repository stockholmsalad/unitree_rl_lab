# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""D435i 깊이 이미지 → 학습 때와 **같은** 192점 점군 (base frame, [x,y,z,valid]).

시뮬레이터는 깊이 카메라를 렌더하지 않는다. `pc_scanner` 는 RayCaster 이고, 정책이
본 것은 16(yaw)×12(pitch) 각도 격자를 따라 쏜 광선의 지면 교점이다. 따라서 실기에서
할 일은 "깊이 영상을 점군으로 바꾸기"가 아니라 **그 192개 광선 방향의 거리만 읽어내기**다.
640×480 을 통째로 다운샘플하면 격자가 어긋나고, 그러면 occlusion 증강이 가정한
행/열 구조(하단 대역 = 발밑)도 같이 어긋난다.

좌표계가 셋이다. 헷갈리면 여기서 다 틀어지므로 명시한다.
  패턴 프레임  x 전방, y 좌, z 상  — frustum_camera_pattern 이 방향을 내는 곳.
                                     카메라가 물리적으로 35° 숙여 달려 있으므로
                                     이 프레임은 **카메라 자신의 프레임**과 같다.
  광학 프레임  z 전방, x 우, y 하  — RealSense 가 깊이를 주는 관례. 투영은 여기서 한다.
  base 프레임  x 전방, y 좌, z 상  — 정책이 받는 프레임. 마운트 회전·평행이동으로 간다.

상태추정이 들어가지 않는다. base_pos_w/base_quat_w 가 필요한 건 시뮬레이터가 world 를
경유하기 때문이고, 실기에서는 카메라 → 고정 외부파라미터 → base 로 끝난다. IMU 도
SLAM 도 이 경로에 없다 → 사족보행 진동으로 지도가 흐르는 문제가 원천적으로 없다.

  # ★ Z790 (자기검증만 — 로봇 없이 돈다)
  python deploy_real/pc_from_depth.py --selftest
"""

from __future__ import annotations

import math

import numpy as np

# ── 시뮬레이터와 일치해야 하는 상수 ─────────────────────────────────────────
# 출처: je_loco/rsl_rl_pc/env_cfg.py 의 PCSceneCfg.pc_scanner 와
#       je_loco/rsl_rl_pc/mdp_pc.py 의 FrustumPatternCfg.
# 하나라도 어긋나면 정책은 학습 때와 다른 지형을 보게 된다. check_against_sim() 참조.
# 2026-09-11 실측(D435i 깊이 640×480). 시뮬 PinholePatternCfg 와 같은 값이어야 한다.
FX = FY = 390.330
CX, CY = 316.090, 239.581
IMG_W, IMG_H = 640, 480
# 구(등각) 격자용. PINHOLE=False 로 되돌릴 때만 쓰인다.
HFOV_DEG = 78.7
VFOV_DEG = 63.1
# True  = 영상 평면 균등 타일링(PinholePatternCfg, V7~). 광선 낭비 0.
# False = 각도 등간격(FrustumPatternCfg, V2~V6). 실측상 192 중 42 개가 영상 밖.
PINHOLE = True
WIDTH = 16              # yaw 열 (좌→우)
HEIGHT = 12             # pitch 행 (상→하)
MOUNT_POS = (0.325, 0.0, 0.045)   # base 원점 기준 카메라 위치 [m]
MOUNT_TILT_DEG = 35.0             # 하향 틸트 (+ = 아래)
MAX_DISTANCE = 2.0                # 이 너머는 무효(홀). D435i 스펙 3m 가 아니다.
COORD_CLIP = 2.0                  # 관측 항의 clip=(-2, 2)
MIN_DEPTH = 0.28                  # D435i 최소 측정 거리 — 그 아래는 쓰레기값

NUM_POINTS = WIDTH * HEIGHT


def pattern_directions() -> np.ndarray:
    """시뮬 ray 패턴과 **같은 순서**의 단위 방향 (192, 3), 패턴 프레임.

    행 = 상→하, 열 = 좌→우, row-major. 이 순서가 곧 occlusion/hole 증강이 쓰는
    (12, 16) 격자다. PINHOLE 이 시뮬 cfg 와 어긋나면 정책이 학습 때와 다른 점을 받는다.
    """
    if PINHOLE:
        u = (np.arange(WIDTH) + 0.5) * (IMG_W / WIDTH)
        v = (np.arange(HEIGHT) + 0.5) * (IMG_H / HEIGHT)
        vv, uu = np.meshgrid(v, u, indexing="ij")
        x_opt = (uu.reshape(-1) - CX) / FX
        y_opt = (vv.reshape(-1) - CY) / FY
        d = np.stack([np.ones_like(x_opt), -x_opt, -y_opt], axis=-1)
        return d / np.linalg.norm(d, axis=-1, keepdims=True)
    hh = math.radians(HFOV_DEG) * 0.5
    vh = math.radians(VFOV_DEG) * 0.5
    yaw = np.linspace(hh, -hh, WIDTH)        # 좌(+y) → 우(-y)
    pitch = np.linspace(vh, -vh, HEIGHT)     # 상(+z) → 하(-z)
    pit, ya = np.meshgrid(pitch, yaw, indexing="ij")
    pit, ya = pit.reshape(-1), ya.reshape(-1)
    d = np.stack([np.cos(pit) * np.cos(ya),
                  np.cos(pit) * np.sin(ya),
                  np.sin(pit)], axis=-1)
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def tilt_matrix(deg: float = MOUNT_TILT_DEG) -> np.ndarray:
    """패턴 프레임 → base 프레임 회전. +X 광축을 Y축 기준으로 deg 만큼 숙인다.

    mdp_pc.tilt_quat_y 와 같은 회전(양수 = 하향)을 행렬로 쓴 것. 부호를 틀리면
    지형이 통째로 위아래로 뒤집혀 들어오고, 정책은 멀쩡히 걷는 척하다 넘어진다.
    """
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


class DepthToPointCloud:
    """깊이 영상 → (768,) float32 관측. 매 제어 스텝 한 번 호출한다."""

    def __init__(self, fx: float, fy: float, cx: float, cy: float,
                 width: int, height: int, depth_scale: float = 0.001) -> None:
        """fx..cy 는 D435i 깊이 스트림의 내부 파라미터(rs.intrinsics 에서 그대로).

        depth_scale = 깊이 정수 1 단위의 미터값. D435i 기본 0.001.
        """
        self.img_w, self.img_h = width, height
        self.depth_scale = depth_scale
        self.dirs = pattern_directions()                    # (192, 3) 패턴 프레임
        self.R = tilt_matrix()
        self.t = np.asarray(MOUNT_POS, dtype=np.float64)

        # 각 광선을 광학 프레임으로 옮겨 한 번만 투영해 둔다 — 카메라가 강체 장착이라
        # 픽셀 좌표는 매 프레임 같다. 루프에서는 인덱싱만 한다.
        x_opt = -self.dirs[:, 1]
        y_opt = -self.dirs[:, 2]
        z_opt = self.dirs[:, 0]
        u = fx * (x_opt / z_opt) + cx
        v = fy * (y_opt / z_opt) + cy
        self.cos_axis = z_opt                # 광축과 이루는 각의 cos — 거리 환산에 쓴다
        self.u = np.rint(u).astype(np.int32)
        self.v = np.rint(v).astype(np.int32)
        # FoV 가 내부 파라미터와 조금 안 맞으면 가장자리 광선이 영상 밖으로 나간다.
        # 그건 영구히 무효인 점이므로 여기서 한 번 걸러 두고, 개수를 보고한다.
        self.in_image = ((self.u >= 0) & (self.u < width)
                         & (self.v >= 0) & (self.v < height))
        self.u = np.clip(self.u, 0, width - 1)
        self.v = np.clip(self.v, 0, height - 1)

    @property
    def num_outside(self) -> int:
        """영상 밖으로 나가 항상 무효인 광선 수. 0 이 아니면 마운트/내부파라미터를 의심한다."""
        return int((~self.in_image).sum())

    def __call__(self, depth: np.ndarray) -> np.ndarray:
        """depth (H, W) uint16 또는 float32 → (768,) float32 = 192 × [x, y, z, valid]."""
        raw = depth[self.v, self.u]
        z_cam = raw.astype(np.float64) * (self.depth_scale if raw.dtype != np.float32 else 1.0)

        # 광학 z(깊이)를 광선 방향 거리로 환산. 가장자리 광선일수록 cos 가 작아
        # 거리가 깊이보다 길다. 이걸 빼먹으면 시야 가장자리 지형이 앞으로 당겨진다.
        rng = z_cam / self.cos_axis

        valid = (self.in_image & np.isfinite(rng)
                 & (z_cam > MIN_DEPTH) & (rng <= MAX_DISTANCE))
        # 무효 거리(inf/NaN)를 여기서 0 으로 눌러 둔다. 그냥 두면 아래 행렬곱이 매
        # 프레임 RuntimeWarning 을 뱉는다 — 50 Hz 루프에서는 로그가 못 쓰게 된다.
        rng = np.where(valid, rng, 0.0)

        pts_cam = self.dirs * rng[:, None]                       # 패턴(=카메라) 프레임
        pts_base = pts_cam @ self.R.T + self.t                   # base 프레임
        pts_base = np.clip(pts_base, -COORD_CLIP, COORD_CLIP)
        # 무효 점의 좌표는 어디로 가든 max-pool 에서 제외되지만, NaN/inf 가 그래프에
        # 들어가면 ONNX 가 전체를 오염시킨다. 시뮬레이터가 무효 hit 을 센서 위치로
        # 채우는 것과 같은 이유로 여기서도 유한한 값으로 눌러 둔다.
        pts_base = np.where(valid[:, None], pts_base, self.t)

        out = np.concatenate([pts_base, valid[:, None].astype(np.float64)], axis=-1)
        return out.reshape(-1).astype(np.float32)


def check_against_sim() -> None:
    """이 파일의 상수가 시뮬레이터 cfg 와 같은지 대조한다. Isaac 이 있는 기계에서만 돈다.

    상수 드리프트는 조용히 일어나고 실기에서만 드러난다 — 그때는 원인을 못 찾는다.
    """
    from unitree_rl_lab.je_loco.rsl_rl_pc.env_cfg import PCSceneCfg

    cfg = PCSceneCfg().pc_scanner
    pat = cfg.pattern_cfg
    sim_pinhole = hasattr(pat, "fx")
    if sim_pinhole != PINHOLE:
        raise SystemExit(
            f"격자 모델 불일치: deploy PINHOLE={PINHOLE}, sim={'pinhole' if sim_pinhole else 'frustum'}")
    bad = []
    pairs = ([("fx", FX, getattr(pat, "fx", None)), ("fy", FY, getattr(pat, "fy", None)),
              ("cx", CX, getattr(pat, "cx", None)), ("cy", CY, getattr(pat, "cy", None))]
             if PINHOLE else
             [("hfov", HFOV_DEG, pat.hfov_deg), ("vfov", VFOV_DEG, pat.vfov_deg)])
    for name, ours, theirs in pairs + [
        ("width", WIDTH, pat.width), ("height", HEIGHT, pat.height),
        ("max_distance", MAX_DISTANCE, cfg.max_distance),
        ("mount_pos", MOUNT_POS, tuple(cfg.offset.pos)),
    ]:
        if ours != theirs:
            bad.append(f"  {name}: deploy={ours} sim={theirs}")
    # 틸트는 cfg 에 쿼터니언으로 들어 있어 각도로 되돌려 본다.
    w = cfg.offset.rot[0]
    sim_tilt = math.degrees(2.0 * math.acos(min(1.0, abs(w))))
    if abs(sim_tilt - MOUNT_TILT_DEG) > 1e-6:
        bad.append(f"  tilt: deploy={MOUNT_TILT_DEG} sim={sim_tilt:.6f}")

    if bad:
        raise SystemExit("시뮬레이터 cfg 와 어긋난다:\n" + "\n".join(bad))
    print("시뮬레이터 cfg 와 모든 상수 일치")

    # 방향 순서까지 대조 — 상수가 같아도 순서가 다르면 격자 구조가 어긋난다.
    import torch

    from unitree_rl_lab.je_loco.rsl_rl_pc.mdp_pc import frustum_camera_pattern
    _, sim_dirs = frustum_camera_pattern(pat, "cpu")
    err = float(torch.abs(sim_dirs - torch.as_tensor(pattern_directions(),
                                                     dtype=sim_dirs.dtype)).max())
    print(f"광선 방향 최대 오차 = {err:.2e}" + ("" if err < 1e-6 else "  ← 순서 불일치!"))


def _selftest() -> None:
    """평지를 보는 합성 깊이 영상을 만들어 되돌아오는 z 가 전부 같은지 본다.

    로봇 없이 전체 사슬(투영 → 역투영 → 틸트 → 평행이동)을 검증한다. 평지 위에
    서 있으면 base 프레임에서 모든 지면 점의 z 는 −(base 높이)로 **똑같아야** 한다.
    한 점이라도 다르면 좌표계 어딘가가 틀린 것이다.
    """
    fx = fy = 390.0
    w, h = 640, 480
    cx, cy = w / 2.0, h / 2.0
    base_height = 0.32                       # Go2 기립 시 base 높이 [m]

    conv = DepthToPointCloud(fx, fy, cx, cy, w, h, depth_scale=1.0)
    print(f"영상 밖으로 나간 광선: {conv.num_outside} / {NUM_POINTS}")

    # 지면 평면을 base 프레임에서 z = −base_height 로 두고, 각 광선의 교점까지의
    # 광학 깊이를 역으로 계산해 깊이 영상에 심는다.
    dirs_base = conv.dirs @ conv.R.T
    denom = dirs_base[:, 2]
    t_ray = np.where(denom < -1e-9, (-base_height - conv.t[2]) / denom, np.inf)
    z_cam = t_ray * conv.cos_axis            # 광선거리 → 광학 깊이

    depth = np.full((h, w), np.inf, dtype=np.float32)
    depth[conv.v, conv.u] = z_cam.astype(np.float32)

    obs = conv(depth).reshape(NUM_POINTS, 4)
    valid = obs[:, 3] > 0.5
    zs = obs[valid, 2]
    print(f"유효 점 {valid.sum()} / {NUM_POINTS}"
          f"   (나머지는 하늘을 보거나 {MAX_DISTANCE} m 너머)")
    print(f"지면 z: 평균 {zs.mean():+.6f}  최대편차 {np.abs(zs + base_height).max():.2e}"
          f"   (기대값 {-base_height:+.3f})")
    xs = obs[valid, 0]
    print(f"지면 x 범위: [{xs.min():.3f}, {xs.max():.3f}] m"
          f"   (논문에 적은 관측 대역과 대조할 것)")
    assert np.abs(zs + base_height).max() < 1e-6, "좌표계 사슬이 틀렸다"
    print("자기검증 통과")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="로봇 없이 좌표계 검증")
    ap.add_argument("--check_sim", action="store_true", help="시뮬레이터 cfg 와 상수 대조")
    args = ap.parse_args()
    if args.check_sim:
        check_against_sim()
    if args.selftest or not args.check_sim:
        _selftest()
