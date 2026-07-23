# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""JE-Loco **TiledCamera** depth 카메라 검증 — 실제 학습에 쓰는 그 카메라의 장착각·FoV·풋프린트 확인.

기존 verify_camera.py 는 ICROS Tier 의 RayCasterCamera 용이다. 이 스크립트는 JE-Loco 가 실제로
학습 입력으로 쓰는 **TiledCamera(RTX 렌더 depth)** 를 검증한다. 헤드리스이므로 GUI 대신:
  - depth/RGB 이미지를 PNG 로 저장 (로봇이 실제로 보는 화면 — 실기 D435i 와 눈으로 대조)
  - 카메라 intrinsics 로부터 HFoV/VFoV 계산 → 실기 D435i 스펙과 비교
  - depth 를 월드로 역투영(unproject) → base 기준 지면 풋프린트(전방 근/원, 좌우 폭) 정량 리포트

사용:
  python scripts/je_loco/verify_depth_camera.py --flat --enable_cameras --headless
  python scripts/je_loco/verify_depth_camera.py --tilt 30 --enable_cameras --headless   # 틸트 즉석 변경
  python scripts/je_loco/verify_depth_camera.py --enable_cameras --headless             # 험지(장애물)
결과: logs/je_loco/camera_check/ 에 depth_*.png, rgb_*.png, footprint_*.png + 리포트.
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="JE-Loco TiledCamera depth 검증")
parser.add_argument("--num_envs", type=int, default=1,
                    help="검증은 1마리 권장(디버그 드로우는 env0만 그림 → 여러 마리면 depth 가 딴 로봇서 나오는 것처럼 보임).")
parser.add_argument("--tilt", type=float, default=None, help="하향 틸트(°) 즉석 override. 미지정 시 cfg(35°).")
parser.add_argument("--cam_x", type=float, default=None, help="base 기준 카메라 전방[m] (기본 0.325).")
parser.add_argument("--cam_y", type=float, default=None, help="base 기준 카메라 좌우[m] (기본 0.0).")
parser.add_argument("--cam_z", type=float, default=None, help="base 기준 카메라 상하[m] (기본 0.045).")
parser.add_argument("--flat", action="store_true", help="평지(plane)에서 검증(FoV/풋프린트 명확).")
parser.add_argument("--hold_z", type=float, default=0.34, help="고정 base 높이[m] (standing).")
parser.add_argument("--settle", type=int, default=40, help="렌더 안정화 스텝 수.")
parser.add_argument("--stairs", type=float, default=None, metavar="STEP_H",
                    help="전방에 오르막 계단(단높이[m], 예 0.12)을 두고 정지 스폰 → depth 로 계단 확인.")
parser.add_argument("--out", type=str, default="logs/je_loco/camera_check")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True  # depth 렌더 필수

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱 기동 후 ----
import os

import numpy as np
import torch
import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isaaclab.utils.math import (
    convert_camera_frame_orientation_convention,
    quat_apply,
    quat_mul,
    transform_points,
    unproject_depth,
)

import unitree_rl_lab.je_loco.envs  # noqa: F401
from unitree_rl_lab.je_loco.envs.je_loco_env_cfg import (
    D435I_FOCAL, D435I_H_APERTURE, D435I_V_APERTURE, D435I_MAX_DIST, D435I_POS, D435I_TILT_DEG,
    JELocoEnvCfg, _tilt_quat,
)


def fov_from_aperture(aperture, focal):
    return math.degrees(2.0 * math.atan(aperture / (2.0 * focal)))


def live_cam_pose(robot, mount_pos, tilt_deg, device):
    """로봇 base 의 **현재** pose + 장착 offset 으로 카메라 월드 pose(ros optical) 계산.

    TiledCamera 의 cam.data.pos_w 는 스폰 pose 에 고정(stale)되어 로봇을 안 따라오므로,
    live base pose(root_pos_w/root_quat_w)로부터 직접 계산한다. mount_pos 는 base-local 오프셋,
    tilt 는 world-convention(+X 전방,+Z 상) 하향 회전.
    """
    base_pos = robot.data.root_pos_w[0:1]              # (1,3)
    base_quat = robot.data.root_quat_w[0:1]            # (1,4) wxyz
    mount = torch.tensor([list(mount_pos)], device=device, dtype=base_pos.dtype)
    tilt_q = torch.tensor([_tilt_quat(tilt_deg)], device=device, dtype=base_quat.dtype)
    cam_pos = base_pos + quat_apply(base_quat, mount)                       # (1,3)
    cam_world_conv = quat_mul(base_quat, tilt_q)                            # world-conv 카메라 방향
    cam_quat_ros = convert_camera_frame_orientation_convention(
        cam_world_conv, origin="world", target="ros")
    return cam_pos[0], cam_quat_ros[0]


def main():
    os.makedirs(args_cli.out, exist_ok=True)
    tilt = args_cli.tilt if args_cli.tilt is not None else D435I_TILT_DEG

    cfg = JELocoEnvCfg(enable_depth=True)
    cfg.scene.num_envs = args_cli.num_envs
    # RGB 도 함께 렌더 (장면 맥락 확인용)
    cfg.scene.depth_camera.data_types = ["distance_to_image_plane", "rgb"]
    cfg.scene.depth_camera.update_period = 0.0  # 매 스텝 렌더
    if args_cli.tilt is not None:
        cfg.scene.depth_camera.offset.rot = _tilt_quat(tilt)
    # base 기준 카메라 장착 위치 override (요청 시). 기본 = D435I_POS 실측값.
    cam_pos_off = (
        args_cli.cam_x if args_cli.cam_x is not None else D435I_POS[0],
        args_cli.cam_y if args_cli.cam_y is not None else D435I_POS[1],
        args_cli.cam_z if args_cli.cam_z is not None else D435I_POS[2],
    )
    cfg.scene.depth_camera.offset.pos = cam_pos_off
    if args_cli.flat:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.curriculum.terrain_levels = None
        for s in (cfg.scene.depth_camera, cfg.scene.heightmap_gt, cfg.scene.height_scanner):
            if s is not None:
                s.mesh_prim_paths = ["/World/ground"]
    elif args_cli.stairs is not None:
        # 전방 오르막 계단 단일 타일(중앙 평판 스폰 → 정면 +x 에 계단). 정지 검증용.
        import isaaclab.terrains as terrain_gen
        from isaaclab.terrains import TerrainGeneratorCfg
        h = args_cli.stairs
        cfg.scene.terrain.terrain_type = "generator"
        cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
            size=(8.0, 8.0), border_width=20.0, num_rows=1, num_cols=1,
            horizontal_scale=0.05, vertical_scale=0.005, slope_threshold=0.75,
            difficulty_range=(1.0, 1.0), use_cache=False, curriculum=False,
            # platform_width 넓게(2.6m) → 로봇이 중앙 평지에 서서 ~1.3m 앞부터 오르막 계단을 봄.
            sub_terrains={"stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=1.0, step_height_range=(h, h), step_width=0.30,
                platform_width=2.6, border_width=1.0, holes=False)},
        )
        cfg.scene.terrain.max_init_terrain_level = 0
        cfg.curriculum.terrain_levels = None
        for s in (cfg.scene.depth_camera, cfg.scene.heightmap_gt, cfg.scene.height_scanner):
            if s is not None:
                s.mesh_prim_paths = ["/World/ground"]

    # 스폰 결정화: 정면(+x) 중앙 스폰, 외란·노이즈 off (검증용).
    cfg.observations.policy.enable_corruption = False
    if getattr(cfg.events, "push_robot", None) is not None:
        cfg.events.push_robot = None
    if getattr(cfg.events, "reset_base", None) is not None:
        cfg.events.reset_base.params["pose_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0), "yaw": (0.0, 0.0)}

    env = gym.make("Unitree-Go2-JELoco", cfg=cfg)
    u = env.unwrapped
    env.reset()
    robot = u.scene["robot"]
    cam = u.scene.sensors["depth_camera"]
    dt = u.physics_dt

    # 로봇을 **물리로** 세운다 (순간이동 금지!). zero action = default 관절목표(standing) PD 유지 →
    # 지형 위 자연 높이로 선다. TiledCamera(RTX)는 write_root_pose 순간이동을 렌더가 안 따라와
    # desync(카메라가 딴 곳에서 쏨) 되므로, env.step 물리 구동으로 카메라가 항상 로봇을 따라오게 한다.
    action_dim = u.action_manager.total_action_dim
    zero_act = torch.zeros((cfg.scene.num_envs, action_dim), device=u.device)
    for _ in range(args_cli.settle):
        env.step(zero_act)

    # --- 데이터 취득 (env 0) ---
    depth = cam.data.output["distance_to_image_plane"][0, ..., 0]  # (H, W)
    H, W = depth.shape
    K = cam.data.intrinsic_matrices[0]                              # (3,3)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    # 카메라 pose 는 stale metadata 대신 live base pose + mount 로 계산 (로봇에 정확히 부착)
    cam_pos, cam_quat = live_cam_pose(robot, cam_pos_off, tilt, u.device)
    base_pos = robot.data.root_pos_w[0]
    rgb = cam.data.output.get("rgb", None)

    # --- FoV ---
    hfov_K = math.degrees(2 * math.atan(W / (2 * fx)))
    vfov_K = math.degrees(2 * math.atan(H / (2 * fy)))
    hfov_cfg = fov_from_aperture(D435I_H_APERTURE, D435I_FOCAL)
    vfov_cfg = fov_from_aperture(D435I_V_APERTURE, D435I_FOCAL)

    # --- 역투영 → base 기준 풋프린트 (live cam pose, ros optical) ---
    pts_cam = unproject_depth(depth.unsqueeze(0), K.unsqueeze(0), is_ortho=True)  # (1, H*W, 3)
    d_flat = depth.reshape(-1)
    valid = torch.isfinite(d_flat) & (d_flat > 0.05) & (d_flat < D435I_MAX_DIST - 1e-3)
    pts_world = transform_points(pts_cam, cam_pos.unsqueeze(0), cam_quat.unsqueeze(0))[0]
    rel = pts_world - base_pos                                       # base 기준
    rv = rel[valid]
    rv = rv[torch.isfinite(rv).all(dim=-1)]                          # inf/nan 픽셀 제거(평면 가장자리 miss 등)

    # --- 리포트 ---
    lines = []
    lines.append("==================== JE-Loco TiledCamera depth 검증 ====================")
    lines.append(f"이미지 크기        : {W}x{H}  (W×H)")
    lines.append(f"장착(base→cam)     : pos={tuple(round(v,3) for v in cam_pos_off)} m, 하향 틸트={tilt:.1f}°")
    lines.append(f"카메라 월드 pos    : ({float(cam_pos[0]):.3f},{float(cam_pos[1]):.3f},{float(cam_pos[2]):.3f})  "
                 f"base=({float(base_pos[0]):.3f},{float(base_pos[1]):.3f},{float(base_pos[2]):.3f})")
    lines.append(f"카메라 높이(지면위): {float(cam_pos[2]):.3f} m")
    lines.append("")
    lines.append(f"FoV (intrinsics)   : H={hfov_K:.1f}°  V={vfov_K:.1f}°   [fx={fx:.1f} fy={fy:.1f}]")
    lines.append(f"FoV (cfg aperture) : H={hfov_cfg:.1f}°  V={vfov_cfg:.1f}°")
    lines.append(f"실기 D435i depth   : 공칭 87°×58°(±3°) / 실측 intrinsics 기반 78.7°×63.1°")
    lines.append("")
    if rv.numel() > 0:
        lines.append(f"지면 풋프린트(base 기준, 유효 depth {rv.shape[0]}/{W*H} 픽셀):")
        lines.append(f"  전방 x : [{float(rv[:,0].min()):.2f}, {float(rv[:,0].max()):.2f}] m")
        lines.append(f"  좌우 y : [{float(rv[:,1].min()):.2f}, {float(rv[:,1].max()):.2f}] m")
        lines.append(f"  높이 z : [{float(rv[:,2].min()):.2f}, {float(rv[:,2].max()):.2f}] m (base 기준; 발밑 평지면 ≈ base_z 아래)")
    else:
        lines.append("경고: 유효 depth 픽셀 없음 (카메라가 지면을 못 봄 — 틸트/높이 확인).")
    lines.append(f"depth 값[m] min/mean/max : {float(d_flat[torch.isfinite(d_flat)].min()):.2f}/"
                 f"{float(d_flat[torch.isfinite(d_flat)].mean()):.2f}/"
                 f"{float(d_flat[torch.isfinite(d_flat)].max() if torch.isfinite(d_flat).any() else 0):.2f}")
    lines.append("======================================================================")
    report = "\n".join(lines)
    print(report, flush=True)
    with open(os.path.join(args_cli.out, "report.txt"), "w") as f:
        f.write(report + "\n")

    # --- 이미지 저장 ---
    tag = "flat" if args_cli.flat else "rough"
    dnp = depth.detach().cpu().numpy()
    dnp_v = np.where(np.isfinite(dnp), dnp, D435I_MAX_DIST)
    plt.figure(figsize=(5, 4))
    plt.imshow(dnp_v, cmap="turbo", vmin=0, vmax=D435I_MAX_DIST)
    plt.colorbar(label="depth [m]"); plt.title(f"D435i depth (tilt {tilt:.0f}°, {tag})")
    plt.xlabel("u (→ right)"); plt.ylabel("v (→ down)")
    plt.tight_layout(); plt.savefig(os.path.join(args_cli.out, f"depth_{tag}_tilt{int(tilt)}.png"), dpi=120)
    plt.close()

    if rgb is not None:
        rnp = rgb[0].detach().cpu().numpy()
        if rnp.dtype != np.uint8:
            rnp = np.clip(rnp, 0, 255).astype(np.uint8)
        plt.figure(figsize=(5, 4)); plt.imshow(rnp[..., :3]); plt.title(f"RGB (tilt {tilt:.0f}°, {tag})")
        plt.axis("off"); plt.tight_layout()
        plt.savefig(os.path.join(args_cli.out, f"rgb_{tag}_tilt{int(tilt)}.png"), dpi=120); plt.close()

    if rv.numel() > 0:  # top-down 풋프린트 (base 기준 x-y)
        plt.figure(figsize=(5, 5))
        sc = plt.scatter(rv[:, 1].cpu(), rv[:, 0].cpu(), c=rv[:, 2].cpu(), cmap="viridis", s=4)
        plt.colorbar(sc, label="z rel base [m]")
        plt.scatter([0], [0], c="red", marker="*", s=120, label="base")
        plt.xlabel("y (좌우) [m]"); plt.ylabel("x (전방) [m]"); plt.axis("equal")
        plt.title(f"지면 풋프린트 (base 기준, {tag})"); plt.legend(); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(os.path.join(args_cli.out, f"footprint_{tag}_tilt{int(tilt)}.png"), dpi=120)
        plt.close()

    print(f"[verify] 저장 → {args_cli.out}/ (depth_*.png, rgb_*.png, footprint_*.png, report.txt)", flush=True)

    # ============================ GUI 실시간 시각화 ============================
    # --headless 없이 실행하면 Isaac Sim 창에서 카메라 back-projection 점(높이색) + frustum 을
    # 지형 위에 겹쳐 그린다. 마우스로 시점을 돌려 카메라가 실제로 전방 지면을 재는지 눈으로 확인.
    if not args_cli.headless:
        try:
            from isaacsim.util.debug_draw import _debug_draw
        except ImportError:
            from omni.isaac.debug_draw import _debug_draw

        def height_rgba(z, zmin=-0.35, zmax=0.25):
            t = max(0.0, min(1.0, (z - zmin) / (zmax - zmin)))
            return (t, 0.2 + 0.6 * (1 - abs(2 * t - 1)), 1.0 - t, 0.9)  # 파랑(낮음)→초록→빨강(높음)

        draw = _debug_draw.acquire_debug_draw_interface()
        Dfar = D435I_MAX_DIST
        thx, thy = math.tan(math.radians(hfov_K / 2)), math.tan(math.radians(vfov_K / 2))
        # ros optical 프레임(+Z 전방, +X 우, +Y 하) frustum 코너
        corners_cam = torch.tensor(
            [[sx * thx * Dfar, sy * thy * Dfar, Dfar] for sx in (-1, 1) for sy in (-1, 1)],
            device=u.device)
        print("[verify] GUI 시각화 — 로봇은 물리로 제자리 standing(순간이동 없음 → 카메라가 로봇을 따라옴). "
              "점=카메라가 잰 지면(높이색), 노란 선=FoV frustum. 마우스로 뷰포트를 돌려 확인. 창 닫기로 종료.",
              flush=True)
        # 물리 구동(zero action=default 포즈 유지)으로 카메라·depth 가 항상 로봇에 붙어 있게 한다.
        # 지형 통과 관찰은 정책 학습 후 play.py 로. 여기선 스폰 지점 정지 검증.
        while simulation_app.is_running():
            with torch.inference_mode():
                env.step(zero_act)

                d = cam.data.output["distance_to_image_plane"][0, ..., 0]
                cpos, cquat = live_cam_pose(robot, cam_pos_off, tilt, u.device)
                pc = unproject_depth(d.unsqueeze(0), cam.data.intrinsic_matrices[0:1], is_ortho=True)
                pw = transform_points(pc, cpos.unsqueeze(0), cquat.unsqueeze(0))[0]
                df = d.reshape(-1)
                m = torch.isfinite(df) & (df > 0.05) & (df < Dfar - 1e-3)
                pv = pw[m]
                if pv.numel():
                    pts = pv.tolist()
                    cols = [height_rgba(p[2]) for p in pts]
                    draw.clear_points()
                    draw.draw_points(pts, cols, [6.0] * len(pts))
                # frustum 선 (카메라→4 코너)
                cw = transform_points(corners_cam.unsqueeze(0), cpos.unsqueeze(0), cquat.unsqueeze(0))[0]
                starts = [cpos.tolist()] * 4
                ends = cw.tolist()
                draw.clear_lines()
                draw.draw_lines(starts, ends, [(1.0, 1.0, 0.0, 0.8)] * 4, [2.0] * 4)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
