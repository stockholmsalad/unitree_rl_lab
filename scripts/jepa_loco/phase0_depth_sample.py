"""Phase 0 — Go2 에 D435i extrinsic 으로 TiledCamera 를 붙여 depth 샘플을 저장한다.

지형 4종(평지 / 오르는 계단 / 내려가는 계단 / gap)에 로봇을 하나씩 세우고 기본 자세를 유지한 뒤
두 카메라(같은 extrinsic·FOV)로 렌더링한다:
  - 112×64 직접 렌더링 (정책 입력 경로, intrinsic 스케일)
  - 848×480 → ``downsample_real`` (실기체 변환 경로)
두 경로의 일치도를 보고해 intrinsic 스케일과 실기체 변환 함수를 함께 검증한다.

    python scripts/jepa_loco/phase0_depth_sample.py --headless --enable_cameras
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--out_dir", type=str, default="results/jepa_loco/phase0")
parser.add_argument("--settle_s", type=float, default=2.0, help="기본 자세로 안정화할 시간 [s].")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import json  # noqa: E402
import os  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab.terrains as terrain_gen  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG  # noqa: E402
from unitree_rl_lab.jepa_loco.sensors.d435i_cfg import D435iParams, make_tiled_camera_cfg  # noqa: E402
from unitree_rl_lab.jepa_loco.sensors.depth_proc import apply_left_band, downsample_real, to_two_channel  # noqa: E402

TERRAIN_NAMES = ["flat", "stairs_up", "stairs_down", "gap"]
PARAMS = D435iParams()
POLICY_HW = PARAMS.render_wh[::-1]
CLIP = (PARAMS.min_range, PARAMS.clip_far)

SAMPLE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=1,
    num_cols=4,
    curriculum=True,  # 열 순서 = sub_terrains 순서, env i → 열 i
    difficulty_range=(1.0, 1.0),
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=1.0),
        # 피라미드 꼭대기에 서면 앞이 내려가는 계단, 역피라미드 바닥에 서면 오르는 계단
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(0.14, 0.14), step_width=0.3, platform_width=2.0, border_width=1.0
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=1.0, step_height_range=(0.14, 0.14), step_width=0.3, platform_width=2.0, border_width=1.0
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(proportion=1.0, gap_width_range=(0.2, 0.2), platform_width=2.0),
    },
)


@configclass
class SampleSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=SAMPLE_TERRAINS_CFG,
        max_init_terrain_level=0,
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    d435i = make_tiled_camera_cfg(PARAMS).replace(update_latest_camera_pose=True)  # 높이 실측용
    d435i_full = make_tiled_camera_cfg(PARAMS, "{ENV_REGEX_NS}/Robot/base/d435i_full", wh=PARAMS.native_wh)
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2000.0))


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args_cli.device))
    scene = InteractiveScene(SampleSceneCfg(num_envs=len(TERRAIN_NAMES), env_spacing=8.0))
    sim.reset()

    robot, cam = scene["robot"], scene["d435i"]
    # env 없이 scene 만 쓰므로 reset 이벤트가 하던 일(지형 원점으로 이동)을 직접 한다
    root = robot.data.default_root_state.clone()
    root[:, :3] += scene.env_origins
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    sim_dt = sim.get_physics_dt()
    for _ in range(int(args_cli.settle_s / sim_dt)):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim_dt)

    render = cam.data.output["distance_to_image_plane"][..., 0]  # (N, 64, 112) 정책 경로
    depth = scene["d435i_full"].data.output["distance_to_image_plane"][..., 0]  # (N, 480, 848)
    converted = downsample_real(depth, POLICY_HW, mode="nearest", near=CLIP[0])
    fx_render = PARAMS.intrinsics(PARAMS.render_wh)[0]
    banded = apply_left_band(render, fx_render, PARAMS.baseline)
    two_ch = to_two_channel(banded, *CLIP)
    # 두 경로 일치도: 정규화 depth 절대오차 (둘 다 유효인 픽셀)
    a, b = to_two_channel(render, *CLIP), to_two_channel(converted, *CLIP)
    both = (a[:, 1] * b[:, 1]).bool()
    path_err = (a[:, 0] - b[:, 0]).abs()

    origins = scene.env_origins
    report = {
        "params": PARAMS.to_dict(),
        "render_intrinsics": PARAMS.intrinsics(PARAMS.render_wh),
        "left_band_px_at": {f"{z}m": PARAMS.baseline * fx_render / z for z in (0.3, 0.5, 1.0, 2.0)},
        "per_env": {},
    }
    for i, name in enumerate(TERRAIN_NAMES):
        d = depth[i]
        report["per_env"][name] = {
            "camera_z_above_env_origin": float(cam.data.pos_w[i, 2] - origins[i, 2]),
            "base_z_above_env_origin": float(robot.data.root_pos_w[i, 2] - origins[i, 2]),
            "camera_pitch_down_deg": float(
                torch.rad2deg(torch.asin(-math_utils.quat_apply(cam.data.quat_w_world[i : i + 1], torch.tensor([[1.0, 0, 0]], device=depth.device))[0, 2]))
            ),
            "base_pitch_deg": float(torch.rad2deg(math_utils.euler_xyz_from_quat(robot.data.root_quat_w[i : i + 1])[1])[0]),
            "valid_frac": float((d > 0).float().mean()),
            "depth_min_valid": float(d[d > 0].min()) if (d > 0).any() else None,
            "depth_max": float(d.max()),
            # 하단 중앙 픽셀까지의 거리 (평지라면 지면 시작 거리와 대응)
            "bottom_center_depth": float(d[-1, d.shape[1] // 2]),
            "path_err_median": float(path_err[i][both[i]].median()),
            "path_err_p95": float(path_err[i][both[i]].quantile(0.95)),
            "two_ch_valid_frac_with_band": float(two_ch[i, 1].mean()),
        }

    os.makedirs(args_cli.out_dir, exist_ok=True)
    np.save(os.path.join(args_cli.out_dir, "depth_full.npy"), depth.cpu().numpy())
    np.save(os.path.join(args_cli.out_dir, "depth_render.npy"), render.cpu().numpy())
    with open(os.path.join(args_cli.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    cols = (
        ("848×480", lambda i: depth[i]),
        ("848×480 → 112×64 nearest", lambda i: converted[i]),
        ("112×64 direct render", lambda i: render[i]),
        ("+ left band: mask ch", lambda i: two_ch[i, 1]),
    )
    fig, axes = plt.subplots(len(TERRAIN_NAMES), len(cols), figsize=(4.2 * len(cols), 2.6 * len(TERRAIN_NAMES)))
    for i, name in enumerate(TERRAIN_NAMES):
        for j, (title, get) in enumerate(cols):
            img = get(i).cpu().numpy()
            if j < 3:
                im = axes[i, j].imshow(np.where(img > 0, np.clip(img, *CLIP), np.nan), cmap="viridis", vmin=CLIP[0], vmax=CLIP[1])
            else:
                axes[i, j].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_title(f"{name} — {title}", fontsize=8)
            axes[i, j].axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="depth [m]")
    fig.savefig(os.path.join(args_cli.out_dir, "depth_samples.png"), dpi=110)

    print(json.dumps(report["per_env"], indent=2), flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
