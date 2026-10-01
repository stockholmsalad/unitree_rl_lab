"""Phase 0 — Go2 에 D435i extrinsic 으로 TiledCamera 를 붙여 depth 샘플을 저장한다.

지형 4종(평지 / 오르는 계단 / 내려가는 계단 / gap)에 로봇을 하나씩 세우고 기본 자세를 유지한 뒤
848×480 depth 와 96×64 다운샘플 미리보기를 저장한다. 평지 env 에서 카메라 지면 높이를 실측해
명세(0.38 m)와 비교한다.

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
import torch.nn.functional as F  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab.terrains as terrain_gen  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

from unitree_rl_lab.assets.robots.unitree import UNITREE_GO2_CFG  # noqa: E402
from unitree_rl_lab.jepa_loco.sensors.d435i_cfg import D435iParams, make_tiled_camera_cfg  # noqa: E402

TERRAIN_NAMES = ["flat", "stairs_up", "stairs_down", "gap"]
PARAMS = D435iParams()
POLICY_HW = (64, 96)
CLIP = (0.28, 2.0)

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

    depth = cam.data.output["distance_to_image_plane"][..., 0]  # (N, H, W)
    small = F.interpolate(depth.clamp(*CLIP)[:, None], size=POLICY_HW, mode="area")[:, 0]

    origins = scene.env_origins
    report = {
        "params": PARAMS.to_dict(),
        "camera_x_in_base": PARAMS.camera_x_in_base,
        "render_hw": list(depth.shape[1:]),
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
        }

    os.makedirs(args_cli.out_dir, exist_ok=True)
    np.save(os.path.join(args_cli.out_dir, "depth_full.npy"), depth.cpu().numpy())
    np.save(os.path.join(args_cli.out_dir, "depth_96x64.npy"), small.cpu().numpy())
    with open(os.path.join(args_cli.out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)

    fig, axes = plt.subplots(len(TERRAIN_NAMES), 2, figsize=(10, 2.6 * len(TERRAIN_NAMES)))
    for i, name in enumerate(TERRAIN_NAMES):
        full = depth[i].cpu().numpy()
        shown = np.where(full > 0, np.clip(full, *CLIP), np.nan)
        for ax, img, title in (
            (axes[i, 0], shown, f"{name} — {full.shape[1]}×{full.shape[0]} (0 = invalid, white)"),
            (axes[i, 1], small[i].cpu().numpy(), f"{name} — {POLICY_HW[1]}×{POLICY_HW[0]} area"),
        ):
            im = ax.imshow(img, cmap="viridis", vmin=CLIP[0], vmax=CLIP[1])
            ax.set_title(title, fontsize=9)
            ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.6, label="depth [m]")
    fig.savefig(os.path.join(args_cli.out_dir, "depth_samples.png"), dpi=110)

    print(json.dumps(report["per_env"], indent=2))


if __name__ == "__main__":
    main()
    simulation_app.close()
