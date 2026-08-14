# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""Phase 2-a — Teacher rollout 데이터 수집 (오프라인 JEPA/recon 사전학습용).

Teacher(privileged heightmap 정책)를 시뮬에 굴리며 기록:
  pc      (T, N, 768)  point cloud 192×[x,y,z,valid] — student 의 입력이 될 것 (±0.02 노이즈 포함)
  policy  (T, N, 232)  teacher 관측 = proprio45(노이즈) + heightmap187(clean GT)
                       → 오프라인에서 슬라이스: proprio=[:,:45], hmap(recon 타깃)=[:,45:]
  act     (T, N, 12)   teacher 행동 (JEPA action-conditioning 용)
  done    (T, N)       에피소드 경계 (done-crossing 쌍 제외용 — ④ 교훈의 오프라인 적용)

설계:
  - env = Teacher 학습 cfg + pc_scanner(frustum) 추가. 지형 커리큘럼 정지 + 전 레벨(0..9) 균등
    스폰 → 난이도 전 구간 데이터. push_robot 유지(회복 동작 다양성).
  - 수집은 정책 실행뿐(학습 없음) → 빠름. 샤드(npz, fp16)로 저장.

사용:
  python -u scripts/je_loco/collect_teacher_data.py --checkpoint <teacher ckpt> \
    --num_envs 4096 --steps_per_shard 256 --num_shards 8 --out datasets/teacher_v1 --headless
  (4096×256×8 ≈ 8.4M steps. pc fp16 기준 셔드당 ~1.6GB)
"""

from __future__ import annotations

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Teacher rollout 수집 (Phase 2-a)")
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--steps_per_shard", type=int, default=256)
parser.add_argument("--num_shards", type=int, default=8)
parser.add_argument("--warmup", type=int, default=60, help="스폰 직후 transient 버림")
parser.add_argument("--out", type=str, default="datasets/teacher_v1")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--task", type=str, default="Unitree-Go2-JELoco-Teacher")
import cli_args  # noqa: E402
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab.managers import ObservationGroupCfg as ObsGroup  # noqa: E402
from isaaclab.managers import ObservationTermCfg as ObsTerm  # noqa: E402
from isaaclab.managers import SceneEntityCfg  # noqa: E402
from isaaclab.sensors import RayCasterCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from importlib.metadata import version  # noqa: E402

import unitree_rl_lab.je_loco.rsl_rl_pc  # noqa: F401, E402
from unitree_rl_lab.je_loco.rsl_rl_pc import mdp_pc  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402

torch.backends.cudnn.enabled = False


@configclass
class _PcGroupCfg(ObsGroup):
    """student 입력이 될 pc 관측 — JELoco PC env 와 동일 정의(frustum 192×4, ±0.02 노이즈)."""

    point_cloud = ObsTerm(
        func=mdp_pc.raycaster_pointcloud,
        params={"sensor_cfg": SceneEntityCfg("pc_scanner")},
        noise=Unoise(n_min=-0.02, n_max=0.02),
        clip=(-2.0, 2.0),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


def main():
    # ── env: Teacher 학습 cfg 기반 (push 유지) + pc_scanner + 지형 전 레벨 균등 ──
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, entry_point_key="env_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    # 지형: 커리큘럼 정지, 0..9 균등 스폰 → 전 난이도 데이터
    env_cfg.curriculum.terrain_levels = None
    env_cfg.scene.terrain.max_init_terrain_level = None   # None → num_rows-1 까지 균등
    # pc_scanner (JELoco PC env 의 D435i frustum 과 동일 파라미터)
    env_cfg.scene.pc_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.325, 0.0, 0.045), rot=mdp_pc.tilt_quat_y(35.0)),
        ray_alignment="base",
        pattern_cfg=mdp_pc.FrustumPatternCfg(hfov_deg=78.7, vfov_deg=63.1, width=16, height=12),
        max_distance=2.0,
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    env_cfg.scene.pc_scanner.update_period = env_cfg.decimation * env_cfg.sim.dt
    env_cfg.observations.pointcloud = _PcGroupCfg()

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint),
                load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
                strict=False)
    policy = runner.get_inference_policy(device=uenv.device)

    os.makedirs(args_cli.out, exist_ok=True)
    obs = env.get_observations()
    pc_dim = obs["pointcloud"].shape[-1]
    pol_dim = obs["policy"].shape[-1]
    print(f"[collect] pc={pc_dim} policy={pol_dim} envs={uenv.num_envs} "
          f"→ {args_cli.num_shards}×{args_cli.steps_per_shard} steps/env")

    with torch.inference_mode():
        for _ in range(args_cli.warmup):                      # transient 버림
            obs, _, _, _ = env.step(policy(obs))

        for shard in range(args_cli.num_shards):
            T, N = args_cli.steps_per_shard, uenv.num_envs
            buf_pc = np.empty((T, N, pc_dim), dtype=np.float16)
            buf_pol = np.empty((T, N, pol_dim), dtype=np.float16)
            buf_act = np.empty((T, N, 12), dtype=np.float16)
            buf_done = np.empty((T, N), dtype=bool)
            for t in range(T):
                act = policy(obs)
                buf_pc[t] = obs["pointcloud"].cpu().numpy().astype(np.float16)
                buf_pol[t] = obs["policy"].cpu().numpy().astype(np.float16)
                buf_act[t] = act.cpu().numpy().astype(np.float16)
                obs, _, dones, _ = env.step(act)
                buf_done[t] = dones.cpu().numpy().astype(bool).reshape(-1)
            path = os.path.join(args_cli.out, f"shard_{shard:03d}.npz")
            np.savez(path, pc=buf_pc, policy=buf_pol, act=buf_act, done=buf_done)
            print(f"[collect] {path}  ({T}×{N} steps, done율 {buf_done.mean():.4f})")

    print(f"[collect] 완료 → {args_cli.out}  (총 {args_cli.num_shards * args_cli.steps_per_shard * uenv.num_envs} env-steps)")
    env.close(); simulation_app.close()


if __name__ == "__main__":
    main()
