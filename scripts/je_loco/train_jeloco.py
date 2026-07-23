# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""단계 2 (게이트 0) — JE-Loco 헤드 A(DreamWaQ++ 재현) 학습 러너.

Isaac Lab `Unitree-Go2-JELoco` 환경 rollout → 헤드 불가지론 복합손실 PPO(JELocoPPO).
관측 그룹: policy=45(현재 proprio) / critic=privileged / depth(image-like) / heightmap_gt(441).
proprioception 히스토리(H)는 여기서 롤링 버퍼로 관리해 ProprioVAE 에 라우팅한다.

사용:
  # 게이트 0 학습 (헤드 A). depth ON 은 --enable_cameras 필수.
  python scripts/je_loco/train_jeloco.py --config <cfg>/head_a.yaml --num_envs 2048 --headless --enable_cameras
  # 파이프라인 스모크 (몇 iter 만, 작은 env)
  python scripts/je_loco/train_jeloco.py --config <cfg>/head_a.yaml --num_envs 64 --max_iterations 3 --headless --enable_cameras
"""

import argparse
import os

# 조각화 완화 (PointNet recompute-PPO 활성화 메모리 큼). torch import 전에 설정해야 유효.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="JE-Loco 헤드 A 학습 (게이트 0)")
parser.add_argument("--config", type=str, required=True, help="head_a.yaml / head_b.yaml")
parser.add_argument("--num_envs", type=int, default=None, help="환경 수 (config runner.num_envs 오버라이드).")
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--no_depth", action="store_true", help="depth 끄기(디버그; 헤드 A 는 depth 필요).")
parser.add_argument("--no_curriculum", action="store_true",
                    help="커리큘럼(지형/명령 승급) 끄고 명령범위 전체 고정 — standing-exploit 디버깅.")
parser.add_argument("--fwd_cmd", action="store_true",
                    help="전진 위주 고정 명령: vx[0.5,1.0], vy=0, ωz[-0.5,0.5], 정지 0%. 명령 승급 off.")
parser.add_argument("--simple", action="store_true",
                    help="격리 진단: 순수 MLP actor-critic(raw proprio, 표현학습/VAE/point cloud 없음)로 PPO 검증.")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--log_dir", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if not args_cli.no_depth:
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- 앱 기동 후 ----
import os
import statistics
import time

import gymnasium as gym
import torch
import yaml

# [환경 함정] RTX 50-series(Blackwell, sm_120) + torch2.7/cuDNN9.2 조합은 conv 커널 부재로
# depth CNN 의 첫 conv 에서 CUDNN_STATUS_NOT_INITIALIZED 를 던진다. cuDNN 을 끄면 native CUDA
# conv 로 폴백해 정상 동작(fwd/bwd 확인). 근본 해결은 torch/cuDNN 업그레이드(cuDNN≥9.7).
# MLP-only 이던 기존 태스크는 conv 를 안 써 이 문제를 못 만났다 — depth CNN 도입으로 처음 노출.
torch.backends.cudnn.enabled = False
torch.backends.cuda.matmul.allow_tf32 = True

from isaaclab.utils.math import convert_camera_frame_orientation_convention

import unitree_rl_lab.je_loco.envs  # noqa: F401  (gym 등록)
from unitree_rl_lab.je_loco.envs.je_loco_env_cfg import (
    D435I_MAX_DIST, D435I_POS, D435I_TILT_DEG, JELocoEnvCfg, _tilt_quat,
)
from unitree_rl_lab.je_loco.models.pointcloud import PointCloudMemory, deproject_to_body
from unitree_rl_lab.je_loco.models.simple import SimpleBackbone
from unitree_rl_lab.je_loco.train.builder import build_backbone, build_spec, load_cfg
from unitree_rl_lab.je_loco.train.ppo import JELocoPPO
from unitree_rl_lab.je_loco.train.rollout import ProprioHistory, RolloutStorage


def main():
    cfg = load_cfg(args_cli.config)
    runner_cfg = cfg.get("runner", {}) or {}
    num_envs = args_cli.num_envs or int(runner_cfg.get("num_envs", 2048))
    max_iter = args_cli.max_iterations or int(runner_cfg.get("max_iterations", 50000))
    num_steps = int(runner_cfg.get("num_steps_per_env", 24))
    save_interval = int(runner_cfg.get("save_interval", 200))

    # 환경 (enable_depth 는 생성자 인자로! post_init 순서)
    env_cfg = JELocoEnvCfg(enable_depth=not args_cli.no_depth)
    env_cfg.scene.num_envs = num_envs
    env_cfg.seed = args_cli.seed
    if args_cli.device is not None:
        env_cfg.sim.device = args_cli.device

    if args_cli.no_curriculum:
        # standing-exploit 디버깅: 커리큘럼 승급 끄고, 명령을 처음부터 full-range 고정
        # → 로봇이 "서있기"에 빠지지 않고 처음부터 속도추종을 학습하도록 강제.
        env_cfg.curriculum.terrain_levels = None
        env_cfg.curriculum.lin_vel_cmd_levels = None
        env_cfg.commands.base_velocity.ranges = env_cfg.commands.base_velocity.limit_ranges
        print("[train] no_curriculum: 지형/명령 승급 off, 명령 full-range 고정", flush=True)

    if args_cli.fwd_cmd:
        # 전진 위주 고정 명령 (명령 승급 off, 지형 승급은 유지 → Curriculum 로그 확인용).
        env_cfg.curriculum.lin_vel_cmd_levels = None
        cmd = env_cfg.commands.base_velocity
        for rg in (cmd.ranges, cmd.limit_ranges):
            rg.lin_vel_x = (0.5, 1.0)
            rg.lin_vel_y = (0.0, 0.0)
            rg.ang_vel_z = (-0.5, 0.5)
        cmd.rel_standing_envs = 0.0
        print("[train] fwd_cmd: vx[0.5,1.0] vy=0 ωz[-0.5,0.5], 정지 0%, 명령 승급 off", flush=True)

    env = gym.make("Unitree-Go2-JELoco", cfg=env_cfg)
    u = env.unwrapped
    device = u.device

    # 백본 + PPO (헤드는 config 가 결정 — 학습 루프는 헤드를 모른다)
    spec = build_spec(cfg)
    priv_dim = spec.base_lin_vel_dim + spec.heightmap_dim     # 3 + 441 = 444
    action_dim = u.action_manager.total_action_dim
    if args_cli.simple:
        backbone = SimpleBackbone(spec, privileged_dim=priv_dim, action_dim=action_dim).to(device)
        print("[train] SIMPLE 모드: 순수 MLP actor-critic (표현학습/VAE/point cloud 없음)", flush=True)
    else:
        backbone = build_backbone(cfg, action_dim=action_dim, privileged_dim=priv_dim)
    ppo = JELocoPPO(backbone, cfg, device=device)
    head_name = cfg["head"]["name"]
    print(f"[train] head={head_name} num_envs={num_envs} steps/env={num_steps} "
          f"action_dim={action_dim} priv_dim={priv_dim} device={device}", flush=True)

    robot = u.scene["robot"]
    is_pointcloud = (not args_cli.simple) and spec.ext_encoder == "pointcloud"

    # --- 외수용 point cloud 경로 준비 (주 경로) ---
    if args_cli.simple:
        ext_shape = (1,)   # SimpleBackbone 은 ext_obs 무시 → dummy
    elif is_pointcloud:
        cam = u.scene.sensors["depth_camera"]
        K_intr = cam.data.intrinsic_matrices[0]                       # (3,3)
        cam_pos_body = torch.tensor(D435I_POS, device=device)
        # 카메라 optical(ros) 방향 (body frame): world-conv 하향 틸트 → ros 변환
        _tq = torch.tensor([_tilt_quat(D435I_TILT_DEG)], device=device)
        cam_quat_body = convert_camera_frame_orientation_convention(_tq, origin="world", target="ros")[0]
        pcmem = PointCloudMemory(num_envs, spec.ext_memory_K, spec.num_points, device)
        ext_shape = (spec.ext_points_total, spec.point_feat_dim)      # (K*num_points, 3)
    else:  # depth_cnn 보조 경로
        ext_shape = spec.depth_shape

    # 히스토리 버퍼 + rollout 저장소
    hist = ProprioHistory(num_envs, spec.history_len, spec.proprio_dim, device)
    shapes = {"ext_obs": ext_shape, "obs_hist": (spec.history_len, spec.proprio_dim),
              "proprio": (spec.proprio_dim,), "proprio_next": (spec.proprio_dim,),
              "privileged": (priv_dim,),
              "heightmap_gt": (spec.heightmap_dim,), "v_gt": (spec.base_lin_vel_dim,)}
    storage = RolloutStorage(num_steps, num_envs, shapes, action_dim, device)

    def perceive(obs: dict) -> dict:
        """현재 관측 → 지각 dict. **hist·pcmem 을 각각 한 번 advance** (obs 당 정확히 1회 호출)."""
        proprio = obs["policy"]                                       # (N, 45)
        v_gt = obs["critic"][:, 0:3]                                  # critic 첫 3 = base_lin_vel(GT)
        heightmap_gt = obs["heightmap_gt"]                            # (N, 441)
        if args_cli.simple:
            ext_obs = proprio.new_zeros(num_envs, 1)                  # SimpleBackbone 무시
        elif is_pointcloud:
            depth = obs["depth"].reshape(num_envs, spec.depth_height, spec.depth_width)
            pts_body = deproject_to_body(depth, K_intr, cam_pos_body, cam_quat_body,
                                         spec.num_points, D435I_MAX_DIST)
            # SE(3) 메모리 정렬은 live body pose 로 (cam.data.pos_w 는 stale)
            ext_obs = pcmem.push_and_get(pts_body, robot.data.root_pos_w, robot.data.root_quat_w)
        else:
            ext_obs = obs["depth"].reshape(num_envs, *spec.depth_shape)
        obs_hist = hist.push(proprio)
        privileged = torch.cat([v_gt, heightmap_gt], dim=-1)
        return {"proprio": proprio, "ext_obs": ext_obs, "obs_hist": obs_hist,
                "v_gt": v_gt, "heightmap_gt": heightmap_gt, "privileged": privileged}

    # 로그 디렉터리 + 텐서보드
    log_dir = args_cli.log_dir or os.path.join(
        "logs", "je_loco", cfg.get("experiment_name", "je_loco_head_a"))
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir)
        print(f"[train] TensorBoard: tensorboard --logdir {log_dir}", flush=True)
    except Exception as e:  # noqa: BLE001
        writer = None
        print(f"[train] TensorBoard 비활성({e}). 콘솔 출력만.", flush=True)
    total_time = 0.0
    total_steps = 0

    obs, _ = env.reset()
    p = perceive(obs)   # 리셋 직후 obs 를 한 번 지각(hist·pcmem advance). 이후 obs 당 정확히 1회.
    # 에피소드 통계 추적
    ep_reward = torch.zeros(num_envs, device=device)
    ep_len = torch.zeros(num_envs, device=device)
    gamma = ppo.gamma

    for it in range(max_iter):
        t0 = time.perf_counter()
        storage.reset()
        rollout_rewards, rollout_lens = [], []
        ep_infos = []   # env 의 Episode_Reward/Termination/Curriculum/Metrics 로그 누적

        for _ in range(num_steps):
            action, logp, value, mu, sigma = backbone.rollout_act(
                p["ext_obs"], p["obs_hist"], p["proprio"], p["privileged"])

            obs, reward, terminated, truncated, info = env.step(action)
            if isinstance(info, dict) and "log" in info:
                ep_infos.append(info["log"])
            truncated = truncated.float()
            done = torch.clamp(terminated.float() + truncated, max=1.0)
            reward = reward + gamma * value * truncated   # 타임아웃 부트스트랩 (rsl_rl 규약)
            proprio_next = obs["policy"]                   # o_{t+1} — VAE 미래 재구성 타깃 (Eq.8)

            storage.add(
                {"ext_obs": p["ext_obs"], "obs_hist": p["obs_hist"], "proprio": p["proprio"],
                 "proprio_next": proprio_next, "privileged": p["privileged"],
                 "heightmap_gt": p["heightmap_gt"], "v_gt": p["v_gt"]},
                action, logp, value, reward, done, mu, sigma,
            )

            # 에피소드 통계
            ep_reward += reward
            ep_len += 1
            done_ids = done.nonzero(as_tuple=False).flatten()
            if done_ids.numel() > 0:
                rollout_rewards += ep_reward[done_ids].tolist()
                rollout_lens += ep_len[done_ids].tolist()
                ep_reward[done_ids] = 0.0
                ep_len[done_ids] = 0.0
                hist.reset(done_ids)                      # 새 에피소드 → 버퍼 초기화 (경계 오염 방지)
                if is_pointcloud:
                    pcmem.reset(done_ids)

            p = perceive(obs)   # 스텝 후 새 obs 지각 (done env 는 위에서 버퍼 리셋됨 → 깨끗한 시작)

        # 부트스트랩: p 는 마지막 obs 지각 결과 (재-push 없음, 다음 iter 첫 스텝이 이 p 로 act)
        last_value = backbone.get_value(p["ext_obs"], p["obs_hist"], p["proprio"], p["privileged"])
        storage.compute_returns(last_value, ppo.gamma, ppo.lam)
        logs = ppo.update(storage)

        dt = time.perf_counter() - t0
        total_time += dt
        total_steps += num_steps * num_envs
        fps = int(num_steps * num_envs / dt)
        mean_r = statistics.mean(rollout_rewards) if rollout_rewards else float("nan")
        mean_l = statistics.mean(rollout_lens) if rollout_lens else float("nan")
        noise_std = float(backbone.actor_critic.actor.log_std.exp().mean())
        eta = total_time / (it + 1) * (max_iter - it - 1)

        # env episode 로그 집계 (Episode_Reward/Termination/Curriculum/Metrics)
        ep_agg: dict[str, float] = {}
        if ep_infos:
            for k in set().union(*[d.keys() for d in ep_infos]):
                vals = [(d[k].item() if hasattr(d[k], "item") else float(d[k]))
                        for d in ep_infos if k in d]
                if vals:
                    ep_agg[k] = sum(vals) / len(vals)

        # ---- rsl_rl 스타일 그룹 콘솔 출력 ----
        w = 80
        print("\n" + "=" * w)
        print(f" Learning iteration {it}/{max_iter} ".center(w, "="))
        print(f"{'Computation:':>26} {fps} steps/s  (iter {dt:.2f}s)")
        print(f"{'Mean action noise std:':>26} {noise_std:.3f}")
        print(f"{'Mean reward:':>26} {mean_r:.3f}")
        print(f"{'Mean episode length:':>26} {mean_l:.2f}")

        def _group(prefix: str, title: str):
            items = {k[len(prefix):]: v for k, v in ep_agg.items() if k.startswith(prefix)}
            if items:
                print(("── " + title + " ").ljust(w, "─"))
                for k in sorted(items):
                    print(f"{k:>36}: {items[k]:.4f}")

        _group("Episode_Reward/", "Episode_Reward")
        _group("Episode_Termination/", "Episode_Termination")
        _group("Curriculum/", "Curriculum")
        _group("Metrics/", "Metrics")
        print("── Loss ".ljust(w, "─"))
        for name, key in (("surrogate", "surrogate"), ("value_function", "value_loss"),
                          ("entropy", "entropy"), ("vae_total", "l_vae"), ("vae_est", "l_est"),
                          ("vae_recon", "l_recon"), ("repr_headA", "l_repr"),
                          ("contrastive", "l_contrastive")):
            print(f"{name:>36}: {logs[key]:.4f}")
        print("── Policy ".ljust(w, "─"))
        print(f"{'kl':>36}: {logs['kl']:.4f}")
        print(f"{'learning_rate':>36}: {logs['lr']:.2e}")
        print(f"{'vae_beta':>36}: {logs['beta']:.2f}")
        print("── Perf ".ljust(w, "─"))
        print(f"{'total_timesteps':>36}: {total_steps:,}")
        print(f"{'total_time':>36}: {total_time:.1f} s")
        print(f"{'eta':>36}: {eta / 60:.1f} min")
        print("=" * w, flush=True)

        # ---- TensorBoard (rsl_rl 그룹 규약) ----
        if writer is not None:
            writer.add_scalar("Train/mean_reward", mean_r, it)
            writer.add_scalar("Train/mean_episode_length", mean_l, it)
            writer.add_scalar("Policy/action_noise_std", noise_std, it)
            writer.add_scalar("Policy/entropy", logs["entropy"], it)
            writer.add_scalar("Policy/kl", logs["kl"], it)
            writer.add_scalar("Policy/learning_rate", logs["lr"], it)
            writer.add_scalar("Loss/surrogate", logs["surrogate"], it)
            writer.add_scalar("Loss/value_function", logs["value_loss"], it)
            writer.add_scalar("Loss/vae_total", logs["l_vae"], it)
            writer.add_scalar("Loss/vae_est", logs["l_est"], it)
            writer.add_scalar("Loss/vae_recon", logs["l_recon"], it)
            writer.add_scalar("Loss/repr_headA", logs["l_repr"], it)
            writer.add_scalar("Loss/contrastive", logs["l_contrastive"], it)
            writer.add_scalar("Perf/steps_per_sec", fps, it)
            writer.add_scalar("Perf/total_time", total_time, it)
            for k, v in ep_agg.items():   # Episode_Reward/*, Episode_Termination/*, Curriculum/*, Metrics/*
                writer.add_scalar(k, v, it)

        if it > 0 and it % save_interval == 0:
            ckpt = os.path.join(log_dir, f"model_{it}.pt")
            torch.save({"backbone": backbone.state_dict(), "iter": it, "cfg": cfg}, ckpt)
            print(f"[train] saved {ckpt}", flush=True)

    torch.save({"backbone": backbone.state_dict(), "iter": max_iter, "cfg": cfg},
               os.path.join(log_dir, "model_final.pt"))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
