"""고정 Current teacher 정책 방문 데이터에서 latent 정규화 평균·표준편차를 저장한다."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", default="results/jepa_loco/student/teacher_latent_stats_s42.pt")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

from pathlib import Path  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402, F401
from unitree_rl_lab.jepa_loco.models.student_depth import FrozenCurrentTeacher  # noqa: E402
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


@configclass
class TeacherCalibrationCfg:
    num_envs: int = 64
    steps: int = 500
    min_std: float = 0.01
    seed: int = 43


def main() -> None:
    calibration = TeacherCalibrationCfg()
    task = "Unitree-Go2-JepaLoco-OracleCurrent-EasyStart"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=calibration.num_envs,
                            use_fabric=True, entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed = calibration.seed
    env_cfg.curriculum.terrain_levels = None
    env = gym.make(task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    teacher = FrozenCurrentTeacher.from_checkpoint(args.checkpoint, torch.device(env.device))
    samples = []
    with torch.inference_mode():
        for _ in range(calibration.steps):
            z = teacher.terrain_latent(obs["terrain_current"])
            samples.append(z.float().cpu())
            action = teacher.action_from_latent(obs["policy"], z)
            obs, *_ = env.step(action)
    all_z = torch.cat(samples)
    mean = all_z.mean(dim=0)
    raw_std = all_z.std(dim=0, unbiased=False)
    std = raw_std.clamp(min=calibration.min_std)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"mean": mean, "std": std, "raw_std": raw_std, "count": all_z.shape[0],
                "teacher_checkpoint": str(Path(args.checkpoint).resolve()),
                "calibration": calibration.to_dict()}, path)
    print(f"TEACHER_STATS count={all_z.shape[0]} std_min={raw_std.min().item():.5f} "
          f"std_mean={raw_std.mean().item():.5f} output={path}", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
