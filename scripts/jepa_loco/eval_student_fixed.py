"""Phase 3 student를 teacher와 동일한 고정 계단·평지·gap 프로토콜로 평가한다."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--policy", choices=("student", "teacher"), default="student")
parser.add_argument("--terrain", choices=("stairs", "flat", "gap"), default="stairs")
parser.add_argument("--step_heights_m", type=float, nargs="+", default=None)
parser.add_argument("--envs_per_height", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--episode_steps", type=int, default=None)
parser.add_argument("--spawn_forward_m", type=float, default=None)
parser.add_argument("--low_gap_width_m", type=float, default=None)
parser.add_argument("--high_gap_width_m", type=float, default=None)
parser.add_argument("--teacher_stats", type=str, default=None,
                    help="체크포인트에 기록된 통계 경로가 이 머신에서 다를 때 지정한다.")
parser.add_argument("--output", type=str, default=None)
parser.add_argument("--video", action="store_true")
parser.add_argument("--audit_terrain_only", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

import torch  # noqa: E402

from eval_stairs_fixed import main as run_fixed_eval  # noqa: E402
from unitree_rl_lab.jepa_loco.eval.stairs import StudentFixedEvalCfg  # noqa: E402
from unitree_rl_lab.jepa_loco.models.student_depth import StudentInferencePolicy  # noqa: E402


def main() -> None:
    defaults = StudentFixedEvalCfg()
    for name in ("step_heights_m", "envs_per_height", "seed", "spawn_forward_m"):
        if getattr(args, name) is None:
            setattr(args, name, getattr(defaults, name))
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    method = checkpoint["student_cfg"]["method"]
    iteration = checkpoint["iteration"]
    if method not in ("no_memory", "gru", "gru_jepa", "gru_copy"):
        raise ValueError(f"지원하지 않는 student method: {method}")
    stats_path = Path(args.teacher_stats or checkpoint["teacher_stats"])
    if not stats_path.exists():
        raise FileNotFoundError(f"teacher latent 통계가 없다: {stats_path}; --teacher_stats로 지정해라")
    stats = torch.load(stats_path, map_location="cpu", weights_only=False)
    if "teacher_checkpoint" in stats and stats["teacher_checkpoint"] != checkpoint["teacher_checkpoint"]:
        print("[WARN] checkpoint와 teacher 통계의 원래 경로가 다르다. 파일 이동 여부를 확인해라.", flush=True)
    output = Path(args.output) if args.output else Path("results/jepa_loco/student/eval") / (
        f"{method}_s{args.seed}_{iteration}_{args.terrain}"
        f"{'_teacher' if args.policy == 'teacher' else ''}.json")
    args.output = str(output)
    args.task = "Unitree-Go2-JepaLoco-StudentDepth-EasyStart"
    args.low_step_height_m = None
    args.high_step_height_m = None
    args.envs_per_width = None
    args.probe_terrain = False
    args.flatten_terrain_input = False
    args.contact_diagnostics = False
    heights = tuple(args.step_heights_m) if args.terrain == "stairs" else None
    run_fixed_eval(
        args, step_heights_m=heights,
        policy_factory=lambda path, device: StudentInferencePolicy(path, device, policy_mode=args.policy),
        latent_stats=(stats["mean"], stats["std"]),
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
