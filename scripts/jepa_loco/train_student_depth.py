"""Current-EasyStart teacher의 고정 head를 재사용하는 Depth student DAgger 증류.

본 학습은 명시적인 --max_iterations와 --num_envs를 지정해 별도로 실행한다.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--teacher_checkpoint", required=True)
parser.add_argument("--teacher_stats", required=True)
parser.add_argument("--method", choices=("no_memory", "gru", "gru_jepa", "gru_copy"), required=True)
parser.add_argument("--num_envs", type=int, required=True)
parser.add_argument("--max_iterations", type=int, required=True)
parser.add_argument("--seed", type=int, default=43)
parser.add_argument("--output_dir", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

from pathlib import Path  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402, F401
from unitree_rl_lab.jepa_loco.agents.student_cfg import StudentTrainCfg  # noqa: E402
from unitree_rl_lab.jepa_loco.models.student_depth import FrozenCurrentTeacher, StudentDepthModel  # noqa: E402
from unitree_rl_lab.jepa_loco.models.student_losses import (  # noqa: E402
    distillation_losses, jepa_valid_mask, masked_future_mse,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    train = StudentTrainCfg(method=args.method, num_envs=args.num_envs,
                            max_iterations=args.max_iterations)
    task = "Unitree-Go2-JepaLoco-StudentDepth-EasyStart"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=train.num_envs)
    env_cfg.seed = args.seed
    horizon_steps = train.validate_training(env_cfg.sim.dt * env_cfg.decimation, env_cfg.depth_period_steps)
    torch.manual_seed(args.seed)
    env = gym.make(task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    device = torch.device(env.device)
    teacher = FrozenCurrentTeacher.from_checkpoint(args.teacher_checkpoint, device)
    if any(parameter.requires_grad for parameter in teacher.parameters()) or teacher.training:
        raise AssertionError("teacher는 완전히 고정된 eval 모드여야 한다")
    teacher_initial = {name: value.detach().clone() for name, value in teacher.state_dict().items()}
    stats = torch.load(args.teacher_stats, map_location=device, weights_only=False)
    if str(Path(args.teacher_checkpoint).resolve()) != stats["teacher_checkpoint"]:
        raise ValueError("teacher 통계와 checkpoint가 다르다")
    mean, std = stats["mean"], stats["std"]
    model = StudentDepthModel(
        method=train.method, image_hw=train.image_hw, feature_dim=train.feature_dim,
        context_dim=train.context_dim, terrain_dim=train.terrain_dim,
        terrain_hidden_dim=train.terrain_hidden_dim, predictor_hidden_dim=train.predictor_hidden_dim,
        cnn_channels=train.cnn_channels, cnn_kernels=train.cnn_kernels,
        cnn_strides=train.cnn_strides,
    ).to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=train.learning_rate)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output)
    state = model.initial_state(env.num_envs, device)
    ema_state = model.initial_state(env.num_envs, device)

    for iteration in range(train.max_iterations):
        iteration_start = time.perf_counter()
        beta = train.dagger_beta(iteration)
        h0 = None if state is None else state.detach().clone()
        h0_ema = None if ema_state is None else ema_state.detach().clone()
        memory = {key: [] for key in ("depth", "fresh", "policy", "command", "teacher_z",
                                       "teacher_action", "done")}
        metrics = {}
        with torch.inference_mode():
            for _ in range(train.rollout_steps):
                fresh = obs["depth_fresh"][:, 0] > 0.5
                context, state = model.encode_step(obs["depth"], fresh, state)
                _, ema_state = model.encode_step(obs["depth"], fresh, ema_state, target=True)
                student_z = model.terrain_latent(context)
                teacher_z = teacher.terrain_latent(obs["terrain_current"])
                student_action = teacher.action_from_latent(obs["policy"], student_z)
                teacher_action = teacher.action_from_latent(obs["policy"], teacher_z)
                mixed = torch.rand(env.num_envs, device=device) < beta
                action = torch.where(mixed[:, None], teacher_action, student_action)
                memory["depth"].append(obs["depth"].clone())
                memory["fresh"].append(fresh.clone())
                memory["policy"].append(obs["policy"].clone())
                memory["command"].append(env.command_manager.get_command("base_velocity").clone())
                memory["teacher_z"].append(teacher_z.clone())
                memory["teacher_action"].append(teacher_action.clone())
                obs, _, terminated, truncated, extras = env.step(action)
                done = (terminated | truncated).bool()
                memory["done"].append(done.clone())
                if state is not None:
                    state[done] = 0
                    ema_state[done] = 0
                for name, value in extras.get("log", {}).items():
                    if (name.startswith("Curriculum/terrain_level/") or
                            "obstacle_clear_rate" in name):
                        metrics[name] = value.item() if hasattr(value, "item") else float(value)
        batch = {key: torch.stack(values) for key, values in memory.items()}
        resets = torch.zeros_like(batch["done"])
        resets[1:] = batch["done"][:-1]
        context, _ = model.encode_sequence(batch["depth"], batch["fresh"], resets, h0)
        student_z = model.terrain_latent(context)
        student_action = teacher.action_from_latent(batch["policy"], student_z)
        latent_loss, action_loss = distillation_losses(
            student_z, batch["teacher_z"], student_action, batch["teacher_action"], mean, std,
        )
        jepa_loss = context.new_zeros(())
        coverage = 0.0
        if train.method in ("gru_jepa", "gru_copy") and train.lambda_j:
            with torch.no_grad():
                target_context, _ = model.encode_sequence(
                    batch["depth"], batch["fresh"], resets, h0_ema, target=True,
                )
            mask = jepa_valid_mask(batch["done"], batch["command"],
                                   horizon_steps, train.command_tolerance)
            pred = model.predict(context[:-horizon_steps], batch["command"][:-horizon_steps])
            jepa_loss = masked_future_mse(pred, target_context[horizon_steps:], mask)
            coverage = mask.float().mean().item()
        total = train.lambda_z * latent_loss + train.lambda_a * action_loss + train.lambda_j * jepa_loss
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), train.max_grad_norm)
        optimizer.step()
        model.update_ema(train.ema_tau)
        state = None if state is None else state.detach()
        ema_state = None if ema_state is None else ema_state.detach()

        logs = {"Loss/latent": latent_loss.item(), "Loss/action": action_loss.item(),
                "Loss/jepa": jepa_loss.item(), "Loss/total": total.item(),
                "Diagnosis/action_abs_error": (student_action.detach() - batch["teacher_action"]).abs().mean().item(),
                "Diagnosis/z_abs_error": (student_z.detach() - batch["teacher_z"]).abs().mean().item(),
                "Diagnosis/latent_std": student_z.detach().std(dim=(0, 1)).mean().item(),
                "Diagnosis/jepa_mask_coverage": coverage, "DAgger/beta": beta, **metrics}
        logs["Perf/iteration_s"] = time.perf_counter() - iteration_start
        for name, value in logs.items():
            writer.add_scalar(name, value, iteration)
        print(f"STUDENT_ITER {iteration + 1}/{train.max_iterations} "
              f"method={train.method} loss={total.item():.5f} "
              f"z={latent_loss.item():.5f} action={action_loss.item():.5f} "
              f"jepa={jepa_loss.item():.5f} mask={coverage:.3f} "
              f"iteration_s={logs['Perf/iteration_s']:.2f}", flush=True)
        if (iteration + 1) % train.save_interval == 0 or iteration + 1 == train.max_iterations:
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "iteration": iteration + 1, "student_cfg": train.to_dict(),
                        "teacher_state_dict": teacher.actor.state_dict(),
                        "teacher_checkpoint": str(Path(args.teacher_checkpoint).resolve()),
                        "teacher_stats": str(Path(args.teacher_stats).resolve())},
                       output / f"model_{iteration + 1}.pt")
    if any(not torch.equal(value, teacher_initial[name]) for name, value in teacher.state_dict().items()):
        raise AssertionError("student 업데이트 중 teacher 가중치가 변경됐다")
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
