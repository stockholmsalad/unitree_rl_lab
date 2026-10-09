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
parser.add_argument("--num_mini_batches", type=int)
parser.add_argument("--num_learning_epochs", type=int)
parser.add_argument("--jepa_target_mode", choices=("future", "present_from_past"))
parser.add_argument("--condition_source", choices=("command", "realized_displacement"))
parser.add_argument("--jepa_target", choices=("context", "frame_embedding"))
parser.add_argument("--lambda_j", type=float)
parser.add_argument("--jepa_warmup_iterations", type=int)
parser.add_argument("--no_init_at_random_ep_len", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
simulation_app = AppLauncher(args).app

from pathlib import Path  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

import unitree_rl_lab.tasks  # noqa: E402, F401
from unitree_rl_lab.jepa_loco.agents.student_cfg import StudentTrainCfg  # noqa: E402
from unitree_rl_lab.jepa_loco.models.student_depth import FrozenCurrentTeacher, StudentDepthModel  # noqa: E402
from unitree_rl_lab.jepa_loco.models.student_batches import (  # noqa: E402
    env_batch_indices, initial_episode_lengths, jepa_pair_tensors, select_env_batch,
)
from unitree_rl_lab.jepa_loco.models.student_losses import (  # noqa: E402
    distillation_losses, jepa_fresh_pair_mask, jepa_valid_mask, normalized_jepa_with_copy,
)
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg  # noqa: E402


def main() -> None:
    overrides = {key: value for key in ("num_mini_batches", "num_learning_epochs",
                                         "jepa_target_mode", "condition_source", "jepa_target", "lambda_j",
                                         "jepa_warmup_iterations")
                 if (value := getattr(args, key)) is not None}
    if args.no_init_at_random_ep_len:
        overrides["init_at_random_ep_len"] = False
    train = StudentTrainCfg(method=args.method, num_envs=args.num_envs,
                            max_iterations=args.max_iterations, **overrides)
    task = "Unitree-Go2-JepaLoco-StudentDepth-EasyStart"
    env_cfg = parse_env_cfg(task, device=args.device, num_envs=train.num_envs)
    env_cfg.seed = args.seed
    horizon_steps = train.validate_training(env_cfg.sim.dt * env_cfg.decimation, env_cfg.depth_period_steps)
    torch.manual_seed(args.seed)
    env = gym.make(task, cfg=env_cfg).unwrapped
    obs, _ = env.reset()
    env.episode_length_buf = initial_episode_lengths(
        env.episode_length_buf, int(env.max_episode_length), train.init_at_random_ep_len,
    )
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
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        iteration_start = time.perf_counter()
        beta = train.dagger_beta(iteration)
        lambda_j = (train.effective_lambda_j(iteration)
                    if train.method in ("gru_jepa", "gru_copy") else 0.0)
        h0 = state.detach().clone()
        h0_ema = ema_state.detach().clone()
        memory = {key: [] for key in ("depth", "fresh", "policy", "command", "teacher_z",
                                       "teacher_action", "done")}
        if train.condition_source == "realized_displacement" and train.method in ("gru_jepa", "gru_copy"):
            memory["pose"] = []
        metrics = {}
        teacher_action_count = torch.zeros((), device=device)
        timeout_any = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        max_timeout_step_fraction = torch.zeros((), device=device)
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
                teacher_action_count += mixed.sum()
                action = torch.where(mixed[:, None], teacher_action, student_action)
                memory["depth"].append(obs["depth"].clone())
                memory["fresh"].append(fresh.clone())
                memory["policy"].append(obs["policy"].clone())
                memory["command"].append(env.command_manager.get_command("base_velocity").clone())
                if "pose" in memory:
                    robot = env.scene["robot"].data
                    yaw = math_utils.euler_xyz_from_quat(robot.root_quat_w)[2]
                    memory["pose"].append(torch.cat((robot.root_pos_w[:, :2], yaw[:, None]), dim=-1).clone())
                memory["teacher_z"].append(teacher_z.clone())
                memory["teacher_action"].append(teacher_action.clone())
                obs, _, terminated, truncated, extras = env.step(action)
                done = (terminated | truncated).bool()
                timeout_any |= truncated.bool()
                max_timeout_step_fraction = torch.maximum(max_timeout_step_fraction, truncated.float().mean())
                memory["done"].append(done.clone())
                state[done] = 0
                ema_state[done] = 0
                for name, value in extras.get("log", {}).items():
                    if (name.startswith("Curriculum/terrain_level/") or
                            "obstacle_clear_rate" in name):
                        metrics[name] = value.item() if hasattr(value, "item") else float(value)
        batch = {key: torch.stack(values) for key, values in memory.items()}
        resets = torch.zeros_like(batch["done"])
        resets[1:] = batch["done"][:-1]
        mask_all = None
        if train.method in ("gru_jepa", "gru_copy"):
            mask_all = jepa_valid_mask(batch["done"], batch["command"],
                                       horizon_steps, train.command_tolerance)
            # 두 target을 동일한 새 목표 프레임 시점에서 비교한다.
            mask_all = jepa_fresh_pair_mask(mask_all, batch["fresh"], horizon_steps)
        update_keys = ("Loss/latent", "Loss/action", "Loss/jepa", "Loss/total",
                       "Diagnosis/action_abs_error", "Diagnosis/z_abs_error", "Diagnosis/latent_std",
                       "Diagnosis/jepa_copy_loss", "Diagnosis/jepa_copy_online_loss",
                       "Diagnosis/jepa_target_var", "Diagnosis/jepa_online_context_var")
        sums = {key: 0.0 for key in update_keys}
        for _ in range(train.num_learning_epochs):
            env_batches = env_batch_indices(env.num_envs, train.num_mini_batches, device, shuffle=True)
            for ids in env_batches:
                mini, mini_resets, mini_h0, mini_h0_ema, mini_mask = select_env_batch(
                    batch, resets, h0, h0_ema, ids, mask_all,
                )
                context, _ = model.encode_sequence(mini["depth"], mini["fresh"], mini_resets, mini_h0)
                student_z = model.terrain_latent(context)
                student_action = teacher.action_from_latent(mini["policy"], student_z)
                latent_loss, action_loss = distillation_losses(
                    student_z, mini["teacher_z"], student_action, mini["teacher_action"], mean, std,
                )
                jepa_loss = context.new_zeros(())
                copy_loss = context.new_zeros(())
                copy_online_loss = context.new_zeros(())
                target_var = context.new_zeros(())
                online_context_var = context.new_zeros(())
                if mini_mask is not None:
                    with torch.no_grad():
                        if train.jepa_target == "context":
                            ema_context, _ = model.encode_sequence(
                                mini["depth"], mini["fresh"], mini_resets, mini_h0_ema, target=True,
                            )
                        else:
                            valid_frames = torch.zeros_like(mini["fresh"])
                            valid_frames[:-horizon_steps] |= mini_mask
                            valid_frames[horizon_steps:] |= mini_mask
                            ema_context = model.encode_frame_targets(
                                mini["depth"], valid_frames, train.jepa_frame_batch_size,
                            )
                    source, target, condition = jepa_pair_tensors(
                        context, ema_context, mini["command"], mini.get("pose"), horizon_steps,
                        train.jepa_target_mode, train.condition_source,
                    )
                    source_ema = ema_context[:-horizon_steps]
                    pred = model.predict(source, condition)
                    jepa_loss, copy_loss, copy_online_loss, target_var, online_context_var = normalized_jepa_with_copy(
                        pred, source, source_ema, target, mini_mask,
                        train.jepa_variance_floor,
                    )
                total = train.lambda_z * latent_loss + train.lambda_a * action_loss + lambda_j * jepa_loss
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad), train.max_grad_norm,
                )
                optimizer.step()
                # EMA target이 방금 변경된 online encoder를 바로 따라가도록 매 step 갱신한다.
                model.update_ema(train.ema_tau)
                values = (latent_loss.item(), action_loss.item(), jepa_loss.item(), total.item(),
                          (student_action.detach() - mini["teacher_action"]).abs().mean().item(),
                          (student_z.detach() - mini["teacher_z"]).abs().mean().item(),
                          student_z.detach().std(dim=(0, 1)).mean().item(),
                          copy_loss.item(), copy_online_loss.item(), target_var.item(),
                          online_context_var.item())
                for key, value in zip(update_keys, values):
                    sums[key] += value
        state = state.detach()
        ema_state = ema_state.detach()
        update_count = train.num_learning_epochs * train.num_mini_batches
        logs = {key: value / update_count for key, value in sums.items()}
        logs["Diagnosis/jepa_pred_over_copy"] = (
            logs["Loss/jepa"] / logs["Diagnosis/jepa_copy_loss"]
            if logs["Diagnosis/jepa_copy_loss"] > 0 else 1.0
        )
        logs["Diagnosis/jepa_weighted_over_latent"] = (
            lambda_j * logs["Loss/jepa"] / logs["Loss/latent"]
            if logs["Loss/latent"] > 0 else 0.0
        )
        logs.update({"Diagnosis/jepa_mask_coverage": 0.0 if mask_all is None else mask_all.float().mean().item(),
                     "Loss/lambda_j": lambda_j,
                     "Rollout/timeout_env_fraction": timeout_any.float().mean().item(),
                     "Rollout/max_timeout_step_fraction": max_timeout_step_fraction.item(),
                     "DAgger/beta": beta,
                     "DAgger/teacher_action_rate": teacher_action_count.item() / (env.num_envs * train.rollout_steps),
                     "Perf/optimizer_steps": update_count, **metrics})
        if device.type == "cuda":
            logs["Perf/gpu_peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            logs["Perf/gpu_peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
        logs["Perf/iteration_s"] = time.perf_counter() - iteration_start
        for name, value in logs.items():
            writer.add_scalar(name, value, iteration)
        print(f"STUDENT_ITER {iteration + 1}/{train.max_iterations} "
              f"method={train.method} loss={logs['Loss/total']:.5f} "
              f"z={logs['Loss/latent']:.5f} action={logs['Loss/action']:.5f} "
              f"jepa={logs['Loss/jepa']:.5f} mask={logs['Diagnosis/jepa_mask_coverage']:.3f} "
              f"copy={logs['Diagnosis/jepa_copy_loss']:.5f} "
              f"copy_online={logs['Diagnosis/jepa_copy_online_loss']:.5f} "
              f"pred_over_copy={logs['Diagnosis/jepa_pred_over_copy']:.3f} "
              f"target_var={logs['Diagnosis/jepa_target_var']:.6f} "
              f"online_var={logs['Diagnosis/jepa_online_context_var']:.6f} "
              f"lambda_j={lambda_j:.4f} "
              f"weighted_over_latent={logs['Diagnosis/jepa_weighted_over_latent']:.3f} "
              f"timeout_env_frac={logs['Rollout/timeout_env_fraction']:.3f} "
              f"max_timeout_step_frac={logs['Rollout/max_timeout_step_fraction']:.3f} "
              f"teacher_rate={logs['DAgger/teacher_action_rate']:.3f} updates={update_count} "
              f"peak_gib={logs.get('Perf/gpu_peak_allocated_gib', 0.0):.2f} "
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
