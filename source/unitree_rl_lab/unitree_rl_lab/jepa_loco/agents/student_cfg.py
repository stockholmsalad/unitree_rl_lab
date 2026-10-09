"""Phase 3 Current teacher 증류의 수치·방법 설정."""

from isaaclab.utils import configclass


@configclass
class StudentTrainCfg:
    method: str = "gru_jepa"
    num_envs: int = 64
    rollout_steps: int = 100
    num_mini_batches: int = 4
    num_learning_epochs: int = 2
    max_iterations: int = 3000
    save_interval: int = 100
    learning_rate: float = 3.0e-4
    max_grad_norm: float = 1.0
    lambda_z: float = 1.0
    lambda_a: float = 1.0
    lambda_j: float = 0.1
    jepa_horizon_s: float = 0.5
    jepa_target_mode: str = "future"
    condition_source: str = "command"
    ema_tau: float = 0.996
    command_tolerance: float = 0.01
    beta_initial: float = 1.0
    beta_final: float = 0.0
    beta_decay_iterations: int = 500
    use_frozen_teacher_head: bool = True
    train_new_head: bool = False  # 선택 ablation 자리, 기본 경로는 고정 head
    enable_future_terrain_distill: bool = False  # 선택 ablation 자리
    image_hw: tuple[int, int] = (64, 112)
    feature_dim: int = 128
    context_dim: int = 128
    terrain_dim: int = 32
    terrain_hidden_dim: int = 128
    predictor_hidden_dim: int = 128
    cnn_channels: tuple[int, ...] = (16, 32, 64)
    cnn_kernels: tuple[int, ...] = (5, 3, 3)
    cnn_strides: tuple[int, ...] = (2, 2, 2)

    def validate_training(self, control_dt: float, depth_period_steps: int) -> int:
        if self.method not in ("no_memory", "gru", "gru_jepa", "gru_copy"):
            raise ValueError("지원하지 않는 student 비교군")
        if self.train_new_head or not self.use_frozen_teacher_head or self.enable_future_terrain_distill:
            raise NotImplementedError("새 head·future terrain은 확정 후 별도 ablation으로 구현한다")
        if min(self.lambda_z, self.lambda_a, self.lambda_j) < 0:
            raise ValueError("손실 계수는 0 이상이어야 한다")
        if self.rollout_steps < 2 or self.num_envs < 1 or self.max_iterations < 1:
            raise ValueError("rollout/env/iteration 크기가 올바르지 않다")
        if self.num_learning_epochs < 1 or self.num_mini_batches < 1 or self.num_envs % self.num_mini_batches:
            raise ValueError("env 수는 미니배치 수로 나누어떨어져야 하고 epoch은 양수여야 한다")
        if self.jepa_target_mode not in ("future", "present_from_past"):
            raise ValueError("JEPA 목표 시점이 올바르지 않다")
        if self.condition_source not in ("command", "realized_displacement"):
            raise ValueError("JEPA predictor 조건이 올바르지 않다")
        horizon_steps = round(self.jepa_horizon_s / control_dt)
        if abs(horizon_steps * control_dt - self.jepa_horizon_s) > 1.0e-6 or horizon_steps % depth_period_steps:
            raise ValueError("JEPA horizon은 depth 주기의 정수배여야 한다")
        if horizon_steps >= self.rollout_steps:
            raise ValueError("JEPA horizon이 rollout보다 길다")
        return horizon_steps

    def dagger_beta(self, iteration: int) -> float:
        if self.beta_decay_iterations <= 0:
            return self.beta_final
        progress = min(max(iteration, 0) / self.beta_decay_iterations, 1.0)
        return self.beta_initial + (self.beta_final - self.beta_initial) * progress
