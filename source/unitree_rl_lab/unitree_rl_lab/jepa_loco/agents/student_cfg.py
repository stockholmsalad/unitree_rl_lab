"""Phase 3 Current teacher 증류의 수치·방법 설정."""

import warnings

from isaaclab.utils import configclass


@configclass
class StudentTrainCfg:
    method: str = "gru_jepa"
    backbone: str = "gru"
    mamba_intermediate_size: int = 256
    mamba_state_size: int = 16
    mamba_conv_kernel: int = 4
    mamba_dt_rank: int | str = "auto"
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
    lambda_r: float = 0.1
    recon_warmup_iterations: int = 50
    recon_variance_floor: float = 0.05
    # 높이맵은 실제 지형 목표라 EMA 붕괴가 없다. 초기 평지 batch 분산이 ~1e-4라 0.05 하한은 손실을 수백 배 줄인다.
    recon_height_variance_floor: float = 1.0e-4
    recon_depth_weight: float = 1.0
    recon_mask_weight: float = 1.0
    recon_hidden_dim: int = 64
    recon_seed_channels: int = 4
    recon_height_hidden_dim: int = 256
    recon_visibility_tolerance_m: float = 0.10
    jepa_warmup_iterations: int = 50
    init_at_random_ep_len: bool = True
    jepa_horizon_s: float = 0.5
    jepa_target: str = "context"  # frame_embedding 후보는 스모크 후 기본값 확정
    jepa_variance_floor: float = 1.0e-6
    jepa_frame_batch_size: int = 1024
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
    heightmap_dim: int = 187
    terrain_hidden_dim: int = 128
    predictor_hidden_dim: int = 128
    proprio_input: bool = False
    proprio_dim: int = 45
    proprio_hidden_dims: tuple[int, ...] = (64, 64)
    proprio_window: int = 5
    cnn_channels: tuple[int, ...] = (16, 32, 64)
    cnn_kernels: tuple[int, ...] = (5, 3, 3)
    cnn_strides: tuple[int, ...] = (2, 2, 2)

    def validate_training(self, control_dt: float, depth_period_steps: int) -> int:
        if self.method not in ("no_memory", "gru", "gru_jepa", "gru_copy",
                               "gru_recon_depth", "gru_recon_height"):
            raise ValueError("지원하지 않는 student 비교군")
        if self.backbone not in ("gru", "mamba") or (self.method == "no_memory" and self.backbone == "mamba"):
            raise ValueError("backbone은 gru/mamba이고 no_memory에는 mamba를 쓸 수 없다")
        if min(self.mamba_intermediate_size, self.mamba_state_size, self.mamba_conv_kernel) < 1 or (
            self.mamba_dt_rank != "auto" and
            (not isinstance(self.mamba_dt_rank, int) or self.mamba_dt_rank < 1)
        ):
            raise ValueError("Mamba 크기와 dt_rank가 올바르지 않다")
        if self.train_new_head or not self.use_frozen_teacher_head or self.enable_future_terrain_distill:
            raise NotImplementedError("새 head·future terrain은 확정 후 별도 ablation으로 구현한다")
        if min(self.lambda_z, self.lambda_a, self.lambda_j, self.lambda_r) < 0:
            raise ValueError("손실 계수는 0 이상이어야 한다")
        if (self.recon_warmup_iterations < 0 or min(self.recon_variance_floor, self.recon_height_variance_floor) <= 0 or
                min(self.recon_depth_weight, self.recon_mask_weight) < 0 or
                min(self.recon_hidden_dim, self.recon_seed_channels, self.recon_height_hidden_dim) < 1 or
                self.recon_visibility_tolerance_m < 0):
            raise ValueError("복원 보조 손실 설정이 올바르지 않다")
        if self.jepa_warmup_iterations < 0:
            raise ValueError("JEPA warmup iteration은 0 이상이어야 한다")
        if self.jepa_target not in ("context", "frame_embedding"):
            raise ValueError("JEPA target은 context 또는 frame_embedding이어야 한다")
        if self.jepa_variance_floor <= 0 or self.jepa_frame_batch_size < 1:
            raise ValueError("JEPA 분산 하한과 frame batch 크기는 양수여야 한다")
        if self.rollout_steps < 2 or self.num_envs < 1 or self.max_iterations < 1:
            raise ValueError("rollout/env/iteration 크기가 올바르지 않다")
        if self.num_learning_epochs < 1 or self.num_mini_batches < 1 or self.num_envs % self.num_mini_batches:
            raise ValueError("env 수는 미니배치 수로 나누어떨어져야 하고 epoch은 양수여야 한다")
        if self.jepa_target_mode not in ("future", "present_from_past"):
            raise ValueError("JEPA 목표 시점이 올바르지 않다")
        if self.condition_source not in ("command", "realized_displacement"):
            raise ValueError("JEPA predictor 조건이 올바르지 않다")
        if self.proprio_input and (self.proprio_dim != 45 or not self.proprio_hidden_dims or
                                   any(width < 1 for width in self.proprio_hidden_dims)):
            raise ValueError("policy 관측은 45차원이고 proprio MLP 층은 양수여야 한다")
        if self.proprio_window < 1:
            raise ValueError("proprio_window는 1 이상이어야 한다")
        if self.proprio_input and self.proprio_window != depth_period_steps:
            warnings.warn("proprio_window와 depth_period_steps가 다르다; 설정을 유지한다", stacklevel=2)
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

    def effective_lambda_j(self, iteration: int) -> float:
        if self.jepa_warmup_iterations == 0:
            return self.lambda_j
        return self.lambda_j * min(max(iteration, 0) / self.jepa_warmup_iterations, 1.0)

    def effective_lambda_r(self, iteration: int) -> float:
        if self.recon_warmup_iterations == 0:
            return self.lambda_r
        return self.lambda_r * min(max(iteration, 0) / self.recon_warmup_iterations, 1.0)
