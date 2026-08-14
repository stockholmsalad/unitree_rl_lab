# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""rsl_rl 5.0.1 RunnerCfg — PointCloudRNNModel actor/critic (GRU + PointNet) + obs_groups.

class_name 은 resolve_callable 의 "모듈경로:클래스" 형식으로 지정 → monkey-patch 불필요.
recurrent(GRU) 이므로 num_steps_per_env=32 (working 계단 실험과 동일).
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg, RslRlRNNModelCfg

_PC_MODEL = "unitree_rl_lab.je_loco.rsl_rl_pc.model:PointCloudRNNModel"


@configclass
class PCModelCfg(RslRlRNNModelCfg):
    """RslRlRNNModelCfg + PointNet 파라미터. class_name 은 커스텀 모델 경로."""

    class_name: str = _PC_MODEL
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    # 외수용: obs["pointcloud"] → PointNet(pc_out_dim)=z_e
    pc_out_dim: int = 64
    pc_group: str = "pointcloud"
    # 고유수용(actor 공통 백본): obs["policy"](45×H) → proprio_encoder(zp_dim)=z_p, v̂=vel_decoder(z_p)
    proprio_group: str = "policy"
    zp_dim: int = 32
    proprio_hidden_dim: int = 128
    vel_hidden_dim: int = 128
    # 표현 헤드(z_e 에만, 교체 지점): "none" | "recon"(Head A) | "jepa"(Head B). actor 만 설정.
    repr_head: str = "none"
    recon_hidden_dim: int = 128
    # Head B VICReg projector(expander): 정규화를 z_e 대신 projector 출력에 걸어 z_e 백화 방지.
    # use_projector=False 면 구 동작(VICReg 을 z_e 에 직접) — projector 효과 ablation 용.
    use_projector: bool = True
    proj_hidden_dim: int = 128
    proj_dim: int = 128
    # Head B predictor conditioning (actor 전용, 단일 소스): 미래 명령/행동을 예측 조건으로.
    # cond_dim(predictor 입력 확장 = 3·command + 12·action)은 모델이 이 플래그로 내부 계산.
    # runner 도 actor 에서 이 플래그를 읽으므로 불일치 불가능.
    jepa_cond_command: bool = False
    jepa_cond_action: bool = False
    # Phase 3 — 사전학습 인코더(pretrain_repr.py 산출) 로드·동결. "" = scratch(랜덤 초기화).
    # 3-way 비교: jepa_v1 / recon_v1 / "" — 유일 변수 = 인코더 초기화. repr_head 는 "none".
    pretrained_encoder: str = ""
    freeze_encoder: bool = True


@configclass
class JELocoPCPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 32          # recurrent → 긴 시퀀스
    max_iterations = 50000
    save_interval = 200
    experiment_name = "je_loco_pc"
    empirical_normalization = False

    # ── velocity 추정 aux 손실 (두 헤드 공통, z_p 경로) ──
    lambda_vel: float = 0.5         # L = L_RL + lambda_vel·MSE(v̂, v). v̂=vel_decoder(z_p).
    vel_num_chunks: int = 4         # backward 를 env 분할(메모리 절감)
    vel_learning_rate: float = 1.0e-3

    # ── 표현 헤드 A(재구성) 손실 ──
    lambda_recon: float = 1.0       # L_repr = lambda_recon·MSE(recon, GT height). 0 이면 off.
    recon_num_chunks: int = 4
    recon_learning_rate: float = 1.0e-3

    # ── 표현 헤드 B(JEPA) 손실 ──
    lambda_jepa: float = 1.0        # L_repr = lambda_jepa·MSE(ẑ_e(t+k), sg·target)
    jepa_k: int = 5                 # 예측 지평 (5~10)
    ema_tau: float = 0.996          # target encoder EMA 계수
    jepa_num_chunks: int = 4
    jepa_learning_rate: float = 1.0e-3
    # VICReg 붕괴 방지(z_e collapse 시 jepa loss→0). EMA 만으론 부족해 추가.
    lambda_var: float = 1.0         # 분산 hinge(각 dim std≥var_gamma)
    lambda_cov: float = 0.04        # 공분산 off-diag 억제(decorrelate)
    var_gamma: float = 1.0          # 목표 std
    jepa_residual: bool = False     # True=Δz(t+k)−z(t) 예측(copy-baseline 과 정렬). False=절대 z(t+k)
    # (predictor conditioning 은 actor(PCModelCfg)에서만 설정 — runner 가 actor 에서 읽음)

    # actor: policy(45×5=225→z_p) + pointcloud(→z_e) / critic: critic(60) + pointcloud(→z_e)
    obs_groups = {
        "actor": ["policy", "pointcloud"],
        "critic": ["critic", "pointcloud"],
    }

    # A/B 통제 비교: **repr_head 만** "recon"↔"jepa" 교체. proprio_encoder(z_p)+vel_decoder+
    # pc_encoder 백본은 두 헤드 공통(동일하게 shaping) → 헤드가 유일한 변수.
    actor: PCModelCfg = PCModelCfg(
        hidden_dims=[256, 128], activation="elu", obs_normalization=False,
        stochastic=True, init_noise_std=1.0, noise_std_type="scalar", state_dependent_std=False,
        repr_head="recon",
    )
    critic: PCModelCfg = PCModelCfg(
        hidden_dims=[256, 128], activation="elu", obs_normalization=False,
        stochastic=False, init_noise_std=1.0, noise_std_type="scalar", state_dependent_std=False,
        repr_head="none",
    )

    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=8,   # recurrent BPTT backward peak 메모리 절반(2048/8=256 env/mb) — OOM 방지
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
