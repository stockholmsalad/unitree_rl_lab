# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""rsl_rl 5.0.1 RunnerCfg — PointCloudRNNModel actor/critic (GRU + PointNet) + obs_groups.

class_name 은 resolve_callable 의 "모듈경로:클래스" 형식으로 지정 → monkey-patch 불필요.
recurrent(GRU) 이므로 num_steps_per_env=32 (working 계단 실험과 동일).
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
    RslRlRNNModelCfg,
)

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
    # 2026-08-31 — ẑ_e(t+k) 를 정책 입력에 포함(예측기를 배포 경로 안으로).
    # cond_command/action 과 동시 사용 불가(추론 시 미래 행동·명령이 없음 → 모델이 막는다).
    predictor_in_policy: bool = False


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


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3b — DAgger 증류 RunnerCfg (2026-08-18)
#
# rsl_rl 5.0.1 `Distillation` = 정확히 DAgger: 환경을 student 행동으로 굴리고(act 에서
# stochastic_output=True), 같은 관측으로 teacher 를 질의해 privileged_actions 로 저장, 손실은
# MSE(student(obs), teacher_actions). gradient_length = truncated BPTT 창.
#
# 비교 축 = **student 인코더 초기화**(jepa_v1 / recon_v1 / "" ) — 나머지 전부 동일.
# 동결 여부는 freeze_encoder 로 별도 축(DeFM 의 frozen vs finetuned 대조에 대응).
# ═════════════════════════════════════════════════════════════════════════════
@configclass
class TeacherMLPCfg(RslRlMLPModelCfg):
    """Phase 1 teacher(`BasePPORunnerCfg.policy`) actor 와 **동일 구조** — 체크포인트
    `actor_state_dict` 를 그대로 로드하므로 hidden_dims·activation 이 어긋나면 안 된다."""

    hidden_dims: list[int] = [512, 256, 128]
    activation: str = "elu"
    obs_normalization: bool = False          # teacher 학습 시 empirical_normalization=False
    stochastic: bool = True
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    state_dependent_std: bool = False


@configclass
class JELocoDistillRunnerCfg(RslRlDistillationRunnerCfg):
    # ── 롤아웃 길이는 예측 지평이 정한다 (2026-08-31) ──────────────────────────
    # JEPA 손실은 저장된 롤아웃 안에서 (t, t+k) 쌍을 만든다. num_steps_per_env ≤ k 면 쌍이
    # 하나도 없어 손실이 조용히 0 이 된다. k=100 → 128 스텝이면 28 개 시점이 예측 출발점이 된다.
    # **환경 스텝 예산은 유지한다**: 32×8000 = 128×2000. 최적화 스텝 수도 T/gradient_length 라
    # 함께 비례해 거의 그대로다. 늘어나는 건 storage 메모리(런당 ~+0.5GB)뿐.
    num_steps_per_env = 128
    max_iterations = 2000
    save_interval = 50
    experiment_name = "je_loco_distill"
    empirical_normalization = False

    # ── 학습 중 관측 노후화 (예측기에 실제 할 일을 준다) ──────────────────────
    # 관측이 항상 신선하면 ẑ_e(t+k) 는 z_e(t) 에서 사소하게 유도되어 정책이 무시한다.
    # 근거: 어려운 지형 freeze 1.0 = −35pp 로, 지형 정보를 통째로 지운 blind(−26pp)보다
    # 9pp 더 해롭다. 노후화가 이 과제에서 가장 비싼 고장이다. 0 = 끔(구 동작).
    train_staleness_max: int = 25    # d ~ U[0,25] 스텝(0.5s), 에피소드 내 상수

    # ── 표현 보조손실 (증류 내내 유지 = 지속 개입) ───────────────────────────
    # 조건별로 하나만 켠다: jepa(자기지도) / recon(특권) / 둘 다 0(대조군).
    lambda_vel: float = 0.0          # 증류에선 critic 관측이 없어 기본 OFF
    vel_num_chunks: int = 4
    vel_learning_rate: float = 1.0e-3

    lambda_recon: float = 0.0
    recon_num_chunks: int = 4
    recon_learning_rate: float = 1.0e-3

    lambda_jepa: float = 0.0
    jepa_k: int = 100                # Phase 2 지평 스윕 실측 최적(skill 0.584 @k=100 = 2.0s)
    ema_tau: float = 0.996
    jepa_num_chunks: int = 8         # T=128 이라 청크를 늘려 backward peak 억제
    jepa_learning_rate: float = 1.0e-3
    lambda_var: float = 1.0
    lambda_cov: float = 0.04
    var_gamma: float = 1.0
    jepa_residual: bool = True       # Δz 예측(copy-baseline 과 정렬) — skill 해석이 직접적

    # student 는 pc+proprio(배포 가능), teacher 는 privileged heightmap 232.
    obs_groups = {
        "student": ["policy", "pointcloud"],
        "teacher": ["teacher"],
    }

    # 학생 = 기존 PointCloudRNNModel 그대로(GRU+PointNet). repr_head="none" — 증류에선 표현
    # 보조손실을 쓰지 않는다(행동 감독이 주 신호이고, 인코더 초기화만이 비교 변수여야 함).
    # init_noise_std 0.1: 롤아웃 탐색용 소량 행동 노이즈. 문헌(Parkour in the Wild)이 증류 중
    # action noise 를 권장 — 이후 RL fine-tune 안정성에도 기여.
    # predictor_in_policy=True 는 **세 조건 공통**이다. 정책 입력이 [z_e(o), z_p, ẑ_e(o+k)] 가
    # 되고 예측기가 배포 시에도 살아 있어, 행동손실이 직접 통과한다 → 사전학습 초기화처럼
    # 씻겨나갈 수 없다. 구조·파라미터 수가 셋 다 같으므로 유일한 변수는 **이 예측기 MLP 를
    # 무엇이 학습시키느냐**뿐이다(jepa 자기지도 / recon 특권 / 없음).
    student: PCModelCfg = PCModelCfg(
        hidden_dims=[256, 128], activation="elu", obs_normalization=False,
        stochastic=True, init_noise_std=0.1, noise_std_type="scalar", state_dependent_std=False,
        repr_head="none",
        predictor_in_policy=True,
        pretrained_encoder="", freeze_encoder=False,
    )
    teacher: TeacherMLPCfg = TeacherMLPCfg()

    algorithm: RslRlDistillationAlgorithmCfg = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=1,
        learning_rate=1.0e-3,
        gradient_length=15,     # Agarwal CoRL 2022 의 24-step 언롤과 같은 자리의 knob
        max_grad_norm=1.0,
        loss_type="mse",
        optimizer="adam",
    )
