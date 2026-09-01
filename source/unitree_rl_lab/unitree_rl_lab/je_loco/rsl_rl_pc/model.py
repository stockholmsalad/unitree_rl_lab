# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""PointCloudRNNModel — rsl_rl 5.0.1 RNNModel + 논문 백본(z_e, z_p, GRU, 교체 표현헤드).

논문 아키텍처(두 헤드 공통 백본, 헤드만 교체 = 통제 비교):
  외수용:  point cloud → PointNet → z_e (64)
  고유수용: proprio(45×H=5=225 히스토리) → proprio encoder → z_p (32)
  정책:    GRU([z_e, z_p]) → Actor(PPO). critic 은 비대칭(privileged, z_p 없음).
  속도:    v̂ = vel_decoder(z_p)  (CENet, 두 헤드 공통)
  표현헤드(z_e 에만, 교체): "recon"(z_e→height) | "jepa"(z_e(t)→ẑ_e(t+k)) | "none"

rsl_rl 규약: 확장점은 get_latent 와 _get_obs_dim. proprio_encoder/vel_decoder/표현헤드는
actor(=proprio_group 를 관측에 가진 모델)만 생성. critic 은 privileged 관측을 raw 로 사용.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import RNNModel
from rsl_rl.modules.rnn import HiddenState


class PointCloudEncoder(nn.Module):
    """PointNet-lite: (M, P*4) → (M, out_dim). 점당 공유 MLP + **마스크드** max-pool (순서 불변).

    각 점 = [x, y, z, valid]. valid=0 인 점(무효 hit·결손)은 max-pool 에서 −inf 로 제외 →
    **어떤 좌표값도 max-pool 에 주입되지 않음** (측정 ④에서 확인된 sentinel-주입 max-pool hijack
    아티팩트 제거). 결손 = valid 채널 0 설정만으로 표현(mdp_pc), 좌표 무주입.
    """

    def __init__(self, num_points: int = 96, out_dim: int = 64) -> None:
        super().__init__()
        self.P = num_points
        self.out_dim = out_dim
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 32), nn.ELU(),
            nn.Linear(32, 64), nn.ELU(),
            nn.Linear(64, 128), nn.ELU(),
        )
        self.global_mlp = nn.Sequential(nn.Linear(128, out_dim), nn.ELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = x.shape[0]
        pc = x.reshape(m, self.P, 4)           # (M, P, 4) = [x,y,z,valid]
        pts = pc[..., :3]                       # (M, P, 3) 좌표
        valid = pc[..., 3] > 0.5               # (M, P)   유효 마스크(노이즈 ±0.02 견고)
        feat = self.point_mlp(pts)             # (M, P, 128)
        feat = feat.masked_fill(~valid.unsqueeze(-1), float("-inf"))   # 무효 점 max 제외
        glob = feat.max(dim=1)[0]              # (M, 128) 마스크드 max-pool
        # 전부 무효(100% 결손 = 카메라 실명)면 max=−inf → 0 벡터로 대체(NaN 방지). 정책은 z_p 로만 보행.
        none_valid = ~valid.any(dim=1)                                 # (M,)
        glob = torch.where(none_valid.unsqueeze(-1), torch.zeros_like(glob), glob)
        return self.global_mlp(glob)           # (M, out_dim)


class PointCloudRNNModel(RNNModel):
    """논문 백본: z_e(PointNet) + z_p(proprio enc) → GRU → policy; 교체 표현헤드(z_e)."""

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        pc_group: str = "pointcloud",
        pc_out_dim: int = 64,
        proprio_group: str = "policy",
        zp_dim: int = 32,
        proprio_hidden_dim: int = 128,
        vel_hidden_dim: int = 128,
        repr_head: str = "none",
        recon_hidden_dim: int = 128,
        height_map_group: str = "height_map",
        use_projector: bool = True,
        proj_hidden_dim: int = 128,
        proj_dim: int = 128,
        jepa_cond_command: bool = False,   # predictor 에 명령 c_t(3) 조건
        jepa_cond_action: bool = False,    # predictor 에 지평 평균행동(12) 조건
        predictor_in_policy: bool = False,  # ẑ_e(t+k) 를 정책 입력에 포함 (2026-08-31)
        pretrained_encoder: str = "",      # Phase 3: 사전학습 pc_encoder.pt 경로 ("" = scratch)
        freeze_encoder: bool = True,       # 사전학습 인코더 동결(RL gradient 차단)
        **kwargs,
    ) -> None:
        # _get_obs_dim(super 초기화 중 호출)이 참조 → super 전에 세팅
        self._pc_group = pc_group
        self._pc_out_dim = pc_out_dim
        self._pc_num_points = obs[pc_group].shape[-1] // 4   # 각 점 [x,y,z,valid] (마스킹)
        self._proprio_group = proprio_group
        self._zp_dim = zp_dim
        # actor 만 proprio 를 z_p 로 인코딩(critic 은 privileged 관측을 raw 사용)
        self._encode_proprio = proprio_group in obs_groups[obs_set]
        # 예측기를 정책 입력에 포함할지 — _get_obs_dim 이 참조하므로 super 전에 세팅
        self._predictor_in_policy = predictor_in_policy and self._encode_proprio

        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)

        self.pc_encoder = PointCloudEncoder(self._pc_num_points, pc_out_dim)

        # ── Phase 3: 사전학습 인코더 로드 + 동결 (jepa/recon/scratch 3-way 비교의 유일 변수) ──
        # pretrain_repr.py 산출물(pc_encoder.pt)을 그대로 로드. freeze 시 RL gradient 가
        # 인코더를 안 건드림 → "사전학습 표현이 정책에 얼마나 유용한가"만 측정됨.
        self._encoder_frozen = False
        if pretrained_encoder:
            sd = torch.load(pretrained_encoder, map_location="cpu", weights_only=False)
            self.pc_encoder.load_state_dict(sd)
            print(f"[JELoco] pretrained pc_encoder 로드: {pretrained_encoder}")
            if freeze_encoder:
                for p in self.pc_encoder.parameters():
                    p.requires_grad_(False)
                self.pc_encoder.eval()
                self._encoder_frozen = True
                print("[JELoco] pc_encoder FROZEN (RL gradient 차단)")

        # ── 고유수용 백본 (actor 공통, 두 헤드 동일) ──────────────────────────
        # proprio(H=5 히스토리) → z_p, 그리고 v̂ = vel_decoder(z_p) (CENet).
        # z_p 는 GRU 입력이자 jepa predictor 의 조건. velocity 로만 shaping(두 헤드 동일).
        if self._encode_proprio:
            proprio_dim = obs[proprio_group].shape[-1]          # 45×H (flatten 히스토리)
            self.proprio_encoder = nn.Sequential(
                nn.Linear(proprio_dim, proprio_hidden_dim), nn.ELU(),
                nn.Linear(proprio_hidden_dim, zp_dim), nn.ELU(),
            )
            self.vel_decoder = nn.Sequential(
                nn.Linear(zp_dim, vel_hidden_dim), nn.ELU(),
                nn.Linear(vel_hidden_dim, 3),
            )

        # ── 표현 헤드 (z_e 에만, 교체 지점 = 통제 비교의 유일한 변수) ──────────
        self.repr_head = repr_head
        if repr_head == "recon":
            hm = obs[height_map_group].shape[-1] if height_map_group in obs.keys() else 96
            self._height_map_dim = hm
            self.recon_decoder = nn.Sequential(
                nn.Linear(pc_out_dim, recon_hidden_dim), nn.ELU(),
                nn.Linear(recon_hidden_dim, hm),
            )
        # ── 예측기(보조 MLP) ────────────────────────────────────────────────
        # jepa 헤드가 쓰거나, 정책 입력에 ẑ 를 넣을 때 생성. 세 조건 모두 predictor_in_policy=True
        # 로 두면 **구조와 파라미터 수가 동일**해지고, 다른 것은 이 MLP 를 무엇이 학습시키느냐뿐이다
        # (jepa: 행동손실+JEPA / recon: 행동손실만 + z_e 에 특권 재구성 / none: 행동손실만).
        if repr_head == "jepa" or self._predictor_in_policy:
            # predictor 입력 = [z_e(t), z_p(t), (conditioning: command c_t · 지평평균 action)].
            # 미래 행동/명령을 조건으로 → 미래 관측 z_e(t+k) 예측 가능성↑ (z_p 에 명령은 이미 암묵
            # 포함되나 raw 로 추가; action 은 z_p 에 없는 실제 변위 신호). runner 가 이 플래그를 읽어
            # 같은 조건을 구성 = 단일 소스(불일치 원천 차단).
            self._cond_command = jepa_cond_command
            self._cond_action = jepa_cond_action
            self._cond_dim = (3 if jepa_cond_command else 0) + (12 if jepa_cond_action else 0)
            # 정책 입력용 예측기는 추론 시 미래 행동/명령을 조건으로 쓸 수 없다(아직 없는 값).
            # 조용히 어긋나면 학습·추론이 다른 입력을 받으므로 여기서 막는다.
            if self._predictor_in_policy and self._cond_dim != 0:
                raise ValueError(
                    "predictor_in_policy=True 에서는 jepa_cond_command/action 을 쓸 수 없다 "
                    "(추론 시 미래 행동·명령이 없음). 명령은 이미 proprio 관측을 통해 z_p 에 들어 있다.")
            self.jepa_predictor = nn.Sequential(
                nn.Linear(pc_out_dim + zp_dim + self._cond_dim, recon_hidden_dim), nn.ELU(),
                nn.Linear(recon_hidden_dim, pc_out_dim),
            )

        if repr_head == "jepa":
            self.target_pc_encoder = copy.deepcopy(self.pc_encoder)   # EMA target + stop-grad
            for p in self.target_pc_encoder.parameters():
                p.requires_grad_(False)
            # VICReg projector(expander) — anti-collapse 정규화를 z_e 가 아니라 이 출력에 건다.
            # 이유(2026-07 진단): VICReg 을 z_e 에 직접 걸면 z_e 를 등방·std=1 로 백화 → 시간 구조 파괴
            # (1스텝에 79% 탈상관) + 정책 입력 스케일이 학습 중 70배 드리프트. SSL 표준(VICReg/BYOL)은
            # projection 위에 정규화를 걸어 표현 z_e 를 자유롭게 둔다. z_e 붕괴는 projector 역전파로
            # 여전히 방지된다(z_e→상수면 proj 도 상수→VICReg 위반). use_projector=False 로 ablation.
            self.use_projector = use_projector
            if use_projector:
                self.vic_projector = nn.Sequential(
                    nn.Linear(pc_out_dim, proj_hidden_dim), nn.ELU(),
                    nn.Linear(proj_hidden_dim, proj_dim),
                )

    # ── rsl_rl 확장점 ────────────────────────────────────────────────────────
    def _get_obs_dim(self, obs, obs_groups, obs_set):
        """pc→pc_out_dim, proprio(actor)→zp_dim, 그 외 raw 로 GRU 입력 차원 계산."""
        active = obs_groups[obs_set]
        dim = 0
        for g in active:
            if g == self._pc_group:
                dim += self._pc_out_dim
            elif g == self._proprio_group and self._encode_proprio:
                dim += self._zp_dim
            else:
                dim += obs[g].shape[-1]
        if self._predictor_in_policy:
            dim += self._pc_out_dim        # ẑ_e(t+k) 를 GRU 입력에 추가
        return active, dim

    def _enc_group(self, g: str, x: torch.Tensor) -> torch.Tensor:
        """그룹별 인코딩: pc→z_e, proprio(actor)→z_p, 그 외 raw. (N,D)·(T,N,D) 모두 처리."""
        if g == self._pc_group:
            enc = self.pc_encoder
        elif g == self._proprio_group and self._encode_proprio:
            enc = self.proprio_encoder
        else:
            return x
        if x.dim() == 3:
            T, N, D = x.shape
            return enc(x.reshape(T * N, D)).reshape(T, N, -1)
        return enc(x)

    def train(self, mode: bool = True):
        """동결된 pc_encoder 는 항상 eval 유지 (runner 의 train_mode() 가 되돌리지 못하게)."""
        super().train(mode)
        if getattr(self, "_encoder_frozen", False):
            self.pc_encoder.eval()
        return self

    def get_latent(self, obs, masks=None, hidden_state: HiddenState = None):
        """[z_p, z_e, (ẑ_e)] concat → 정규화 → GRU."""
        parts = [self._enc_group(g, obs[g]) for g in self.obs_groups]
        if self._predictor_in_policy:
            # 예측된 미래 잠재를 정책 입력으로 — 예측기가 **배포 시에도 살아 있고**, 행동손실이
            # 직접 통과한다. 사전학습 초기화와 달리 학습 중에 씻겨나갈 수 없다.
            z_e = parts[self.obs_groups.index(self._pc_group)]
            z_p = parts[self.obs_groups.index(self._proprio_group)]
            z_hat = self.jepa_predict(z_e, z_p, None)
            # 게이트 4(기전 검증): eval 에서 ẑ 블록만 0 으로 → 성공률 낙폭 = 정책이 예측에
            # 실제로 의존하는 정도. 강인성은 간접 증거지만 이건 직접 측정이다.
            if getattr(self, "ablate_predictor", False):
                z_hat = torch.zeros_like(z_hat)
            parts.append(z_hat)
        latent = torch.cat(parts, dim=-1)
        latent = self.obs_normalizer(latent)               # obs_normalization=False → Identity
        latent = self.rnn(latent, masks, hidden_state).squeeze(0)
        return latent

    # ── 백본 latent (runner aux 손실용) ──────────────────────────────────────
    def encode_exteroception(self, pc_obs: torch.Tensor) -> torch.Tensor:
        """(M, P*3) → z_e. Head A/B 공용 외수용 latent."""
        return self.pc_encoder(pc_obs)

    def encode_proprio(self, proprio_obs: torch.Tensor) -> torch.Tensor:
        """(M, 45*H) → z_p. 두 헤드 공통 고유수용 latent."""
        return self.proprio_encoder(proprio_obs)

    def predict_vel(self, proprio_obs: torch.Tensor) -> torch.Tensor:
        """v̂ = vel_decoder(z_p) (CENet). velocity 로 z_p·proprio_encoder shaping (두 헤드 동일)."""
        return self.vel_decoder(self.proprio_encoder(proprio_obs))

    # ── Head A (recon) ───────────────────────────────────────────────────────
    def reconstruct_height(self, pc_obs: torch.Tensor) -> torch.Tensor:
        """z_e → GT height map (grad→pc_encoder+recon_decoder, z_e 를 재구성 쪽 shaping)."""
        return self.recon_decoder(self.pc_encoder(pc_obs))

    # ── Head B (jepa) ────────────────────────────────────────────────────────
    def jepa_predict(self, z_e_t: torch.Tensor, z_p_t: torch.Tensor,
                     cond: torch.Tensor | None = None) -> torch.Tensor:
        """[z_e(t), z_p(t), cond] → ẑ_e(t+k). z_p·cond 는 runner 에서 detach(z_e 만 shaping).

        cond = 미래 행동 a_{t:t+k} 와/또는 명령 c_t (conditioning). cond_dim=0 이면 안 넘김(구동작).
        """
        parts = [z_e_t, z_p_t]
        if cond is not None:
            parts.append(cond)
        return self.jepa_predictor(torch.cat(parts, dim=-1))

    def vic_project(self, z_e: torch.Tensor) -> torch.Tensor:
        """VICReg 정규화 대상 = projector(z_e). use_projector=False 면 z_e 그대로(구 동작)."""
        return self.vic_projector(z_e) if getattr(self, "use_projector", False) else z_e

    @torch.no_grad()
    def target_encode(self, pc_obs: torch.Tensor) -> torch.Tensor:
        return self.target_pc_encoder(pc_obs)

    @torch.no_grad()
    def ema_update_target(self, tau: float = 0.996) -> None:
        for tp, op in zip(self.target_pc_encoder.parameters(), self.pc_encoder.parameters()):
            tp.mul_(tau).add_(op.detach(), alpha=1.0 - tau)
