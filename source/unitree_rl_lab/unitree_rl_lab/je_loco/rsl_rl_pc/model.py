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
    """PointNet-lite: (M, P*3) → (M, out_dim). 점당 공유 MLP + global max-pool (순서 불변)."""

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
        pts = x.reshape(m, self.P, 3)          # (M, P, 3)
        feat = self.point_mlp(pts)             # (M, P, 128)
        glob = feat.max(dim=1)[0]              # (M, 128) max-pool
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
        **kwargs,
    ) -> None:
        # _get_obs_dim(super 초기화 중 호출)이 참조 → super 전에 세팅
        self._pc_group = pc_group
        self._pc_out_dim = pc_out_dim
        self._pc_num_points = obs[pc_group].shape[-1] // 3
        self._proprio_group = proprio_group
        self._zp_dim = zp_dim
        # actor 만 proprio 를 z_p 로 인코딩(critic 은 privileged 관측을 raw 사용)
        self._encode_proprio = proprio_group in obs_groups[obs_set]

        super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)

        self.pc_encoder = PointCloudEncoder(self._pc_num_points, pc_out_dim)

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
        elif repr_head == "jepa":
            self.jepa_predictor = nn.Sequential(
                nn.Linear(pc_out_dim + zp_dim, recon_hidden_dim), nn.ELU(),
                nn.Linear(recon_hidden_dim, pc_out_dim),
            )
            self.target_pc_encoder = copy.deepcopy(self.pc_encoder)   # EMA target + stop-grad
            for p in self.target_pc_encoder.parameters():
                p.requires_grad_(False)

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

    def get_latent(self, obs, masks=None, hidden_state: HiddenState = None):
        """[z_p, z_e] concat → 정규화 → GRU."""
        parts = [self._enc_group(g, obs[g]) for g in self.obs_groups]
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
    def jepa_predict(self, z_e_t: torch.Tensor, z_p_t: torch.Tensor) -> torch.Tensor:
        """[z_e(t), z_p(t)] → ẑ_e(t+k). z_p 는 runner 에서 detach 해 넘김(z_e 만 shaping)."""
        return self.jepa_predictor(torch.cat([z_e_t, z_p_t], dim=-1))

    @torch.no_grad()
    def target_encode(self, pc_obs: torch.Tensor) -> torch.Tensor:
        return self.target_pc_encoder(pc_obs)

    @torch.no_grad()
    def ema_update_target(self, tau: float = 0.996) -> None:
        for tp, op in zip(self.target_pc_encoder.parameters(), self.pc_encoder.parameters()):
            tp.mul_(tau).add_(op.detach(), alpha=1.0 - tau)
