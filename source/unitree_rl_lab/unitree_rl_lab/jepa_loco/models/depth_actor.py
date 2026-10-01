"""rsl_rl 5 용 depth + proprio 순환 actor (Phase 2 기준선, JEPA 없음).

    Depth_t [N, 2, 64, 112] → DepthCNN → f_t [N, 128] → SequenceBackbone(GRU) → z_t [N, 128]
    [proprio(정규화) 45, z_t] → MLP → 관절 명령 12

depth 는 10 Hz 로만 새 프레임이 온다. 관측 그룹 ``depth_fresh`` (1 이면 이번 스텝에 새 프레임)가
backbone 갱신을 게이트하고, 나머지 스텝은 마지막 z_t 를 재사용한다(CLAUDE.md §3).

PPO 연결 (rsl_rl 5.0.1 기준):
- rollout: ``act()`` 가 스텝마다 ``get_hidden_state()`` 를 저장한 뒤 forward(masks=None) 로 1스텝 진행.
  ``process_env_step()`` 이 ``reset(dones)`` 로 끝난 env 의 상태만 0 으로 만든다.
- update: ``ResetMaskRolloutStorage`` (agents/ppo.py) 가 env 별 rollout 시퀀스 [T=num_steps_per_env, B] 를
  패딩·복사 없이 주고, rollout 시작 시점의 저장된 상태를 h0, 에피소드 경계를 ``traj_start`` 로 준다.
  forward 가 시퀀스를 다시 펼치며 경계에서 상태를 0 으로 = rollout 과 같은 계산, truncated BPTT 길이 T.
  (rsl_rl 기본 trajectory 패딩도 지원하지만 depth 에서는 메모리가 넘어짐 횟수에 비례해 OOM.)
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories

from .backbone import make_backbone
from .depth_cnn import DepthCNN


class DepthRecurrentActor(MLPModel):
    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        depth_group: str = "depth",
        fresh_group: str = "depth_fresh",
        backbone: str = "gru",
        latent_dim: int = 128,
        cnn_cfg: dict | None = None,
    ) -> None:
        self.depth_group, self.fresh_group, self.latent_dim = depth_group, fresh_group, latent_dim
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )
        d = obs[depth_group]
        self.cnn = DepthCNN(in_channels=d.shape[1], in_hw=tuple(d.shape[2:4]), **(cnn_cfg or {}))
        self.backbone = make_backbone(backbone, self.cnn.feat_dim, latent_dim)
        self.h: torch.Tensor | None = None
        self.last_z: torch.Tensor | None = None
        """최근 z_t (detach). latent 차원별 표준편차 로깅용."""

    # ---- 관측 분리 ----
    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        groups = obs_groups[obs_set]
        for g in (self.depth_group, self.fresh_group):
            if g not in groups:
                raise ValueError(f"'{g}' 관측 그룹이 {obs_set} 에 없다: {groups}")
        proprio = [g for g in groups if g not in (self.depth_group, self.fresh_group)]
        for g in proprio:
            if obs[g].dim() != 2:
                raise ValueError(f"proprio 그룹 '{g}' 는 1D 여야 한다: {tuple(obs[g].shape)}")
        return proprio, sum(obs[g].shape[-1] for g in proprio)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.latent_dim

    # ---- forward ----
    def _encode(self, depth: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        """valid 인 프레임만 CNN 에 통과(10 Hz 프레임은 5스텝에 1번)."""
        f = torch.zeros(*valid.shape, self.cnn.feat_dim, device=depth.device)
        if valid.any():
            f[valid] = self.cnn(depth[valid].float())  # 관측은 fp16 로 저장된다
        return f

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        proprio = self.obs_normalizer(torch.cat([obs[g] for g in self.obs_groups], dim=-1))
        depth = obs[self.depth_group]
        fresh = obs[self.fresh_group][..., 0] > 0.5
        if masks is None:  # rollout: [N, ...]
            if self.h is None or self.h.shape[0] != depth.shape[0]:
                self.h = self.backbone.initial_state(depth.shape[0], depth.device)
            z, self.h = self.backbone.gated_step(self._encode(depth, fresh), self.h, fresh)
            self.last_z = z.detach()
            return torch.cat([proprio, z], dim=-1)
        # update: [T, B, ...] — rsl_rl 패딩 trajectory, 또는 패딩 없는 env 시퀀스 + traj_start (ResetMaskRolloutStorage)
        if hidden_state is None:
            raise ValueError("배치 모드에는 저장된 hidden state 가 필요하다")
        valid = fresh & masks
        reset = obs["traj_start"][..., 0].bool() if "traj_start" in obs.keys() else None
        z, _ = self.backbone.gated_sequence(self._encode(depth, valid), hidden_state[0], valid, reset)
        return unpad_trajectories(torch.cat([proprio, z], dim=-1), masks)

    # ---- 순환 상태 ----
    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        if dones is None:
            self.h = None if hidden_state is None else hidden_state[0]
        elif self.h is not None:
            self.h[dones.bool()] = 0.0

    def get_hidden_state(self) -> HiddenState:
        return None if self.h is None else self.h.unsqueeze(0)

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        if self.h is not None:
            self.h = self.h.detach()

    def as_jit(self):
        raise NotImplementedError("배포용 내보내기는 Phase 8")

    def as_onnx(self, verbose: bool = False):
        raise NotImplementedError("배포용 내보내기는 Phase 8")
