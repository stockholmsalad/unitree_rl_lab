"""카메라 없는 proprio-only GRU 대조군 actor."""

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import HiddenState
from rsl_rl.utils import unpad_trajectories

from .backbone import make_backbone


class ProprioRecurrentActor(MLPModel):
    is_recurrent: bool = True

    def __init__(self, obs: TensorDict, obs_groups: dict, obs_set: str, output_dim: int,
                 hidden_dims=(512, 256, 128), activation="elu", obs_normalization=False,
                 distribution_cfg=None, backbone="gru", latent_dim=128):
        self.latent_dim = latent_dim
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, distribution_cfg)
        self.frame_encoder = nn.Linear(self.obs_dim, latent_dim)
        self.backbone = make_backbone(backbone, latent_dim, latent_dim)
        self.h: torch.Tensor | None = None
        self.last_z: torch.Tensor | None = None

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.latent_dim

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None,
                   hidden_state: HiddenState = None) -> torch.Tensor:
        proprio = self.obs_normalizer(torch.cat([obs[g] for g in self.obs_groups], dim=-1))
        f = torch.tanh(self.frame_encoder(proprio))
        if masks is None:
            if self.h is None or self.h.shape[0] != proprio.shape[0]:
                self.h = self.backbone.initial_state(proprio.shape[0], proprio.device)
            z, self.h = self.backbone.step(f, self.h)
            self.last_z = z.detach()
            return torch.cat([proprio, z], dim=-1)
        if hidden_state is None:
            raise ValueError("배치 모드에는 저장된 hidden state가 필요하다")
        reset = obs["traj_start"][..., 0].bool() if "traj_start" in obs.keys() else None
        z, _ = self.backbone.gated_sequence(f, hidden_state[0], masks,
                                             reset)
        return unpad_trajectories(torch.cat([proprio, z], dim=-1), masks)

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
