"""특권 heightscan을 terrain latent로 압축하는 PPO oracle actor."""

from __future__ import annotations

import torch
from tensordict import TensorDict
from torch import nn

from rsl_rl.models.mlp_model import MLPModel


class OracleTerrainActor(MLPModel):
    """[proprio, E_T(heightscan)]에서 행동을 출력한다.

    ``terrain_encoder``의 출력이 student distillation의 고정 teacher target이다.
    """

    def __init__(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str, output_dim: int,
                 hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128), activation: str = "elu",
                 obs_normalization: bool = False, distribution_cfg: dict | None = None,
                 terrain_group: str = "terrain_current", terrain_latent_dim: int = 32,
                 terrain_hidden_dim: int = 128, future_group: str | None = None) -> None:
        self.terrain_group = terrain_group
        self.terrain_latent_dim = terrain_latent_dim
        self.future_group = future_group
        super().__init__(obs, obs_groups, obs_set, output_dim, hidden_dims, activation,
                         obs_normalization, distribution_cfg)
        terrain_dim = obs[terrain_group].shape[-1]
        if future_group is not None and obs[future_group].shape[-1] != terrain_dim:
            raise ValueError("현재·미래 heightmap 차원은 공유 terrain encoder 때문에 같아야 한다")
        self.terrain_encoder = nn.Sequential(
            nn.Linear(terrain_dim, terrain_hidden_dim), nn.ELU(),
            nn.Linear(terrain_hidden_dim, terrain_latent_dim), nn.ELU(),
        )
        self.last_z: torch.Tensor | None = None

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        groups = obs_groups[obs_set]
        terrain_groups = (self.terrain_group,) if self.future_group is None else (self.terrain_group, self.future_group)
        if any(group not in groups or obs[group].ndim != 2 for group in terrain_groups):
            raise ValueError(f"actor에는 1D '{self.terrain_group}' 관측이 필요하다")
        proprio = [group for group in groups if group not in terrain_groups]
        if any(obs[group].ndim != 2 for group in proprio):
            raise ValueError("proprio 관측은 모두 [N,D]여야 한다")
        return proprio, sum(obs[group].shape[-1] for group in proprio)

    def _get_latent_dim(self) -> int:
        return self.obs_dim + self.terrain_latent_dim * (2 if self.future_group else 1)

    def get_terrain_latent(self, heightscan: torch.Tensor) -> torch.Tensor:
        if heightscan.ndim != 2:
            raise ValueError("heightscan은 [N,D]여야 한다")
        return self.terrain_encoder(heightscan)

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None) -> torch.Tensor:
        proprio = self.obs_normalizer(torch.cat([obs[group] for group in self.obs_groups], dim=-1))
        z = self.get_terrain_latent(obs[self.terrain_group])
        if self.future_group is None:
            self.last_z = z.detach()
            return torch.cat((proprio, z), dim=-1)
        z_future = self.get_terrain_latent(obs[self.future_group])
        self.last_z = torch.cat((z, z_future), dim=-1).detach()
        return torch.cat((proprio, z, z_future), dim=-1)
