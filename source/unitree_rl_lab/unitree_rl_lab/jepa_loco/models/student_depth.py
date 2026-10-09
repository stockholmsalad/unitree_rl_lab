"""Current teacher의 고정 정책 head에 넣을 depth terrain latent를 학습한다."""

from __future__ import annotations

import copy

import torch
from tensordict import TensorDict
from torch import nn

from .backbone import make_backbone
from .depth_cnn import DepthCNN
from .oracle_actor import OracleTerrainActor


class FrozenCurrentTeacher(nn.Module):
    """OracleCurrent checkpoint의 encoder·정규화·행동 head를 모두 동결한다."""

    def __init__(self, state_dict: dict[str, torch.Tensor], proprio_dim: int = 45,
                 terrain_dim: int = 187, action_dim: int = 12,
                 latent_dim: int = 32, terrain_hidden_dim: int = 128,
                 hidden_dims: tuple[int, ...] = (512, 256, 128)):
        super().__init__()
        sample = TensorDict({"policy": torch.zeros(1, proprio_dim),
                             "terrain_current": torch.zeros(1, terrain_dim)}, batch_size=[1])
        self.actor = OracleTerrainActor(
            sample, {"actor": ["policy", "terrain_current"]}, "actor", action_dim,
            hidden_dims=hidden_dims, obs_normalization=True,
            distribution_cfg={"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
            terrain_latent_dim=latent_dim, terrain_hidden_dim=terrain_hidden_dim,
        )
        self.actor.load_state_dict(state_dict, strict=True)
        self.eval()
        self.requires_grad_(False)

    @classmethod
    def from_checkpoint(cls, path: str, device: torch.device, **kwargs):
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        return cls(checkpoint["actor_state_dict"], **kwargs).to(device)

    def terrain_latent(self, heightmap: torch.Tensor) -> torch.Tensor:
        return self.actor.get_terrain_latent(heightmap)

    def action_from_latent(self, proprio: torch.Tensor, latent: torch.Tensor) -> torch.Tensor:
        normalized = self.actor.obs_normalizer(proprio)
        return self.actor.mlp(torch.cat((normalized, latent), dim=-1))


class StudentDepthModel(nn.Module):
    """한 프레임 CNN, 선택적 GRU 기억, 32차원 terrain head, JEPA predictor."""

    def __init__(self, method: str = "gru", image_hw: tuple[int, int] = (64, 112),
                 feature_dim: int = 128, context_dim: int = 128, terrain_dim: int = 32,
                 terrain_hidden_dim: int = 128, predictor_hidden_dim: int = 128,
                 cnn_channels: tuple[int, ...] = (16, 32, 64),
                 cnn_kernels: tuple[int, ...] = (5, 3, 3),
                 cnn_strides: tuple[int, ...] = (2, 2, 2)):
        super().__init__()
        if method not in ("no_memory", "gru", "gru_jepa", "gru_copy"):
            raise ValueError(f"student method={method!r}")
        self.method = method
        self.context_dim = context_dim
        self.cnn = DepthCNN(in_hw=image_hw, channels=cnn_channels, kernels=cnn_kernels,
                            strides=cnn_strides, feat_dim=feature_dim)
        if method == "no_memory":
            self.frame_projection = nn.Sequential(nn.Linear(feature_dim, context_dim), nn.ELU())
            self.backbone = None
        else:
            self.frame_projection = None
            self.backbone = make_backbone("gru", feature_dim, context_dim)
        self.terrain_head = nn.Sequential(nn.Linear(context_dim, terrain_hidden_dim), nn.ELU(),
                                          nn.Linear(terrain_hidden_dim, terrain_dim))
        self.predictor = nn.Sequential(nn.Linear(context_dim + 3, predictor_hidden_dim), nn.ELU(),
                                       nn.Linear(predictor_hidden_dim, context_dim))
        self.ema_cnn = copy.deepcopy(self.cnn)
        self.ema_backbone = copy.deepcopy(self.backbone)
        self.ema_frame_projection = copy.deepcopy(self.frame_projection)
        for module in (self.ema_cnn, self.ema_backbone, self.ema_frame_projection):
            if module is not None:
                module.requires_grad_(False)

    def initial_state(self, batch: int, device: torch.device) -> torch.Tensor:
        if self.backbone is None:
            return torch.zeros(batch, self.context_dim, device=device)
        return self.backbone.initial_state(batch, device)

    def encode_step(self, depth: torch.Tensor, fresh: torch.Tensor,
                    state: torch.Tensor | None, target: bool = False
                    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cnn = self.ema_cnn if target else self.cnn
        backbone = self.ema_backbone if target else self.backbone
        projection = self.ema_frame_projection if target else self.frame_projection
        if self.method == "no_memory":
            if state is None:
                state = self.initial_state(depth.shape[0], depth.device)
            if not fresh.any():
                return state, state
            updated = projection(cnn(depth[fresh].float()))
            # index_copy는 이전 프레임의 gradient graph를 제자리에서 바꾸지 않는다.
            state = state.index_copy(0, fresh.nonzero(as_tuple=True)[0], updated)
            return state, state
        if state is None:
            state = backbone.initial_state(depth.shape[0], depth.device)
        features = torch.zeros(depth.shape[0], cnn.feat_dim, device=depth.device)
        if fresh.any():
            features[fresh] = cnn(depth[fresh].float())
        return backbone.gated_step(features, state, fresh)

    def encode_sequence(self, depth: torch.Tensor, fresh: torch.Tensor,
                        resets: torch.Tensor, h0: torch.Tensor | None = None,
                        target: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        """[T,N,2,H,W]를 에피소드 경계와 10 Hz 새 프레임 gate에 맞춰 재생한다."""
        if depth.ndim != 5 or fresh.shape != depth.shape[:2] or resets.shape != fresh.shape:
            raise ValueError("depth [T,N,2,H,W], fresh/resets [T,N]이어야 한다")
        state = h0
        outs = []
        for t in range(depth.shape[0]):
            if state is not None:
                state = torch.where(resets[t, :, None], torch.zeros_like(state), state)
            z, state = self.encode_step(depth[t], fresh[t], state, target)
            outs.append(z)
        return torch.stack(outs), state

    def terrain_latent(self, context: torch.Tensor) -> torch.Tensor:
        return self.terrain_head(context)

    def predict(self, context: torch.Tensor, command: torch.Tensor) -> torch.Tensor:
        if self.method == "gru_copy":
            return context
        return self.predictor(torch.cat((context, command), dim=-1))

    @torch.no_grad()
    def update_ema(self, tau: float) -> None:
        if not 0 <= tau <= 1:
            raise ValueError("EMA tau는 [0,1]이어야 한다")
        for online, target in ((self.cnn, self.ema_cnn),
                               (self.backbone, self.ema_backbone),
                               (self.frame_projection, self.ema_frame_projection)):
            if online is None:
                continue
            for p, q in zip(online.parameters(), target.parameters()):
                q.lerp_(p, 1 - tau)


class StudentInferencePolicy:
    """기존 고정 지형 평가 스크립트의 policy(obs)·reset(dones) 인터페이스."""

    def __init__(self, checkpoint_path: str, device: torch.device):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = checkpoint["student_cfg"]
        self.model = StudentDepthModel(
            method=cfg["method"], image_hw=tuple(cfg["image_hw"]),
            feature_dim=cfg["feature_dim"], context_dim=cfg["context_dim"],
            terrain_dim=cfg["terrain_dim"], terrain_hidden_dim=cfg["terrain_hidden_dim"],
            predictor_hidden_dim=cfg["predictor_hidden_dim"],
            cnn_channels=tuple(cfg["cnn_channels"]), cnn_kernels=tuple(cfg["cnn_kernels"]),
            cnn_strides=tuple(cfg["cnn_strides"]),
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.teacher = FrozenCurrentTeacher(checkpoint["teacher_state_dict"]).to(device)
        self.state: torch.Tensor | None = None
        self.last_z: torch.Tensor | None = None
        # no_memory도 10 Hz 프레임을 유지하는 상태가 있으므로 종료 시 초기화한다.
        self.is_recurrent = True

    @torch.inference_mode()
    def __call__(self, obs: TensorDict) -> torch.Tensor:
        fresh = obs["depth_fresh"][..., 0] > 0.5
        context, self.state = self.model.encode_step(obs["depth"], fresh, self.state)
        z = self.model.terrain_latent(context)
        self.last_z = z
        return self.teacher.action_from_latent(obs["policy"], z)

    def reset(self, dones: torch.Tensor) -> None:
        if self.state is not None:
            self.state[dones.bool()] = 0
