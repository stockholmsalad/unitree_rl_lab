"""Current teacher의 고정 정책 head에 넣을 depth terrain latent를 학습한다."""

from __future__ import annotations

import copy

import torch
from tensordict import TensorDict
from torch import nn

from .backbone import make_backbone
from .depth_cnn import DepthCNN
from .oracle_actor import OracleTerrainActor
from .student_batches import ProprioHistory, proprio_windows
from unitree_rl_lab.jepa_loco.eval.latent_noise import TeacherLatentNoise


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
                 cnn_strides: tuple[int, ...] = (2, 2, 2),
                 proprio_input: bool = False, proprio_dim: int = 45,
                 proprio_hidden_dims: tuple[int, ...] = (64, 64),
                 proprio_window: int = 1, backbone: str = "gru",
                 mamba_intermediate_size: int = 256, mamba_state_size: int = 16,
                 mamba_conv_kernel: int = 4, mamba_dt_rank: int | str = "auto"):
        super().__init__()
        if method not in ("no_memory", "gru", "gru_jepa", "gru_copy"):
            raise ValueError(f"student method={method!r}")
        if backbone not in ("gru", "mamba") or (method == "no_memory" and backbone == "mamba"):
            raise ValueError(f"student backbone={backbone!r}는 method={method!r}에 사용할 수 없다")
        self.method = method
        self.backbone_name = backbone
        self.context_dim = context_dim
        self.proprio_input = proprio_input
        if proprio_window < 1:
            raise ValueError("proprio_window는 1 이상이어야 한다")
        self.proprio_window = proprio_window
        if proprio_input and (proprio_dim < 1 or not proprio_hidden_dims or
                              any(width < 1 for width in proprio_hidden_dims)):
            raise ValueError("proprio 입력과 MLP 층 크기는 양수여야 한다")
        self.proprio_dim = proprio_dim
        self.proprio_feature_dim = proprio_hidden_dims[-1] if proprio_input else 0
        self.cnn = DepthCNN(in_hw=image_hw, channels=cnn_channels, kernels=cnn_kernels,
                            strides=cnn_strides, feat_dim=feature_dim)
        if method == "no_memory":
            self.frame_projection = nn.Sequential(nn.Linear(feature_dim + self.proprio_feature_dim, context_dim), nn.ELU())
            self.backbone = None
        else:
            # 기본 128→128은 항등이다. 차원이 다를 때만 같은 projection을
            # GRU 입력과 frame_embedding target 양쪽에 적용한다.
            self.frame_projection = (nn.Identity() if feature_dim == context_dim else
                                     nn.Linear(feature_dim, context_dim))
            backbone_kwargs = ({"intermediate_size": mamba_intermediate_size, "state_size": mamba_state_size,
                                "conv_kernel": mamba_conv_kernel, "dt_rank": mamba_dt_rank}
                               if backbone == "mamba" else {})
            self.backbone = make_backbone(backbone, context_dim + self.proprio_feature_dim,
                                          context_dim, **backbone_kwargs)
        self.terrain_head = nn.Sequential(nn.Linear(context_dim, terrain_hidden_dim), nn.ELU(),
                                          nn.Linear(terrain_hidden_dim, terrain_dim))
        self.predictor = nn.Sequential(nn.Linear(context_dim + self.proprio_feature_dim + 3, predictor_hidden_dim), nn.ELU(),
                                       nn.Linear(predictor_hidden_dim, context_dim))
        if proprio_input:
            widths = (proprio_dim, *proprio_hidden_dims)
            self.proprio_encoder = nn.Sequential(*[
                layer for index in range(len(widths) - 1)
                for layer in (nn.Linear(widths[index], widths[index + 1]), nn.ELU())
            ])
            self.ema_proprio_encoder = copy.deepcopy(self.proprio_encoder)
            self.ema_proprio_encoder.requires_grad_(False)
            if proprio_window > 1:
                summary_in = proprio_window * (self.proprio_feature_dim + 1)
                self.proprio_summary = nn.Sequential(
                    nn.Linear(summary_in, self.proprio_feature_dim), nn.ELU(),
                )
                self.ema_proprio_summary = copy.deepcopy(self.proprio_summary)
                self.ema_proprio_summary.requires_grad_(False)
            else:
                self.proprio_summary = None
                self.ema_proprio_summary = None
        else:
            self.proprio_encoder = None
            self.ema_proprio_encoder = None
            self.proprio_summary = None
            self.ema_proprio_summary = None
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

    def encode_proprio(self, proprio: torch.Tensor | None, normalizer: nn.Module | None,
                       target: bool = False, valid: torch.Tensor | None = None) -> torch.Tensor:
        if not self.proprio_input or proprio is None or normalizer is None:
            raise ValueError("proprio 옵션에는 policy 관측과 고정 teacher normalizer가 필요하다")
        if proprio.shape[-1] != self.proprio_dim:
            raise ValueError(f"policy 관측 마지막 차원은 {self.proprio_dim}이어야 한다")
        encoder = self.ema_proprio_encoder if target else self.proprio_encoder
        if self.proprio_window == 1:
            return encoder(normalizer(proprio))
        if proprio.shape[-2] != self.proprio_window or valid is None or valid.shape != proprio.shape[:-1]:
            raise ValueError("K>1이면 proprio [...,K,45]와 valid [...,K]가 필요하다")
        embeddings = encoder(normalizer(proprio))
        embeddings = torch.where(valid[..., None], embeddings, torch.zeros_like(embeddings))
        summary_input = torch.cat((embeddings.flatten(-2), valid.to(embeddings.dtype)), dim=-1)
        summary = self.ema_proprio_summary if target else self.proprio_summary
        return summary(summary_input)

    def encode_step(self, depth: torch.Tensor, fresh: torch.Tensor,
                    state: torch.Tensor | None, target: bool = False,
                    proprio: torch.Tensor | None = None, proprio_normalizer: nn.Module | None = None,
                    proprio_valid: torch.Tensor | None = None,
                    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        cnn = self.ema_cnn if target else self.cnn
        backbone = self.ema_backbone if target else self.backbone
        projection = self.ema_frame_projection if target else self.frame_projection
        if self.method == "no_memory":
            if state is None:
                state = self.initial_state(depth.shape[0], depth.device)
            if not fresh.any():
                return state, state
            features = cnn(depth[fresh].float())
            if self.proprio_input:
                features = torch.cat((features, self.encode_proprio(
                    proprio[fresh] if proprio is not None else None, proprio_normalizer, target,
                    proprio_valid[fresh] if proprio_valid is not None else None)), dim=-1)
            updated = projection(features)
            # index_copy는 이전 프레임의 gradient graph를 제자리에서 바꾸지 않는다.
            state = state.index_copy(0, fresh.nonzero(as_tuple=True)[0], updated)
            return state, state
        if state is None:
            state = backbone.initial_state(depth.shape[0], depth.device)
        features = torch.zeros(depth.shape[0], self.context_dim + self.proprio_feature_dim, device=depth.device)
        if fresh.any():
            frame = projection(cnn(depth[fresh].float()))
            if self.proprio_input:
                frame = torch.cat((frame, self.encode_proprio(
                    proprio[fresh] if proprio is not None else None, proprio_normalizer, target,
                    proprio_valid[fresh] if proprio_valid is not None else None)), dim=-1)
            features[fresh] = frame
        return backbone.gated_step(features, state, fresh)

    def encode_frame_targets(self, depth: torch.Tensor, valid: torch.Tensor,
                             batch_size: int) -> torch.Tensor:
        """유효한 새 프레임만 EMA CNN(+projection)으로 인코딩한다."""
        if depth.ndim != 5 or valid.shape != depth.shape[:2] or batch_size < 1:
            raise ValueError("depth [T,N,2,H,W], valid [T,N], batch_size>0이어야 한다")
        flat_depth = depth.flatten(0, 1)
        ids = valid.flatten().nonzero(as_tuple=True)[0]
        output = torch.zeros(flat_depth.shape[0], self.context_dim, device=depth.device)
        with torch.no_grad():
            for chunk in ids.split(batch_size):
                embedding = self.ema_frame_projection(self.ema_cnn(flat_depth[chunk].float()))
                output[chunk] = embedding
        return output.reshape(*depth.shape[:2], self.context_dim)

    def encode_sequence(self, depth: torch.Tensor, fresh: torch.Tensor,
                        resets: torch.Tensor, h0: torch.Tensor | None = None,
                        target: bool = False, proprio: torch.Tensor | None = None,
                        proprio_normalizer: nn.Module | None = None,
                        proprio_prefix: torch.Tensor | None = None,
                        proprio_prefix_valid: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor | None]:
        """[T,N,2,H,W]를 에피소드 경계와 10 Hz 새 프레임 gate에 맞춰 재생한다."""
        if depth.ndim != 5 or fresh.shape != depth.shape[:2] or resets.shape != fresh.shape:
            raise ValueError("depth [T,N,2,H,W], fresh/resets [T,N]이어야 한다")
        if self.proprio_input and (proprio is None or proprio.shape[:2] != fresh.shape):
            raise ValueError("proprio sequence는 [T,N,45] 형태여야 한다")
        windows = masks = None
        if self.proprio_input and self.proprio_window > 1:
            if proprio_prefix is None or proprio_prefix_valid is None:
                raise ValueError("K>1인 sequence에는 rollout 시작 prefix와 valid가 필요하다")
            windows, masks = proprio_windows(proprio, resets, proprio_prefix, proprio_prefix_valid)
        state = h0
        outs = []
        for t in range(depth.shape[0]):
            if state is not None:
                state = torch.where(resets[t, :, None], torch.zeros_like(state), state)
            z, state = self.encode_step(depth[t], fresh[t], state, target,
                                        (windows[t] if windows is not None else
                                         None if proprio is None else proprio[t]), proprio_normalizer,
                                        None if masks is None else masks[t])
            outs.append(z)
        return torch.stack(outs), state

    def terrain_latent(self, context: torch.Tensor) -> torch.Tensor:
        return self.terrain_head(context)

    def predict(self, context: torch.Tensor, command: torch.Tensor,
                proprio: torch.Tensor | None = None, proprio_normalizer: nn.Module | None = None,
                proprio_valid: torch.Tensor | None = None) -> torch.Tensor:
        if self.method == "gru_copy":
            return context
        if self.proprio_input:
            return self.predictor(torch.cat((context, self.encode_proprio(
                proprio, proprio_normalizer, valid=proprio_valid), command), dim=-1))
        return self.predictor(torch.cat((context, command), dim=-1))

    @torch.no_grad()
    def update_ema(self, tau: float) -> None:
        if not 0 <= tau <= 1:
            raise ValueError("EMA tau는 [0,1]이어야 한다")
        for online, target in ((self.cnn, self.ema_cnn),
                               (self.backbone, self.ema_backbone),
                               (self.frame_projection, self.ema_frame_projection),
                               (self.proprio_encoder, self.ema_proprio_encoder),
                               (self.proprio_summary, self.ema_proprio_summary)):
            if online is None:
                continue
            for p, q in zip(online.parameters(), target.parameters()):
                q.lerp_(p, 1 - tau)


class StudentInferencePolicy:
    """기존 고정 지형 평가 스크립트의 policy(obs)·reset(dones) 인터페이스."""

    def __init__(self, checkpoint_path: str, device: torch.device, policy_mode: str = "student",
                 teacher_std: torch.Tensor | None = None, latent_noise_std: float = 0.0,
                 latent_noise_mode: str = "step", noise_seed: int = 0):
        if policy_mode not in ("student", "teacher"):
            raise ValueError("policy_mode는 student 또는 teacher여야 한다")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = checkpoint["student_cfg"]
        self.method = cfg["method"]
        self.policy_mode = policy_mode
        self.model = StudentDepthModel(
            method=cfg["method"], image_hw=tuple(cfg["image_hw"]),
            feature_dim=cfg["feature_dim"], context_dim=cfg["context_dim"],
            terrain_dim=cfg["terrain_dim"], terrain_hidden_dim=cfg["terrain_hidden_dim"],
            predictor_hidden_dim=cfg["predictor_hidden_dim"],
            cnn_channels=tuple(cfg["cnn_channels"]), cnn_kernels=tuple(cfg["cnn_kernels"]),
            cnn_strides=tuple(cfg["cnn_strides"]),
            proprio_input=cfg.get("proprio_input", False),
            proprio_dim=cfg.get("proprio_dim", 45),
            proprio_hidden_dims=tuple(cfg.get("proprio_hidden_dims", (64, 64))),
            proprio_window=cfg.get("proprio_window", 1),
            backbone=cfg.get("backbone", "gru"),
            mamba_intermediate_size=cfg.get("mamba_intermediate_size", 256),
            mamba_state_size=cfg.get("mamba_state_size", 16),
            mamba_conv_kernel=cfg.get("mamba_conv_kernel", 4),
            mamba_dt_rank=cfg.get("mamba_dt_rank", "auto"),
        ).to(device)
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.model.eval()
        self.teacher = FrozenCurrentTeacher(checkpoint["teacher_state_dict"]).to(device)
        if latent_noise_std > 0 and (policy_mode != "teacher" or teacher_std is None):
            raise ValueError("teacher 잡음은 teacher 정책과 teacher std가 필요하다")
        self.latent_noise = (TeacherLatentNoise(teacher_std, latent_noise_std, latent_noise_mode, noise_seed)
                             if policy_mode == "teacher" and teacher_std is not None else None)
        self.state: torch.Tensor | None = None
        self.proprio_history: ProprioHistory | None = None
        self.last_z: torch.Tensor | None = None
        self.last_action_latent: torch.Tensor | None = None
        # no_memory도 10 Hz 프레임을 유지하는 상태가 있으므로 종료 시 초기화한다.
        self.is_recurrent = True

    @torch.inference_mode()
    def __call__(self, obs: TensorDict) -> torch.Tensor:
        fresh = obs["depth_fresh"][..., 0] > 0.5
        policy_input, policy_valid = obs["policy"], None
        if self.model.proprio_input and self.model.proprio_window > 1:
            if self.proprio_history is None:
                self.proprio_history = ProprioHistory(self.model.proprio_window, len(fresh),
                                                      self.model.proprio_dim, obs["policy"].device)
            policy_input, policy_valid = self.proprio_history.append(obs["policy"])
        context, self.state = self.model.encode_step(
            obs["depth"], fresh, self.state, proprio=policy_input,
            proprio_normalizer=self.teacher.actor.obs_normalizer, proprio_valid=policy_valid,
        )
        z = self.model.terrain_latent(context)
        self.last_z = z
        if self.policy_mode == "teacher":
            z = self.teacher.terrain_latent(obs["terrain_current"])
            if self.latent_noise is not None:
                z = self.latent_noise.apply(z)
        self.last_action_latent = z
        return self.teacher.action_from_latent(obs["policy"], z)

    def reset(self, dones: torch.Tensor) -> None:
        if self.state is not None:
            self.state = torch.where(dones.bool()[:, None], torch.zeros_like(self.state), self.state)
        if self.proprio_history is not None:
            self.proprio_history.reset(dones.bool())
        if self.latent_noise is not None:
            self.latent_noise.reset(dones)
