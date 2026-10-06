"""Phase 2 depth GRU PPO 러너 설정."""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class DepthActorCfg(RslRlMLPModelCfg):
    class_name: str = "unitree_rl_lab.jepa_loco.models.depth_actor:DepthRecurrentActor"
    depth_group: str = "depth"
    fresh_group: str = "depth_fresh"
    backbone: str = "gru"
    """시간축 backbone. 'gru' (Phase 6 에서 'mamba' 추가)."""
    latent_dim: int = 128
    cnn_cfg: dict = {"channels": (16, 32, 64), "kernels": (5, 3, 3), "strides": (2, 2, 2), "feat_dim": 128}


@configclass
class DepthGRUPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 100
    """rollout = truncated BPTT 길이. 100 제어 스텝 = 2.0 s = depth 20 프레임 (CLAUDE.md §3 context 길이)."""
    max_iterations = 3000
    save_interval = 100
    experiment_name = "jepa_loco_depth_gru"
    obs_groups = {"actor": ["policy", "depth", "depth_fresh"], "critic": ["critic"]}
    actor = DepthActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(hidden_dims=[512, 256, 128], activation="elu", obs_normalization=True)
    algorithm = RslRlPpoAlgorithmCfg(
        class_name="unitree_rl_lab.jepa_loco.agents.ppo:LatentLoggingPPO",
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class BlindActorCfg(RslRlMLPModelCfg):
    class_name: str = "unitree_rl_lab.jepa_loco.models.proprio_actor:ProprioRecurrentActor"
    backbone: str = "gru"
    latent_dim: int = 128


@configclass
class BlindGRUPPORunnerCfg(DepthGRUPPORunnerCfg):
    experiment_name = "jepa_loco_blind_gru"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = BlindActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )


@configclass
class BlindMLPPPORunnerCfg(DepthGRUPPORunnerCfg):
    """기억 없는 proprio-only 대조군: depth 정책의 MLP head 와 같은 구조에 depth 경로(CNN+GRU)만 뺀다.

    depth 정책: [proprio 45, z_t 128] → MLP 512-256-128 → 12
    이 정책 : [proprio 45]           → MLP 512-256-128 → 12
    rollout 길이·PPO·critic 은 depth 와 동일. 비순환이라 rsl_rl 기본 미니배치를 쓴다.
    """

    experiment_name = "jepa_loco_blind_mlp"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )


@configclass
class OracleActorCfg(RslRlMLPModelCfg):
    class_name: str = "unitree_rl_lab.jepa_loco.models.oracle_actor:OracleTerrainActor"
    terrain_group: str = "terrain_current"
    terrain_latent_dim: int = 32
    terrain_hidden_dim: int = 128
    future_group: str | None = None


@configclass
class OracleCurrentPPORunnerCfg(DepthGRUPPORunnerCfg):
    experiment_name = "jepa_loco_oracle_current"
    obs_groups = {"actor": ["policy", "terrain_current"], "critic": ["critic"]}
    actor = OracleActorCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )


@configclass
class OracleWidePPORunnerCfg(OracleCurrentPPORunnerCfg):
    experiment_name = "jepa_loco_oracle_wide"
    obs_groups = {"actor": ["policy", "terrain_wide"], "critic": ["critic"]}
    actor = OracleActorCfg(
        terrain_group="terrain_wide",
        terrain_latent_dim=64,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )


@configclass
class OracleCurrentFuturePPORunnerCfg(OracleCurrentPPORunnerCfg):
    experiment_name = "jepa_loco_oracle_current_future"
    obs_groups = {"actor": ["policy", "terrain_current", "terrain_future"], "critic": ["critic"]}
    actor = OracleActorCfg(
        terrain_group="terrain_current",
        future_group="terrain_future",
        terrain_latent_dim=32,
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
