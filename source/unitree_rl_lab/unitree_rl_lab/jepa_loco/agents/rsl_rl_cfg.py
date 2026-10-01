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
