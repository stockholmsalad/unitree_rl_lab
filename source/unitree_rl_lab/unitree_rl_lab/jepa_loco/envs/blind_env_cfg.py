"""Depth 기준선과 보상·명령·지형을 공유하는 카메라 없는 대조군."""

from __future__ import annotations

from isaaclab.utils import configclass

from .depth_env_cfg import JepaDepthEnvCfg, JepaDepthEnvCfg_PLAY, JepaObservationsCfg, JepaSceneCfg


@configclass
class BlindSceneCfg(JepaSceneCfg):
    d435i = None


@configclass
class BlindObservationsCfg(JepaObservationsCfg):
    depth = None
    depth_fresh = None


@configclass
class JepaBlindEnvCfg(JepaDepthEnvCfg):
    scene: BlindSceneCfg = BlindSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: BlindObservationsCfg = BlindObservationsCfg()


@configclass
class JepaBlindEnvCfg_PLAY(JepaDepthEnvCfg_PLAY):
    scene: BlindSceneCfg = BlindSceneCfg(num_envs=32, env_spacing=2.5)
    observations: BlindObservationsCfg = BlindObservationsCfg()
