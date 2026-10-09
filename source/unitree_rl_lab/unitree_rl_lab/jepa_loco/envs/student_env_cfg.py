"""Current-EasyStart oracle의 지형·보상·진단을 유지하고 Depth 관측을 더한 student 환경."""

from __future__ import annotations

from isaaclab.utils import configclass

from unitree_rl_lab.jepa_loco.sensors.d435i_cfg import D435iParams, make_tiled_camera_cfg

from .blind_env_cfg import BlindSceneCfg
from .depth_env_cfg import JepaObservationsCfg
from .oracle_env_cfg import (
    OracleCurrentEasyStartEnvCfg, OracleObservationsCfg,
)


@configclass
class StudentSceneCfg(BlindSceneCfg):
    d435i = make_tiled_camera_cfg(D435iParams()).replace(update_period=0.0)


@configclass
class StudentObservationsCfg(OracleObservationsCfg):
    depth: JepaObservationsCfg.DepthCfg = JepaObservationsCfg.DepthCfg()
    depth_fresh: JepaObservationsCfg.FreshCfg = JepaObservationsCfg.FreshCfg()


@configclass
class StudentDepthEasyStartEnvCfg(OracleCurrentEasyStartEnvCfg):
    scene: StudentSceneCfg = StudentSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: StudentObservationsCfg = StudentObservationsCfg()
    camera: D435iParams = D435iParams()
    depth_period_steps: int = 5

    def __post_init__(self):
        super().__post_init__()
        self.scene.d435i = make_tiled_camera_cfg(self.camera).replace(update_period=0.0)
        frame = self.observations.depth.frame
        frame.params["fx"] = self.camera.intrinsics(self.camera.render_wh)[0]
        frame.params["near"] = self.camera.min_range
        frame.params["far"] = self.camera.clip_far
        frame.params["noise"]["clip_far"] = self.camera.clip_far
        frame.params["noise"]["baseline"] = self.camera.baseline
        self.sim.render_interval = self.decimation * self.depth_period_steps


@configclass
class StudentDepthEasyStartEnvCfg_PLAY(StudentDepthEasyStartEnvCfg):
    scene: StudentSceneCfg = StudentSceneCfg(num_envs=32, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.initial_ranges = self.commands.base_velocity.final_ranges.copy()
        self.commands.base_velocity.ranges = self.commands.base_velocity.final_ranges.copy()
        self.scene.terrain.max_init_terrain_level = self.easy_start.num_rows - 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
