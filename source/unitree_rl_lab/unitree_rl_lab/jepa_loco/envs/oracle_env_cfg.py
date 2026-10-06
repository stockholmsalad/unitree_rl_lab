"""현재 지형 heightscan을 보는 oracle PPO 환경.

카메라를 만들지 않는다. heightscan은 teacher actor와 critic에서만 사용한다.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .blind_env_cfg import BlindObservationsCfg, BlindSceneCfg, JepaBlindEnvCfg, JepaBlindEnvCfg_PLAY
from .oracle_heightmap import future_height_scan
from .path_curriculum import terrain_levels_path


@configclass
class OraclePathCurriculumCfg:
    terrain_levels = CurrTerm(
        func=terrain_levels_path,
        params={"command_name": "base_velocity", "up_fraction": 0.5, "down_command_fraction": 0.5},
    )


@configclass
class OracleObservationsCfg(BlindObservationsCfg):
    @configclass
    class TerrainCurrentCfg(ObsGroup):
        scan = ObsTerm(func=mdp.height_scan,
                       params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 1.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    terrain_current: TerrainCurrentCfg = TerrainCurrentCfg()


@configclass
class OracleCurrentEnvCfg(JepaBlindEnvCfg):
    scene: BlindSceneCfg = BlindSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: OracleObservationsCfg = OracleObservationsCfg()
    curriculum: OraclePathCurriculumCfg = OraclePathCurriculumCfg()
    terrain_column_assignment_eps: float = 0.001


@configclass
class OracleCurrentEnvCfg_PLAY(JepaBlindEnvCfg_PLAY):
    scene: BlindSceneCfg = BlindSceneCfg(num_envs=32, env_spacing=2.5)
    observations: OracleObservationsCfg = OracleObservationsCfg()
    curriculum: OraclePathCurriculumCfg = OraclePathCurriculumCfg()
    terrain_column_assignment_eps: float = 0.001


@configclass
class OracleWideSceneCfg(BlindSceneCfg):
    wide_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=(4.0, 3.0), ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class OracleWideObservationsCfg(OracleObservationsCfg):
    terrain_current = None

    @configclass
    class TerrainWideCfg(ObsGroup):
        scan = ObsTerm(func=mdp.height_scan,
                       params={"sensor_cfg": SceneEntityCfg("wide_height_scanner")}, clip=(-1.0, 1.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    terrain_wide: TerrainWideCfg = TerrainWideCfg()


@configclass
class OracleCurrentFutureObservationsCfg(OracleObservationsCfg):
    @configclass
    class TerrainFutureCfg(ObsGroup):
        scan = ObsTerm(func=future_height_scan,
                       params={"wide_sensor_cfg": SceneEntityCfg("wide_height_scanner"),
                               "current_sensor_cfg": SceneEntityCfg("height_scanner"),
                               "command_name": "base_velocity", "horizon_s": 0.5},
                       clip=(-1.0, 1.0))

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    terrain_future: TerrainFutureCfg = TerrainFutureCfg()


@configclass
class OracleWideEnvCfg(OracleCurrentEnvCfg):
    scene: OracleWideSceneCfg = OracleWideSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: OracleWideObservationsCfg = OracleWideObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.wide_height_scanner.update_period = self.sim.dt


@configclass
class OracleCurrentFutureEnvCfg(OracleCurrentEnvCfg):
    scene: OracleWideSceneCfg = OracleWideSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: OracleCurrentFutureObservationsCfg = OracleCurrentFutureObservationsCfg()
    future_horizon_s: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self.scene.wide_height_scanner.update_period = self.sim.dt
        self.observations.terrain_future.scan.params["horizon_s"] = self.future_horizon_s


@configclass
class OracleWideEnvCfg_PLAY(OracleWideEnvCfg):
    scene: OracleWideSceneCfg = OracleWideSceneCfg(num_envs=32, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.initial_ranges = self.commands.base_velocity.final_ranges.copy()
        self.commands.base_velocity.ranges = self.commands.base_velocity.final_ranges.copy()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.max_init_terrain_level = self.scene.terrain.terrain_generator.num_rows - 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


@configclass
class OracleCurrentFutureEnvCfg_PLAY(OracleCurrentFutureEnvCfg):
    scene: OracleWideSceneCfg = OracleWideSceneCfg(num_envs=32, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.initial_ranges = self.commands.base_velocity.final_ranges.copy()
        self.commands.base_velocity.ranges = self.commands.base_velocity.final_ranges.copy()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.max_init_terrain_level = self.scene.terrain.terrain_generator.num_rows - 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
