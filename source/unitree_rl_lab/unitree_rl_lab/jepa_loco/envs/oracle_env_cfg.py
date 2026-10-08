"""현재 지형 heightscan을 보는 oracle PPO 환경.

카메라를 만들지 않는다. heightscan은 teacher actor와 critic에서만 사용한다.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp

from .blind_env_cfg import BlindObservationsCfg, BlindSceneCfg, JepaBlindEnvCfg, JepaBlindEnvCfg_PLAY
from .depth_env_cfg import JepaRewardsCfg
from .easy_start_terrain import EasyStartTerrainCfg, make_easy_start_terrain
from .oracle_heightmap import future_height_scan
from .path_curriculum import terrain_levels_path
from .rewards import nominal_pose_hip, nominal_pose_thigh_calf, swing_foot_clearance


@configclass
class OraclePathCurriculumCfg:
    terrain_levels = CurrTerm(
        func=terrain_levels_path,
        params={"command_name": "base_velocity", "up_fraction": 0.5, "down_command_fraction": 0.5},
    )


@configclass
class OracleProgressCfg:
    clearance_margin_m: float = 0.1
    clearance_hold_steps: int = 10
    stair_approach_margin_m: float = 0.4
    first_step_progress_margin_m: float = 0.05
    stall_speed_mps: float = 0.1
    stall_hold_steps: int = 50
    forward_attempt_speed_mps: float = 0.2
    warmup_levels: int = 0
    first_tread_contact_threshold_n: float = 1.0


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
class OracleSwingClearanceCfg:
    target_clearance: float = 0.10
    foot_radius: float = 0.022
    radius: float = 0.10
    contact_threshold: float = 1.0
    min_air_time: float = 0.05
    command_threshold: float = 0.1
    # 20 s, dt=0.02, two swing feet: <= 0.5 * weight * 0.02 * 1000 = 10*weight.
    # All four feet in swing give the absolute bound 20*weight; weight=1 is below
    # the linear tracking term's 20 s maximum of 30 under the trot assumption.
    weight: float = 1.0


@configclass
class OracleNominalPoseCfg:
    hip_joint_regex: str = ".*_hip_joint"
    thigh_joint_regex: str = ".*_thigh_joint"
    calf_joint_regex: str = ".*_calf_joint"
    hip_weight: float = -0.7
    thigh_calf_weight: float = -0.15  # model_800 평지 평가: 단위항 -1.953/20 s → -0.293/20 s
    stand_still_scale: float = 5.0
    velocity_threshold: float = 0.3
    normalization: str = "command_xy_sq_floor"
    normalization_floor: float = 1.0


@configclass
class OracleCurrentRewardsCfg(JepaRewardsCfg):
    joint_pos = None
    nominal_hip = RewTerm(
        func=nominal_pose_hip, weight=-0.7,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
            "command_name": "base_velocity", "stand_still_scale": 5.0,
            "velocity_threshold": 0.3, "normalization": "command_xy_sq_floor",
            "normalization_floor": 1.0,
        },
    )
    nominal_thigh_calf = RewTerm(
        func=nominal_pose_thigh_calf, weight=-0.15,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_thigh_joint", ".*_calf_joint"]),
            "command_name": "base_velocity", "stand_still_scale": 5.0,
            "velocity_threshold": 0.3, "normalization": "command_xy_sq_floor",
            "normalization_floor": 1.0,
        },
    )
    swing_foot_clearance = RewTerm(
        func=swing_foot_clearance, weight=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "ray_cfg": SceneEntityCfg("height_scanner"),
            "command_name": "base_velocity",
            "target_clearance": 0.10, "foot_radius": 0.022, "radius": 0.10,
            "contact_threshold": 1.0, "min_air_time": 0.05,
            "command_threshold": 0.1,
        },
    )


def configure_current_rewards(cfg):
    swing = cfg.rewards.swing_foot_clearance
    swing.weight = cfg.swing_clearance.weight
    for key in ("target_clearance", "foot_radius", "radius", "contact_threshold",
                "min_air_time", "command_threshold"):
        swing.params[key] = getattr(cfg.swing_clearance, key)
    nominal = cfg.nominal_pose
    for name, weight, joint_names in (
        ("nominal_hip", nominal.hip_weight, nominal.hip_joint_regex),
        ("nominal_thigh_calf", nominal.thigh_calf_weight,
         [nominal.thigh_joint_regex, nominal.calf_joint_regex]),
    ):
        term = getattr(cfg.rewards, name)
        term.weight = weight
        term.params["asset_cfg"].joint_names = joint_names
        for key in ("stand_still_scale", "velocity_threshold", "normalization", "normalization_floor"):
            term.params[key] = getattr(nominal, key)


@configclass
class OracleCurrentEnvCfg(JepaBlindEnvCfg):
    scene: BlindSceneCfg = BlindSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: OracleObservationsCfg = OracleObservationsCfg()
    rewards: OracleCurrentRewardsCfg = OracleCurrentRewardsCfg()
    curriculum: OraclePathCurriculumCfg = OraclePathCurriculumCfg()
    progress: OracleProgressCfg = OracleProgressCfg()
    swing_clearance: OracleSwingClearanceCfg = OracleSwingClearanceCfg()
    nominal_pose: OracleNominalPoseCfg = OracleNominalPoseCfg()
    terrain_column_assignment_eps: float = 0.001

    def __post_init__(self):
        super().__post_init__()
        if not hasattr(self.rewards, "swing_foot_clearance"):
            return
        configure_current_rewards(self)


@configclass
class OracleCurrentEnvCfg_PLAY(JepaBlindEnvCfg_PLAY):
    scene: BlindSceneCfg = BlindSceneCfg(num_envs=32, env_spacing=2.5)
    observations: OracleObservationsCfg = OracleObservationsCfg()
    rewards: OracleCurrentRewardsCfg = OracleCurrentRewardsCfg()
    curriculum: OraclePathCurriculumCfg = OraclePathCurriculumCfg()
    progress: OracleProgressCfg = OracleProgressCfg()
    swing_clearance: OracleSwingClearanceCfg = OracleSwingClearanceCfg()
    nominal_pose: OracleNominalPoseCfg = OracleNominalPoseCfg()
    terrain_column_assignment_eps: float = 0.001

    def __post_init__(self):
        super().__post_init__()
        configure_current_rewards(self)


@configclass
class OracleCurrentEasyStartEnvCfg(OracleCurrentEnvCfg):
    easy_start: EasyStartTerrainCfg = EasyStartTerrainCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = make_easy_start_terrain(
            self.scene.terrain.terrain_generator, self.easy_start,
        )
        self.progress.warmup_levels = self.easy_start.warmup_rows


@configclass
class OracleCurrentEasyStartEnvCfg_PLAY(OracleCurrentEasyStartEnvCfg):
    scene: BlindSceneCfg = BlindSceneCfg(num_envs=32, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        configure_easy_start_play(self)


def configure_easy_start_play(cfg):
    """Current와 Current+Future 평가가 같은 명령·지형·이벤트 설정을 쓰게 한다."""
    cfg.commands.base_velocity.initial_ranges = cfg.commands.base_velocity.final_ranges.copy()
    cfg.commands.base_velocity.ranges = cfg.commands.base_velocity.final_ranges.copy()
    cfg.scene.terrain.max_init_terrain_level = cfg.easy_start.num_rows - 1
    cfg.observations.policy.enable_corruption = False
    cfg.events.push_robot = None


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


def configure_future_observation(cfg):
    """미래 보간 전 clip과 최종 관측 clip에 같은 cfg 값을 적용한다."""
    scan = cfg.observations.terrain_future.scan
    scan.params["horizon_s"] = cfg.future_horizon_s
    scan.params["clip_range"] = scan.clip


@configclass
class OracleWideEnvCfg(OracleCurrentEnvCfg):
    scene: OracleWideSceneCfg = OracleWideSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: OracleWideObservationsCfg = OracleWideObservationsCfg()
    rewards: JepaRewardsCfg = JepaRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.wide_height_scanner.update_period = self.sim.dt


@configclass
class OracleCurrentFutureEnvCfg(OracleCurrentEnvCfg):
    scene: OracleWideSceneCfg = OracleWideSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: OracleCurrentFutureObservationsCfg = OracleCurrentFutureObservationsCfg()
    rewards: JepaRewardsCfg = JepaRewardsCfg()
    future_horizon_s: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self.scene.wide_height_scanner.update_period = self.sim.dt
        configure_future_observation(self)


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


@configclass
class OracleCurrentFutureEasyStartEnvCfg(OracleCurrentEasyStartEnvCfg):
    """Current-EasyStart 보상·지형·진단을 공유하는 Current+Future teacher."""

    scene: OracleWideSceneCfg = OracleWideSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: OracleCurrentFutureObservationsCfg = OracleCurrentFutureObservationsCfg()
    future_horizon_s: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        self.scene.wide_height_scanner.update_period = self.sim.dt
        configure_future_observation(self)


@configclass
class OracleCurrentFutureEasyStartEnvCfg_PLAY(OracleCurrentFutureEasyStartEnvCfg):
    scene: OracleWideSceneCfg = OracleWideSceneCfg(num_envs=32, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()
        configure_easy_start_play(self)
