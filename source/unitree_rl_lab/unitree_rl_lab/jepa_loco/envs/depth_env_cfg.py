"""Phase 2 — Go2 depth + proprio PPO 기준선 환경 (CLAUDE.md §2–§6).

관측 그룹:
  policy      : proprio 45 (ang vel 3, gravity 3, cmd 3, joint pos 12, joint vel 12, last action 12), 노이즈 포함
  depth       : [N, 2, 64, 112] (정규화 depth, mask), 10 Hz + 지연 + 노이즈
  depth_fresh : [N, 1] 새 프레임 플래그
  critic      : 특권 정보 (proprio + 선속도 + 관절 토크 + heightscan). 이후 privileged teacher 에도 재사용 가능.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    CommandsCfg,
    ObservationsCfg,
    RobotEnvCfg,
    RobotSceneCfg,
    RewardsCfg,
)

from ..sensors.d435i_cfg import D435iParams, make_tiled_camera_cfg
from .command_schedule import ScheduledVelocityCommandCfg
from .depth_obs import DepthFrame, depth_fresh

D435I = D435iParams()

# 지형: 열 = 종류, 행 = 난이도(curriculum). 계단 8→20 cm, gap 10→30 cm (CLAUDE.md §6).
# IsaacLab 관례: 피라미드 계단은 중앙 꼭대기에서 출발 → 내려가는 계단, 역피라미드는 구덩이 출발 → 오르는 계단.
JEPA_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
        "rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "stairs_up": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.25, step_height_range=(0.08, 0.20), step_width=0.3, platform_width=3.0, border_width=1.0
        ),
        "stairs_down": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25, step_height_range=(0.08, 0.20), step_width=0.3, platform_width=3.0, border_width=1.0
        ),
        "gap": terrain_gen.MeshGapTerrainCfg(proportion=0.2, gap_width_range=(0.10, 0.30), platform_width=3.0),
    },
)


@configclass
class DepthNoiseCfg:
    """depth_noise() 인자. 초기값 — 실기 통계로 조정."""

    gauss_coef: float = 0.01
    edge_rel_thresh: float = 0.2
    clip_far: float = D435I.clip_far
    edge_drop_prob: float = 0.5
    hole_prob: float = 0.5
    max_holes: int = 3
    hole_size_px: tuple[int, int] = (2, 8)
    left_band: bool = True
    baseline: float = D435I.baseline


@configclass
class JepaSceneCfg(RobotSceneCfg):
    d435i = make_tiled_camera_cfg(D435I).replace(update_period=0.0)


@configclass
class JepaObservationsCfg:
    policy: ObservationsCfg.PolicyCfg = ObservationsCfg.PolicyCfg()

    @configclass
    class DepthCfg(ObsGroup):
        frame = ObsTerm(
            func=DepthFrame,
            params={
                "sensor_cfg": SceneEntityCfg("d435i"),
                "fx": D435I.intrinsics(D435I.render_wh)[0],
                "near": D435I.min_range,
                "far": D435I.clip_far,
                "delay_range": (1, 3),  # 실제 지연(제어 스텝), CLAUDE.md §3
                "inherent_delay": 0,  # ★ IsaacLab 고유 지연 d0 — 측정 후 갱신
                "noise": DepthNoiseCfg().to_dict(),
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class FreshCfg(ObsGroup):
        fresh = ObsTerm(func=depth_fresh)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan, params={"sensor_cfg": SceneEntityCfg("height_scanner")}, clip=(-1.0, 1.0)
        )

    depth: DepthCfg = DepthCfg()
    depth_fresh: FreshCfg = FreshCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class JepaCommandsCfg(CommandsCfg):
    """처음 500 iteration에 걸쳐 명세의 명령 범위까지 선형 확장."""

    base_velocity = ScheduledVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        heading_command=False,
        debug_vis=True,
        ranges=ScheduledVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-0.1, 0.1)
        ),
    )


@configclass
class JepaRewardsCfg(RewardsCfg):
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)


@configclass
class JepaCurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class JepaDepthEnvCfg(RobotEnvCfg):
    scene: JepaSceneCfg = JepaSceneCfg(num_envs=1024, env_spacing=2.5)
    observations: JepaObservationsCfg = JepaObservationsCfg()
    commands: JepaCommandsCfg = JepaCommandsCfg()
    rewards: JepaRewardsCfg = JepaRewardsCfg()
    curriculum: JepaCurriculumCfg = JepaCurriculumCfg()

    depth_period_steps: int = 5
    """depth 갱신 주기 [제어 스텝]. 50 Hz / 5 = 10 Hz."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = JEPA_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        # 렌더링은 depth 주기에만 (물리 스텝 단위)
        self.sim.render_interval = self.decimation * self.depth_period_steps


@configclass
class JepaDepthEnvCfg_PLAY(JepaDepthEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.initial_ranges = self.commands.base_velocity.final_ranges.copy()
        self.commands.base_velocity.ranges = self.commands.base_velocity.final_ranges.copy()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        # 육안 판정이 쉬운 지형에만 머물지 않도록 전 난이도에 스폰
        self.scene.terrain.max_init_terrain_level = self.scene.terrain.terrain_generator.num_rows - 1
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
