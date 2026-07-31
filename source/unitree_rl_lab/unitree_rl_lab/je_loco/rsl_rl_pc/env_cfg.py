# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""JE-Loco point cloud 속도추종 env — 검증된 Go2 velocity 위에 pointcloud 관측 그룹 추가.

관측 그룹: policy(45, actor) / pointcloud(192×4=768, [x,y,z,valid]) / critic(60, privileged).
scene: 표준 velocity 씬 + pc_scanner 전방 프러스텀(192점) + height_scanner 전방정렬 격자(16×9=144, Head A GT).
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.go2.velocity_env_cfg import (
    ObservationsCfg,
    RewardsCfg,
    RobotEnvCfg,
    RobotSceneCfg,
)

from . import mdp_pc


@configclass
class PCSceneCfg(RobotSceneCfg):
    """velocity 씬 + D435i **전방 프러스텀** point cloud(정책 입력) + top-down height_scanner(Head A GT).

    pc_scanner: 실기 D435i(전방 32.5cm·위 4.5cm 장착, 35° 하향, FoV 78.7°×63.1°, max 3m) 시야를
    raycaster 프러스텀으로 재현(렌더 없이). ray_alignment='base' 로 몸 자세 따라감(실기 강체 장착).
    """

    # (1) 정책 입력 — D435i 전방 프러스텀 (16×12=192점)
    pc_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.325, 0.0, 0.045), rot=mdp_pc.tilt_quat_y(35.0)),
        ray_alignment="base",
        pattern_cfg=mdp_pc.FrustumPatternCfg(hfov_deg=78.7, vfov_deg=63.1, width=16, height=12),
        max_distance=2.0,          # D435i locomotion 유효거리(근거리 집중) → 그 너머 무효(홀)
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # (2) Head A 재구성 GT — top-down 격자를 **카메라 전방 footprint 에 정렬**(16×9=144).
    # 이전(2026-07 이전): base 중심 x∈[−0.55,0.55] → pc_scanner 시야(x∈[0.5,2.3])와 8셀만 겹침.
    # = Head A 가 관측 불가능한 지형(타깃의 ~92%)을 재구성하도록 학습 → baseline crippling.
    # 정렬: 중심을 base 앞 1.25m 로 이동, x∈[0.5,2.0]·y∈[−0.4,0.4] → 전부 카메라 시야 안.
    # (검증: height_scan = sensor_z − hit_z − offset 은 z 만 쓰므로 xy 이동이 높이값 안 깨뜨림.
    #  recon_decoder 출력차원은 model.py 가 height_map dim=144 로 자동 조정.)
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(1.25, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.5, 0.8], ordering="yx"),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


@configclass
class PCObservationsCfg(ObservationsCfg):
    """표준 policy(45)/critic(60) + pointcloud(768=192×4 [x,y,z,valid], 정책입력) + height_map(144, GT)."""

    @configclass
    class PointCloudCfg(ObsGroup):
        point_cloud = ObsTerm(
            func=mdp_pc.raycaster_pointcloud,
            params={"sensor_cfg": SceneEntityCfg("pc_scanner")},
            noise=Unoise(n_min=-0.02, n_max=0.02),
            clip=(-2.0, 2.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class HeightMapCfg(ObsGroup):
        """Head A 재구성 타깃 — 카메라 전방 footprint 에 정렬된 **clean** height map(144). 정책 입력 아님.

        노이즈 있는 pointcloud(정책 입력)로부터 이 clean height map 을 복원하도록 z_e 를 학습
        → graceful degradation(depth 결손에도 지형 표현 유지)의 기반. 타깃 격자는 height_scanner
        정의(전방 x∈[0.5,2.0]·y∈[−0.4,0.4], 16×9)를 따름 — pc_scanner 시야 안이라 재구성 가능.
        """

        # offset=0: raycaster pos_w 가 base(~0.35) 를 보고하므로 height = base_z − terrain_z
        # (≈0.3, 지형 상대높이). offset=20 이면 pos_w 가 20 을 안 담아 −19.65 로 어긋남(버그).
        # clip: 어려운 지형에서 일부 광선이 지형 못 맞힘→ray_hits=inf→height=inf 방지.
        # 무효 셀은 −1.0(홀 sentinel)로 클리핑. 유효 범위(대략 −0.2~0.9)는 그대로.
        height_map = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
            clip=(-1.0, 1.5),
        )

        def __post_init__(self):
            self.enable_corruption = False   # GT: 노이즈 없음
            self.concatenate_terms = True

    pointcloud: PointCloudCfg = PointCloudCfg()
    height_map: HeightMapCfg = HeightMapCfg()


@configclass
class PCRewardsCfg(RewardsCfg):
    """표준 velocity 보상 + feet_gait(trot 접촉 패턴 양의 보상) 추가.

    stair 실험 참고: 명령이 있을 때(cmd>0.1) trot 위상에 맞춰 발을 딛으면 +보상 →
    **스텝을 밟는 게 서 있는 것보다 이득** → stand-still local optimum 탈출의 핵심.
    """

    feet_gait = RewTerm(
        func=mdp.feet_gait,
        weight=0.2,
        params={
            "period": 0.5,
            "offset": [0.0, 0.5, 0.5, 0.0],  # trot (대각선 쌍)
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "threshold": 0.5,
            "command_name": "base_velocity",
        },
    )

    # stair 의 핵심 스텝 보상 — 발이 **수평 이동하며** 목표 높이까지 들리면 +보상.
    # foot_velocity_tanh 항 때문에 제자리(발 안 움직임)엔 보상 없음 → march-in-place 직접 억제.
    foot_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,                  # 0.5→1.0 (stair값) — 발 들기 유인 강화
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "target_height": 0.15,   # 계단(최대 18cm)·박스 넘게 발 높이 상향(0.06→0.12→0.15)
            "std": 0.05,
            "tanh_mult": 2.0,
        },
    )


@configclass
class JELocoPCEnvCfg(RobotEnvCfg):
    """point cloud 속도추종 env (rsl_rl 학습용)."""

    scene: PCSceneCfg = PCSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: PCObservationsCfg = PCObservationsCfg()
    rewards: PCRewardsCfg = PCRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.pc_scanner.update_period = self.decimation * self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # proprio 관측을 H=5 히스토리로 → obs["policy"] = 45×5 = 225. proprio_encoder→z_p 입력.
        # (논문: proprio 45-d, history H=5 → z_p. 실기 배포 시 5프레임 버퍼 필요.)
        self.observations.policy.history_length = 5
        self.observations.policy.flatten_history_dim = True

        # ── rough 지형(set A) 활성화 — depth/pc 가 인지할 지형 구조 확보 ──
        # 기본 velocity_env_cfg 는 flat 전용(외수용 무의미 → 표현 헤드 공회전).
        # configclass 인스턴스별 deep-copy 라 sub_terrains 교체는 공유 cfg 오염 없음.
        # curriculum 플래그는 super().__post_init__ 에서 이미 True 설정됨.
        # 쉬운 지형(flat+rough) 비중 45% → 갓 태어난 정책이 기본 보행 먼저 학습(공유 정책 전이).
        # 계단/박스는 줄이되 유지(0.05~0.18m, 커리큘럼이 난이도로 스케일). 오르막 계단 강조.
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.2),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.25, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
            ),
            "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
            ),
            "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
                proportion=0.1, slope_range=(0.0, 0.3), platform_width=2.0, border_width=0.25
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.1, grid_width=0.45, grid_height_range=(0.025, 0.12), platform_width=2.0
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(  # 중앙 높음 → 내려감(descending)
                proportion=0.05, step_height_range=(0.025, 0.12), step_width=0.30,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(  # 중앙 낮음 → 올라감(ascending)
                proportion=0.2, step_height_range=(0.025, 0.12), step_width=0.30,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
        }

        # ── 개방루프 커리큘럼 (헤드/seed 간 훈련 지형 분포 통제) ──
        # 표준 terrain_levels_vel 은 성능 의존이라 A/B·seed 마다 훈련 지형이 달라진다(내생적 교란).
        # 실측(2026-07): Head B 는 seed 별로 커리큘럼을 0(실패)~5.0(정상) 로 요동 → A/B 비교 불가.
        # scripted_terrain_levels 로 교체: 난이도 상한을 iteration 함수로 고정 → 모든 run 동일 분포.
        # steps_per_iter 는 agent_cfg 의 num_steps_per_env(=32) 와 반드시 일치.
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.scripted_terrain_levels,
            params={"steps_per_iter": 32},
        )
        # scripted 는 매 reset 시 [0, C(t)] 로 직접 배정하므로 max_init_terrain_level 은 무의미.
        self.scene.terrain.max_init_terrain_level = 0

        # ── 전진 명령 (stair 방식: 항상 ≥0.1 전진, 제자리·회전 없음) ──
        # 서 있기 차단의 3요소: (1) min>0 항상 전진, (2) rel_standing_envs=0 제자리 명령 없음,
        # (3) PCRewardsCfg.feet_gait 스텝 보상. 이전 (0,0.6)+standing0.1+yaw±0.5 는 작은 명령을
        # 서서 추종 가능 → stand-still 수렴 원인. yaw=0 은 순변위 확보(terrain 승급)에도 유리.
        cmd = self.commands.base_velocity
        cmd.rel_standing_envs = 0.0
        cmd.ranges.lin_vel_x = (0.3, 0.6)      # 하한 0.3 — 제자리로는 추종 불가(전진 강제)
        cmd.ranges.lin_vel_y = (0.0, 0.0)
        cmd.ranges.ang_vel_z = (0.0, 0.0)
        cmd.limit_ranges.lin_vel_x = (0.3, 1.0)
        cmd.limit_ranges.lin_vel_y = (0.0, 0.0)
        cmd.limit_ranges.ang_vel_z = (0.0, 0.0)

        # 움직임 억제 완화(stair 값) → 걷기에 필요한 관절 움직임/동역학 허용.
        #  - joint_pos -0.7→-0.3: 기본 자세 이탈 페널티. -0.7 은 다리를 못 움직이게 해
        #    "제자리 움찔거림(march-in-place)"의 주원인 → stair 처럼 -0.3.
        #  - action_rate -0.1→-0.05.
        self.rewards.joint_pos.weight = -0.3
        self.rewards.action_rate.weight = -0.05
        # 보폭 확대(더듬듯 짧게 → 성큼) — swing 체공시간 보상 강화
        self.rewards.feet_air_time.weight = 0.25   # 0.1→0.25


@configclass
class JELocoPCPlayEnvCfg(JELocoPCEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        if self.scene.terrain.terrain_generator is not None:
            # 학습과 동일한 난이도 범위(10행) + 5개 타입 전부(5열) 를 보이게.
            # (기존 2×2 는 레벨 0~1·타입 2개만 → 평지+슬로프만 보였음)
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = True
        # 학습 도달 레벨(~5)에서 스폰 → 실제 훈련한 rough 지형에서 걷는 모습 확인
        self.scene.terrain.max_init_terrain_level = 5
        self.observations.policy.enable_corruption = False
        self.observations.pointcloud.enable_corruption = False
        self.events.push_robot = None
        # point cloud 검증용 — raycaster hit 점을 GUI 에 마커로 표시(로봇 따라다니며 지형 hit)
        self.scene.pc_scanner.debug_vis = True
