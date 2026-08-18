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
class JELocoPCFootholdEnvCfg(JELocoPCEnvCfg):
    """Stage 2 — 계단 정밀 foothold. 계단 위주 지형 + foothold 보상 2항.

    규율: foot_scan(발밑 GT height)은 **보상 계산에만**(privileged, 배포 시 제거). 정책 입력(actor obs)은
    point cloud(전방 카메라)만 — Stage 1 과 동일, real 배포 가능. A(recon)·B(jepa) 를 이 env 로 재학습해
    "예측이 결손 하 발딛기를 재구성보다 정확히 하나"(예측이 이길 축)를 시험.
    """

    def __post_init__(self):
        super().__post_init__()

        # ── 발밑 dense 스캔 (보상 전용 privileged. 정책 관측 아님) ──
        self.scene.foot_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.04, size=[1.0, 0.7], ordering="yx"),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.foot_scan.update_period = self.decimation * self.sim.dt

        # ── 지형 = 계단 + gap + 불규칙블록 + 디딤돌 + 계단구멍 (시각-필수·비정형·결손민감 다양화) ──
        # foothold 의 핵심은 "안 보이면 빠지는" gap/디딤돌/구멍 — 여기서 결손 시 예측(B)이 재구성(A)보다
        # 유리할 여지가 큼(가려진 발밑을 예측). 계단만으론 그 차이가 약함.
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.2, step_height_range=(0.03, 0.15), step_width=0.30,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2, step_height_range=(0.03, 0.15), step_width=0.30,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
            # 계단에 구멍(발 빠지면 실패) — foothold 정밀도가 진짜 필요
            "stairs_holes": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.1, step_height_range=(0.03, 0.12), step_width=0.32,
                platform_width=3.0, border_width=1.0, holes=True,
            ),
            # gap: 착지 잘못하면 빠지는 구멍
            # gap 폭 상한 0.3m — Go2 trot 보폭 한계(~0.3m) 내. 0.5m 는 점프 없이 불가능해
            # teacher 가 가장자리서 엉거주춤(과제 불능, 2026-08-13 play 관찰). platform 2.0 = 도움닫기.
            "gaps": terrain_gen.MeshGapTerrainCfg(
                proportion=0.1, gap_width_range=(0.05, 0.3), platform_width=2.0,
            ),
            # 불규칙 높이 블록 — 발 위치 예측 어려움
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.1, grid_width=0.45, grid_height_range=(0.03, 0.15), platform_width=2.0,
            ),
            # 디딤돌 — foothold 연구 정석(돌 위에만 정확히 딛기)
            "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
                proportion=0.1, stone_height_max=0.1, stone_width_range=(0.3, 0.5),
                stone_distance_range=(0.05, 0.2), holes_depth=-1.0, platform_width=2.0,
            ),
        }

        # ── foothold 보상 2항 (단일 변수: Stage 1 대비 이 둘만 추가) ──
        foot = SceneEntityCfg("robot", body_names=".*_foot")
        hs = SceneEntityCfg("foot_scan")
        cs = SceneEntityCfg("contact_forces", body_names=".*_foot")
        # ① 지형 상대 클리어런스(사용자 제안): 스윙 발이 발밑 지형 위로 충분히 뜨게
        self.rewards.foot_clearance_terrain = RewTerm(
            func=mdp.foot_clearance_terrain, weight=1.0,
            params={"height_sensor_cfg": hs, "asset_cfg": foot,
                    "target_clearance": 0.10, "std": 0.05, "tanh_mult": 2.0},
        )
        # ② foothold safety: 착지점이 면 중앙(평탄)이면 +, edge/gap 이면 −
        self.rewards.foothold_safety = RewTerm(
            func=mdp.foothold_safety, weight=0.5,
            params={"height_sensor_cfg": hs, "contact_sensor_cfg": cs, "asset_cfg": foot, "edge_scale": 12.0},
        )
        # 기존 절대높이 클리어런스는 계단에서 부적합 → 제거(지형 상대판으로 대체)
        self.rewards.foot_clearance = None

        # ── 보상 균형·gait·명령 (사용자 관찰 반영. 값은 knob) ──
        # ③ 대각선 trot: offset[0,0.5,0.5,0]=FL+RR / FR+RL(이미 대각선). 가중치 0.2→0.75.
        self.rewards.feet_gait.weight = 0.75
        # ★ 추종 vs 발동작 균형 재조정(2026-08-11): 발동작 보상이 속도·자세 추종을 압도해
        #   error_vel_xy 0.5·yaw 0.85 로 커짐(정책이 '명령대로 걷기'보다 '발 딛기' 우선). →
        #   추종 대폭↑ + 발동작↓ 로 "걸으면서 딛기"가 되게. (foothold 는 걷기 대신이 아님)
        self.rewards.track_lin_vel_xy.weight = 3.0    # 선속도 추종 (xy 0.5 대응)
        self.rewards.track_ang_vel_z.weight = 2.0     # 회전 추종 강화 (yaw 0.85 대응, 기본 0.75)
        self.rewards.flat_orientation_l2.weight = -4.0  # 자세 불안정 억제 (기본 −2.5)
        self.rewards.foot_clearance_terrain.weight = 0.6   # 발동작 비중↓ (1.0→0.6)
        self.rewards.foothold_safety.weight = 0.35         # 〃 (0.5→0.35)

        # ★★ teacher 와 정합 (2026-08-17 play 진단): student 셋(jepa/recon/scratch) 모두 다리를
        # 바깥으로 벌린 '거미 자세'로 수렴 → 표현 비교가 오염됨. 원인 = 조상 JELocoPCEnvCfg 의
        # 옛 설정 잔재를 물려받은 것. teacher(RobotEnvCfg 직접 상속)는 순정값이라 멀쩡했음.
        #   joint_pos −0.3 → −0.7 : 기본자세 이탈 페널티. 약하면 지지다각형 넓히기로 도망감(거미).
        #   action_rate −0.05 → −0.1 : 순정값 복원(동작 부드러움).
        #   명령: 회전·횡이동 복원 — 0 으로 막으면 회전을 못 배워 지형서 몸 틀어져도 못 잡음.
        self.rewards.joint_pos.weight = -0.7
        self.rewards.action_rate.weight = -0.1
        self.rewards.feet_air_time.weight = 0.5   # 발 시원하게 들기(teacher 와 동일. 옛 0.25 잔재)
        cmd = self.commands.base_velocity
        cmd.rel_standing_envs = 0.05
        cmd.ranges.lin_vel_x = (0.0, 0.8)
        cmd.limit_ranges.lin_vel_x = (0.0, 1.5)
        cmd.ranges.lin_vel_y = (-0.2, 0.2)
        cmd.limit_ranges.lin_vel_y = (-0.4, 0.4)
        cmd.ranges.ang_vel_z = (-0.5, 0.5)
        cmd.limit_ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class JELocoPCFootholdPlayEnvCfg(JELocoPCFootholdEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.max_init_terrain_level = 3
        self.observations.policy.enable_corruption = False
        self.observations.pointcloud.enable_corruption = False
        self.events.push_robot = None


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


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1 — Teacher (privileged heightmap 으로 험지를 '잘' 걷는 정책)
#
# 목적: Phase 2(오프라인 JEPA/recon 사전학습)의 데이터 공장. 시각(pc) 없이 GT heightmap 을
# actor 에 직접 줘서(=teacher) 계단·gap·디딤돌에서 깔끔한 보행을 확립한다.
# 설계 근거(EXPERIMENTS.md·2026-08-11/12 논의):
#   - 순정 Go2-Velocity 는 잘 걸음(사용자 육안 확인) → 그 명령·보상 골격에서 출발
#   - je_loco 가 좁힌 명령(회전0·횡0·0.3~0.6)이 더듬거림·yaw 불안정의 원인 → 정상화
#   - 걷기와 foothold 동시학습이 기형 유발 → teacher 는 순수 PPO(순정 train 파이프라인)
# 통과 게이트: G1 추종(xy<0.25,yaw<0.2) · G2 play 육안(대각선 trot·발 들림) · G3 계단 12cm+
# 학습: train_pc.py --task Unitree-Go2-JELoco-Teacher (JELoco aux 는 hasattr 가드로 전부 OFF,
#       plain MLP PPO 로 동작. pc 렌더 없음 → 4096 env 가능)
# ═════════════════════════════════════════════════════════════════════════════
@configclass
class TeacherObservationsCfg(ObservationsCfg):
    """순정 관측 + privileged height_scan(187=17×11) 을 actor·critic 모두에 (teacher 의 '완벽한 눈')."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        # privileged — noise 없음(항목별 noise 미지정이라 corruption 켜져도 clean)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
            clip=(-1.0, 1.5),
        )

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "offset": 0.0},
            clip=(-1.0, 1.5),
        )

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class JELocoTeacherEnvCfg(RobotEnvCfg):
    """Phase 1 Teacher — 순정 RobotEnvCfg 직접 상속(pc·GRU·스크립트커리큘럼 무관). 적응형 커리큘럼."""

    observations: TeacherObservationsCfg = TeacherObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # ── height_scanner 전방 이동: 커버 x∈[−0.5,+1.1] (몸 아래 + 전방 발딛기 계획) ──
        self.scene.height_scanner.offset = RayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 20.0))

        # ── foot_scan (foothold 보상 전용 privileged. 관측 아님) ──
        self.scene.foot_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.04, size=[1.0, 0.7], ordering="yx"),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.foot_scan.update_period = self.decimation * self.sim.dt

        # ── 지형 = foothold 8종 (Foothold env 와 동일 분포 — student 가 뛸 지형에서 teacher 도 걸어야) ──
        self.scene.terrain.terrain_generator.sub_terrains = {
            "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.1),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
            ),
            "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
                proportion=0.2, step_height_range=(0.03, 0.15), step_width=0.30,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
            "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.2, step_height_range=(0.03, 0.15), step_width=0.30,
                platform_width=3.0, border_width=1.0, holes=False,
            ),
            "stairs_holes": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
                proportion=0.1, step_height_range=(0.03, 0.12), step_width=0.32,
                platform_width=3.0, border_width=1.0, holes=True,
            ),
            # gap 폭 상한 0.3m — Go2 trot 보폭 한계(~0.3m) 내. 0.5m 는 점프 없이 불가능해
            # teacher 가 가장자리서 엉거주춤(과제 불능, 2026-08-13 play 관찰). platform 2.0 = 도움닫기.
            "gaps": terrain_gen.MeshGapTerrainCfg(
                proportion=0.1, gap_width_range=(0.05, 0.3), platform_width=2.0,
            ),
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.1, grid_width=0.45, grid_height_range=(0.03, 0.15), platform_width=2.0,
            ),
            "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
                proportion=0.1, stone_height_max=0.1, stone_width_range=(0.3, 0.5),
                stone_distance_range=(0.05, 0.2), holes_depth=-1.0, platform_width=2.0,
            ),
        }
        # (커리큘럼: RobotEnvCfg 기본 = 적응형 terrain_levels_vel. teacher 는 단일정책이라 통제 불필요)

        # ── 명령 정상화 (je_loco 가 죽인 회전·횡이동 복원. 후진만 제외 — 사용자 결정) ──
        cmd = self.commands.base_velocity
        cmd.rel_standing_envs = 0.05
        cmd.ranges.lin_vel_x = (0.0, 0.8)
        cmd.limit_ranges.lin_vel_x = (0.0, 1.5)
        cmd.ranges.lin_vel_y = (-0.2, 0.2)
        cmd.limit_ranges.lin_vel_y = (-0.4, 0.4)
        cmd.ranges.ang_vel_z = (-0.5, 0.5)
        cmd.limit_ranges.ang_vel_z = (-1.0, 1.0)

        # ── 보상: fhA3 재조정판(추종 우선) + 발 들기 강화 + 대각선 trot + foothold(저가중) ──
        self.rewards.track_lin_vel_xy.weight = 3.0
        self.rewards.track_ang_vel_z.weight = 2.0
        self.rewards.flat_orientation_l2.weight = -4.0
        # feet_air_time: 짧은 스텝 페널티 성격(실현값 음수) → 가중치↑ = 보폭·체공 늘리는 압력
        self.rewards.feet_air_time.weight = 0.5
        foot = SceneEntityCfg("robot", body_names=".*_foot")
        hs = SceneEntityCfg("foot_scan")
        cs = SceneEntityCfg("contact_forces", body_names=".*_foot")
        # 대각선 trot (base RewardsCfg 엔 없음 → 신설. offset[0,.5,.5,0]=FL+RR/FR+RL)
        self.rewards.feet_gait = RewTerm(
            func=mdp.feet_gait, weight=0.75,
            params={"period": 0.5, "offset": [0.0, 0.5, 0.5, 0.0], "sensor_cfg": cs,
                    "threshold": 0.5, "command_name": "base_velocity"},
        )
        # foothold 2종 (motion-gated — 정지 편법 차단 검증됨). teacher 가 디딤돌·gap 에서
        # 발을 골라 딛어야 Phase 2 데이터가 '정답 foothold' 를 담음. 기형 재발 시 1번 제거 knob.
        self.rewards.foot_clearance_terrain = RewTerm(
            func=mdp.foot_clearance_terrain, weight=0.6,
            params={"height_sensor_cfg": hs, "asset_cfg": foot,
                    "target_clearance": 0.10, "std": 0.03, "tanh_mult": 4.0},
        )
        self.rewards.foothold_safety = RewTerm(
            func=mdp.foothold_safety, weight=0.35,
            params={"height_sensor_cfg": hs, "contact_sensor_cfg": cs, "asset_cfg": foot,
                    "edge_scale": 12.0, "vel_gate": 0.2},
        )


@configclass
class JELocoTeacherPlayEnvCfg(JELocoTeacherEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 10
            self.scene.terrain.terrain_generator.num_cols = 8
            self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.max_init_terrain_level = 5
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None


# ═════════════════════════════════════════════════════════════════════════════
# Phase 3b — DAgger 증류 (2026-08-18 전환)
#
# 배경: Phase 3(인코더 동결 + PPO 재학습)은 **세 조건 모두 제대로 못 걷는** 결과로 끝났다
# (S2 iter 4000: 지표·육안 모두 무차별, teacher 는 잘 걸음). LITERATURE.md 축 3 이 그대로
# 예측한 실패 — 우리가 **행동 감독**과 **on-policy 보정**을 둘 다 뺐기 때문. 표준 레시피는
# DAgger: 환경을 [학생 행동]으로 굴리고 방문 상태마다 teacher 에게 a_t 를 물어 ‖â−a‖² 최소화.
#
# 연구질문은 유지된다 — jepa/recon/scratch 를 **인코더 초기화**로 비교. 이는 DeFM(2026) 의
# 설계와 동일(동결 DeFM vs scratch CNN, 단 distillation 안에서). 차이가 날 자리는 결손 평가.
#
# 관측 그룹: student = policy(45×5) + pointcloud(768) / teacher = teacher(45+187=232)
# ═════════════════════════════════════════════════════════════════════════════
@configclass
class DistillObservationsCfg(PCObservationsCfg):
    """PC 관측(학생) + teacher 관측 그룹을 **한 env 에서 동시** 제공."""

    @configclass
    class TeacherCfg(ObservationsCfg.PolicyCfg):
        """Phase 1 teacher 가 학습 때 본 것과 **완전히 동일**해야 한다(45 + 187 = 232).

        - 항목 순서: ObservationsCfg.PolicyCfg 6항 + height_scan (TeacherObservationsCfg 와 동일한
          상속 방식이라 dataclass 필드 순서가 그대로 재현됨).
        - history 없음(H=1): teacher env 는 policy 그룹에 history_length 를 설정하지 않았다.
          (이 env 의 `policy` 그룹은 학생용이라 history_length=5 — 별도 그룹이라 충돌 없음.)
        - corruption: PolicyCfg.__post_init__ 이 True 로 켬 = teacher 학습 시와 동일한 noise.
          라벨 분포를 학습 때와 맞추기 위해 그대로 둔다.
        """

        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("teacher_scanner"), "offset": 0.0},
            clip=(-1.0, 1.5),
        )

    teacher: TeacherCfg = TeacherCfg()


@configclass
class JELocoDistillEnvCfg(JELocoPCFootholdEnvCfg):
    """Foothold env + teacher 관측. 보상·지형·명령은 Foothold 와 동일(증류엔 보상 미사용이나
    커리큘럼·종료조건이 동일해야 teacher 가 자기 학습 분포와 비슷한 상태를 본다)."""

    observations: DistillObservationsCfg = DistillObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        # ── teacher 전용 스캐너 ────────────────────────────────────────────
        # Phase 1 teacher 학습 시 geometry 와 **정확히 동일**해야 체크포인트가 의미를 갖는다:
        #   RobotEnvCfg 기본 격자(resolution 0.1, size [1.6,1.0] → 17×11 = 187, **기본 ordering**)
        #   + JELocoTeacherEnvCfg 의 전방 이동 offset (0.3, 0, 20).
        # 이 env 의 `height_scanner` 는 Head A 재구성 타깃용(144, ordering="yx", offset 1.25)이라
        # 재사용 불가 — 반드시 별도 센서.
        self.scene.teacher_scanner = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.3, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )
        self.scene.teacher_scanner.update_period = self.decimation * self.sim.dt


@configclass
class JELocoDistillPlayEnvCfg(JELocoDistillEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.max_init_terrain_level = 3
        self.observations.policy.enable_corruption = False
        self.observations.pointcloud.enable_corruption = False
        self.observations.teacher.enable_corruption = False
        self.events.push_robot = None
