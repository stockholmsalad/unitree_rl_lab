# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""JE-Loco observation specification — single source of truth for dimensions & layout.

CLAUDE.md 절대 규율: proprioception 관측 = **45차원** (DreamWaQ / START / TRANS 공통).

    o_t = [ ω_t(3), g_t(3), cmd_t(3), θ_t(12), θ̇_t(12), a_{t-1}(12) ] ∈ R^45

base **선속도 v_t 는 이 45차원에 넣지 않는다** (실기 드리프트·노이즈).
    - critic(privileged) 에만 GT v_t 를 준다 (asymmetric actor-critic).
    - proprio VAE 가 z_p 와 함께 v̂_t 를 추정한다 (CENet 방식, L_est = MSE(v̂, v)).
    - 즉 선속도는 "입력 차원"이 아니라 "추정 출력". 47차원으로 맞추지 말 것.

이 파일은 학습 코드(models/*)와 Isaac Lab 환경(envs/*)이 **동일한 인덱스 규약**을
공유하도록 강제하는 유일한 정의처다. 환경의 ObservationTerm 순서는 반드시 아래
`PROPRIO_TERMS` 순서와 일치해야 한다.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# --------------------------------------------------------------------------------------------- #
# Proprioception (o_t) 항목 정의 — (이름, 차원). 순서 = 환경 ObsTerm 순서 = 관측 벡터 슬라이스 순서.
# --------------------------------------------------------------------------------------------- #
PROPRIO_TERMS: list[tuple[str, int]] = [
    ("base_ang_vel", 3),      # ω_t   : base frame 각속도
    ("projected_gravity", 3),  # g_t   : 중력 투영 벡터 (base frame)
    ("velocity_commands", 3),  # cmd_t : (vx, vy, ωz) 명령
    ("joint_pos_rel", 12),     # θ_t   : 관절 위치 (default 대비 상대)
    ("joint_vel_rel", 12),     # θ̇_t   : 관절 속도
    ("last_action", 12),       # a_{t-1}: 직전 액션
]

PROPRIO_DIM: int = sum(d for _, d in PROPRIO_TERMS)  # == 45


def _index_ranges(terms: list[tuple[str, int]]) -> dict[str, tuple[int, int]]:
    """각 항목의 [start, end) 슬라이스 범위를 계산한다."""
    ranges: dict[str, tuple[int, int]] = {}
    cursor = 0
    for name, dim in terms:
        ranges[name] = (cursor, cursor + dim)
        cursor += dim
    return ranges


# 항목별 슬라이스 범위 (config·주석·디버깅에서 참조). 예: PROPRIO_INDEX["joint_pos_rel"] == (9, 21)
PROPRIO_INDEX: dict[str, tuple[int, int]] = _index_ranges(PROPRIO_TERMS)


@dataclass
class JELocoObsSpec:
    """JE-Loco 관측/센서 형상 명세. 환경·모델·config 가 공유하는 계약(contract)."""

    # -- proprioception --
    proprio_dim: int = PROPRIO_DIM                 # 45
    history_len: int = 5                           # H: DreamWaQ=5(100ms@50Hz), TRANS=10. 기본 5.

    # -- privileged (critic 전용) --
    base_lin_vel_dim: int = 3                      # GT v_t : critic privileged + VAE 추정 타깃

    # -- exteroception (depth 렌더 → point cloud, 주 경로: rule 2 / 2026-07-08) --
    depth_height: int = 48                         # D435i depth 이미지 세로 (TiledCamera 렌더 크기)
    depth_width: int = 64                          # D435i depth 이미지 가로
    depth_channels: int = 1                        # depth 렌더 (그래프 안에서 point cloud 로 deproject)
    ext_encoder: str = "pointcloud"                # "pointcloud"(기본) | "depth_cnn"(보조/ablation)
    num_points: int = 128                          # 프레임당 PointNet 입력 점 수 (depth 픽셀 서브샘플).
    #  총 점 수 = num_points × ext_memory_K (recompute-PPO 라 PointNet 활성화 메모리에 민감 → 작게).
    point_feat_dim: int = 3                        # 점당 차원 (xyz). 필요시 확장.
    ext_memory_K: int = 5                          # SE(3) body-frame 외수용 메모리 프레임 수 (DreamWaQ++ K=5)

    # -- GT height map (헤드 A 학습 신호 / probe decoder 타깃) --
    # 로봇 주변 격자 height scan. 헤드 A 는 z_e → 이 격자를 재구성한다.
    heightmap_rows: int = 21                       # x 방향 셀 수
    heightmap_cols: int = 21                       # y 방향 셀 수

    # -- latent 차원 --
    z_p_dim: int = 16                              # proprioception latent (내수용)
    z_e_dim: int = 32                              # exteroception latent (외수용) — A·B 간 동일 고정

    proprio_terms: list[tuple[str, int]] = field(default_factory=lambda: list(PROPRIO_TERMS))

    @property
    def proprio_history_dim(self) -> int:
        """VAE 입력으로 flatten 했을 때 차원 = H * 45."""
        return self.history_len * self.proprio_dim

    @property
    def depth_shape(self) -> tuple[int, int, int]:
        return (self.depth_channels, self.depth_height, self.depth_width)

    @property
    def ext_points_total(self) -> int:
        """PointNet 에 들어가는 총 점 수 = 메모리 K프레임 × 프레임당 점 수."""
        return self.ext_memory_K * self.num_points

    @property
    def heightmap_dim(self) -> int:
        return self.heightmap_rows * self.heightmap_cols

    def index_of(self, term: str) -> tuple[int, int]:
        return PROPRIO_INDEX[term]


# 기본 스펙 인스턴스 (import 해서 바로 쓰기)
DEFAULT_SPEC = JELocoObsSpec()


if __name__ == "__main__":  # 스모크 테스트: 차원·인덱스 정합성
    assert PROPRIO_DIM == 45, PROPRIO_DIM
    assert PROPRIO_INDEX["base_ang_vel"] == (0, 3)
    assert PROPRIO_INDEX["velocity_commands"] == (6, 9)
    assert PROPRIO_INDEX["joint_pos_rel"] == (9, 21)
    assert PROPRIO_INDEX["last_action"] == (33, 45)
    s = DEFAULT_SPEC
    assert s.proprio_history_dim == 5 * 45 == 225
    assert s.depth_shape == (1, 48, 64)
    assert s.heightmap_dim == 441
    print("[obs_spec] OK — proprio_dim=45, index ranges:")
    for name, (a, b) in PROPRIO_INDEX.items():
        print(f"    {name:20s} [{a:2d}:{b:2d})  dim={b - a}")
    print(f"    proprio_history_dim (H={s.history_len}) = {s.proprio_history_dim}")
    print(f"    depth_shape = {s.depth_shape},  heightmap = {s.heightmap_rows}x{s.heightmap_cols}")
