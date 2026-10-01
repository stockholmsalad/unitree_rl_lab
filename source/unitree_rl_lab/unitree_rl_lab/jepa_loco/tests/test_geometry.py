"""D435i 기하 단위 테스트 — Isaac 앱 불필요."""

import math

import pytest

from unitree_rl_lab.jepa_loco.sensors.geometry import (
    camera_offset_in_base,
    ground_visible_distance,
    intrinsic_matrix,
    vfov_deg,
)


def _rotate(q, v):
    w, x, y, z = q
    # v' = q v q*  (단위 쿼터니언)
    R = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return [sum(R[i][j] * v[j] for j in range(3)) for i in range(3)]


def test_offset_points_forward_and_down():
    pos, q = camera_offset_in_base(0.38, 30.0, 0.322, 0.328)
    assert pos == pytest.approx((0.328, 0.0, 0.058))
    fwd = _rotate(q, [1.0, 0.0, 0.0])
    assert fwd[0] == pytest.approx(math.cos(math.radians(30)))
    assert fwd[2] == pytest.approx(-math.sin(math.radians(30)))  # 아래로 숙임
    assert sum(c * c for c in q) == pytest.approx(1.0)


def test_intrinsics_shape_and_hfov():
    K = intrinsic_matrix(848, 480, 87.0)
    assert len(K) == 9
    assert 2 * math.degrees(math.atan(424 / K[0])) == pytest.approx(87.0)
    assert K[0] == K[4]
    assert vfov_deg(848, 480, 87.0) == pytest.approx(56.5, abs=0.1)


def test_ground_visible_distance_matches_spec():
    # CLAUDE.md §2: 명목 VFOV 58° 에서 약 0.23 m
    assert ground_visible_distance(0.38, 30.0, 58.0) == pytest.approx(0.23, abs=0.005)
