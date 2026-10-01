"""D435i 기하 단위 테스트 — Isaac 앱 불필요."""

import math
import os

import pytest

from unitree_rl_lab.jepa_loco.sensors.geometry import (
    default_intrinsics,
    depth_origin_in_base,
    fov_deg,
    ground_visible_distance,
    left_band_width_px,
    pitch_down_quat,
    scale_intrinsics,
    urdf_joint_origin,
)

URDF = os.path.join(
    os.path.dirname(__file__), *[".."] * 5, "unitree_ros/robots/go2_description/urdf/go2_description.urdf"
)


def _rotate(q, v):
    w, x, y, z = q
    R = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ]
    return [sum(R[i][j] * v[j] for j in range(3)) for i in range(3)]


def test_pitch_quat_points_forward_and_down():
    q = pitch_down_quat(30.0)
    fwd = _rotate(q, [1.0, 0.0, 0.0])
    assert fwd[0] == pytest.approx(math.cos(math.radians(30)))
    assert fwd[2] == pytest.approx(-math.sin(math.radians(30)))
    assert sum(c * c for c in q) == pytest.approx(1.0)


def test_urdf_hip_origin():
    assert urdf_joint_origin(URDF, "FL_hip_joint") == pytest.approx((0.1934, 0.0465, 0.0))
    assert urdf_joint_origin(URDF, "FR_hip_joint") == pytest.approx((0.1934, -0.0465, 0.0))
    with pytest.raises(KeyError):
        urdf_joint_origin(URDF, "no_such_joint")


def test_depth_origin_conversion():
    hip = (0.1934, 0.0465, 0.0)
    # 오프셋 0 이면 hip 축 + 측정값, y 는 중심선
    assert depth_origin_in_base(hip, 0.10, 0.05, 30.0, (0, 0, 0)) == pytest.approx((0.2934, 0.0, 0.05))
    # 왼쪽 IR 횡오프셋은 pitch 와 무관하게 +y
    assert depth_origin_in_base(hip, 0.0, 0.0, 30.0, (0, 0.0175, 0))[1] == pytest.approx(0.0175)
    # 광축 방향 -4.2 mm(후방)는 숙인 광축을 따라 뒤·위로 이동
    p = depth_origin_in_base(hip, 0.0, 0.0, 30.0, (-0.0042, 0, 0))
    assert p[0] == pytest.approx(0.1934 - 0.0042 * math.cos(math.radians(30)))
    assert p[2] == pytest.approx(0.0042 * math.sin(math.radians(30)))


def test_intrinsics_scaling_112x64():
    native = default_intrinsics(848, 480, 87.0)
    assert fov_deg(native[0], 848) == pytest.approx(87.0)
    fx, fy, cx, cy = scale_intrinsics(*native, (848, 480), (112, 64))
    assert (cx, cy) == pytest.approx((55.5, 31.5))  # 저해상도에서도 이미지 중앙
    assert fov_deg(fx, 112) == pytest.approx(87.0)  # 크롭 없음 → HFOV 보존
    assert fov_deg(fy, 64) == pytest.approx(fov_deg(native[1], 480))
    assert abs(fx / fy - 1) < 0.01  # 종횡비 1.767→1.75 로 생기는 비정사각 0.95%


def test_ground_visible_distance_matches_old_spec():
    assert ground_visible_distance(0.38, 30.0, 58.0) == pytest.approx(0.23, abs=0.005)


def test_left_band_width():
    fx = scale_intrinsics(*default_intrinsics(848, 480, 87.0), (848, 480), (112, 64))[0]
    assert left_band_width_px(0.05, fx, 0.5) == pytest.approx(0.05 * fx / 0.5)
    assert left_band_width_px(0.05, fx, 0.5) > left_band_width_px(0.05, fx, 2.0)
