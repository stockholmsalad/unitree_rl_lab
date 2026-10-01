"""D435i 카메라 기하 — Isaac 앱 없이 import 가능한 순수 함수.

좌표계: 몸통(base) 프레임 x 전방, y 좌측, z 상방. 쿼터니언은 IsaacLab 관례 (w, x, y, z).
카메라 프레임은 IsaacLab ``OffsetCfg(convention="world")`` 관례(전방 +X, 상방 +Z)를 따른다.
"""

from __future__ import annotations

import math


def camera_offset_in_base(
    height_above_ground: float,
    pitch_down_deg: float,
    nominal_base_height: float,
    camera_x_in_base: float,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """기본 자세에서 지면 기준 높이/아래 pitch 를 만족하는 base 프레임 오프셋.

    Returns:
        (pos, quat_wxyz). world 관례에서 +y 축 회전 +θ 는 전방축(+x)을 아래로 숙인다.
    """
    z = height_above_ground - nominal_base_height
    half = math.radians(pitch_down_deg) / 2.0
    return (camera_x_in_base, 0.0, z), (math.cos(half), 0.0, math.sin(half), 0.0)


def intrinsic_matrix(width: int, height: int, hfov_deg: float) -> list[float]:
    """정사각 픽셀 핀홀 intrinsic (row-major 9원소). fx = fy 는 수평 FOV 로 정한다."""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return [fx, 0.0, width / 2.0, 0.0, fx, height / 2.0, 0.0, 0.0, 1.0]


def vfov_deg(width: int, height: int, hfov_deg: float) -> float:
    """정사각 픽셀일 때 수평 FOV 로부터 유도되는 수직 FOV."""
    fx = intrinsic_matrix(width, height, hfov_deg)[0]
    return math.degrees(2.0 * math.atan((height / 2.0) / fx))


def ground_visible_distance(height_above_ground: float, pitch_down_deg: float, vfov: float) -> float:
    """평지에서 시야 하단 광선이 지면에 닿는 카메라 기준 수평 거리."""
    return height_above_ground / math.tan(math.radians(pitch_down_deg + vfov / 2.0))
