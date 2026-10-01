"""D435i 카메라 기하 — Isaac 앱 없이 import 가능한 순수 함수.

좌표계: 몸통(base) 프레임 x 전방, y 좌측, z 상방. 쿼터니언은 IsaacLab 관례 (w, x, y, z).
카메라 프레임은 IsaacLab ``OffsetCfg(convention="world")`` 관례(전방 +X, 상방 +Z)를 따른다.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET


def pitch_down_quat(pitch_down_deg: float) -> tuple[float, float, float, float]:
    """world 관례에서 +y 축 회전 +θ 는 전방축(+x)을 아래로 숙인다."""
    half = math.radians(pitch_down_deg) / 2.0
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def urdf_joint_origin(urdf_path: str, joint_name: str) -> tuple[float, float, float]:
    """URDF 관절의 parent 기준 원점 xyz. hip 관절의 parent 는 base 다."""
    joint = ET.parse(urdf_path).getroot().find(f"joint[@name='{joint_name}']")
    if joint is None:
        raise KeyError(joint_name)
    return tuple(float(v) for v in joint.find("origin").get("xyz").split())


def depth_origin_in_base(
    hip_origin: tuple[float, float, float],
    meas_dx: float,
    meas_dz: float,
    pitch_down_deg: float,
    ref_to_depth_origin: tuple[float, float, float],
) -> tuple[float, float, float]:
    """앞다리 hip 관절 축 기준 실측값 → base 기준 depth 원점(왼쪽 IR) 위치.

    Args:
        hip_origin: base 기준 hip 관절 원점. 좌우 hip 의 y 는 무시하고 몸통 중심선(y=0)에 둔다.
        meas_dx: hip 관절 축에서 카메라 기준점까지 전방 거리 [m].
        meas_dz: hip 관절 축에서 카메라 기준점까지 높이 [m].
        pitch_down_deg: 카메라 아래 방향 pitch.
        ref_to_depth_origin: 카메라 하우징 프레임(전방 x, 좌측 y, 상방 z)에서 기준점 → depth 원점 벡터.
    """
    ref = (hip_origin[0] + meas_dx, 0.0, hip_origin[2] + meas_dz)
    th = math.radians(pitch_down_deg)
    ox, oy, oz = ref_to_depth_origin
    # 하우징이 pitch 만큼 숙어 있으므로 하우징 프레임 오프셋을 y 축으로 회전
    return (
        ref[0] + math.cos(th) * ox + math.sin(th) * oz,
        ref[1] + oy,
        ref[2] - math.sin(th) * ox + math.cos(th) * oz,
    )


def default_intrinsics(width: int, height: int, hfov_deg: float) -> tuple[float, float, float, float]:
    """실측 calibration 이 없을 때의 정사각 픽셀 핀홀 (fx, fy, cx, cy). fx = fy 는 수평 FOV 로 정한다.

    주점은 OpenCV/librealsense 관례(픽셀 중심이 정수 좌표)로 이미지 중앙 (W-1)/2, (H-1)/2.
    """
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return fx, fx, (width - 1) / 2.0, (height - 1) / 2.0


def scale_intrinsics(
    fx: float, fy: float, cx: float, cy: float, src_wh: tuple[int, int], dst_wh: tuple[int, int]
) -> tuple[float, float, float, float]:
    """크롭 없이 해상도만 바꿀 때의 intrinsic. 픽셀 중심 관례 (c + 0.5) * s - 0.5."""
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    return fx * sx, fy * sy, (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5


def intrinsic_matrix(fx: float, fy: float, cx: float, cy: float) -> list[float]:
    """row-major 9원소."""
    return [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]


def fov_deg(f: float, size: int) -> float:
    return math.degrees(2.0 * math.atan((size / 2.0) / f))


def ground_visible_distance(height_above_ground: float, pitch_down_deg: float, vfov: float) -> float:
    """평지에서 시야 하단 광선이 지면에 닿는 카메라 기준 수평 거리."""
    return height_above_ground / math.tan(math.radians(pitch_down_deg + vfov / 2.0))


def left_band_width_px(baseline: float, fx: float, depth: float) -> float:
    """D435 왼쪽 결측 띠 폭 [px] = baseline × fx / Z. fx 는 렌더링 해상도 기준."""
    return baseline * fx / depth
