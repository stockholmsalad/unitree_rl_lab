"""D435i depth 카메라 IsaacLab 설정 (CLAUDE.md §2, §6). Isaac 앱 실행 후 import 할 것."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .geometry import camera_offset_in_base, intrinsic_matrix


@configclass
class D435iParams:
    """CLAUDE.md 수치를 하드코딩하지 않기 위한 단일 진리원."""

    height_above_ground: float = 0.38
    """지면 기준 카메라 높이 [m] (기본 자세)."""
    pitch_down_deg: float = 30.0
    """아래 방향 pitch [deg]."""
    hfov_deg: float = 87.0
    """수평 FOV [deg]. 정사각 픽셀 가정 → 수직 FOV 는 유도값(848×480 에서 ≈56.5°)."""
    width: int = 848
    height: int = 480
    min_range: float = 0.28
    """최소 측정 거리 [m]. 렌더링 near clip 으로도 쓴다."""
    max_range: float = 10.0
    """렌더링 far clip [m]. 정책 입력 클리핑([0.28, 2.0])은 전처리 단계에서 따로 한다."""
    nominal_base_height: float = 0.322
    """기본 관절각에서 지면–base 원점 높이 [m]. URDF FK: 앞발 z=-0.300, 발 구 반지름 0.022."""
    front_foot_x_in_base: float = 0.178
    """기본 관절각에서 앞발 착지점의 base x [m] (URDF FK)."""
    camera_to_front_foot_x: float = 0.15
    """카메라–앞발 수평 거리 [m]. ★가정값 — 실제 D435i 마운트 위치 확인 필요 (CLAUDE.md §2)."""
    update_period: float = 0.1
    """10 Hz (CLAUDE.md §3)."""

    @property
    def camera_x_in_base(self) -> float:
        return self.front_foot_x_in_base + self.camera_to_front_foot_x


def make_tiled_camera_cfg(
    params: D435iParams, prim_path: str = "{ENV_REGEX_NS}/Robot/base/d435i", width: int | None = None, height: int | None = None
) -> TiledCameraCfg:
    """FOV 는 유지하고 해상도만 바꿔 렌더링할 수 있게 width/height 를 따로 받는다."""
    w = width or params.width
    h = height or params.height
    pos, rot = camera_offset_in_base(
        params.height_above_ground, params.pitch_down_deg, params.nominal_base_height, params.camera_x_in_base
    )
    return TiledCameraCfg(
        prim_path=prim_path,
        update_period=params.update_period,
        width=w,
        height=h,
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",  # 0 은 결손(노이즈 모델) 전용으로 남긴다
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="world"),
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix(w, h, params.hfov_deg),
            width=w,
            height=h,
            clipping_range=(params.min_range, params.max_range),
        ),
    )
