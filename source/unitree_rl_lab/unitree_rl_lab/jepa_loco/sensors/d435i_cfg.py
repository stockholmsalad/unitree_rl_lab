"""D435i depth 카메라 IsaacLab 설정 (CLAUDE.md §2, §4, §6). Isaac 앱 실행 후 import 할 것."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from .geometry import default_intrinsics, intrinsic_matrix, pitch_down_quat, scale_intrinsics


@configclass
class D435iParams:
    """CLAUDE.md 수치를 하드코딩하지 않기 위한 단일 진리원.

    카메라는 base link 고정 오프셋으로 정의한다. ``depth_origin_in_base`` 는 depth 원점(왼쪽 IR)의
    base 기준 위치이며, 실측 후 ``geometry.depth_origin_in_base`` 로 hip 기준 측정값에서 변환해 넣는다.
    """

    depth_origin_in_base: tuple[float, float, float] = (0.328, 0.0, 0.058)
    """★임시값 — 앞다리 hip 기준 실측 대기. Phase 0 의 높이 기반 배치(카메라–앞발 0.15 m)를 그대로 둔 것."""
    pitch_down_deg: float = 30.0
    """아래 방향 pitch [deg]."""

    native_wh: tuple[int, int] = (848, 480)
    """D435i depth 원본 해상도."""
    native_intrinsics: tuple[float, float, float, float] | None = None
    """원본 해상도 (fx, fy, cx, cy). None 이면 HFOV 로부터 정사각 픽셀 가정. 실기 calibration 확보 시 교체."""
    hfov_deg: float = 87.0
    """native_intrinsics 가 None 일 때만 사용."""
    render_wh: tuple[int, int] = (112, 64)
    """시뮬레이터 직접 렌더링 = 정책 입력 해상도. 크롭 없이 intrinsic 만 스케일."""

    baseline: float = 0.050
    """스테레오 baseline [m] (D435 데이터시트). 왼쪽 결측 띠 폭 = baseline × fx / Z."""
    min_range: float = 0.28
    """최소 측정 거리 [m]. 렌더링 near clip 이자 전처리 유효 하한."""
    clip_far: float = 2.0
    """전처리 정규화 상한 [m]. 초과는 1.0 으로 clamp, 유효."""
    render_far: float = 10.0
    """렌더링 far clip [m]. 초과는 render_far 로 채워져 전처리에서 1.0(유효)이 된다."""
    update_period: float = 0.1
    """10 Hz (CLAUDE.md §3)."""

    def intrinsics(self, wh: tuple[int, int]) -> tuple[float, float, float, float]:
        native = self.native_intrinsics or default_intrinsics(*self.native_wh, self.hfov_deg)
        return scale_intrinsics(*native, self.native_wh, wh)


@configclass
class D435iRandomization:
    """CLAUDE.md §6 카메라 랜덤화 — base 기준 위치 ±2 cm, pitch ±3°."""

    pos_range: float = 0.02
    pitch_range_deg: float = 3.0


def make_tiled_camera_cfg(
    params: D435iParams, prim_path: str = "{ENV_REGEX_NS}/Robot/base/d435i", wh: tuple[int, int] | None = None
) -> TiledCameraCfg:
    """기본은 render_wh(112×64) 직접 렌더링. wh 로 다른 해상도(예: 848×480 검증용)도 같은 FOV 로 렌더링한다.

    Omniverse 는 비정사각 픽셀을 지원하지 않아 fx, fy 평균을 쓴다(848×480→112×64 에서 차이 0.95%).
    """
    w, h = wh or params.render_wh
    return TiledCameraCfg(
        prim_path=prim_path,
        update_period=params.update_period,
        width=w,
        height=h,
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",  # 0 은 결손(노이즈 모델) 전용으로 남긴다
        offset=TiledCameraCfg.OffsetCfg(
            pos=params.depth_origin_in_base, rot=pitch_down_quat(params.pitch_down_deg), convention="world"
        ),
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=intrinsic_matrix(*params.intrinsics((w, h))),
            width=w,
            height=h,
            clipping_range=(params.min_range, params.render_far),
        ),
    )
