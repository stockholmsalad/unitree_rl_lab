# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""실기 D435i 의 **실제** 깊이 내부 파라미터를 읽는다. 로봇에 카메라가 붙은 채로 실행.

시뮬레이터 cfg 의 FoV 78.7°×63.1° 는 "fx=390 이라고 치면" 에서 나온 값이고, D435i 의
공표 깊이 FoV 는 87°×58° 다. 수직은 시뮬이 카메라보다 **넓게** 가정하고 있다 — 즉
정책이 학습 때 본 위·아래 끝 광선을 실기에서는 아예 못 받는다. 추측하지 말고 잰다.

  # ★ 로봇 노트북 (카메라 연결 상태)
  python deploy_real/read_intrinsics.py
"""

from __future__ import annotations

import math

import pyrealsense2 as rs


def main() -> None:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(cfg)
    try:
        s = prof.get_stream(rs.stream.depth).as_video_stream_profile()
        i = s.get_intrinsics()
        scale = prof.get_device().first_depth_sensor().get_depth_scale()
        hfov = math.degrees(2 * math.atan(i.width / (2 * i.fx)))
        vfov = math.degrees(2 * math.atan(i.height / (2 * i.fy)))
        print(f"해상도      {i.width} × {i.height}")
        print(f"fx, fy      {i.fx:.3f}, {i.fy:.3f}")
        print(f"cx, cy      {i.ppx:.3f}, {i.ppy:.3f}")
        print(f"왜곡        {i.model} {['%.4f' % c for c in i.coeffs]}")
        print(f"depth_scale {scale}")
        print(f"실측 FoV    수평 {hfov:.2f}°  수직 {vfov:.2f}°")
        print()
        print(f"시뮬 가정    수평 78.70°  수직 63.10°")
        for name, real, sim in [("수평", hfov, 78.70), ("수직", vfov, 63.10)]:
            if sim > real:
                print(f"  ← {name}: 시뮬이 {sim - real:.2f}° 더 넓다. "
                      f"그만큼의 광선은 실기에서 영원히 무효다.")
            else:
                print(f"  ← {name}: 여유 {real - sim:.2f}°. 문제 없음.")
    finally:
        pipe.stop()


if __name__ == "__main__":
    main()
