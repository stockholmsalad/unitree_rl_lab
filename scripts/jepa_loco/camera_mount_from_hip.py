"""앞다리 hip 관절 축 기준 D435i 실측값 → base 기준 depth 원점(왼쪽 IR) 위치.

측정 기준점: D435i 하우징 **전면 유리 중앙**. hip 관절 축(FL/FR 공통, base x=0.1934, z=0)에서
기준점까지 전방 거리 dx, 높이 dz 를 잰다. 결과를 ``D435iParams.depth_origin_in_base`` 에 넣는다.

    python scripts/jepa_loco/camera_mount_from_hip.py --dx 0.12 --dz 0.06 --pitch 30
"""

import argparse
import os

from unitree_rl_lab.jepa_loco.sensors.geometry import depth_origin_in_base, urdf_joint_origin

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
URDF = os.path.join(ROOT, "unitree_ros/robots/go2_description/urdf/go2_description.urdf")

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--dx", type=float, required=True, help="hip 축 → 전면 유리 중앙, 전방 [m]")
parser.add_argument("--dz", type=float, required=True, help="hip 축 → 전면 유리 중앙, 높이 [m]")
parser.add_argument("--pitch", type=float, default=30.0, help="아래 방향 pitch [deg]")
# 하우징 프레임(전방 x, 좌측 y, 상방 z)에서 전면 유리 중앙 → 왼쪽 IR depth 원점.
# ★ D435 데이터시트 값 — 사용 전 확인할 것: 왼쪽 imager 는 하우징 중심에서 좌측 17.5 mm,
#   depth 원점(ground zero)은 전면 유리에서 후방 4.2 mm.
parser.add_argument("--ir_back", type=float, default=0.0042)
parser.add_argument("--ir_left", type=float, default=0.0175)
parser.add_argument("--urdf", type=str, default=URDF)
args = parser.parse_args()

hip_l = urdf_joint_origin(args.urdf, "FL_hip_joint")
hip_r = urdf_joint_origin(args.urdf, "FR_hip_joint")
assert hip_l[0] == hip_r[0] and hip_l[2] == hip_r[2], "좌우 hip 축 x, z 가 달라 기준이 모호하다"

pos = depth_origin_in_base(hip_l, args.dx, args.dz, args.pitch, (-args.ir_back, args.ir_left, 0.0))
print(f"hip 축 base 좌표: x={hip_l[0]:.4f}, z={hip_l[2]:.4f}  (y=±{abs(hip_l[1]):.4f})")
print(f"depth_origin_in_base = ({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
print(f"pitch_down_deg = {args.pitch}")
