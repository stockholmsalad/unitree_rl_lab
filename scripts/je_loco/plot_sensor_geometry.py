#!/usr/bin/env python3
"""그림 1 — D435i 장착 기하: 측면도(수직 시야)와 평면도(수평 시야).

수치는 전부 env_cfg.py 의 pc_scanner 설정에서 온다. 손으로 적은 값이 없다:
    offset  pos=(0.325, 0.0, 0.045),  rot=tilt_quat_y(35°)
    pattern FrustumPatternCfg(hfov=78.7°, vfov=63.1°, width=16, height=12)
    max_distance = 2.0                     ← D435i 하드웨어 스펙 3m 이 아니다
그림에 그 유효 거리를 명시하는 이유는 예측 지평 k 의 상한이 여기서 유도되기 때문이다.

사용:
  python scripts/je_loco/plot_sensor_geometry.py --out docs/figs/fig1_sensor.png
"""
import argparse, math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, Arc

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, GRID = "#1a1f24", "#5c666c", "#d9dfe2"
BLUE, GROUND, BODY = "#2c6fbb", "#c9b79a", "#8b969c"

MOUNT_X, MOUNT_Z = 0.325, 0.045     # base 기준 장착 오프셋 (m)
TILT, HFOV, VFOV = 35.0, 78.7, 63.1  # 하향각 · 시야 (deg)
RANGE = 2.0                          # max_distance (m)
BASE_H = 0.35                        # Go2 기립 시 base 높이 (근사, 그림 축척용)
NX, NY = 16, 12


def side(ax):
    camx, camz = MOUNT_X, BASE_H + MOUNT_Z
    lo = math.radians(TILT + VFOV / 2)      # 하단 광선 (지면에 더 가파르게)
    up = math.radians(TILT - VFOV / 2)

    # 지면
    ax.axhline(0, color=GROUND, lw=3, zorder=2)
    ax.text(2.55, -0.055, "지면", color=INK2, fontsize=9, ha="right")

    # 몸통 (개략)
    ax.add_patch(Polygon([(-0.30, BASE_H - 0.06), (0.28, BASE_H - 0.06),
                          (0.28, BASE_H + 0.06), (-0.30, BASE_H + 0.06)],
                         closed=True, fc=BODY, alpha=.28, ec=BODY, lw=1.2, zorder=3))
    ax.text(-0.01, BASE_H, "Go2 base", ha="center", va="center", fontsize=9, color=INK)
    for fx in (-0.22, 0.20):
        ax.plot([fx, fx], [0, BASE_H - 0.06], color=BODY, lw=2.4, zorder=3)

    # 시야 사다리꼴 — 광선을 지면 또는 유효거리 중 먼저 닿는 쪽에서 끊는다
    pts = [(camx, camz)]
    for ang in (up, lo):
        t_ground = camz / math.sin(ang) if math.sin(ang) > 1e-6 else 1e9
        t = min(RANGE, t_ground)
        pts.append((camx + t * math.cos(ang), camz - t * math.sin(ang)))
    ax.add_patch(Polygon(pts, closed=True, fc=BLUE, alpha=.13, ec=BLUE, lw=1.4, zorder=4))

    # 유효거리 호
    a0, a1 = -math.degrees(lo), -math.degrees(up)
    ax.add_patch(Arc((camx, camz), 2 * RANGE, 2 * RANGE, angle=0, theta1=a0, theta2=a1,
                     color=BLUE, lw=1.6, ls=(0, (5, 3)), zorder=5))
    mid = math.radians(TILT)
    ax.annotate(f"max_distance {RANGE:.1f} m", xy=(camx + RANGE * math.cos(mid) * .99,
                                                   camz - RANGE * math.sin(mid) * .99),
                xytext=(-30, 40), textcoords="offset points", fontsize=9, color=BLUE,
                ha="center",
                fontweight="medium",
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=.9))

    # 카메라
    ax.plot([camx], [camz], "s", ms=10, color=BLUE, mec="white", mew=1.6, zorder=7)
    ax.annotate(f"D435i\n전방 {MOUNT_X*100:.1f} cm · 상방 {MOUNT_Z*100:.1f} cm",
                xy=(camx, camz), xytext=(-6, 34), textcoords="offset points",
                ha="center", fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=.9))

    # 하향각
    ax.plot([camx, camx + 0.62], [camz, camz], color=INK2, lw=1, ls=(0, (3, 3)), zorder=5)
    ax.add_patch(Arc((camx, camz), 0.72, 0.72, theta1=-TILT, theta2=0, color=INK2, lw=1.3, zorder=6))
    ax.text(camx + 0.44, camz - 0.10, f"{TILT:.0f}° 하향", fontsize=9, color=INK2)
    ax.text(camx + 0.16, camz - 0.30, f"수직 시야\n{VFOV:.1f}°", fontsize=9, color=BLUE,
            fontweight="medium")

    # 지면 관측 구간
    x_near = camx + (camz / math.tan(lo))
    x_far = camx + RANGE * math.cos(up)
    ax.add_patch(FancyArrowPatch((x_near, -0.11), (x_far, -0.11), arrowstyle="<|-|>",
                                 mutation_scale=11, color=INK, lw=1.3, zorder=6))
    ax.text((x_near + x_far) / 2, -0.175, f"지면 관측 구간  x ∈ [{x_near:.2f}, {x_far:.2f}] m",
            ha="center", fontsize=9.5, color=INK, fontweight="medium")
    for x in (x_near, x_far):
        ax.plot([x, x], [-0.13, 0], color=INK, lw=.9, zorder=6)

    ax.set_xlim(-0.42, 2.62); ax.set_ylim(-0.30, 0.86)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(a) 측면도 — 수직 시야와 유효 거리", fontsize=11.5, color=INK,
                 pad=8, loc="left")


def top(ax):
    half = math.radians(HFOV / 2)
    camx = MOUNT_X
    arc_pts = [(camx + RANGE * math.cos(half * (-1 + 2 * i / 48)),
                RANGE * math.sin(half * (-1 + 2 * i / 48))) for i in range(49)]
    ax.add_patch(Polygon([(camx, 0)] + arc_pts, closed=True,
                         fc=BLUE, alpha=.13, ec=BLUE, lw=1.4, zorder=3))
    ax.add_patch(Arc((camx, 0), 2 * RANGE, 2 * RANGE, theta1=-HFOV / 2, theta2=HFOV / 2,
                     color=BLUE, lw=1.6, ls=(0, (5, 3)), zorder=4))
    ax.add_patch(Polygon([(-0.30, -0.10), (0.28, -0.10), (0.28, 0.10), (-0.30, 0.10)],
                         closed=True, fc=BODY, alpha=.28, ec=BODY, lw=1.2, zorder=3))
    ax.text(-0.01, 0, "Go2 base", ha="center", va="center", fontsize=9, color=INK)
    ax.plot([camx], [0], "s", ms=10, color=BLUE, mec="white", mew=1.6, zorder=6)
    ax.plot([camx, camx + RANGE], [0, 0], color=INK2, lw=1, ls=(0, (3, 3)), zorder=4)
    ax.add_patch(Arc((camx, 0), 1.5, 1.5, theta1=0, theta2=HFOV / 2, color=INK2, lw=1.3, zorder=5))
    ax.text(camx + 0.80, 0.30, f"수평 시야\n{HFOV:.1f}°", fontsize=9, color=BLUE,
            fontweight="medium")

    # 샘플 격자 (16×12 중 표시용으로 성기게)
    for i in range(NX + 1):
        a = (-1 + 2 * i / NX) * half
        ax.plot([camx, camx + RANGE * math.cos(a)], [0, RANGE * math.sin(a)],
                color=BLUE, lw=.5, alpha=.35, zorder=2)
    ax.text(camx + RANGE * 0.55, -1.16, f"{NX} × {NY} = {NX*NY} 점\n(실기 64×48 다운샘플)",
            ha="center", fontsize=9, color=INK)
    ax.text(camx + RANGE * math.cos(half) * 0.5, RANGE * math.sin(half) * 0.86,
            f"max_distance {RANGE:.1f} m", fontsize=9, color=BLUE, fontweight="medium")

    ax.set_xlim(-0.42, 2.62); ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("(b) 평면도 — 수평 시야와 표본 격자", fontsize=11.5, color=INK,
                 pad=8, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figs/fig1_sensor.png")
    a = ap.parse_args()
    fig, axes = plt.subplots(1, 2, figsize=(13.4, 4.4), dpi=200,
                             gridspec_kw={"width_ratios": [1.5, 1]})
    side(axes[0]); top(axes[1])
    fig.text(0.5, 0.005,
             "수치는 env_cfg.py 의 pc_scanner 설정값 — "
             "offset (0.325, 0, 0.045) · 하향 35° · FoV 78.7°×63.1° · max_distance 2.0 m",
             ha="center", fontsize=8.5, color=INK2)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    lo = math.radians(TILT + VFOV / 2); up = math.radians(TILT - VFOV / 2)
    camz = BASE_H + MOUNT_Z
    print(f"[saved] {a.out}\n  지면 구간 x ∈ [{MOUNT_X + camz/math.tan(lo):.2f}, "
          f"{MOUNT_X + RANGE*math.cos(up):.2f}] m  (설정 주석 [0.5, 2.3] 과 대조)")


if __name__ == "__main__":
    main()
