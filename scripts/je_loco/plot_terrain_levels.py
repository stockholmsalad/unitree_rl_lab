#!/usr/bin/env python3
"""그림 11 — 지형 난이도에 따른 학생·교사 성능.

학생(전방 깊이 192점)이 교사(특권 높이 스캔 187차원)의 몇 %를 유지하는지,
그리고 그 격차가 난이도에 따라 벌어지는지를 본다. 이 그림의 주장은
"학생이 교사를 이긴다" 가 아니라 "격차가 난이도에 무관하게 일정하다" 이다.

두 정책은 **같은 지형 격자**에서 측정해야 한다. 교사 play cfg 만 num_cols=8
이던 시기의 값(97.3/97.0/96.6)은 디딤돌이 0% 라 무효다(커밋 e75d76e).

단일 시드 측정이므로 레벨 사이의 순서는 해석하지 않는다 — 둘 다 단조롭지 않다.

사용:
  python scripts/je_loco/plot_terrain_levels.py --out docs/figs/fig11_terrain.png
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, GRID = "#1a1f24", "#5c666c", "#d9dfe2"
C_STU, C_TEA = "#2c6fbb", "#d1780a"

# 2026-09-11 측정 (지형 격자 일치 후). eval_seed 42, 256 env, 1500 step.
LEVELS = [5, 7, 9]
STAIR = ["9.0–10.2 cm", "~12 cm", "~14.4 cm"]
STUDENT = [93.4, 90.5, 90.9]
TEACHER = [99.2, 96.6, 98.1]
STU_SPD = [0.34, 0.33, 0.33]
TEA_SPD = [0.40, 0.40, 0.40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/figs/fig11_terrain.png")
    a = ap.parse_args()

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(9.4, 3.9), gridspec_kw={"width_ratios": [1.55, 1]})

    # ── 왼쪽: 성공률 ──
    ax.plot(LEVELS, TEACHER, "o-", color=C_TEA, lw=2.2, ms=7,
            label="교사 — 특권 높이 스캔 (187)")
    ax.plot(LEVELS, STUDENT, "s-", color=C_STU, lw=2.2, ms=7,
            label="학생 — 전방 깊이 점군 (192)")
    for lv, s, t in zip(LEVELS, STUDENT, TEACHER):
        ax.annotate("", xy=(lv, t), xytext=(lv, s),
                    arrowprops=dict(arrowstyle="<->", color=INK2, lw=0.9))
        ax.text(lv + 0.12, (s + t) / 2, f"{t - s:.1f} pp",
                color=INK2, fontsize=9, va="center")
    ax.set_xticks(LEVELS)
    ax.set_xticklabels([f"레벨 {l}\n{h}" for l, h in zip(LEVELS, STAIR)], fontsize=9)
    ax.set_ylabel("성공률 (%)", color=INK)
    ax.set_ylim(84, 102)
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    ax.set_title("지형 난이도에 따른 성공률", color=INK, fontsize=11, loc="left")

    # ── 오른쪽: 유지율 ──
    keep = [100 * s / t for s, t in zip(STUDENT, TEACHER)]
    ax2.bar([str(l) for l in LEVELS], keep, color=C_STU, width=0.55)
    for i, k in enumerate(keep):
        ax2.text(i, k + 0.6, f"{k:.1f}%", ha="center", color=INK, fontsize=9.5)
    ax2.axhline(100, color=C_TEA, lw=1.4, ls="--")
    ax2.text(2.45, 100.4, "교사", color=C_TEA, fontsize=9, ha="right")
    ax2.set_ylim(80, 104)
    ax2.set_xlabel("지형 레벨", color=INK)
    ax2.set_ylabel("교사 대비 유지율 (%)", color=INK)
    ax2.set_title("특권 관측 없이 유지되는 비율", color=INK, fontsize=11, loc="left")

    for x in (ax, ax2):
        x.grid(axis="y", color=GRID, lw=0.8)
        x.set_axisbelow(True)
        for sp in ("top", "right"):
            x.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            x.spines[sp].set_color(GRID)
        x.tick_params(colors=INK2, labelsize=9)

    cap = (f"학생 속도 {STU_SPD[0]:.2f}→{STU_SPD[-1]:.2f} m/s, "
           f"교사 {TEA_SPD[0]:.2f} m/s. eval_seed 42 · 256 env × 1500 step · "
           f"두 정책 동일 지형 격자 · 단일 시드(레벨 간 순서는 해석하지 않음)")
    fig.text(0.012, 0.015, cap, color=INK2, fontsize=8)
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.savefig(a.out, dpi=200, facecolor="white")
    print(f"저장: {a.out}")
    print(f"유지율 {keep[0]:.1f} / {keep[1]:.1f} / {keep[2]:.1f} %")


if __name__ == "__main__":
    main()
