#!/usr/bin/env python3
"""증류 3조건 × 2시드 depth 결손 저하 곡선 (논문 figure).

plot_degradation.py 는 recon/jepa 2조건 전용이라 인코더 초기화 3조건에는 못 쓴다.
조건별 = 시드 평균 선 + 시드 min~max 밴드, 결손 종류별 패널.

사용:
  python scripts/je_loco/plot_distill_degradation.py \
      --dir results/distill_degradation --out results/distill_degradation/degradation_3cond.png
"""

import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 한글 라벨 (Noto Sans CJK 는 한글 글리프 포함). 없으면 라벨이 두부(□)로 깨진다.
plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np

# 검증 통과 팔레트(validate_palette.js, light): 모든 인접쌍 CVD ΔE ≥ 21
COLORS = {"jepa": "#2c6fbb", "recon": "#d1780a", "scratch": "#7c4bb8"}
LABELS = {"jepa": "JEPA (예측)", "recon": "Recon (재구성)", "scratch": "Scratch (랜덤init)"}
CONDS = ["jepa", "recon", "scratch"]
DEGS = [("dropout", "dropout (i.i.d. 점 결손)"), ("hole", "hole (블록 결손)"),
        ("occlusion", "occlusion (하단 대역 차폐)")]
INK, MUTED, GRID = "#1a1a1a", "#5c5c5c", "#dcdcdc"


def load(d, metric):
    """(deg, cond) -> (levels, [seed별 값])"""
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*_curve_*.csv"))):
        deg = os.path.basename(f).split("_curve_")[0]
        tag = os.path.basename(f).split("_curve_")[1].replace("_model_7999.csv", "")
        cond = tag.split("_D_")[1].rsplit("_s", 1)[0]
        rows = list(csv.DictReader(open(f)))
        lv = np.array([float(r[list(r.keys())[0]]) for r in rows])
        out.setdefault((deg, cond), [lv, []])[1].append(np.array([float(r[metric]) for r in rows]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/distill_degradation")
    p.add_argument("--metric", default="success_rate")
    p.add_argument("--out", default="results/distill_degradation/degradation_3cond.png")
    a = p.parse_args()

    data = load(a.dir, a.metric)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4), sharey=True)
    fig.patch.set_facecolor("white")

    for ax, (deg, title) in zip(axes, DEGS):
        for c in CONDS:
            lv, seeds = data[(deg, c)]
            Y = np.vstack(seeds) * 100
            mean, lo, hi = Y.mean(0), Y.min(0), Y.max(0)
            ax.fill_between(lv, lo, hi, color=COLORS[c], alpha=0.16, linewidth=0)  # 시드 밴드
            ax.plot(lv, mean, color=COLORS[c], lw=2, marker="o", ms=6,
                    mec="white", mew=1.2, label=LABELS[c], zorder=3)
        ax.axhline(50, color=MUTED, lw=1, ls=(0, (4, 3)), alpha=0.7, zorder=1)
        ax.set_title(title, fontsize=11, color=INK, pad=9)
        ax.set_xlabel("결손 레벨", fontsize=10, color=MUTED)
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-3, 103)
        ax.set_xticks(np.arange(0, 1.01, 0.2))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=9, length=0)

    axes[0].set_ylabel("성공률 (%)", fontsize=10, color=MUTED)
    axes[0].text(0.02, 52, "50%", fontsize=8, color=MUTED, va="bottom")
    axes[0].legend(frameon=False, fontsize=9.5, loc="lower left", labelcolor=INK)

    fig.suptitle("DAgger 증류 학생 정책의 depth 결손 강인성 — 인코더 초기화 3조건 × 2시드",
                 fontsize=12.5, color=INK, y=1.0)
    fig.text(0.5, -0.04, "선 = 시드 평균 · 밴드 = 시드 min~max · 256 envs × 1500 steps/레벨 · "
             "eval_seed 42 고정(동일 지형) · model_7999",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
