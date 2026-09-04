#!/usr/bin/env python3
"""그림 6 — 게이트 판정의 근거: 조건 간 격차 대 조건 내 시드폭.

판정 규칙은 "조건 간 평균 격차가 최대 조건내 시드폭을 넘을 때만 신호"다. 막대그래프로
평균만 보이면 그 규칙이 보이지 않는다. 조건마다 시드 min~max 범위를 그대로 그려서
**범위가 겹치는지**를 독자가 직접 보게 한다 — 겹치면 그것이 곧 판정력 없음이다.

사용:
  python scripts/je_loco/plot_gate_summary.py --dir results/full_matrix_hard \
      --degs occlusion,freeze,latency,lowfps --out docs/figs/fig6_gates.png
"""
import argparse, csv, glob, os, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, GRID = "#1a1f24", "#5c666c", "#d9dfe2"
COLORS = {"jepa": "#2c6fbb", "recon": "#d1780a", "scratch": "#7c4bb8", "none": "#7c4bb8"}
LABELS = {"jepa": "JEPA", "recon": "Recon", "scratch": "Scratch", "none": "None"}
ORDER = ["jepa", "recon", "scratch", "none"]
TITLES = {"dropout": "dropout", "hole": "hole", "occlusion": "occlusion (공간)",
          "freeze": "freeze (시간)", "latency": "latency (시간)",
          "lowfps": "lowfps (시간)", "blind": "blind"}
RUN_RE = re.compile(r"_(?:D|V2)_([a-z]+)_s(\d+)_model_")


def aulc(curve):
    """레벨 축 위 성공률의 사다리꼴 평균 (judge_gates.py 와 동일 정의)."""
    return sum((curve[i+1][0] - curve[i][0]) * (curve[i+1][1] + curve[i][1]) / 2
               for i in range(len(curve) - 1)) / (curve[-1][0] - curve[0][0])


def load(d):
    """(deg, cond) -> {seed: AULC}"""
    out = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(d, "*_curve_*.csv"))):
        base = os.path.basename(f)
        deg = base.split("_curve_")[0]
        m = RUN_RE.search(base)
        if not m or deg == "abl":
            continue
        rows = list(csv.DictReader(open(f)))
        lv = list(rows[0].keys())[0]
        curve = sorted((float(r[lv]), float(r["success_rate"])) for r in rows)
        if len(curve) > 1:
            out[(deg, m.group(1))][int(m.group(2))] = aulc(curve)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--degs", default="occlusion,freeze,latency,lowfps")
    p.add_argument("--out", default="docs/figs/fig6_gates.png")
    p.add_argument("--suptitle", default="")
    a = p.parse_args()

    data = load(a.dir)
    degs = [d for d in a.degs.split(",") if any(k[0] == d for k in data)]
    if not degs:
        raise SystemExit(f"요청 결손 없음. 있는 것: {sorted({k[0] for k in data})}")
    conds = [c for c in ORDER if any(k[1] == c for k in data)]

    fig, axes = plt.subplots(1, len(degs), figsize=(3.5 * len(degs), 3.2), sharex=True)
    axes = [axes] if len(degs) == 1 else list(axes)

    for ax, deg in zip(axes, degs):
        stats, ys = {}, {}
        for i, c in enumerate(conds):
            v = list(data.get((deg, c), {}).values())
            if not v:
                continue
            stats[c] = (sum(v) / len(v) * 100, min(v) * 100, max(v) * 100, len(v))
            ys[c] = len(conds) - 1 - i
        for c, (mu, lo, hi, n) in stats.items():
            ax.plot([lo, hi], [ys[c]] * 2, color=COLORS[c], lw=6, alpha=0.32,
                    solid_capstyle="round", zorder=2)
            ax.plot([mu], [ys[c]], "o", color=COLORS[c], ms=9, mec="white", mew=1.6, zorder=3)
            ax.text(hi + 0.8, ys[c], f"{mu:.1f}", va="center", fontsize=9,
                    color=INK, fontweight="medium")

        gap = max(s[0] for s in stats.values()) - min(s[0] for s in stats.values())
        spread = max(s[2] - s[1] for s in stats.values())
        sig = gap > spread
        ax.set_title(f"{TITLES.get(deg, deg)}", fontsize=11, color=INK, pad=22, loc="left")
        ax.text(0, 1.03, f"격차 {gap:.2f}pp  {'>' if sig else '≤'}  시드폭 {spread:.2f}pp"
                         f"   →  {'신호' if sig else '판정력 없음'}",
                transform=ax.transAxes, fontsize=8.8, color=INK if sig else INK2,
                fontweight="medium" if sig else "normal")

        ax.set_yticks(list(ys.values()), [LABELS[c] for c in ys])
        ax.set_ylim(-0.6, len(conds) - 0.4)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9.5, length=0)

    # 값 라벨이 오른쪽 경계에 붙어 잘린다 → 여백. sharex 라 한 축만 건드리면 된다.
    axes[0].set_xlim(right=max(axes[0].get_xlim()[1], 104))
    axes[0].set_xlabel("AULC — 결손 레벨 전 구간 평균 성공률 (%)",
                       fontsize=9.5, color=INK2, loc="left")
    n = max(s for v in data.values() for s in [len(v)])
    fig.text(0.5, -0.06, f"점 = 시드 평균 · 막대 = 시드 min~max (n={n}) · "
             f"범위가 겹치면 조건 간 차이는 시드 노이즈 안이다",
             ha="center", fontsize=8.5, color=INK2)
    if a.suptitle:
        fig.suptitle(a.suptitle, fontsize=12.5, color=INK, y=1.06)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"[saved] {a.out}   결손 {degs}  조건 {conds}")


if __name__ == "__main__":
    main()
