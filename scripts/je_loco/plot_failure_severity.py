#!/usr/bin/env python3
"""그림 7 — 어떤 고장이 얼마나 아픈가: 최대 강도에서의 성공률.

핵심은 blind 와 freeze 의 대비다.
  · blind  = 지형 정보가 0 인 점군(시각을 통째로 뺀 것과 같다) → 시각의 기여분
  · freeze = 낡은 관측이 계속 들어온다
freeze 가 blind 보다 **더 나쁘면**, 문제는 정보 부족이 아니라 틀린 정보이며
그것이 곧 "예측이 처방인 자리"다. v2 설계의 근거가 이 한 장에 있다.

조건(jepa/recon/…) 간 차이가 아니라 **고장 종류 간 차이**를 보는 그림이므로
조건과 시드를 모두 합친다. 조건별 비교는 그림 6 이 맡는다.

사용:
  python scripts/je_loco/plot_failure_severity.py --dir results/full_matrix_hard \
      --out docs/figs/fig7_severity.png
"""
import argparse, csv, glob, os, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, GRID = "#1a1f24", "#5c666c", "#d9dfe2"
BAR, HILITE = "#9fb3c0", "#2c6fbb"
TITLES = {"dropout": "dropout · i.i.d. 점 결손", "hole": "hole · 블록 결손",
          "occlusion": "occlusion · 하단 대역 차폐", "freeze": "freeze · 관측 정지",
          "latency": "latency · 관측 지연", "lowfps": "lowfps · 갱신률 저하",
          "blind": "blind · 지형 정보 0"}
RUN_RE = re.compile(r"_(?:D|V2)_([a-z]+)_s(\d+)_model_")


def load(d):
    """deg -> ([level0 성공률...], [최대강도 성공률...])  — 조건·시드 전부 합침"""
    base, worst = defaultdict(list), defaultdict(list)
    for f in sorted(glob.glob(os.path.join(d, "*_curve_*.csv"))):
        b = os.path.basename(f)
        deg = b.split("_curve_")[0]
        if deg == "abl" or not RUN_RE.search(b):
            continue
        rows = list(csv.DictReader(open(f)))
        lv = list(rows[0].keys())[0]
        c = sorted((float(r[lv]), float(r["success_rate"])) for r in rows)
        if len(c) > 1:
            base[deg].append(c[0][1] * 100)
            worst[deg].append(c[-1][1] * 100)
    return base, worst


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--out", default="docs/figs/fig7_severity.png")
    p.add_argument("--hilite", default="freeze,blind", help="강조할 고장")
    p.add_argument("--suptitle", default="")
    a = p.parse_args()

    base, worst = load(a.dir)
    if not worst:
        raise SystemExit(f"곡선 CSV 가 없다: {a.dir}")
    b0 = sum(v for vs in base.values() for v in vs) / sum(len(v) for v in base.values())
    if "blind" not in worst:
        print("  ! blind 곡선이 없다 — 이 그림의 논지(낡은 관측 > 무관측)는 blind 대비로만"
              " 성립한다. blind CSV 가 있는 디렉터리로 다시 돌려라.")
    elif "freeze" in worst:
        f, bl = (sum(worst[k]) / len(worst[k]) for k in ("freeze", "blind"))
        print(f"  freeze {f:.1f}% vs blind {bl:.1f}%  → "
              + ("낡은 관측이 무관측보다 해롭다 ★" if f < bl else "blind 가 더 해롭다"))
    degs = sorted(worst, key=lambda d: sum(worst[d]) / len(worst[d]))   # 아픈 것부터
    hi = set(a.hilite.split(","))

    fig, ax = plt.subplots(figsize=(7.6, 0.52 * len(degs) + 1.9), dpi=200)
    ys = range(len(degs))
    for y, d in zip(ys, degs):
        v = worst[d]
        mu = sum(v) / len(v)
        c = HILITE if d in hi else BAR
        ax.barh(y, mu, height=0.6, color=c, zorder=3,
                edgecolor="white", linewidth=2)
        ax.plot([min(v), max(v)], [y, y], color="white", lw=1.6, zorder=4)
        ax.plot([min(v), max(v)], [y, y], color=INK, lw=1.1, alpha=.55, zorder=5)
        ax.text(max(mu, max(v)) + 1.6, y, f"{mu:.1f}%   낙폭 {b0 - mu:+.1f}pp".replace("+", "−"),
                va="center", fontsize=9.5, color=INK,
                fontweight="medium" if d in hi else "normal", zorder=6)

    ax.axvline(b0, color=INK2, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.text(b0 - 1.2, len(degs) - 0.35, f"무결손 {b0:.1f}%", ha="right", va="center",
            fontsize=9, color=INK2)

    ax.set_yticks(list(ys), [TITLES.get(d, d) for d in degs])
    ax.set_xlabel("최대 강도(레벨 1.0)에서의 성공률 (%)", fontsize=10, color=INK)
    ax.set_xlim(0, max(b0, max(sum(v) / len(v) for v in worst.values())) + 26)
    ax.set_ylim(-0.6, len(degs) - 0.35)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5, length=0)
    if a.suptitle:
        ax.set_title(a.suptitle, fontsize=12.5, color=INK, pad=13, loc="left")
    n = sum(len(v) for v in worst.values()) // len(worst)
    fig.text(0.5, -0.02, f"막대 = 조건·시드 평균 (런 {n}개) · 가는 선 = 런 min~max · "
             f"조건 간 비교는 그림 6", ha="center", fontsize=8.5, color=INK2)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    order = "  ".join(f"{d} {sum(worst[d])/len(worst[d]):.1f}%" for d in degs)
    print(f"[saved] {a.out}\n  무결손 {b0:.1f}%  |  {order}")


if __name__ == "__main__":
    main()
