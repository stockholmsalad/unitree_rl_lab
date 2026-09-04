#!/usr/bin/env python3
"""그림 3 — 교사 명령 봉투: 속도 × 회전 격자에서의 성공률.

증류 타깃 명령 범위를 넓히기 전에 "교사가 실제로 어디까지 걷는가"를 실측한 결과다.
v2 가 0.6~1.5 m/s · ±0.6 rad/s 를 쓰는 근거이며, 추측이 아니라 이 격자에서 나왔다.

입력: eval_cmd_envelope.py 가 저장한 CSV
      v_cmd,w_cmd,success_rate,speed,yaw_rate,err_xy,err_yaw,ep_frac,n_ep

사용:
  python scripts/je_loco/plot_cmd_envelope.py --csv cmd_envelope_*.csv --out docs/figs/envelope.png
"""
import argparse, csv, glob, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, GRID = "#1a1f24", "#5c666c", "#d9dfe2"
# 순차 램프 = 단일 색상 밝음→어두움 (무지개 금지). 조건 색 #2c6fbb 계열로 통일.
RAMP = LinearSegmentedColormap.from_list("jeloco_seq", ["#f2f6fa", "#9dc0e2", "#2c6fbb", "#123f70"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--out", default="docs/figs/cmd_envelope.png")
    ap.add_argument("--metric", default="success_rate")
    ap.add_argument("--title", default="교사 명령 봉투 — 지형 레벨 5")
    a = ap.parse_args()

    paths = [p for pat in a.csv for p in sorted(glob.glob(os.path.expanduser(pat)))]
    cells = {}
    for p in paths:
        with open(p) as f:
            for r in csv.DictReader(f):
                cells[(float(r["v_cmd"]), float(r["w_cmd"]))] = float(r[a.metric])
    if not cells:
        raise SystemExit(f"셀이 없다: {paths}")

    vs = sorted({v for v, _ in cells})
    ws = sorted({w for _, w in cells}, reverse=True)   # 위가 +회전
    grid = [[cells.get((v, w)) for v in vs] for w in ws]

    fig, ax = plt.subplots(figsize=(1.15 * len(vs) + 2.4, 0.95 * len(ws) + 2.1), dpi=200)
    im = ax.imshow([[0 if c is None else c for c in row] for row in grid],
                   cmap=RAMP, vmin=0.0, vmax=1.0, aspect="auto")

    # 셀마다 값을 적는다 — 전 셀이 높으면 색만으로는 차이가 안 읽힌다.
    for i, row in enumerate(grid):
        for j, c in enumerate(row):
            if c is None:
                ax.text(j, i, "—", ha="center", va="center", color=INK2, fontsize=10)
                continue
            ax.text(j, i, f"{100*c:.0f}", ha="center", va="center", fontsize=11,
                    fontweight="medium", color="white" if c > 0.55 else INK)

    ax.set_xticks(range(len(vs)), [f"{v:.1f}" for v in vs])
    ax.set_yticks(range(len(ws)), [f"{w:+.1f}" for w in ws])
    ax.set_xlabel("전진 속도 명령  v  (m/s)", fontsize=10, color=INK)
    ax.set_ylabel("회전 속도 명령  ω  (rad/s)", fontsize=10, color=INK)
    ax.set_title(a.title, fontsize=12.5, color=INK, pad=11, loc="left")
    ax.tick_params(colors=INK2, labelsize=9.5, length=0)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.set_xticks([x - .5 for x in range(1, len(vs))], minor=True)
    ax.set_yticks([y - .5 for y in range(1, len(ws))], minor=True)
    ax.grid(which="minor", color="white", lw=2)      # 셀 사이 2px 간격

    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("성공률 (%)", fontsize=9.5, color=INK)
    cb.set_ticks([0, .25, .5, .75, 1.0])
    cb.ax.set_yticklabels(["0", "25", "50", "75", "100"])
    cb.ax.tick_params(colors=INK2, labelsize=9)
    cb.outline.set_edgecolor(GRID)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[saved] {a.out}   {len(cells)} 셀  최저 {100*min(cells.values()):.1f}%  "
          f"최고 {100*max(cells.values()):.1f}%")


if __name__ == "__main__":
    main()
