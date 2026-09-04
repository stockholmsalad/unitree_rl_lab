#!/usr/bin/env python3
"""그림 4 — Phase 2 예측 지평선 스윕: 지평 k 에 따른 copy 대비 예측 우위(skill).

skill = 1 − loss_pred / copy_mse.  0 = "직전 프레임 복사"와 동점, >0 = 진짜 예측.
k 가 작으면 다음 프레임이 현재와 거의 같아 copy 가 이미 정답이므로 예측할 여지가 없다.
k 를 늘리면 copy 가 나빠져 예측이 값을 갖는다 — 그 곡선이 v2 의 k=100 선택 근거다.

로그 형식(pretrain_repr.py):
    [ep  20/20] pred=0.4232  skill_k15=0.4609  skill_k25=0.4789  ...
마지막 epoch 행의 skill_k* 를 그 로그의 최종값으로 삼는다.

사용:
  python scripts/je_loco/plot_horizon_sweep.py --logs ~/pretrain_jepa.log --out docs/figs/horizon.png
  python scripts/je_loco/plot_horizon_sweep.py --logs "~/pretrain_jepa*.log" --out ...
"""
import argparse, glob, os, re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, GRID = "#1a1f24", "#5c666c", "#d9dfe2"
SERIES = ["#2c6fbb", "#d1780a", "#7c4bb8"]      # 조건 색 (논문 전체 공통)
EP_RE = re.compile(r"\[ep\s+(\d+)/\s*(\d+)\]")
SKILL_RE = re.compile(r"skill_k(\d+)\s*=\s*(-?[\d.]+)")


def parse(path):
    """마지막 epoch 행의 {k: skill}. 없으면 None."""
    last = None
    with open(path) as f:
        for line in f:
            if EP_RE.search(line):
                found = SKILL_RE.findall(line)
                if found:
                    last = {int(k): float(v) for k, v in found}
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True, help="pretrain 로그 (glob 가능)")
    ap.add_argument("--out", default="docs/figs/horizon_sweep.png")
    ap.add_argument("--mark_k", type=int, default=100, help="채택한 지평 (강조 표시)")
    ap.add_argument("--labels", nargs="*", default=None,
                    help="범례 이름 (로그 순서대로). 없으면 파일 이름을 쓴다")
    a = ap.parse_args()

    paths = [p for pat in a.logs for p in sorted(glob.glob(os.path.expanduser(pat)))]
    series = [(os.path.basename(p).replace(".log", ""), d)
              for p in paths if (d := parse(p))]
    if a.labels:
        if len(a.labels) != len(series):
            raise SystemExit(f"--labels {len(a.labels)}개 ≠ 파싱된 로그 {len(series)}개")
        series = [(lab, d) for lab, (_, d) in zip(a.labels, series)]
    if not series:
        raise SystemExit(f"skill_k* 를 담은 로그가 없다: {paths}")

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=200)
    ax.axhline(0, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("copy 기준선 — 여기 아래면 예측이 복사보다 못하다",
                xy=(0.015, 0.015), xycoords=("axes fraction", "data"),
                ha="left", va="bottom", fontsize=8.5, color=INK2)

    peak = None
    for i, (name, d) in enumerate(series):
        ks = sorted(d)
        ys = [d[k] for k in ks]
        c = SERIES[i % len(SERIES)]
        ax.plot(ks, ys, color=c, lw=2, marker="o", ms=6.5, mew=1.6, mec="white",
                zorder=3, label=name if len(series) > 1 else None)
        best = max(ks, key=lambda k: d[k])
        if peak is None or d[best] > peak[1]:
            peak = (best, d[best], c)

    k, v, c = peak
    ax.annotate(f"k={k}  skill {v:.3f}", xy=(k, v), xytext=(0, 13),
                textcoords="offset points", ha="center", fontsize=9.5,
                color=INK, fontweight="medium")
    if a.mark_k in [k for _, d in series for k in d]:
        ax.axvline(a.mark_k, color=GRID, lw=1.2, zorder=0)

    ys = [v for _, d in series for v in d.values()]
    ax.set_ylim(min(0, min(ys)) - 0.03, max(ys) + 0.10)   # 최고점 라벨 자리
    ax.set_xlabel("예측 지평 k  (스텝, 1스텝 = 0.02 s)", fontsize=10, color=INK)
    ax.set_ylabel("skill  = 1 − loss_pred / copy_mse", fontsize=10, color=INK)
    ax.set_title("예측 지평이 길수록 예측이 값을 갖는다", fontsize=12.5,
                 color=INK, pad=11, loc="left")
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=9, loc="center left")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[saved] {a.out}")
    for name, d in series:
        print(f"  {name}: " + "  ".join(f"k{k}={d[k]:.4f}" for k in sorted(d)))


if __name__ == "__main__":
    main()
