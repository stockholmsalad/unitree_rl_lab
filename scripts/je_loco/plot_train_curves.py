#!/usr/bin/env python3
"""그림 8 — v2 증류 학습 곡선: 조건별 시드 평균 + min~max 밴드.

세 지표는 스케일이 달라 한 축에 못 올린다(이중 축 금지). 패널을 나눈다:
  · 행동손실   세 조건 공통. 보조손실이 모방을 해쳤는지 — 게이트 1 의 전제.
  · jepa_skill jepa 전용. >0 = copy 보다 나은 진짜 예측. v2 의 전제가 성립하는지.
  · z_e_std    jepa 전용. 표현 붕괴 감시.

로그 형식(rsl-rl): "Learning iteration N/M" 뒤에 "Mean <지표> loss: x" 행들.
런 이름은 <접두어>_<조건>_s<시드> 로 가정한다(V2_jepa_s1 등).

사용:
  python scripts/je_loco/plot_train_curves.py --logs "~/distill_logs/V2_*.log" \
      --out docs/figs/v2_train.png
"""
import argparse, glob, os, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, GRID = "#1a1f24", "#5c666c", "#d9dfe2"
COLORS = {"jepa": "#2c6fbb", "recon": "#d1780a", "none": "#7c4bb8", "scratch": "#7c4bb8"}
LABELS = {"jepa": "JEPA (예측)", "recon": "Recon (재구성)",
          "none": "None (행동손실만)", "scratch": "Scratch"}
ORDER = ["jepa", "recon", "none", "scratch"]

ANSI = re.compile(r"\x1b\[[0-9;]*m")
IT_RE = re.compile(r"Learning iteration\s+(\d+)/")
PANELS = [("behavior", "행동손실", True), ("jepa_skill", "jepa_skill", False),
          ("z_e_std", "z_e_std", False)]


def parse(path):
    """{지표: [(iter, value), ...]}"""
    out = defaultdict(list)
    it = None
    pat = re.compile(r"Mean (" + "|".join(k for k, _, _ in PANELS) + r")[a-z_ ]* loss:\s+(-?[\d.]+)")
    with open(path, errors="ignore") as f:
        for raw in f:
            line = ANSI.sub("", raw)
            m = IT_RE.search(line)
            if m:
                it = int(m.group(1))
                continue
            m = pat.search(line)
            if m and it is not None:
                out[m.group(1)].append((it, float(m.group(2))))
    return out


def band(ax, runs, key, color, label):
    """시드별 곡선 → 공통 iter 격자 위 평균선 + min~max 밴드."""
    curves = [dict(r[key]) for r in runs if r.get(key)]
    if not curves:
        return False
    its = sorted(set.intersection(*(set(c) for c in curves)))
    if len(its) < 2:
        return False
    lo = [min(c[i] for c in curves) for i in its]
    hi = [max(c[i] for c in curves) for i in its]
    mu = [sum(c[i] for c in curves) / len(curves) for i in its]
    if len(curves) > 1:
        ax.fill_between(its, lo, hi, color=color, alpha=0.16, lw=0, zorder=2)
    ax.plot(its, mu, color=color, lw=2, zorder=3,
            label=f"{label}  (n={len(curves)})")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--out", default="docs/figs/v2_train.png")
    a = ap.parse_args()

    paths = [p for pat in a.logs for p in sorted(glob.glob(os.path.expanduser(pat)))]
    byc = defaultdict(list)
    for p in paths:
        stem = os.path.basename(p).replace(".log", "")
        parts = stem.split("_")
        cond = next((x for x in parts if x in COLORS), None)
        if cond is None:
            print(f"  ! 조건 파싱 실패, 건너뜀: {stem}")
            continue
        d = parse(p)
        if d:
            byc[cond].append(d)
            print(f"  {stem}: " + "  ".join(f"{k}×{len(v)}" for k, v in sorted(d.items())))
    if not byc:
        raise SystemExit(f"파싱된 런이 없다: {paths}")

    live = [(k, t, lg) for k, t, lg in PANELS
            if any(r.get(k) for rs in byc.values() for r in rs)]
    fig, axes = plt.subplots(1, len(live), figsize=(4.5 * len(live), 3.5), dpi=200)
    axes = [axes] if len(live) == 1 else list(axes)

    for ax, (key, title, logy) in zip(axes, live):
        drew = 0
        for c in ORDER:
            if c in byc and band(ax, byc[c], key, COLORS[c], LABELS[c]):
                drew += 1
        if key == "jepa_skill":
            ax.axhline(0, color=INK2, lw=1.2, ls=(0, (4, 3)), zorder=1)
        if logy:
            ax.set_yscale("log")
        ax.set_title(title, fontsize=11.5, color=INK, pad=9, loc="left")
        ax.set_xlabel("증류 iteration", fontsize=9.5, color=INK2)
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=9)
        if drew > 1:
            ax.legend(frameon=False, fontsize=8.5, loc="best")
        elif drew == 1:
            ax.legend(frameon=False, fontsize=8.5, loc="best")

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[saved] {a.out}   패널 {[k for k,_,_ in live]}")


if __name__ == "__main__":
    main()
