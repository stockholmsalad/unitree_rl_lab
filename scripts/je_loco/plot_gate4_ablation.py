#!/usr/bin/env python3
"""그림 10 — 게이트 4: 정책이 예측을 실제로 쓰는가.

정책 입력의 z_hat 블록만 0 으로 치환하고(그 외 전부 동일) 성공률이 얼마나 떨어지는지 잰다.
강인성 곡선은 간접 증거지만 이 절제는 직접 측정이다.

사전 예측(PREREGISTRATION_V2 §4): 낙폭(jepa) > 낙폭(none).
판정 규칙은 다른 게이트와 같다 — 조건 간 격차가 최대 조건 내 시드폭을 넘을 때만 신호.

입력: run_eval_v2.sh 가 만든 두 종류의 곡선
  abl_curve_*.csv   z_hat 절제 후 level 0 성공률 (한 행)
  그 외 *_curve_*   같은 런의 무결손(level 0) 성공률 — 기준선

사용:
  python scripts/je_loco/plot_gate4_ablation.py --dir results/v2_matrix_hard \
      --out docs/figs/fig10_gate4.png
"""
import argparse, csv, glob, os, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, GRID = "#1a1f24", "#5c666c", "#d9dfe2"
COLORS = {"jepa": "#2c6fbb", "recon": "#d1780a", "none": "#7c4bb8", "scratch": "#7c4bb8"}
LABELS = {"jepa": "JEPA", "recon": "Recon", "none": "None", "scratch": "Scratch"}
ORDER = ["jepa", "recon", "none", "scratch"]
RUN_RE = re.compile(r"_(?:D|V2)_([a-z]+)_s(\d+)_model_")


def load(d):
    """cond -> {seed: (무결손, 절제)}  — 같은 런끼리만 짝짓는다"""
    base, abl = defaultdict(dict), defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(d, "*_curve_*.csv"))):
        b = os.path.basename(f)
        m = RUN_RE.search(b)
        if not m:
            continue
        deg, cond, seed = b.split("_curve_")[0], m.group(1), int(m.group(2))
        rows = list(csv.DictReader(open(f)))
        lv = list(rows[0].keys())[0]
        first = sorted((float(r[lv]), float(r["success_rate"])) for r in rows)[0]
        if deg == "abl":
            abl[cond][seed] = first[1] * 100
        elif first[0] == 0.0:
            base[cond][seed] = first[1] * 100     # 결손 종류 무관하게 동일(게이트 1 이 확인)
    return {c: {s: (base[c][s], abl[c][s]) for s in sorted(abl[c]) if s in base[c]}
            for c in abl}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--out", default="docs/figs/fig10_gate4.png")
    p.add_argument("--suptitle", default="")
    a = p.parse_args()

    data = {c: v for c, v in load(a.dir).items() if v}
    if not data:
        raise SystemExit(f"abl_curve_*.csv 가 없다: {a.dir}  "
                         f"(run_eval_v2.sh 를 ABLATE=1 로 돌렸는지 확인)")
    conds = [c for c in ORDER if c in data]

    fig, ax = plt.subplots(figsize=(8.2, 0.62 * len(conds) + 2.2), dpi=200)
    drops, rows = {}, {}
    for i, c in enumerate(conds):
        y = len(conds) - 1 - i
        pairs = list(data[c].values())
        b = [x for x, _ in pairs]; z = [x for _, x in pairs]
        mb, mz = sum(b) / len(b), sum(z) / len(z)
        d = [x - y_ for x, y_ in pairs]
        drops[c] = d
        rows[c] = y

        # 아령: 무결손 → z_hat 절제
        ax.plot([mz, mb], [y, y], color=COLORS[c], lw=3, solid_capstyle="round", zorder=3)
        ax.plot([min(z), max(z)], [y, y], color=COLORS[c], lw=8, alpha=.22,
                solid_capstyle="round", zorder=2)
        ax.plot([mb], [y], "o", ms=9, mfc="white", mec=COLORS[c], mew=2.2, zorder=4)
        ax.plot([mz], [y], "o", ms=9, color=COLORS[c], mec="white", mew=1.6, zorder=4)
        ax.text((mb + mz) / 2, y + .17, f"−{sum(d)/len(d):.2f}pp", ha="center",
                fontsize=10, color=INK, fontweight="medium")
        ax.text(mb + 1.0, y, f"{mb:.1f}", va="center", fontsize=9, color=INK2)
        ax.text(mz - 1.0, y, f"{mz:.1f}", va="center", ha="right", fontsize=9, color=INK2)

    ax.set_yticks(list(rows.values()), [LABELS[c] for c in rows])
    ax.set_ylim(-0.65, len(conds) - 0.25)
    ax.set_xlabel("결손 없는 상태의 성공률 (%)", fontsize=10, color=INK)
    lo = min(min(v for _, v in d.values()) for d in data.values())
    hi = max(max(v for v, _ in d.values()) for d in data.values())
    ax.set_xlim(lo - 9, hi + 6)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9.5, length=0)

    # 판정 — 사전 등록 규칙 그대로
    verdict = ""
    if "jepa" in drops:
        mj = sum(drops["jepa"]) / len(drops["jepa"])
        spread = max(max(v) - min(v) for v in drops.values())
        parts = []
        for o in [c for c in conds if c != "jepa"]:
            g = mj - sum(drops[o]) / len(drops[o])
            parts.append(f"jepa−{o} {g:+.2f}pp {'★신호' if g > spread else '판정력 없음'}")
        verdict = f"최대 조건 내 시드폭 {spread:.2f}pp   |   " + "   ".join(parts)
    ttl = a.suptitle or "게이트 4 — 정책 입력의 z_hat 블록만 0 으로 절제했을 때"
    ax.set_title(ttl, fontsize=12.5, color=INK, pad=26, loc="left")
    ax.text(0, 1.015, verdict, transform=ax.transAxes, fontsize=9, color=INK2)

    n = max(len(v) for v in data.values())
    fig.text(0.5, -0.02, f"빈 원 = 무결손 · 찬 원 = z_hat 절제 · 굵은 띠 = 절제 시드 min~max "
             f"(n={n}) · 사전 예측: 낙폭(jepa) > 낙폭(none)",
             ha="center", fontsize=8.5, color=INK2)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    fig.tight_layout()
    fig.savefig(a.out, bbox_inches="tight", facecolor="white")
    print(f"[saved] {a.out}")
    for c in conds:
        d = drops[c]
        print(f"  {c:6s} n={len(d)}  낙폭 평균 {sum(d)/len(d):6.2f}pp  "
              f"[{min(d):.2f}, {max(d):.2f}]")


if __name__ == "__main__":
    main()
