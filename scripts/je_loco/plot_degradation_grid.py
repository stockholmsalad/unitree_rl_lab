# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""게이트 2 종합 figure — 3결손(dropout·hole·occlusion) × 2지표(추종오차·생존율) 2×3 그리드.

eval_pc.py 가 저장한 6개 CSV(각 결손 × {recon,jepa})를 읽어 한 장으로. A·B 동일 seed/지형 전제.

사용:
  python scripts/je_loco/plot_degradation_grid.py \
    --recon_tag 2026-07-11_17-59-32_model_24000 --jepa_tag 2026-07-10_17-32-17_model_24000 \
    --out degradation_grid_AvsB.png
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    xs, err, succ = [], [], []
    with open(path) as f:
        rd = csv.DictReader(f)
        xcol = rd.fieldnames[0]
        for r in rd:
            xs.append(float(r[xcol]) * 100.0)
            err.append(float(r["err_xy"]))
            succ.append(float(r["success_rate"]) * 100.0)
    return xs, err, succ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon_tag", required=True, help="예: 2026-07-11_17-59-32_model_24000")
    ap.add_argument("--jepa_tag", required=True)
    ap.add_argument("--out", default="degradation_grid_AvsB.png")
    ap.add_argument("--metric", default="both", choices=["err", "success", "both"],
                    help="err/success=1행 컴팩트(논문 지면 절약), both=2행 전체")
    ap.add_argument("--suptitle", action="store_true", help="상단 제목 표시(기본 off — 캡션으로 대체)")
    args = ap.parse_args()

    degs = [("dropout", "i.i.d. dropout"), ("hole", "clustered holes"), ("occlusion", "bottom-band occlusion")]
    C_R, C_J = "#d1495b", "#2e86ab"
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

    rows = ["err", "success"] if args.metric == "both" else [args.metric]
    figh = 6.4 if len(rows) == 2 else 2.6          # 1행이면 얇은 밴드(논문 지면 절약)
    fig, ax = plt.subplots(len(rows), 3, figsize=(13, figh), sharex=True, squeeze=False)

    for j, (deg, nice) in enumerate(degs):
        xr, er, sr = load(f"{deg}_curve_{args.recon_tag}.csv")
        xj, ej, sj = load(f"{deg}_curve_{args.jepa_tag}.csv")
        data = {"err": (er, ej, "Vel. tracking error [m/s] ↓", (0, 0.62)),
                "success": (sr, sj, "Success rate (no fall) [%] ↑", (40, 102))}
        for i, m in enumerate(rows):
            yr, yj, ylab, ylim = data[m]
            ax[i, j].plot(xr, yr, "-o", color=C_R, lw=2.2, ms=6, label="Reconstruction (A)")
            ax[i, j].plot(xj, yj, "-s", color=C_J, lw=2.2, ms=6, label="Prediction / JEPA (B)")
            ax[i, j].set_ylim(*ylim)
            if i == 0:
                ax[i, j].set_title(f"{chr(97+j)}) {nice}", fontsize=12)
            if i == len(rows) - 1:
                ax[i, j].set_xlabel("Depth degradation (%)")
            if j == 0:
                ax[i, j].set_ylabel(ylab)

    ax[0, 0].legend(frameon=False, fontsize=9.5, loc="upper left")
    if args.suptitle:
        fig.suptitle("Graceful degradation under depth loss: predictive (JEPA) vs. reconstructive representation",
                     fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97] if args.suptitle else None)
    fig.savefig(args.out, bbox_inches="tight")
    print(f"[saved] {args.out}  (metric={args.metric}, {len(rows)}행)")


if __name__ == "__main__":
    main()
