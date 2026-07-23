# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""게이트 2 저하 곡선 플롯 — Head A(recon) vs Head B(jepa) depth dropout 강건성.

eval_pc.py 가 저장한 두 CSV(dropout,err_xy,err_yaw,speed,ep_frac,success_rate,terrain)를 읽어
2-패널 figure(추종오차·성공률 vs dropout%)로 그린다. A·B 동일 seed/지형에서 뽑은 CSV 를 넣을 것.

사용:
  python scripts/je_loco/plot_degradation.py \
    --recon dropout_curve_2026-07-11_17-59-32_model_24000.csv \
    --jepa  dropout_curve_2026-07-10_17-32-17_model_24000.csv \
    --out degradation_curve_AvsB_clean.png --title "Go2 · seed 42 · terrain 2.5 · 256 envs"
"""

import argparse
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load(path):
    xs, err, succ, spd = [], [], [], []
    with open(path) as f:
        rd = csv.DictReader(f)
        xcol = rd.fieldnames[0]   # 첫 열 = 결손 레벨 (dropout/hole/occlusion 무관)
        for r in rd:
            xs.append(float(r[xcol]) * 100.0)
            err.append(float(r["err_xy"]))
            succ.append(float(r["success_rate"]) * 100.0)
            spd.append(float(r["speed"]))
    return xs, err, succ, spd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recon", required=True, help="Head A(recon) CSV")
    ap.add_argument("--jepa", required=True, help="Head B(jepa) CSV")
    ap.add_argument("--out", default="degradation_curve_AvsB_clean.png")
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    xr, er, sr, _ = load(args.recon)
    xj, ej, sj, _ = load(args.jepa)

    C_R, C_J = "#d1495b", "#2e86ab"   # recon=red, jepa=blue
    plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))

    # (1) 추종오차 — 낮을수록 좋음
    ax[0].plot(xr, er, "-o", color=C_R, lw=2.2, ms=7, label="Reconstruction (Head A)")
    ax[0].plot(xj, ej, "-s", color=C_J, lw=2.2, ms=7, label="Prediction / JEPA (Head B)")
    ax[0].set_xlabel("Depth dropout (%)")
    ax[0].set_ylabel("Velocity tracking error  [m/s]  ↓")
    ax[0].set_title("(a) Command tracking under depth loss")
    ax[0].legend(frameon=False, fontsize=10, loc="upper left")
    ax[0].set_ylim(bottom=0)

    # (2) 성공률(비낙상) — 높을수록 좋음
    ax[1].plot(xr, sr, "-o", color=C_R, lw=2.2, ms=7, label="Reconstruction (Head A)")
    ax[1].plot(xj, sj, "-s", color=C_J, lw=2.2, ms=7, label="Prediction / JEPA (Head B)")
    ax[1].set_xlabel("Depth dropout (%)")
    ax[1].set_ylabel("Success rate (no fall)  [%]  ↑")
    ax[1].set_title("(b) Survival under depth loss")
    ax[1].legend(frameon=False, fontsize=10, loc="lower left")
    ax[1].set_ylim(40, 102)

    if args.title:
        fig.suptitle(args.title, fontsize=10, y=1.01, color="#555")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"[saved] {args.out}")

    # 콘솔 요약
    print(f"  clean err  recon {er[0]:.3f} / jepa {ej[0]:.3f}")
    print(f"  100%  err  recon {er[-1]:.3f} / jepa {ej[-1]:.3f}   (배율 recon {er[-1]/er[0]:.1f}x, jepa {ej[-1]/ej[0]:.1f}x)")
    print(f"  100%  succ recon {sr[-1]:.1f}% / jepa {sj[-1]:.1f}%")


if __name__ == "__main__":
    main()
