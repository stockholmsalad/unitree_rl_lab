"""실기체 depth 변환 결과와 시뮬레이터 출력의 통계 비교 (Phase 2 전 sim-to-real 점검).

실기체: 848×480 프레임 → ``downsample_real``(nearest, median 둘 다) → 112×64.
시뮬레이터: 112×64 직접 렌더링 npy (Phase 0 스크립트 ``depth_render.npy``).
양쪽 모두 ``to_two_channel`` 을 거친 뒤 비교한다 — 정책이 실제로 보는 값.

출력: 유효 픽셀 비율(전체·행별·열별 — 열별은 왼쪽 결측 띠 확인용), 유효 depth 히스토그램, summary.json.

    python scripts/jepa_loco/compare_depth_stats.py \\
        --real <dir 또는 .npy> --sim results/jepa_loco/phase0/depth_render.npy --out results/jepa_loco/depth_stats

실기체 입력: (N,480,848) 미터 단위 .npy, 또는 z16 .png(uint16, --depth_scale 로 미터 변환) 디렉터리.
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from unitree_rl_lab.jepa_loco.sensors.depth_proc import apply_left_band, downsample_real, to_two_channel

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--real", type=str, required=True)
parser.add_argument("--sim", type=str, required=True)
parser.add_argument("--out", type=str, default="results/jepa_loco/depth_stats")
parser.add_argument("--depth_scale", type=float, default=0.001, help="z16 png → m")
parser.add_argument("--out_hw", type=int, nargs=2, default=(64, 112))
parser.add_argument("--near", type=float, default=0.28)
parser.add_argument("--far", type=float, default=2.0)
parser.add_argument("--median_min_valid", type=float, default=0.5)
parser.add_argument("--sim_left_band_fx", type=float, default=None, help="주면 시뮬 출력에 왼쪽 결측 띠 적용")
parser.add_argument("--baseline", type=float, default=0.050)
args = parser.parse_args()


def load_real(path: str) -> torch.Tensor:
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path)).float()
    files = sorted(glob.glob(os.path.join(path, "*.npy")) + glob.glob(os.path.join(path, "*.png")))
    if not files:
        raise FileNotFoundError(path)
    frames = []
    for f in files:
        if f.endswith(".npy"):
            frames.append(np.load(f).astype(np.float32))
        else:
            import cv2

            frames.append(cv2.imread(f, cv2.IMREAD_UNCHANGED).astype(np.float32) * args.depth_scale)
    return torch.from_numpy(np.stack(frames))


def stats(two_ch: torch.Tensor) -> dict:
    """two_ch: (N, 2, H, W)."""
    depth, mask = two_ch[:, 0], two_ch[:, 1].bool()
    vals = depth[mask].numpy()
    hist, _ = np.histogram(vals, bins=BINS)
    return {
        "n_frames": int(two_ch.shape[0]),
        "valid_frac": float(mask.float().mean()),
        "valid_frac_row": mask.float().mean((0, 2)).tolist(),
        "valid_frac_col": mask.float().mean((0, 1)).tolist(),
        "clamped_far_frac": float((depth[mask] >= 1.0).float().mean()) if mask.any() else 0.0,
        "depth_hist": (hist / max(hist.sum(), 1)).tolist(),
    }


BINS = np.linspace(0.0, 1.0, 41)
real = load_real(args.real)
sim = torch.from_numpy(np.load(args.sim)).float()
assert tuple(sim.shape[-2:]) == tuple(args.out_hw), f"sim {tuple(sim.shape)} ≠ {args.out_hw}"
if args.sim_left_band_fx:
    sim = apply_left_band(sim, args.sim_left_band_fx, args.baseline)

sets = {
    "real_nearest": to_two_channel(downsample_real(real, args.out_hw, "nearest", args.near), args.near, args.far),
    "real_median": to_two_channel(
        downsample_real(real, args.out_hw, "median", args.near, args.median_min_valid), args.near, args.far
    ),
    "sim": to_two_channel(sim, args.near, args.far),
}
summary = {k: stats(v) for k, v in sets.items()}

os.makedirs(args.out, exist_ok=True)
with open(os.path.join(args.out, "summary.json"), "w") as f:
    json.dump(summary, f)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
centers = (BINS[:-1] + BINS[1:]) / 2 * (args.far - args.near) + args.near
for k, s in summary.items():
    axes[0].plot(centers, s["depth_hist"], label=f"{k} (valid {s['valid_frac']:.3f})")
    axes[1].plot(s["valid_frac_row"], label=k)
    axes[2].plot(s["valid_frac_col"], label=k)
axes[0].set(xlabel="depth [m] (valid only)", ylabel="fraction", title="depth histogram")
axes[1].set(xlabel="row (top→bottom)", ylabel="valid fraction", title="valid fraction by row")
axes[2].set(xlabel="column (left→right)", ylabel="valid fraction", title="valid fraction by column (left band)")
for ax in axes:
    ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(args.out, "depth_stats.png"), dpi=110)
for k, s in summary.items():
    print(f"{k:13s} frames={s['n_frames']:4d} valid={s['valid_frac']:.3f} far_clamped={s['clamped_far_frac']:.3f}")
