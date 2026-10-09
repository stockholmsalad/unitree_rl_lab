"""서로 다른 seed의 저장된 student 데이터로 linear probe를 학습·평가한다."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--train_data", required=True)
parser.add_argument("--test_data", required=True)
parser.add_argument("--visibility_json", default="results/jepa_loco/blind_spot/visibility.json")
parser.add_argument("--output", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
simulation_app = AppLauncher(args).app

import torch
from isaaclab.utils import configclass

from unitree_rl_lab.jepa_loco.eval.linear_probe import (
    fit_linear_probe, masked_reconstruction_mse, probe_predict,
)


@configclass
class ProbeFitCfg:
    heights_m: tuple[float, float] = (0.09, 0.13)
    ridge_alpha: float = 1.0
    distance_halfwidth_m: float = 0.15


def main():
    cfg = ProbeFitCfg()
    train = torch.load(args.train_data, map_location="cpu", weights_only=False)
    test = torch.load(args.test_data, map_location="cpu", weights_only=False)
    if train["seed"] == test["seed"] or train["checkpoint"] != test["checkpoint"]:
        raise ValueError("probe 학습·평가는 서로 다른 seed의 같은 student checkpoint여야 한다")
    visibility = json.loads(Path(args.visibility_json).read_text())
    rows = visibility["stairs"]
    if len(visibility["flat_visible_mask"]) != 187:
        raise ValueError("사각지대 mask가 teacher 187점과 맞지 않는다")
    if train["heightmap"].shape[1] != 187 or test["heightmap"].shape[1] != 187:
        raise ValueError("teacher 높이맵 차원이 187이 아니다")
    coefficients = fit_linear_probe(train["features"], train["heightmap"], cfg.ridge_alpha)
    predictions = probe_predict(test["features"], coefficients)
    evaluation = []
    for height_index, height in enumerate(cfg.heights_m):
        for row in rows:
            if abs(row["step_height_m"] - height) > 1.0e-6:
                continue
            distance = row["distance_to_first_riser_m"]
            select = ((test["height_index"] == height_index) &
                      ((test["distance"] - distance).abs() <= cfg.distance_halfwidth_m))
            visible = torch.tensor(row["render_visible_mask"], dtype=torch.bool)
            item = {"step_height_m": height, "distance_to_first_riser_m": distance,
                    "n": int(select.sum())}
            if select.any():
                item["hidden_mse"] = masked_reconstruction_mse(
                    predictions[select], test["heightmap"][select], ~visible).item()
                item["visible_mse"] = masked_reconstruction_mse(
                    predictions[select], test["heightmap"][select], visible).item()
            else:
                item["hidden_mse"] = item["visible_mse"] = None
            evaluation.append(item)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": test["checkpoint"], "train_seed": train["seed"],
              "test_seed": test["seed"], "visibility_json": str(Path(args.visibility_json).resolve()),
              "config": cfg.to_dict(), "train_samples": len(train["features"]),
              "test_samples": len(test["features"]), "rows": evaluation}
    output.write_text(json.dumps(report, indent=2))
    torch.save({"coefficients": coefficients, "train_seed": train["seed"],
                "test_seed": test["seed"]}, output.with_suffix(".pt"))
    print("STUDENT_BLIND_SPOT_PROBE", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
