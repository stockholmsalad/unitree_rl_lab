"""Export the depth-only GRU student and fixed teacher head for NumPy inference."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch


def export(checkpoint_path, output_prefix):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["student_cfg"]
    if (cfg["method"] not in ("gru", "gru_recon_depth", "gru_recon_height") or
            cfg.get("backbone", "gru") != "gru" or cfg.get("proprio_input", False)):
        raise ValueError("이 export는 depth-only GRU checkpoint 전용이다")
    student = checkpoint["model_state_dict"]
    teacher = checkpoint["teacher_state_dict"]
    selected = {}
    for source, names in ((student, ("cnn.", "frame_projection.", "backbone.cell.", "terrain_head.")),
                          (teacher, ("obs_normalizer.", "mlp."))):
        for key, value in source.items():
            if key.startswith(names) and not key.startswith("obs_normalizer.count"):
                selected[("teacher." if source is teacher else "student.") + key] = value.detach().cpu().numpy().astype(np.float32)
    if "teacher.obs_normalizer._std" not in selected:
        raise ValueError("teacher normalizer가 checkpoint에 없다")
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(prefix) + ".npz", **selected)
    metadata = {"format": "jepa_depth_gru_numpy_v1", "student_cfg": cfg,
                "teacher_checkpoint": checkpoint.get("teacher_checkpoint"),
                "normalizer_eps": 0.01, "checkpoint_iteration": checkpoint.get("iteration")}
    Path(str(prefix) + ".json").write_text(json.dumps(metadata, indent=2, default=list))
    return prefix


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_prefix", required=True)
    args = parser.parse_args()
    print(export(args.checkpoint, args.output_prefix))
