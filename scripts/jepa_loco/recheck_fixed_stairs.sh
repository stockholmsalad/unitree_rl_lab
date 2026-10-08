#!/usr/bin/env bash
# 2026-10-08 EasyStart 고정 계단 재평가: seed 2개 × 높이 쌍 4개 × 반복 2회.
set -euo pipefail

python_bin=${PYTHON_BIN:-python}
task=Unitree-Go2-JepaLoco-OracleCurrent-EasyStart
output_root=results/jepa_loco/stair_eval/height_sweep_verified
lows=(0.05 0.09 0.13 0.17)
highs=(0.07 0.11 0.15 0.19)

for repeat in 1 2; do
    for seed in 42 43; do
        if [[ ${seed} == 42 ]]; then
            checkpoint=logs/rsl_rl/jepa_loco_oracle_current/2026-10-07_17-57-46_oracle_current_easystart_s42/model_2999.pt
        else
            checkpoint=logs/rsl_rl/jepa_loco_oracle_current/2026-10-07_18-08-21_oracle_current_easystart_s43/model_2999.pt
        fi
        for i in 0 1 2 3; do
            low=${lows[$i]}
            high=${highs[$i]}
            output=${output_root}/run${repeat}/easystart_s${seed}_${low}_${high}.json
            log=/tmp/jepa_easystart_s${seed}_${low}_${high}_run${repeat}.log
            mkdir -p "$(dirname "${output}")"
            "${python_bin}" scripts/jepa_loco/eval_stairs_fixed.py --task "${task}" --checkpoint "${checkpoint}" --output "${output}" --terrain stairs --spawn_forward_m 0 --envs_per_height 128 --seed 43 --low_step_height_m "${low}" --high_step_height_m "${high}" --headless > "${log}" 2>&1
            if [[ ! -s ${output} ]]; then
                tail -n 40 "${log}"
                exit 1
            fi
            "${python_bin}" - "${output}" <<'PY'
import json
import sys

path = sys.argv[1]
report = json.load(open(path))
rows = report["per_env"]
origins = report["initial_scan_geometry"]["terrain_origin_z"]
count = report["stair_count"]
assert len(rows) == len(origins) == 256
bad = [row["env_id"] for row, z in zip(rows, origins)
       if abs(z + (count + 1) * row["step_height_m"]) > 1.0e-4]
if bad:
    raise SystemExit(f"INVALID terrain columns in {path}: {bad[:10]}")
print(f"VERIFIED {path}: low={report['groups']['low']['success_rate']:.6f} "
      f"high={report['groups']['high']['success_rate']:.6f} flat=0", flush=True)
PY
        done
    done
done
