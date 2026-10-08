#!/usr/bin/env bash
# Fixed gap comparison: Current s42/s43, Current+Future s42/s43, two identical repeats.
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 CURRENT_S42 CURRENT_S43 FUTURE_S42 FUTURE_S43 OUTPUT_DIR" >&2
    exit 2
fi

for checkpoint in "${@:1:4}"; do
    if [[ ! -f "$checkpoint" ]]; then
        echo "missing checkpoint: $checkpoint" >&2
        exit 2
    fi
done

checkpoints=("$1" "$2" "$3" "$4")
tasks=(
    Unitree-Go2-JepaLoco-OracleCurrent-EasyStart
    Unitree-Go2-JepaLoco-OracleCurrent-EasyStart
    Unitree-Go2-JepaLoco-OracleCurrentFuture-EasyStart
    Unitree-Go2-JepaLoco-OracleCurrentFuture-EasyStart
)
names=(current_s42 current_s43 future_s42 future_s43)
low_widths=(0.10 0.20 0.30)
high_widths=(0.15 0.25 0.35)
python_bin="${PYTHON_BIN:-python}"
output_dir="$5"
mkdir -p "$output_dir"

for repeat in 1 2; do
    for index in 0 1 2 3; do
        for pair in 0 1 2; do
            output="$output_dir/${names[$index]}_${low_widths[$pair]}_${high_widths[$pair]}_repeat${repeat}.json"
            "$python_bin" scripts/jepa_loco/eval_stairs_fixed.py \
                --task "${tasks[$index]}" --checkpoint "${checkpoints[$index]}" \
                --output "$output" --terrain gap \
                --low_gap_width_m "${low_widths[$pair]}" --high_gap_width_m "${high_widths[$pair]}" \
                --envs_per_width 128 --spawn_forward_m 0 --seed 43 --headless
        done
    done
done
