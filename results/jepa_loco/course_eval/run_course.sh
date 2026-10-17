#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../../.."
PYTHON_BIN=${PYTHON_BIN:-/home/user/miniconda3/envs/env_test/bin/python}
DEST=results/jepa_loco/course_eval
mkdir -p "$DEST/logs"
teacher=logs/rsl_rl/jepa_loco_oracle_current/teacher_cont_s42/model_17000.pt
names=(teacher gru gru_jepa_vf05 gru_jepa_propw5 gru_propw5 mamba)
if (($#)); then selected=("$@"); else selected=("${names[@]}"); fi
for name in "${selected[@]}"; do
  out="$DEST/${name}_course_s43.json"
  if [[ -f "$out" ]]; then echo "skip existing $out"; continue; fi
  case "$name" in
    teacher)
      checkpoint="$teacher"
      task=Unitree-Go2-JepaLoco-OracleCurrent-EasyStart
      evaluator=scripts/jepa_loco/eval_stairs_fixed.py
      ;;
    gru) checkpoint=logs/student/gru_t17000_s43/model_3000.pt ;;
    gru_jepa_vf05) checkpoint=logs/student/gru_jepa_t17000_vf05_s43/model_3000.pt ;;
    gru_jepa_propw5) checkpoint=logs/student/gru_jepa_propw5_t17000_vf05_s43/model_3000.pt ;;
    gru_propw5) checkpoint=logs/student/gru_propw5_t17000_s43/model_3000.pt ;;
    mamba) checkpoint=logs/student/mamba_t17000_s43/model_3000.pt ;;
    *) echo "unknown model: $name" >&2; exit 2 ;;
  esac
  [[ -f "$checkpoint" ]] || { echo "missing checkpoint: $checkpoint" >&2; exit 2; }
  if [[ "$name" == teacher ]]; then
    "$PYTHON_BIN" "$evaluator" --task "$task" --checkpoint "$checkpoint" --terrain course --course_step_heights_m 0.10 0.12 0.14 --envs_per_height 128 --seed 43 --output "$out" --headless > "$DEST/logs/${name}.log" 2>&1
  else
    "$PYTHON_BIN" scripts/jepa_loco/eval_student_fixed.py --checkpoint "$checkpoint" --teacher_stats results/jepa_loco/student/teacher_latent_stats_t17000.pt --terrain course --course_step_heights_m 0.10 0.12 0.14 --envs_per_height 128 --seed 43 --output "$out" --headless > "$DEST/logs/${name}.log" 2>&1
  fi
  echo "done $name: $out"
done
"$PYTHON_BIN" "$DEST/summarize_course.py" "$DEST"/*_course_s43.json --output "$DEST/summary.md"
