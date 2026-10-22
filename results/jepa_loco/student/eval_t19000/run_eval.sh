#!/bin/bash
# teacher 19000 student 평가: 3000 전체 + 2500/2800 오르기 (eval_t17000/propw5와 같은 구성)
cd ~/graduation/unitree_rl_lab
D=results/jepa_loco/student/eval_t19000
PY="conda run -n env_test --no-capture-output python scripts/jepa_loco/eval_student_fixed.py --teacher_stats results/jepa_loco/student/teacher_latent_stats_t19000.pt --seed 43 --envs_per_height 128 --headless"
run() { name=$1; shift; [ -f $D/$name.json ] && return; $PY "$@" --output $D/$name.json > $D/logs/$name.log 2>&1; echo "$(date +%H:%M) $name exit=$?"; }
for m in "$@"; do
  for it in 3000 2800 2500; do
    C=logs/student/$m/model_$it.pt; [ -f $C ] || { echo "skip $C"; continue; }
    run ${m}_${it}_stairs_up --checkpoint $C --terrain stairs --step_heights_m 0.09 0.10 0.11 0.12 0.13 0.14 0.15
    [ $it = 3000 ] || continue
    run ${m}_${it}_stairs_down --checkpoint $C --terrain stairs_down --step_heights_m 0.15 0.16 0.17 0.18 0.19 0.20
    run ${m}_${it}_rough --checkpoint $C --terrain rough --rough_max_heights_m 0.06 0.08 0.10 0.12
    run ${m}_${it}_flat --checkpoint $C --terrain flat
  done
done
echo ALL_DONE
