#!/bin/bash
# 천장 효과 확인: 오르기 16~18 cm, teacher 19000 + student 2500/2800/3000
cd ~/graduation/unitree_rl_lab
D=results/jepa_loco/student/eval_t19000
H="--step_heights_m 0.16 0.17 0.18 --seed 43 --envs_per_height 128 --headless --terrain stairs"
[ -f $D/teacher19000_up_hard.json ] || conda run -n env_test --no-capture-output python scripts/jepa_loco/eval_stairs_fixed.py --task Unitree-Go2-JepaLoco-OracleCurrent-EasyStart --checkpoint logs/rsl_rl/teacher_v2/progress_model_19000.pt $H --output $D/teacher19000_up_hard.json > $D/logs/teacher19000_up_hard.log 2>&1; echo "$(date +%H:%M) teacher exit=$?"
for m in "$@"; do for it in 3000 2800 2500; do
  o=$D/${m}_${it}_up_hard.json; [ -f $o ] && continue
  conda run -n env_test --no-capture-output python scripts/jepa_loco/eval_student_fixed.py --checkpoint logs/student/$m/model_$it.pt --teacher_stats results/jepa_loco/student/teacher_latent_stats_t19000.pt $H --output $o > $D/logs/${m}_${it}_up_hard.log 2>&1; echo "$(date +%H:%M) ${m}_$it exit=$?"
done; done
echo ALL_DONE
