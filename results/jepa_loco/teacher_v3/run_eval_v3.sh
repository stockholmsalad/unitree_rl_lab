#!/bin/bash
# teacher v3 평가: v2 구성(오르기·내리기·비정형) + 평지, 이어서 연속 코스(보조)
cd ~/graduation/unitree_rl_lab
while tmux has-session -t ev_mamba_jepa 2>/dev/null; do sleep 60; done
N="progress_18000 progress_18500 progress_19000 progress_19500 progress_20000"
bash results/jepa_loco/teacher_v2/eval/run_eval.sh $N
D=results/jepa_loco/teacher_v2/eval
for n in $N; do
  [ -f $D/${n}_flat.json ] && continue
  conda run -n env_test --no-capture-output python scripts/jepa_loco/eval_stairs_fixed.py --task Unitree-Go2-JepaLoco-OracleCurrent-EasyStart --seed 43 --envs_per_height 128 --headless --checkpoint logs/rsl_rl/teacher_v2/progress_model_${n##*_}.pt --terrain flat --output $D/${n}_flat.json > $D/logs/${n}_flat.log 2>&1
  echo "$(date +%H:%M) ${n}_flat exit=$?"
done
bash results/jepa_loco/course_eval/run_course.sh teacher_v3_18000 teacher_v3_18500 teacher_v3_19000 teacher_v3_19500 teacher_v3_20000
echo V3_ALL_DONE
