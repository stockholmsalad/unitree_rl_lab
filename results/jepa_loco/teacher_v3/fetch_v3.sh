#!/bin/bash
# pilab teacher v3 checkpoint -> Z790 logs/rsl_rl/teacher_v2/progress_model_<iter>.pt (v2 평가 스크립트 이름 규칙)
cd ~/graduation/unitree_rl_lab
R=taehyeong_pilab@155.230.24.142:workspaces/unitree_rl_lab/logs/rsl_rl/jepa_loco_oracle_current_progress/2026-10-17_22-09-28_progress_s42_from17000
mkdir -p /tmp/v3
scp "$R/model_{18000,18500,19000,19500,20000}.pt" /tmp/v3/ || exit 1
for f in /tmp/v3/model_*.pt; do mv "$f" logs/rsl_rl/teacher_v2/progress_$(basename "$f"); done
ls -la logs/rsl_rl/teacher_v2/progress_*
