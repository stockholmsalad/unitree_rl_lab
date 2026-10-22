#!/bin/bash
# Z790: pilab student checkpoint 2500/2800/3000 가져오기. 사용법: bash fetch.sh <run_dir> [...]
cd ~/graduation/unitree_rl_lab
R=taehyeong_pilab@155.230.24.142:workspaces/unitree_rl_lab
[ -f results/jepa_loco/student/teacher_latent_stats_t19000.pt ] || scp $R/results/jepa_loco/student/teacher_latent_stats_t19000.pt results/jepa_loco/student/ || exit 1
for m in "$@"; do mkdir -p logs/student/$m && scp "$R/logs/student/$m/model_{2500,2800,3000}.pt" logs/student/$m/ || exit 1; done
for m in "$@"; do ls -la logs/student/$m; done
