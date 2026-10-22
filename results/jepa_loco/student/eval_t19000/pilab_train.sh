#!/bin/bash
# pilab: teacher 19000 student 학습. 사용법: [SEED=44] bash pilab_train.sh <run> [<run>]
# run: mamba_jepa_propw5 | mamba_jepa_fe_propw5 | mamba_propw5 | mamba_recon_depth | mamba_recon_depth_lr03
#      | mamba_recon_height | gru_jepa_propw5 | gru_propw5
cd ~/workspaces/unitree_rl_lab
SEED=${SEED:-43}
T=logs/rsl_rl/jepa_loco_oracle_current_progress/2026-10-17_22-09-28_progress_s42_from17000/model_19000.pt
C="--teacher_checkpoint $T --teacher_stats results/jepa_loco/student/teacher_latent_stats_t19000.pt --proprio_input --proprio_window 5 --num_envs 2048 --max_iterations 3000 --seed $SEED --headless"
for r in "$@"; do
  case $r in
    mamba_jepa_propw5) o=mamba_jepa_propw5_t19000_vf05; a="--method gru_jepa --backbone mamba --jepa_variance_floor 0.05" ;;
    mamba_jepa_fe_propw5) o=mamba_jepa_fe_propw5_t19000_vf1e4; a="--method gru_jepa --backbone mamba --jepa_variance_floor 1e-4 --jepa_target frame_embedding" ;;
    mamba_propw5) o=mamba_propw5_t19000; a="--method gru --backbone mamba" ;;
    mamba_recon_depth) o=mamba_recon_depth_propw5_t19000_rvf05; a="--method gru_recon_depth --backbone mamba" ;;
    mamba_recon_depth_lr03) o=mamba_recon_depth_lr03_propw5_t19000_rvf05; a="--method gru_recon_depth --backbone mamba --lambda_r 0.03" ;;
    mamba_recon_height) o=mamba_recon_height_propw5_t19000; a="--method gru_recon_height --backbone mamba" ;;
    gru_jepa_propw5) o=gru_jepa_propw5_t19000_vf05; a="--method gru_jepa --jepa_variance_floor 0.05" ;;
    gru_propw5) o=gru_propw5_t19000; a="--method gru" ;;
    *) echo "unknown $r"; exit 2 ;;
  esac
  o=${o}_s$SEED; s=st_${r}_s$SEED
  [ -e logs/student/$o ] && { echo "exists logs/student/$o"; exit 2; }
  tmux new -d -s $s "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ~/miniconda3/envs/env_isaaclab/bin/python scripts/jepa_loco/train_student_depth.py $a $C --output_dir logs/student/$o > logs/student/$o.log 2>&1"
  echo "started $s -> logs/student/$o"
done
