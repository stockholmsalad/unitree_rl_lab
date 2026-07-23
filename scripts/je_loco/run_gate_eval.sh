#!/bin/bash
# =============================================================================
# JE-Loco 게이트 1 + 게이트 2 평가 (한 seed 쌍에 대해 전체 결손 스윕 + figure)
#
#   게이트 1 = 스윕의 dropout 0% 지점(완전관측) → A·B 대등성
#   게이트 2 = 전체 곡선(0→100%) 3결손(dropout·hole·occlusion) → B 완만 저하(핵심 주장)
#   둘 다 아래 6회 스윕(3결손 × {recon,jepa})에서 동시에 나온다.
#
# 사용:
#   bash scripts/je_loco/run_gate_eval.sh <RECON_RUN> <JEPA_RUN> [CKPT]
# 예:
#   bash scripts/je_loco/run_gate_eval.sh 2026-07-11_17-59-32 2026-07-10_17-32-17 model_24000.pt
#
# 새 학습 seed 를 평가할 땐 run 디렉토리 두 개만 바꿔 그대로 재실행.
# 출력 CSV/그림은 run 디렉토리 tag 로 이름이 갈리므로 seed 끼리 덮어쓰지 않는다.
# =============================================================================
set -e
source /home/user/miniconda3/etc/profile.d/conda.sh && conda activate go2_isaaclab
cd /home/user/icros/unitree_rl_lab

RECON_RUN=${1:?RECON_RUN(예: 2026-07-11_17-59-32) 필요}
JEPA_RUN=${2:?JEPA_RUN(예: 2026-07-10_17-32-17) 필요}
CKPT=${3:-model_24000.pt}

# 평가 공통 설정 (A·B 공정 비교의 핵심)
ENVS=256            # 표본 수
STEPS=1500          # 레벨당 측정 스텝(에피소드 1000 완주 위해 >1000)
SEED=42             # A·B 동일 지형 배정(공정 비교) — 헤드 무관 같은 seed
LEVELS=0,0.2,0.4,0.6,0.8,1.0
CKTAG=$(basename $CKPT .pt)

echo "############ 게이트 1+2 평가 시작 ############"
echo "  recon=$RECON_RUN  jepa=$JEPA_RUN  ckpt=$CKPT"
echo "  envs=$ENVS steps=$STEPS seed=$SEED  결손=dropout,hole,occlusion"

for DEG in dropout hole occlusion; do
  for RUN in $RECON_RUN $JEPA_RUN; do
    echo "===== [$DEG] $RUN 스윕 ====="
    python -u scripts/je_loco/eval_pc.py --headless \
      --num_envs $ENVS --steps $STEPS --eval_seed $SEED \
      --degradation $DEG --dropout_levels $LEVELS \
      --load_run $RUN --checkpoint $CKPT
    # --fix_terrain 은 기본 on(eval 중 지형 커리큘럼 정지 → 모든 레벨 동일 지형)
  done
done

# ── figure ── (개별 3장 + 종합 2×3 그리드)
RTAG=${RECON_RUN}_${CKTAG}
JTAG=${JEPA_RUN}_${CKTAG}
echo "===== figure 생성 ====="
python scripts/je_loco/plot_degradation.py --recon dropout_curve_${RTAG}.csv    --jepa dropout_curve_${JTAG}.csv    --out degradation_curve_dropout_${RTAG}.png    --title "dropout · seed $SEED"
python scripts/je_loco/plot_degradation.py --recon hole_curve_${RTAG}.csv       --jepa hole_curve_${JTAG}.csv       --out degradation_curve_hole_${RTAG}.png       --title "hole · seed $SEED"
python scripts/je_loco/plot_degradation.py --recon occlusion_curve_${RTAG}.csv  --jepa occlusion_curve_${JTAG}.csv  --out degradation_curve_occlusion_${RTAG}.png  --title "occlusion · seed $SEED"
python scripts/je_loco/plot_degradation_grid.py --recon_tag $RTAG --jepa_tag $JTAG --out degradation_grid_${RTAG}.png

echo "############ 완료 — degradation_grid_${RTAG}.png (게이트2 대표) ############"
echo "  게이트1 = 각 CSV 의 첫 행(level 0). 게이트2 = 전체 곡선."
