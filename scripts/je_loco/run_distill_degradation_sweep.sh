#!/bin/bash
# =============================================================================
# DAgger 증류 6런 × 3결손 depth 저하 스윕  (Phase 3b 실제 판정)
#
#   비교 축 = 인코더 초기화 (jepa / recon / scratch) × seed(s1,s2)
#   Loss/behavior 는 학습분포 모방정확도일 뿐 → 논문 질문은 "결손 하 강인성"인 이 곡선.
#   DeFM 도 in-distribution 무승부(90.14 vs 90.45), OOD 에서만 갈렸다(0.876 vs 0.486).
#
# 사용:
#   bash scripts/je_loco/run_distill_degradation_sweep.sh              # 기본(6런×공간3종)
#   DEGS="freeze latency lowfps" OUTDIR=results/distill_temporal bash ...   # 게이트 3(시간 결손)
#   ENVS=512 LEVELS=0,0.25,0.5,0.75,1.0 bash ... run_distill_degradation_sweep.sh
#   TERRAIN_LEVEL=5 OUTDIR=results/deg_hard bash ... run_distill_degradation_sweep.sh
#
# 사전조건: conda 환경 활성화 상태(env_isaaclab)에서 실행. 리포 루트에서 실행.
# =============================================================================
set -uo pipefail

TASK=${TASK:-Unitree-Go2-JELoco-Distill}
CKPT=${CKPT:-model_7999.pt}
ENVS=${ENVS:-256}            # 표본 수
STEPS=${STEPS:-1500}         # 레벨당 측정 스텝(에피소드 1000 완주 위해 >1000)
SEED=${SEED:-42}             # 모든 조건이 동일 지형을 밟게 — 공정 비교의 핵심
LEVELS=${LEVELS:-0,0.2,0.4,0.6,0.8,1.0}
DEGS=${DEGS:-"dropout hole occlusion"}   # 시간 결손: "freeze latency lowfps" (게이트 3)
TERRAIN_LEVEL=${TERRAIN_LEVEL:--1}   # -1=max_init 분포(=3), N=전 env 강제 레벨 N
OUTDIR=${OUTDIR:-results/distill_degradation}
LOGROOT=logs/rsl_rl/je_loco_distill

RUNS=(
  2026-08-18_10-11-39_D_jepa_s1
  2026-08-18_10-13-39_D_jepa_s2
  2026-08-18_10-12-19_D_recon_s1
  2026-08-18_10-14-19_D_recon_s2
  2026-08-18_10-12-59_D_scratch_s1
  2026-08-18_10-15-00_D_scratch_s2
)

mkdir -p "$OUTDIR"
TL_ARG=""; [ "$TERRAIN_LEVEL" -ge 0 ] && TL_ARG="--terrain_level $TERRAIN_LEVEL"

# ── 사전 점검: 체크포인트 6개가 다 있고 증류 키인지 ──
echo "############ 사전 점검 ############"
for R in "${RUNS[@]}"; do
  P="$LOGROOT/$R/$CKPT"
  [ -f "$P" ] || { echo "!! 없음: $P"; exit 1; }
done
python3 - "$LOGROOT/${RUNS[0]}/$CKPT" <<'PY'
import sys, torch
k = list(torch.load(sys.argv[1], map_location='cpu', weights_only=False).keys())
print("checkpoint keys:", k)
assert 'student_state_dict' in k, "증류 키 아님 — eval_pc.py 분기 확인 필요"
print("OK: 증류 체크포인트 (eval_pc.py 의 DistillationRunner 분기로 로드됨)")
PY
[ $? -ne 0 ] && exit 1

TOTAL=$(( ${#RUNS[@]} * $(echo $DEGS | wc -w) )); i=0
echo "############ 스윕 시작: $TOTAL 회 ############"
echo "  task=$TASK ckpt=$CKPT envs=$ENVS steps=$STEPS seed=$SEED levels=$LEVELS"
echo "  결손=$DEGS  terrain_level=$TERRAIN_LEVEL  → $OUTDIR"

FAILED=()
for DEG in $DEGS; do
  for R in "${RUNS[@]}"; do
    i=$((i+1))
    echo; echo "===== [$i/$TOTAL] $DEG · $R ====="
    python -u scripts/je_loco/eval_pc.py --headless \
      --task "$TASK" --num_envs "$ENVS" --steps "$STEPS" --eval_seed "$SEED" \
      --degradation "$DEG" --dropout_levels "$LEVELS" \
      --load_run "$R" --checkpoint "$LOGROOT/$R/$CKPT" $TL_ARG \
      2>&1 | tee "$OUTDIR/log_${DEG}_${R}.txt"
    # eval_pc.py 는 CSV 를 CWD 에 <deg>_curve_<run>_<ckpt>.csv 로 떨어뜨린다 → 수거
    CSV="${DEG}_curve_${R}_${CKPT%.pt}.csv"
    if [ -f "$CSV" ]; then mv "$CSV" "$OUTDIR/"; echo "[saved] $OUTDIR/$CSV"
    else echo "!! CSV 없음: $CSV"; FAILED+=("$DEG/$R"); fi
  done
done

echo; echo "############ 완료 ############"
echo "CSV $(ls "$OUTDIR"/*.csv 2>/dev/null | wc -l) / $TOTAL 개  → $OUTDIR"
[ ${#FAILED[@]} -gt 0 ] && { echo "실패: ${FAILED[*]}"; exit 1; }
echo "다음: 조건×시드 곡선 비교 (level 0 = 대등성 확인, 전체 = 강인성 판정)"
