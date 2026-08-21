#!/usr/bin/env bash
# =============================================================================
# 무인 실행 큐 — 자리를 비우는 동안 순차 실행 (2026-08-20)
#
#   Stage A. 시드 증설  : 3조건 × seed {3,4,5} = 9런 학습 (~2~3일)
#   Stage B. 전체 재평가: 15런(기존6+신규9) × {occlusion, freeze, latency, lowfps} (~3~5시간)
#
# 설계 원칙 — 무인이므로 "한 곳이 죽어도 나머지는 간다":
#   · 각 스테이지는 독립. 앞 스테이지가 부분 실패해도 다음으로 넘어간다.
#   · 개별 런 실패는 기록만 하고 큐를 멈추지 않는다.
#   · 상태를 STATUS 파일에 계속 갱신 → 학회장에서 `ssh <pilab> cat <STATUS>` 한 줄로 확인.
#
# 사용:
#   nohup bash scripts/je_loco/run_unattended_queue.sh > ~/queue.log 2>&1 &
#   MAX_PARALLEL=4 nohup bash ... &        # GPU 를 남과 나눠 쓸 때 보수적으로
#   STAGES="B" nohup bash ... &            # 특정 스테이지만
#
# 원격 확인:
#   ssh <pilab> 'cat ~/jeloco_queue_status.txt'
#   ssh <pilab> 'tail -5 ~/queue.log'
# =============================================================================
set -u
cd "$(dirname "$0")/../.."
REPO=$(pwd)

STATUS="${STATUS:-$HOME/jeloco_queue_status.txt}"
STAGES="${STAGES:-A B}"
ITER="${ITER:-8000}"
NEW_SEEDS="${NEW_SEEDS:-3 4 5}"
NUM_ENVS="${NUM_ENVS:-1024}"
EVAL_DEGS="${EVAL_DEGS:-occlusion freeze latency lowfps}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_STEPS="${EVAL_STEPS:-1500}"
LEVELS="${LEVELS:-0,0.2,0.4,0.6,0.8,1.0}"
OUTDIR="${OUTDIR:-results/full_matrix}"
LOGROOT=logs/rsl_rl/je_loco_distill

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$STATUS"; }

: > "$STATUS"
say "큐 시작 (PID $$)  stages=[$STAGES]"

# ── 사전 점검: 무인 실행에서 가장 흔한 사망 원인부터 ────────────────────────
FREE_GB=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc '0-9')
say "디스크 여유 ${FREE_GB}GB (9런 체크포인트 ≈ 3GB 필요)"
if [ "$FREE_GB" -lt 15 ]; then
  say "★ 중단: 디스크 부족(${FREE_GB}GB). 무인 실행 중 디스크가 차면 전부 잃는다."
  exit 1
fi
nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader | while read -r l; do say "GPU: $l"; done

# ═══ Stage A — 시드 증설 ════════════════════════════════════════════════════
if [[ " $STAGES " == *" A "* ]]; then
  LAST=$(( ITER - 1 ))
  # 재실행 안전성: 이미 3조건 모두 완주한 시드는 건너뛴다. 원격에서 큐가 죽어
  # 다시 띄울 때 끝난 학습을 처음부터 반복하지 않게 하는 장치.
  TODO=""
  for s in $NEW_SEEDS; do
    n=0
    for c in jepa recon scratch; do
      ls $LOGROOT/*_D_${c}_s${s}/model_${LAST}.pt >/dev/null 2>&1 && n=$((n+1))
    done
    if [ "$n" -eq 3 ]; then say "skip seed $s (3조건 완주 확인)"; else TODO="$TODO $s"; fi
  done
  TODO=$(echo $TODO | xargs)
  if [ -z "$TODO" ]; then
    say "Stage A: 할 일 없음 — 모든 시드 완주됨"; STAGES="${STAGES/A/}"
  fi
fi

if [[ " $STAGES " == *" A "* ]]; then
  say "───── Stage A 시작: 시드 {$TODO} × 3조건 = $(( $(echo $TODO | wc -w) * 3 ))런"
  NEW_SEEDS="$TODO"
  BEFORE=$(ls -d $LOGROOT/*/ 2>/dev/null | wc -l)
  # run_distill_matrix.sh 가 GPU 메모리 기반으로 슬롯을 계산해 큐잉한다(OOM 가드).
  bash scripts/je_loco/run_distill_matrix.sh "$ITER" "$NEW_SEEDS" 2>&1 | tail -40
  AFTER=$(ls -d $LOGROOT/*/ 2>/dev/null | wc -l)
  say "Stage A 종료: 런 디렉터리 $BEFORE → $AFTER"

  # 완주 검증 — max_iterations-1 체크포인트 존재 여부
  LAST=$(( ITER - 1 ))
  DONE_N=0; FAIL=""
  for d in $LOGROOT/*/; do
    [ -f "$d/model_${LAST}.pt" ] && DONE_N=$((DONE_N+1)) || FAIL="$FAIL $(basename $d)"
  done
  say "완주 $DONE_N 런 (model_${LAST}.pt 보유)"
  [ -n "$FAIL" ] && say "미완주:$FAIL"
fi

# ═══ Stage B — 전체 재평가 ══════════════════════════════════════════════════
if [[ " $STAGES " == *" B "* ]]; then
  LAST=$(( ITER - 1 ))
  mapfile -t RUNS < <(for d in $LOGROOT/*/; do
      [ -f "$d/model_${LAST}.pt" ] && basename "$d"; done)
  say "───── Stage B 시작: ${#RUNS[@]}런 × $(echo $EVAL_DEGS | wc -w)결손 = $(( ${#RUNS[@]} * $(echo $EVAL_DEGS | wc -w) ))회"
  mkdir -p "$OUTDIR"
  OK=0; NG=0
  for DEG in $EVAL_DEGS; do
    for R in "${RUNS[@]}"; do
      CSV="${DEG}_curve_${R}_model_${LAST}.csv"
      if [ -f "$OUTDIR/$CSV" ]; then say "skip(이미 있음) $DEG/$R"; OK=$((OK+1)); continue; fi
      python -u scripts/je_loco/eval_pc.py --headless \
        --task Unitree-Go2-JELoco-Distill --num_envs "$EVAL_ENVS" --steps "$EVAL_STEPS" \
        --eval_seed 42 --degradation "$DEG" --dropout_levels "$LEVELS" \
        --load_run "$R" --checkpoint "$LOGROOT/$R/model_${LAST}.pt" \
        > "$OUTDIR/log_${DEG}_${R}.txt" 2>&1
      if [ -f "$CSV" ]; then mv "$CSV" "$OUTDIR/"; OK=$((OK+1)); say "ok  $DEG/$R  ($OK)"
      else NG=$((NG+1)); say "FAIL $DEG/$R — $OUTDIR/log_${DEG}_${R}.txt 확인"; fi
    done
  done
  say "Stage B 종료: 성공 $OK · 실패 $NG · CSV $(ls "$OUTDIR"/*.csv 2>/dev/null | wc -l)개"
fi

say "큐 완료. 결과: $OUTDIR"
say "돌아와서 할 일: CSV 를 로컬로 rsync → plot_distill_degradation.py 로 판정"
