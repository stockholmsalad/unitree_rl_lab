#!/usr/bin/env bash
# =============================================================================
# blind 대조군 — "시각 입력이 도대체 얼마나 기여하는가" (2026-08-31)
#
# 왜 필요한가 (n=5 판정에서 드러난 모순):
#   · freeze 1.0 (카메라 영구 정지) @ terrain_level 5  →  성공률 91%
#   · occlusion 0.4 (발밑 시야 40% 차단) @ 같은 지형    →  성공률 32~51%
#   둘 다 참이려면 occlusion 은 "공간정보 상실"이 아니라 **분포 밖 입력**을 재고 있고,
#   정책은 신선한 시각에 거의 의존하지 않는다는 뜻이다.
#
# blind 는 마스킹을 일절 안 쓰고(valid=1 유지) 좌표만 지형 무관 기준 점군으로 바꾼다.
#   → 분포 안 · 지형 상호정보량 0 = 순수 "눈 감은" 조건.
#
# 판정:
#   blind 1.0 ≈ 무결손   →  인코더가 과제에 거의 기여 안 함. 비교축(인코더 초기화) 검정력 없음.
#                           (Phase 3 무효 · 사전학습 우위 증발 · 게이트 3 무반응을 한 번에 설명)
#   blind 1.0 ≪ 무결손   →  시각은 중요. 게이트 3 무효는 시간 결손 설계 문제로 좁혀진다.
#
# 어려운 지형을 먼저 돈다 — 시각이 가장 필요한 조건이라 판정력이 거기 있다.
#
# 사용:
#   tmux new -d -s blind "conda run -n env_isaaclab bash scripts/je_loco/run_blind_control.sh"
#   ssh <pilab> 'cat ~/jeloco_blind_status.txt'
# =============================================================================
set -u
cd "$(dirname "$0")/../.."

STATUS="${STATUS:-$HOME/jeloco_blind_status.txt}"
ITER="${ITER:-8000}"
LEVELS="${LEVELS:-0,0.25,0.5,0.75,1.0}"
EVAL_ENVS="${EVAL_ENVS:-256}"
EVAL_STEPS="${EVAL_STEPS:-1500}"
HARD_LEVEL="${HARD_LEVEL:-5}"
LOGROOT=logs/rsl_rl/je_loco_distill
LAST=$(( ITER - 1 ))

say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; echo "[$(date '+%m-%d %H:%M:%S')] $*" >> "$STATUS"; }

: > "$STATUS"
mapfile -t RUNS < <(for d in $LOGROOT/*/; do [ -f "$d/model_${LAST}.pt" ] && basename "$d"; done)
say "blind 대조군 시작 (PID $$) — ${#RUNS[@]}런 × 2지형, levels=$LEVELS"
[ "${#RUNS[@]}" -eq 0 ] && { say "★ 중단: model_${LAST}.pt 를 가진 런이 없다"; exit 1; }

run_one_terrain() {
  local out="$1" extra="$2"
  mkdir -p "$out"
  local OK=0 NG=0
  for R in "${RUNS[@]}"; do
    local CSV="blind_curve_${R}_model_${LAST}.csv"
    if [ -f "$out/$CSV" ]; then OK=$((OK+1)); continue; fi     # 재실행 시 이어하기
    python -u scripts/je_loco/eval_pc.py --headless \
      --task Unitree-Go2-JELoco-Distill --num_envs "$EVAL_ENVS" --steps "$EVAL_STEPS" \
      --eval_seed 42 --degradation blind --dropout_levels "$LEVELS" \
      --load_run "$R" --checkpoint "$LOGROOT/$R/model_${LAST}.pt" $extra \
      > "$out/log_blind_${R}.txt" 2>&1
    if [ -f "$CSV" ]; then mv "$CSV" "$out/"; OK=$((OK+1)); say "  ok  $R ($OK/${#RUNS[@]})"
    else NG=$((NG+1)); say "  FAIL $R — $out/log_blind_${R}.txt"; fi
  done
  say "  종료: 성공 $OK · 실패 $NG → $out"
}

say "───── 어려운 지형 (--terrain_level $HARD_LEVEL) — 판정력이 여기 있다"
run_one_terrain results/full_matrix_hard "--terrain_level $HARD_LEVEL"

say "───── 기본 지형 (비교용)"
run_one_terrain results/full_matrix ""

say "완료. 판정: python scripts/je_loco/judge_gates.py results/full_matrix results/full_matrix_hard"
