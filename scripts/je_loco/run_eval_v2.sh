#!/bin/bash
# =============================================================================
# v2 평가 스윕 — PREREGISTRATION_V2.md 의 게이트 1~4 를 한 번에 채운다 (2026-09-03)
#
# 게이트별로 무엇을 쓰는지:
#   게이트 1  무결손 대등성   ← 모든 곡선의 level 0 행 (별도 실행 불필요)
#   게이트 2  공간 결손       ← occlusion 곡선
#   게이트 3  시간 결손 ★핵심 ← freeze / latency / lowfps 곡선
#                              latency 는 --latency_max_steps 50 (학습 d<=25 의 2배)
#                              사전등록 §4: "학습 범위 밖에서 판정한다"
#   게이트 4  예측기 기여도   ← --ablate_predictor 로 ẑ 블록만 0 치환, level 0 성공률
#                              출력을 abl_curve_*.csv 로 개명해 dropout 과 섞이지 않게 한다
#   (blind)   시각 기여분     ← 판정 아님. 게이트 3 해석의 맥락
#
# 중지 규칙 (2026-09-03 결정, 게이트 판정 전):
#   분석 대상은 09-05 12:00 까지 완주한 시드. 조건 간 시드 수가 다르면 세 조건이 모두
#   갖춰진 최대 시드까지만 쓴다. 이 스크립트가 그 규칙을 자동 집행한다.
#
# 사용:
#   PARALLEL=3 nohup bash scripts/je_loco/run_eval_v2.sh > ~/eval_v2.log 2>&1 &
#   DEGS=occlusion bash scripts/je_loco/run_eval_v2.sh      # 일부만
#   학습이 도는 중에는 PARALLEL=1 로도 걸지 마라 — CPU 경합이 학습을 2.6배 늦춘다(실측).
#
# 재실행 안전: 이미 있는 CSV 는 건너뛴다. 중단 후 같은 명령으로 이어받으면 된다.
# =============================================================================
set -uo pipefail
cd "$(dirname "$0")/../.."

TASK="${TASK:-Unitree-Go2-JELoco-Distill}"
CKPT="${CKPT:-model_1999.pt}"
ENVS="${ENVS:-256}"
STEPS="${STEPS:-1500}"
SEED="${SEED:-42}"                 # 모든 조건이 동일 지형을 밟게 — 공정 비교의 핵심
LEVELS="${LEVELS:-0,0.2,0.4,0.6,0.8,1.0}"
LATENCY_MAX="${LATENCY_MAX:-50}"   # 사전등록 v2 §4 — 학습 노후화 상한 25 의 2배
HARD_LEVEL="${HARD_LEVEL:-5}"
DEGS="${DEGS:-occlusion freeze latency lowfps blind}"
ABLATE="${ABLATE:-1}"              # 게이트 4 수행 여부
PARALLEL="${PARALLEL:-1}"
LOGROOT=logs/rsl_rl/je_loco_distill

# ── 중지 규칙 집행: 세 조건이 모두 완주한 최대 시드까지 ──
declare -A DIR
MAXSEED=0
for s in 1 2 3 4 5; do
  ok=1
  for c in jepa recon none; do
    d=$(ls -d "$LOGROOT"/*_V2_${c}_s${s} 2>/dev/null | head -1)
    if [ -n "$d" ] && [ -f "$d/$CKPT" ]; then DIR[${c}_$s]=$(basename "$d"); else ok=0; fi
  done
  [ "$ok" -eq 1 ] && MAXSEED=$s || break
done
[ "$MAXSEED" -eq 0 ] && { echo "!! 세 조건이 모두 완주한 시드가 없다. 학습 진행 확인."; exit 1; }

RUNS=()
for c in jepa recon none; do for s in $(seq 1 $MAXSEED); do RUNS+=("${DIR[${c}_$s]}"); done; done

echo "############ v2 평가 스윕 ############"
echo "  중지 규칙 적용 → n=$MAXSEED (세 조건 × seed 1~$MAXSEED = ${#RUNS[@]} 런)"
echo "  결손=$DEGS   게이트4 절제=$ABLATE   병렬=$PARALLEL"
echo "  envs=$ENVS steps=$STEPS eval_seed=$SEED levels=$LEVELS latency_max=$LATENCY_MAX"
for s in $(seq 1 $MAXSEED); do printf "    seed %d: %s | %s | %s\n" "$s" "${DIR[jepa_$s]}" "${DIR[recon_$s]}" "${DIR[none_$s]}"; done

# ── 사전 점검: 증류 체크포인트인지 (v1 에서 로드 분기 문제로 한 번 태운 적 있음) ──
python3 - "$LOGROOT/${RUNS[0]}/$CKPT" <<'PY' || exit 1
import sys, torch
k = list(torch.load(sys.argv[1], map_location='cpu', weights_only=False).keys())
assert 'student_state_dict' in k, f"증류 키 아님: {k}"
print(f"  사전 점검 OK — 증류 체크포인트 ({len(k)} 키)")
PY

FAILDB=$(mktemp)   # 실패 집계. one() 은 & 로 서브셸에서 도니 배열은 부모에 안 남는다
# $1=OUTDIR  $2=지형인자  $3=deg  $4=run  $5=추가인자  $6=결과 CSV 접두어
one() {
  local out=$1 tl=$2 deg=$3 run=$4 extra=$5 prefix=$6
  local csv="${prefix}_curve_${run}_${CKPT%.pt}.csv"
  [ -f "$out/$csv" ] && { echo "  skip $csv"; return 0; }
  local src="${deg}_curve_${run}_${CKPT%.pt}.csv"
  # CWD 에 남은 이전 CSV(스모크 테스트 잔여물 등)를 먼저 지운다. 안 지우면 이번 eval 이
  # 죽었을 때 낡은 파일이 결과로 수거돼 조용히 오염된다.
  rm -f "$src"
  local lat=""; [ "$deg" = latency ] && lat="--latency_max_steps $LATENCY_MAX"
  python -u scripts/je_loco/eval_pc.py --headless \
    --task "$TASK" --num_envs "$ENVS" --steps "$STEPS" --eval_seed "$SEED" \
    --degradation "$deg" --dropout_levels "${7:-$LEVELS}" \
    --load_run "$run" --checkpoint "$LOGROOT/$run/$CKPT" \
    $tl $lat $extra > "$out/log_${prefix}_${run}.txt" 2>&1
  # eval_pc.py 는 CWD 에 <deg>_curve_<run>_<ckpt>.csv 를 떨군다 → 수거(+게이트4는 개명)
  if [ -f "$src" ]; then mv "$src" "$out/$csv"; echo "  [saved] $out/$csv"
  else echo "  !! CSV 없음: $src  (로그: $out/log_${prefix}_${run}.txt)"; echo "$prefix/$run" >> "$FAILDB"; fi
}

slot() { while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL" ]; do wait -n 2>/dev/null || sleep 5; done; }

# ── 지형 2종 × (결손 스윕 + 게이트4 절제) ──
for terrain in easy hard; do
  if [ "$terrain" = hard ]; then OUT=results/v2_matrix_hard; TL="--terrain_level $HARD_LEVEL"
  else                          OUT=results/v2_matrix;      TL=""; fi
  mkdir -p "$OUT"
  echo; echo "===== 지형: $terrain → $OUT ====="

  for deg in $DEGS; do
    for r in "${RUNS[@]}"; do slot; one "$OUT" "$TL" "$deg" "$r" "" "$deg" & done
  done
  wait

  # 게이트 4: ẑ 절제. level 0 하나만 재면 되므로 6분의 1 비용.
  if [ "$ABLATE" = 1 ]; then
    echo "----- 게이트 4 · 예측기 절제 ($terrain) -----"
    for r in "${RUNS[@]}"; do slot; one "$OUT" "$TL" dropout "$r" "--ablate_predictor" abl "0.0" & done
    wait
  fi
done

echo; echo "############ 완료 ############"
echo "  v2_matrix      CSV $(ls results/v2_matrix/*.csv      2>/dev/null | wc -l) 개"
echo "  v2_matrix_hard CSV $(ls results/v2_matrix_hard/*.csv 2>/dev/null | wc -l) 개"
NF=$(wc -l < "$FAILDB")
if [ "$NF" -gt 0 ]; then echo "  실패 $NF 건:"; sed "s/^/    /" "$FAILDB"; fi
rm -f "$FAILDB"
echo
echo "판정: python scripts/je_loco/judge_gates.py --v2 results/v2_matrix results/v2_matrix_hard"
