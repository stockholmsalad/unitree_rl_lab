#!/usr/bin/env bash
# JE-Loco Phase 3b — DAgger 증류 매트릭스. 인코더 초기화 3종 × seed 를 한 번에 실행.
#
# 배경(2026-08-18 전환): Phase 3(인코더 동결 + PPO 재학습)은 세 조건 **모두 제대로 못 걷는**
# 결과로 끝났다(S2 iter 4000: 지표·육안 무차별, teacher 는 잘 걸음). LITERATURE.md 축 3 이
# 예측한 실패 — 행동 감독과 on-policy 보정을 둘 다 뺐기 때문. 표준 레시피 = DAgger.
# 연구질문은 유지: jepa/recon/scratch 를 **인코더 초기화**로 비교(= DeFM 2026 의 설계).
#
# ★ GPU 메모리 주의: 1024 env 한 런이 **약 11.2 GB** 쓴다(2026-08-18 실측, Z790 RTX 5070 Ti).
#   96GB(pilab)면 6런 병렬 OK. 16GB 급이면 1런씩만 들어간다 — 이 스크립트가 자동으로 슬롯을
#   계산해 큐잉하므로, 작은 GPU 에서도 죽지 않고 순차 실행된다.
#   (2026-08-18: 이 가드가 없어 Z790 에서 6런 동시 실행 → 5런 CUDA OOM 사망.)
#
# 사용법:
#   bash scripts/je_loco/run_distill_matrix.sh                      # 기본 8000 iter, seed 1 2
#   bash scripts/je_loco/run_distill_matrix.sh 8000 "1 2"
#   NUM_ENVS=512 bash scripts/je_loco/run_distill_matrix.sh         # 작은 GPU 에서 환경 수 축소
#   MAX_PARALLEL=1 bash scripts/je_loco/run_distill_matrix.sh       # 강제 순차
#   TEACHER=logs/.../model_19999.pt bash scripts/je_loco/run_distill_matrix.sh
#
# 로그: ~/distill_logs/<name>.log  ·  체크포인트: logs/rsl_rl/je_loco_distill/<타임스탬프>_<name>/
# 중단: pkill -f train_pc.py   ·  상태: nvidia-smi, ls ~/distill_logs/

set -u
cd "$(dirname "$0")/../.."

ITER="${1:-8000}"
SEEDS="${2:-1 2}"
NUM_ENVS="${NUM_ENVS:-1024}"
# Phase 1 teacher (2026-08-13_15-20-59_teacher_s2, 19999 iter, G2 육안 통과)
TEACHER="${TEACHER:-logs/rsl_rl/unitree_go2_jeloco_teacher/2026-08-13_15-20-59_teacher_s2/model_19999.pt}"
mkdir -p ~/distill_logs

if [ ! -f "$TEACHER" ]; then
  echo "★ teacher 체크포인트 없음: $TEACHER"
  echo "  TEACHER=<경로> 로 지정하거나 파일을 확인하세요."
  exit 1
fi

# ── 병렬 슬롯 계산 ────────────────────────────────────────────────────────
# 런당 사용량 ≈ 3000 MiB(Isaac/드라이버 고정분) + 8 MiB × env  (1024 env → ~11.2 GB, 실측 일치)
PER_RUN=$(( 3000 + NUM_ENVS * 8 ))
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
AUTO=$(( FREE / PER_RUN ))
[ "$AUTO" -lt 1 ] && AUTO=1
MAX_PARALLEL="${MAX_PARALLEL:-$AUTO}"

echo "teacher      = $TEACHER"
echo "iter         = $ITER   seeds = $SEEDS   num_envs = $NUM_ENVS"
echo "GPU 여유     = ${FREE} MiB   런당 예상 = ${PER_RUN} MiB"
echo "병렬 슬롯    = $MAX_PARALLEL  (초과분은 큐잉되어 순차 실행)"
if [ "$AUTO" -lt 1 ]; then
  echo "★ 경고: 여유 메모리가 런 하나에도 부족할 수 있습니다. NUM_ENVS 를 줄이세요."
fi
echo

launch() {  # $1=라벨 $2=인코더경로("" = scratch) $3=seed
  local name="D_$1_s$3"
  nohup python -u scripts/je_loco/train_pc.py \
    --task Unitree-Go2-JELoco-Distill --num_envs "$NUM_ENVS" --headless \
    --max_iterations "$ITER" --seed "$3" --run_name "$name" \
    --teacher_checkpoint "$TEACHER" \
    agent.student.pretrained_encoder="$2" \
    > ~/distill_logs/"$name".log 2>&1 &
  echo "  launched $name : encoder='${2:-scratch}' seed=$3 (pid $!)"
  sleep 40   # Isaac 기동 겹침 완화(동시 기동 시 메모리 스파이크 방지)
}

# 슬롯이 빌 때까지 대기 — 이게 OOM 방지의 핵심.
wait_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n 2>/dev/null || sleep 10
  done
}

for s in $SEEDS; do
  wait_slot; launch jepa    "pretrained/jepa_v1/pc_encoder.pt"  "$s"
  wait_slot; launch recon   "pretrained/recon_v1/pc_encoder.pt" "$s"
  wait_slot; launch scratch ""                                  "$s"
done

echo
echo "전체 투입됨(큐 포함). 확인: tail -f ~/distill_logs/D_jepa_s1.log"
echo "판정 지표: Loss/behavior (teacher 행동 모방 오차) — 낮을수록 teacher 를 잘 따라함."
echo "※ MAX_PARALLEL < 6 이면 이 셸이 끝까지 살아 있어야 큐가 진행됩니다 — tmux 안에서 실행하세요."
wait
echo "ALL DONE"
