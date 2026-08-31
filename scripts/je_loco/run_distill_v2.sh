#!/usr/bin/env bash
# =============================================================================
# 증류 v2 — 비교축을 "인코더 초기화"에서 "증류 중 보조 목적함수"로 바꾼 실험 (2026-08-31)
#
# 왜 바꾸는가 (v1 n=5 판정 결과):
#   v1 은 사전학습 인코더 초기화만 달랐고, 8000 iter 행동 감독이 그 차이를 씻어냈다.
#   무결손·공간결손·시간결손 세 게이트 전부에서 조건 간 격차 < 시드폭. 그런데 Phase 2
#   로그는 JEPA 가 예측을 **배웠음**을 보여준다(copy 대비 skill 0.584 @k=100). 배운 능력이
#   정책에 도달하지 못한 것이다 — 관측이 항상 신선하면 쓸 자리가 없으니까.
#
# v2 에서 바뀐 셋:
#   1. 예측기가 정책 안에 있다 — 입력 [z_e(o), z_p, ẑ_e(o+k)], 배포 시에도 살아 있음
#   2. 보조손실이 증류 내내 걸린다 (지속 개입, 씻겨나갈 수 없음)
#   3. 학습 중 관측 노후화 d ~ U[0,25] 주입 — 예측기에 실제 할 일을 준다
#   + 속도 0.6~1.5 m/s · 회전 ±0.6 (teacher 봉투 실측으로 검증된 범위)
#
# 통제: 세 조건 모두 predictor_in_policy=True → 구조·배포 파라미터 수 동일(498,427).
#       유일한 변수 = 그 예측기를 무엇이 학습시키느냐.
#         jepa   λ_jepa=1  자기지도(EMA 타깃)        → 실기 로그로 학습 가능
#         recon  λ_recon=1 특권(시뮬 GT 높이맵)      → 시뮬 전용
#         none   보조손실 없음                        → 행동손실만
#
# 사용:
#   MAX_PARALLEL=2 nohup bash scripts/je_loco/run_distill_v2.sh > ~/v2.log 2>&1 &
#   SEEDS="1 2 3" bash scripts/je_loco/run_distill_v2.sh
# =============================================================================
set -u
cd "$(dirname "$0")/../.."

ITER="${ITER:-2000}"          # num_steps_per_env=128 → 환경 스텝 예산은 v1(32×8000)과 동일
SEEDS="${SEEDS:-1 2 3 4 5}"
NUM_ENVS="${NUM_ENVS:-1024}"
TEACHER="${TEACHER:-logs/rsl_rl/unitree_go2_jeloco_teacher/2026-08-13_15-20-59_teacher_s2/model_19999.pt}"
MAX_PARALLEL="${MAX_PARALLEL:-2}"
LOGDIR=~/distill_logs
mkdir -p "$LOGDIR"

[ -f "$TEACHER" ] || { echo "★ teacher 체크포인트 없음: $TEACHER"; exit 1; }
echo "teacher = $TEACHER"
echo "seeds=[$SEEDS]  iter=$ITER  envs=$NUM_ENVS  병렬=$MAX_PARALLEL"

# $1=라벨  $2=보조손실 인자들  $3=seed
launch() {
  local name="V2_$1_s$3"
  if ls logs/rsl_rl/je_loco_distill/*_${name}/model_$((ITER-1)).pt >/dev/null 2>&1; then
    echo "  skip $name (완주 확인)"; return
  fi
  nohup python -u scripts/je_loco/train_pc.py \
    --task Unitree-Go2-JELoco-Distill --num_envs "$NUM_ENVS" --headless \
    --max_iterations "$ITER" --seed "$3" --run_name "$name" \
    --teacher_checkpoint "$TEACHER" \
    $2 \
    > "$LOGDIR/$name.log" 2>&1 &
  echo "  launched $name : $2 (pid $!)"
  sleep 40    # Isaac 동시 기동 메모리 스파이크 완화
}

wait_slot() {
  while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
    wait -n 2>/dev/null || sleep 10
  done
}

# repr_head 는 모델이 어떤 헤드를 **만들지**, lambda_* 는 러너가 어떤 손실을 **걸지** 정한다.
# 둘 다 맞춰야 한다 — repr_head 만 켜고 lambda 를 0 으로 두면 헤드가 학습되지 않는다.
JEPA_ARGS='agent.student.repr_head=jepa  agent.lambda_jepa=1.0  agent.lambda_recon=0.0'
RECON_ARGS='agent.student.repr_head=recon agent.lambda_jepa=0.0  agent.lambda_recon=1.0'
NONE_ARGS='agent.student.repr_head=none  agent.lambda_jepa=0.0  agent.lambda_recon=0.0'

for s in $SEEDS; do
  wait_slot; launch jepa  "$JEPA_ARGS"  "$s"
  wait_slot; launch recon "$RECON_ARGS" "$s"
  wait_slot; launch none  "$NONE_ARGS"  "$s"
done
wait
echo "전체 완료. 확인: grep jepa_skill $LOGDIR/V2_jepa_s1.log | tail"
