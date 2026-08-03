#!/usr/bin/env bash
# JE-Loco JEPA conditioning ablation — predictor 에 command/action 조건을 넣어 예측이 살아나는지.
# 기존 maskB(조건 없음)와 비교하려면 동일 조건: 마스킹, seed 1·2, jepa, k=5.
#
# 조건 4종(한 번에 한 변수):
#   B-none : 조건 없음 (기존 maskB 와 동일 = 기준선, 재현용)
#   B-cmd  : 명령 c_t(3)             agent.actor.jepa_cond_command=true
#   B-act  : 지평 평균 행동(12)        agent.actor.jepa_cond_action=true
#   B-both : 둘 다
#
# 사용법:  bash scripts/je_loco/run_cond_matrix.sh            # 기본 30000 iter, seed 1,2
#          bash scripts/je_loco/run_cond_matrix.sh 30000 "1 2"
# 로그: ~/mask_logs/<name>.log   중단: pkill -f train_pc.py

set -u
cd "$(dirname "$0")/../.."
ITER="${1:-30000}"
SEEDS="${2:-1 2}"
mkdir -p ~/mask_logs

launch() {  # $1=라벨 $2=override(공백가능) $3=seed
  local name="Bcond$1_s$3"
  nohup python -u scripts/je_loco/train_pc.py \
    --task Unitree-Go2-JELoco-PC --num_envs 1024 --headless \
    --max_iterations "$ITER" --seed "$3" --run_name "$name" \
    agent.actor.repr_head=jepa $2 > ~/mask_logs/"$name".log 2>&1 &
  echo "  launched $name : $2 (pid $!)"
  sleep 40
}

echo "=== JEPA conditioning ablation (iter=$ITER, seeds=$SEEDS) ==="
for s in $SEEDS; do launch cmd  "agent.actor.jepa_cond_command=true" "$s"; done
for s in $SEEDS; do launch act  "agent.actor.jepa_cond_action=true"  "$s"; done
for s in $SEEDS; do launch both "agent.actor.jepa_cond_command=true agent.actor.jepa_cond_action=true" "$s"; done

echo ""
echo "총 $(echo $SEEDS | wc -w)×3 개 실행됨 (기준선 B-none 은 기존 maskB_s* 재사용)."
echo "확인:  grep -h 'cond_' ~/mask_logs/Bcond*.log | grep JELoco"
echo "중단:  pkill -f train_pc.py"
