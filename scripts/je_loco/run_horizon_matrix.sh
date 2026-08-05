#!/usr/bin/env bash
# JE-Loco JEPA 지평(horizon) 스윕 — ①. 예측 지평 k 를 늘리면 예측이 살아나나(skill>0)?
# ④done마스킹 + ②skill score + residual 타깃 위에서, k 만 바꿈(한 번에 한 변수).
#
# 진단: k=5(0.1초)=near-identity → copy 가 이미 정답 → skill≤0(예측 불가능). k 를 늘리면
# copy 가 나빠져 예측 여지 생김. **판정 지표 = Loss/jepa_skill (>0 이면 copy 보다 나음=진짜 예측).**
# T=num_steps_per_env=32 제약: k=27 이면 valid_t=5(빈약하나 ×1024env). k>27 은 T 확대 필요.
#
# 사용법:  bash scripts/je_loco/run_horizon_matrix.sh            # k∈{5,15,27} × seed 1,2 (6런), 30000 iter
#          bash scripts/je_loco/run_horizon_matrix.sh 30000 "1"  # seed 1 만(3런, 빠른 확인)
# 로그: ~/mask_logs/Bhz*.log   중단: pkill -f train_pc.py
# 확인:  grep -h 'k=' ~/mask_logs/Bhz*.log | grep JELoco   (k·residual 확인)

set -u
cd "$(dirname "$0")/../.."
ITER="${1:-30000}"
SEEDS="${2:-1 2}"
KS="5 15 27"
mkdir -p ~/mask_logs

launch() {  # $1=k $2=seed
  local name="Bhz_k$1_s$2"
  nohup python -u scripts/je_loco/train_pc.py \
    --task Unitree-Go2-JELoco-PC --num_envs 1024 --headless \
    --max_iterations "$ITER" --seed "$2" --run_name "$name" \
    agent.actor.repr_head=jepa agent.jepa_k=$1 agent.jepa_residual=true \
    > ~/mask_logs/"$name".log 2>&1 &
  echo "  launched $name : k=$1 seed=$2 residual=true (pid $!)"
  sleep 40
}

echo "=== JEPA horizon 스윕 (k=$KS, seeds=$SEEDS, iter=$ITER, residual=true) ==="
for k in $KS; do for s in $SEEDS; do launch "$k" "$s"; done; done

echo ""
echo "판정: Loss/jepa_skill 이 k 클수록 0 을 넘어가나(>0=예측 성공). k=5 는 ≤0(copy 못이김) 예상."
echo "확인:  grep -h 'k=' ~/mask_logs/Bhz*.log | grep JELoco"
echo "중단:  pkill -f train_pc.py"
