#!/usr/bin/env bash
# JE-Loco 마스킹 재학습 매트릭스 — A(recon)·B(jepa) × seed 를 백그라운드로 한 번에 실행.
#
# 배경: 긴 train 명령을 tmux 창마다 붙여넣으면 잘림/오타로 override(--seed, repr_head)가
# 누락되는 사고가 반복됨(2026-07-31: 4개가 전부 recon+seed42 로 실행됨). 이 스크립트가
# 전체 명령을 원자적으로 담아 그 사고를 원천 차단한다. nohup 이라 로그아웃해도 계속 돈다.
#
# 사용법:
#   bash scripts/je_loco/run_mask_matrix.sh            # 기본: A/B × seed 1,2 (4런), 30000 iter
#   bash scripts/je_loco/run_mask_matrix.sh 30000 "1 2 3"   # A/B × seed 1,2,3 (6런)
#
# 로그: ~/mask_logs/<name>.log  ·  체크포인트: logs/rsl_rl/je_loco_pc/<타임스탬프>_<name>/
# 중단: pkill -f train_pc.py   ·  상태: nvidia-smi, ls ~/mask_logs/

set -u
cd "$(dirname "$0")/../.."

ITER="${1:-30000}"
SEEDS="${2:-1 2}"
mkdir -p ~/mask_logs

launch() {  # $1=라벨(A/B) $2=repr_head $3=seed
  local name="mask$1_s$3"
  nohup python -u scripts/je_loco/train_pc.py \
    --task Unitree-Go2-JELoco-PC --num_envs 1024 --headless \
    --max_iterations "$ITER" --seed "$3" --run_name "$name" \
    agent.actor.repr_head="$2" > ~/mask_logs/"$name".log 2>&1 &
  echo "  launched $name : repr_head=$2 seed=$3 (pid $!)"
  sleep 40   # Isaac 기동 겹침 완화
}

echo "=== 마스킹 매트릭스 시작 (iter=$ITER, seeds=$SEEDS) ==="
for s in $SEEDS; do launch A recon "$s"; done
for s in $SEEDS; do launch B jepa  "$s"; done

echo ""
echo "전부 백그라운드 실행됨. 로그: ~/mask_logs/"
echo "확인:  grep -h 'JELoco' ~/mask_logs/*.log      # 각 run 의 헤드 (A=recon, B=jepa) 확인"
echo "상태:  nvidia-smi | grep python                # 프로세스 살아있나"
echo "중단:  pkill -f train_pc.py"
