#!/usr/bin/env bash
# JE-Loco depth 벤치마크 스윕 드라이버.
# Isaac Lab SimulationContext 는 프로세스당 싱글턴이라, 각 (num_envs, depth) 설정을
# **별도 프로세스**로 기동해서 측정한다. 결과는 logs/je_loco/benchmark_depth.json 에 누적.
#
# 사용:
#   bash scripts/je_loco/benchmark_depth_sweep.sh 256 1024 2048 4096
#   STEPS=300 bash scripts/je_loco/benchmark_depth_sweep.sh 1024 2048
set -u

cd "$(dirname "$0")/../.." || exit 1

ENV_COUNTS=("$@")
if [ ${#ENV_COUNTS[@]} -eq 0 ]; then
  ENV_COUNTS=(256 1024 2048 4096)
fi
STEPS="${STEPS:-200}"
WARMUP="${WARMUP:-20}"
OUT="logs/je_loco/benchmark_depth.json"

# 새 스윕은 기존 결과를 백업하고 새로 시작
if [ -f "$OUT" ]; then
  mv "$OUT" "${OUT%.json}_$(date +%Y%m%d_%H%M%S).json"
fi

echo "[sweep] env_counts=${ENV_COUNTS[*]} steps=$STEPS  → $OUT"
for n in "${ENV_COUNTS[@]}"; do
  for mode in off on; do
    if [ "$mode" = "off" ]; then
      DEPTH_FLAG="--no_depth"
    else
      DEPTH_FLAG="--enable_cameras"
    fi
    echo ""
    echo "======== num_envs=$n depth=$mode ========"
    python scripts/je_loco/benchmark_depth.py \
      --num_envs "$n" --steps "$STEPS" --warmup "$WARMUP" \
      --headless $DEPTH_FLAG --out "$OUT"
  done
done

echo ""
echo "==================== 스윕 요약 ===================="
python - "$OUT" <<'PY'
import json, sys
res = json.load(open(sys.argv[1]))["results"]
by = {}
for r in res:
    if "error" in r: continue
    by.setdefault(r["num_envs"], {})[r["depth"]] = r
print(f"{'num_envs':>9} | {'OFF env-steps/s':>16} | {'ON env-steps/s':>16} | {'slowdown':>9} | {'ON gpu(GB)':>10}")
for n in sorted(by):
    off = by[n].get(False); on = by[n].get(True)
    ov = off["env_steps_per_s"] if off else None
    nv = on["env_steps_per_s"] if on else None
    slow = f"{ov/nv:.2f}x" if (ov and nv) else "n/a"
    gpu = on["device_used_gb"] if on else "-"
    print(f"{n:>9} | {str(ov):>16} | {str(nv):>16} | {slow:>9} | {str(gpu):>10}")
PY
echo "=================================================="
