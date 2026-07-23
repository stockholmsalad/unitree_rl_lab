# 단계 1 — D435i depth 렌더링 처리량 벤치마크 (게이트: depth CNN 확정 여부)

**측정일**: 2026-07-07 · **장비**: RTX 5070 Ti (16GB), i7-14700K · **Isaac Lab**: 0.54.4
**설정**: Go2 rough-generator 지형 + D435i **TiledCamera** depth(48×64, `distance_to_image_plane`),
렌더 주기 = decimation(4) → 정책 스텝당 depth 1프레임. headless. steps=150, warmup=20.
재현: `bash scripts/je_loco/benchmark_depth_sweep.sh 256 512 1024 2048 4096`

## 결과

| num_envs | depth OFF (env-steps/s) | depth ON (env-steps/s) | slowdown | ON GPU |
|---------:|------------------------:|-----------------------:|:--------:|-------:|
|      256 |                  10,358 |                  4,664 |  2.22×   | 7.4 GB |
|      512 |                  15,745 |                  6,789 |  2.32×   | 7.8 GB |
|     1024 |                  18,991 |                  9,593 |  1.98×   | 8.5 GB |
|     2048 |                  27,232 |                 13,502 |  2.02×   | 10.2 GB |
|     4096 |                  32,289 |                 16,744 |  1.93×   | 13.2 GB |

(env-steps/s = 병렬환경 × 정책스텝/s. GPU = 디바이스 전체 사용량 `mem_get_info`, RTX 렌더러 포함.)

## 결론 (Stage-1 게이트)

1. **image-like depth(TiledCamera)는 학습 규모에서 실행 가능** — 4096 병렬환경 + depth ON 이
   16GB 카드에 **13.2GB 로 들어간다**(OOM 아님). 처리량은 depth OFF 대비 **일관되게 ~2× 저하**
   (1.93–2.32×, 규모 무관). 재앙적 병목 아님.
2. → **depth CNN encoder 를 기본 경로로 확정**(CLAUDE.md 절대 규율 2 유지). point cloud + PointNeXt
   변환은 처리량 때문에 강제되지 않음 — 부록 실험 옵션으로 남긴다.
3. 4096 depth ON = **16,744 env-steps/s**. DreamWaQ 급 ~1e9 스텝 학습 ≈ 16–17시간(이 GPU 기준).
   더 큰 GPU/멀티GPU 로 단축 가능. 메모리 여유(13.2/16.3GB)로 카메라 해상도·env 수 상향 여지 있음.

## 함정(디버깅 기록)

- **`enable_depth` 는 반드시 생성자 인자로**: `JELocoEnvCfg(enable_depth=False)`. construct 후 필드만
  바꾸면 `__post_init__`(카메라 제거)이 이미 끝나 카메라가 안 지워져 `--enable_cameras` 에러.
- **SimulationContext 는 프로세스당 싱글턴**: 한 프로세스에서 env 여러 번 생성 시 재귀 에러.
  스윕은 반드시 프로세스를 분리(`benchmark_depth_sweep.sh`)해서 돌린다.
- **초기 4096 depth-ON 크래시는 TiledCamera 한계가 아니라, 앞선 실패 실행의 좀비 프로세스가
  GPU 9GB 를 점유한 자원 경합 탓**이었다. 클린 상태에선 4096 ON 이 정상 실행됨. → 벤치 전
  `pkill -f benchmark_depth` 로 잔여 프로세스 정리 권장.
- torch `max_memory_allocated` 는 RTX 렌더러 메모리(allocator 밖)를 못 잡는다 →
  `torch.cuda.mem_get_info()` 로 디바이스 전체 사용량 측정.
