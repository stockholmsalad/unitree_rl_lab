# JEPA-Locomotion: 제어가능한 잠재공간 기반 Go2 보행 정책

석사 졸업논문 구현 저장소. 이 문서는 확정된 설계 명세다. 명세에 없는 결정이 필요하면 임의로 정하지 말고 선택지와 근거를 제시한 뒤 확인을 받을 것.

## 1. 연구 목표

Depth 시퀀스로부터 행동 조건부 미래 latent를 JEPA 방식으로 예측하고, 여러 행동 후보에 대한 예측 latent를 정책 입력으로 사용하는 Unitree Go2 보행 정책을 IsaacLab + PPO로 학습한다. 핵심 주장은 두 가지다.

1. 예측 latent가 접촉 전 지형 foresight를 제공한다.
2. 예측이 행동에 따라 일관되게 반응한다(제어가능성). 이는 반사실 평가로 정량 검증한다.

## 2. 하드웨어 / 센서

- 로봇: Unitree Go2, 관절 12개, 관절 위치 목표 + PD 제어
- Depth: Intel RealSense D435i
  - **장착 위치는 base link 기준 고정 오프셋으로 정의한다.** 지면 기준 높이는 자세·보행에 따라 변하는 결과값이지 명세가 아니다.
  - 실측: 앞다리 hip 관절 축(URDF base 기준 x 0.1934, z 0) 기준으로 하우징 전면 유리 중앙까지 전방 x, 높이 z를 잰다.
    `scripts/jepa_loco/camera_mount_from_hip.py`로 base 기준 좌표로 변환한다. **실측 대기 — 현재 임시값 (0.328, 0, 0.058)**
  - **depth 원점은 왼쪽 IR 카메라**다. 하우징 기준점 → 왼쪽 IR 오프셋을 변환에 포함한다 (D435 데이터시트 값, 확인 필요: 좌측 17.5 mm, 전면 유리 후방 4.2 mm)
  - 아래 방향 pitch 30°, 스테레오 baseline 50 mm
  - depth FOV 약 87° × 58°, 최소 측정 거리 약 0.28 m. 실기 calibration 확보 전에는 정사각 픽셀·HFOV 87°로 intrinsic을 만든다
- URDF 기본 관절각 FK: 앞발 base 기준 x 0.178 m, 명목 base 높이 0.322 m (시뮬 PD 정지 자세 실측 0.280 m). 카메라–앞발 수평 거리는 실측 오프셋 확정 후 기록할 것

## 3. 시간축

| 항목 | 값 |
|---|---|
| 제어 주기 | 50 Hz (dt 0.02 s) |
| Depth 인코딩 / Predictor 주기 | 10 Hz (5 제어 스텝마다 갱신, 사이에는 마지막 값 재사용) |
| Context 길이 | depth 20 프레임 (2.0 s) |
| 예측 horizon H | {0.5, 1.0} s 다중 horizon (horizon embedding으로 조건화) |
| Depth 지연 | 1~3 제어 스텝 랜덤 |

## 4. 아키텍처

모든 시퀀스 모듈은 공통 인터페이스(`SequenceBackbone`) 뒤에 두어 GRU ↔ Mamba 교체가 config 한 줄로 가능해야 한다. hidden state는 환경 reset 시 해당 env만 초기화한다.

1. **Depth 전처리** (`jepa_loco/sensors/depth_proc.py`, 시뮬·실기 공통): 해상도 **112×64** (종횡비 1.75, 크롭 없음), **2채널 (정규화 depth, validity mask)**
   - [0.28, 2.0] m: 유효, `(d − 0.28) / (2.0 − 0.28)` ∈ [0, 1]
   - 2.0 m 초과: 1.0으로 clamp, 유효
   - 0 / NaN / 0.28 m 미만: depth 0, mask 0
   - 시뮬레이터: D435i intrinsic을 112×64로 스케일해 **직접 렌더링** (Omniverse는 비정사각 픽셀 미지원 → fx·fy 평균, 차이 0.95%)
   - 실기: 848×480 → 112×64를 **mask-aware nearest 또는 median**으로 변환. **bilinear·area 금지** (계단 모서리에 가짜 중간 depth 생성)
   - 실기 변환과 시뮬 출력의 통계(유효 픽셀 비율, depth 히스토그램)를 `scripts/jepa_loco/compare_depth_stats.py`로 Phase 2 전에 비교한다
2. **Context 인코더** (backbone 교체 가능): 2채널 입력, 8×8 패치 → 112 토큰 → scan → 프레임 latent → 시간축 backbone → `z_t` (dim 128, config)
3. **고유수용감각 인코더 (MLP)**: 각속도 3 + 중력 투영 3 + 속도 명령 3 + 관절 위치 12 + 관절 속도 12 + 이전 행동 12 = 45차원 → proprio 임베딩
4. **Predictor** (backbone 교체 가능): `(z_t, proprio 임베딩, 조건 c, horizon embedding) → ẑ_{t+H}`
5. **Target 인코더**: Context 인코더의 EMA 사본, stop-gradient. Context 인코더와 같은 2채널 입력. 출력 `z̄_{t+H}`
6. **정책 (Actor)**: K개 primitive에 대한 예측 `ẑ^1..ẑ^K` (각 horizon별) → attention pooling → proprio와 결합 → 관절 명령 12
7. **Critic**: 비대칭 AC. 특권 정보(heightscan, 마찰, 질량 등) 사용 가능

### 4.1 행동 조건 c (중요)

- 학습 시: **실제로 실현된** horizon H 동안의 몸통 좌표계(시점 t 기준) 평면 변위 `(Δx, Δy, Δyaw)`
- 추론 시: 아래 primitive의 명령 속도 × H
- **Δz, pitch, roll은 조건에 절대 넣지 않는다.** 실현된 Δz는 지형 높이 정보 누수이며, 추론 시 줄 수도 없다.

### 4.2 Primitive 집합 (K = 8)

| 후보 | (vx, vy, ωz) |
|---|---|
| 정지 | (0, 0, 0) |
| 직진 저속 | (0.3, 0, 0) |
| 직진 중속 | (0.6, 0, 0) |
| 직진 고속 | (0.9, 0, 0) |
| 전진 좌회전 | (0.5, 0, +0.5) |
| 전진 우회전 | (0.5, 0, −0.5) |
| 좌 횡이동 | (0, +0.3, 0) |
| 우 횡이동 | (0, −0.3, 0) |

K = 1(무조건부), 4, 16 변형을 ablation용으로 config에 정의한다. K = 4: 정지, 직진 중속, 좌·우 회전.

### 4.3 정책과 PPO의 관계

K개 후보는 실행할 행동이 아니라 **관측 인코더의 일부**다(I2A 방식). 실행 행동은 항상 π에서 샘플링하므로 PPO on-policy 가정이 유지된다. 후보 중 최선을 골라 실행하는 구조로 바꾸지 말 것.

## 5. 손실

- `L_pred`: 정규화된 `ẑ`와 `sg(z̄)` 간 cosine 거리 (horizon별 평균)
- `L_ctr`: InfoNCE. positive = (실현 조건으로 만든 ẑ, z̄). negative = 배치 내 다른 env의 z̄ + **같은 상태에서 다른 조건으로 만든 ẑ**. 조건 negative의 margin은 조건 간 거리에 비례.
- 붕괴 방지: latent 차원별 표준편차를 로깅하고, 임계값 아래로 떨어지면 경고. 필요 시 VICReg식 variance 항을 config 옵션으로 추가.
- 전체: `L_PPO + λ_pred L_pred + λ_ctr L_ctr`. λ는 config.

## 6. 시뮬레이션 설정

- **Depth 노이즈**: 거리 제곱에 비례하는 가우시안 노이즈, 경계(depth gradient 큰 곳) 결측 패치, 랜덤 hole, 범위 밖 0 처리,
  **D435i 왼쪽 결측 띠** (폭 ≈ baseline × fx / Z 픽셀, fx는 렌더링 해상도 기준 — 112×64에서 fx ≈ 59, Z = 0.5 m일 때 약 6 px)
- **카메라 랜덤화**: base 기준 장착 위치 ±2 cm (축별), pitch ±3°
- **명령 랜덤화**: vx ∈ [−0.3, 1.0], vy ∈ [−0.4, 0.4], ωz ∈ [−0.8, 0.8]
- **커리큘럼** (단계별 성공률 기준 자동 진행): 평지 → 비정형 → 오르는 계단 8~20 cm → **내려가는 계단 8~20 cm** → gap 10~30 cm
- depth 렌더링 비용 때문에 env 수는 GPU 메모리를 보고 결정하고, 결정 근거를 기록할 것.

## 7. 구현 단계와 완료 기준

각 단계는 완료 기준을 만족하고 사용자 확인을 받은 뒤 다음으로 넘어간다.

- **Phase 0 — 환경 점검**: 설치된 IsaacLab 버전과 Go2 관련 기존 task/asset, 카메라(TiledCamera 등) API를 **설치된 소스에서 직접 확인**. 기억에 의존해 API를 추측하지 말 것. 카메라를 위 extrinsic으로 붙여 depth 샘플 이미지를 저장.
- **Phase 1 — Teacher**: heightscan 기반 PPO teacher. 지형 전체 커리큘럼 통과. 이후 반사실 평가의 분기 실행기로 사용한다.
- **Phase 2 — Depth student (GRU, JEPA 없음)**: depth + proprio PPO 기준선. 학습 안정성 확인.
- **Phase 3 — JEPA 보조손실 (GRU, 온라인)**: Predictor + EMA Target 추가. 조건 c 계산 로직은 단위 테스트 필수(몸통 좌표계 변환, yaw wrap-around). 실현 변위 (Δx, Δy, Δyaw) 히스토그램을 지형별로 출력해 primitive 주변 샘플 공백 확인.
- **Phase 4 — 반사실 평가 도구**: 지형별(비정형, 오르는 계단, 내려가는 계단, gap) 저장 상태 500개 × K개 primitive를 teacher로 H 동안 분기 실행 → (ẑ, z̄) 쌍 생성. 지표: retrieval top-1 정확도(우연 1/K), 예측/실제 거리 행렬 Spearman 상관, 조건 셔플 시 오차 증가량. 발 주변 heightmap linear probe(속도 구간별 보고: v ≥ 0.76 m/s는 현재 관측 기반, 미만은 메모리 기반 예견).
- **Phase 5 — 후보 입력 정책**: K개 예측 latent attention pooling을 정책 입력으로. 비교군: (a) 예측기 없음, (b) 무조건부 latent 1개, (c) K개 조건부 latent. 지표: 지형별 성공률, 계단 모서리 발끝 충돌 횟수.
- **Phase 6 — Mamba 교체**: `mamba-ssm` 설치(CUDA 빌드 확인). LocoMamba 공식 코드는 비공개이므로 논문 서술 기반으로 직접 구현. GRU 결과와 동일 조건 비교.
- **Phase 7 — 학습 절차 비교**: 오프라인 사전학습 + freeze vs 온라인 보조손실.
- **Phase 8 — 배포 준비**: Orin 기준 K ∈ {1, 4, 8, 16}별 추론 지연을 GRU / Transformer / Mamba로 측정. ONNX/TensorRT 내보내기.

## 8. 작업 규칙

- 설정값은 IsaacLab 관례(configclass)를 따르고, 이 문서의 숫자는 하드코딩하지 말고 config로 노출.
- 새 모듈마다 shape 테스트와 최소 단위 테스트 작성.
- 1시간 이상 걸리는 학습은 실행 전에 명령과 예상 소요 시간을 보여주고 확인받을 것.
- 로깅: TensorBoard(또는 wandb). PPO 지표, `L_pred`, `L_ctr`, latent 표준편차, 커리큘럼 단계를 항상 기록.
- 커밋은 작게, 단계별로. 실험 결과와 결정 사항은 `docs/experiment_log.md`에 날짜와 함께 기록.
- 이 문서의 설계와 충돌하는 변경이 필요하면 먼저 이유를 설명하고 이 문서를 갱신할 것.
