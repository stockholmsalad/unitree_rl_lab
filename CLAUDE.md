# JEPA-Locomotion: 제어가능한 잠재공간 기반 Go2 보행 정책

석사 졸업논문 구현 저장소. 이 문서는 확정된 설계 명세다. 명세에 없는 결정이 필요하면 임의로 정하지 말고 선택지와 근거를 제시한 뒤 확인을 받을 것.

## 2026-10-06 연구 구조 전환 — 현재 우선 명세

사용자가 제시한 Mamba + JEPA + privileged future terrain distillation 구조와 구현 요청에 따라
이 절이 아래의 기존 §1, §4, §5, §7 중 **서로 충돌하는 부분을 대체**한다. 기존 절은 GRU
기준선과 종전 반사실 연구안의 기록이다. §2 센서·전처리, §3의 50/10 Hz와 2초 context,
§6의 명령 범위·종료 보상, §8 작업 규칙은 계속 적용한다. Oracle 지형 커리큘럼의
승급·강등 판정은 아래 2026-10-06 결정이 §6의 기존 판정을 대체한다.

### 목표 및 모델

1. Depth 한 프레임을 기존 2채널 CNN에 통과시키고, 10 Hz 새 프레임에서만 갱신하는
   시간축 backbone으로 context `h_t`를 만든다. 기존 GRU는 기준선으로 유지하고 Mamba를
   같은 CNN·입력·평가 조건에서 비교한다. 실제 Mamba 구현은 라이브러리 설치·CUDA 빌드 검증 후 진행한다.
2. **현재/미래 공유 terrain head** `g`를 두어 `z_current = g(h_t)`와
   `z_future = g(P(h_t, p_t, c_t, Δ))`를 얻는다. 초기는 벡터 입력 MLP predictor.
   이 설계에서 `c_t`는 현재 velocity command이며, 종전 K=8 primitive와 attention pooling은
   새 모델의 주 경로에서 사용하지 않는다. 단일 command 조건 결과에 대해 종전의
   행동별 반사실 제어가능성 주장은 하지 않는다.
3. 실제 미래 Depth context를 EMA target encoder에 넣고 stop-gradient한
   `h_target(t+Δ)`와 예측 `h_future`의 MSE를 `L_JEPA`로 쓴다.
4. 특권 heightmap teacher는 현재 patch `H_current`, 명령을 SE(2)로 Δ초 적분한 예상 pose
   주변 patch `H_future`, 현재+미래를 포함하는 wide patch를 비교한다. Current/Future는
   동일한 teacher encoder를 공유한다. Wide와 Current+Future의 **policy 입력 총 terrain
   latent 차원**을 맞춘다. teacher 정책은 PPO로 학습하고 확인된 checkpoint를 고정한 뒤
   student distillation에 사용한다.
5. Student는 `L_current = ||g(h_t)-sg(z_current^T)||²`와
   `L_future = ||g(h_future)-sg(z_future^T)||²`를 학습한다. JEPA target의 실제 도착 pose와
   teacher의 명령 외삽 pose가 어긋나므로, 같은 에피소드·명령 유지·위치/yaw 오차·teacher
   patch 가시성을 검사한다. 유효 표본만 미래 손실에 쓰고 **유효 수로 정규화하며 mask
   coverage를 기록**한다. 현재 distillation은 이 mask와 무관하게 학습한다.
6. 초기 student 총 손실은 `λc L_current + m λf L_future + m λJ L_JEPA`.
   세 계수·horizon·mask 허용오차는 configclass. Copy predictor `h_future=h_t`를
   항상 비교해 시간창 겹침으로 생기는 복사 해를 탐지한다.
7. Policy는 `[proprio, command, z_current, z_future]`를 입력으로 하는 MLP PPO.
   PPO actor·critic과 teacher encoder의 파라미터, 학습 및 동결 경계를 명시적으로 분리한다.
   실기 추론에는 Depth, proprio, command만 사용하며 EMA target·teacher heightmap은 제외한다.

### 연구 실험 순서와 통과 조건

1. 기존 blind/depth GRU PPO 기준선과 고정 계단 평가를 보관한다. 두 정책이 9 cm 첫 단에서
   정지한 원인은 `docs/experiment_log.md`에 기록되어 있다.
2. 카메라 없는 privileged **Current / Wide / Current+Future oracle** 세 조건을 같은
   평가에서 학습·비교한다. 먼저 Current+Future가 Current보다 계단을 실제로 더 잘 오르는지
   확인한다. Wide와의 동률은 허용한다. 보상·성공률만으로 통과시키지 않고 play 영상으로 본다.
3. oracle이 실제로 오르는 것을 확인한 뒤 teacher checkpoint를 고정하고, Depth student
   Current/Wide/Current+Future, distillation-only, distillation+JEPA, copy predictor를
   동일 평가 조건에서 비교한다. DAgger 및 truncated BPTT 연결은 이 단계에서 구현·검증한다.
4. GRU 결과를 확보한 뒤 동일 CNN에서 Mamba backbone을 비교한다. 실제 Go2 Depth
   시퀀스가 준비되면 PPO policy와 terrain head를 고정하고 encoder/predictor를 real JEPA +
   simulation replay로 적응시킨다. 실기 적응 실험은 로그 확보 후 시작한다.

### 2026-10-06 Oracle 커리큘럼·비교 결정

- 성능 기반 지형 커리큘럼을 유지한다. 에피소드 시작점과 종료점 사이의 직선거리 대신
  **매 제어 스텝의 월드 XY 이동 거리 합**, 즉 실제 에피소드 경로 길이로 승급·강등을 판정한다.
  종료를 일으킨 마지막 스텝까지 더하고 reset 이동은 제외한다. 기존 임계값인
  승급 `경로 길이 > 지형 패치 길이/2`, 강등 `경로 길이 < ||명령 선속도|| × 최대 에피소드 시간 × 0.5`
  및 승급 우선 규칙은 유지한다. 모든 threshold는 configclass에 노출한다.
- **2026-10-06 보완:** `vx`와 yaw 명령이 독립이고 yaw가 ±1 rad/s까지 나와 순변위만으로
  승급하면 원형 보행을 잘못 강등한다. 경로 길이는 유지하되, 장애물 지형
  `stairs_up`, `stairs_down`, `gap`에는 첫 장애물 너머에 실제 도달해 설정된
  제어 스텝 수만큼 머문 조건을 승급에 추가한다. 계단은 첫 단 너머, gap은 틈 너머를
  판정한다. 평지·rough는 경로 길이만 사용한다. 장애물 경계는 지형 생성 설정에서
  계산하고 통과 여유·유지 스텝은 configclass에 둔다.
  세 oracle에 동일 적용한다. 에피소드별 장애물 통과율과 경로 길이만 충족한
  거짓 승급 비율을 지형별로 로깅한다. 오르는 계단에서는 첫 단 앞 정지와 첫 단
  너머 진행 에피소드의 보상 항별 합 및 swing 발 최고 높이를 함께 기록한다.
- Current, Wide, Current+Future는 같은 보상·명령·지형 생성 seed·커리큘럼 규칙,
  PPO hyperparameter·환경 수·iteration 수(샘플 예산)를 쓴다. 학습 중 평균 지형 레벨을
  flat, rough, stairs_up, stairs_down, gap으로 나눠 기록한다. 성능 기반 승급으로 실제
  노출 난이도가 달라질 수 있으므로 학습 reward를 조건 간 직접 성능 비교값으로 쓰지 않는다.
- **먼저 Current 한 조건만 본 학습**한다. 고정 9 cm 오르는 계단에서 정량 성공률과
  play 영상을 확인한 뒤 통과 여부를 결정한다. 통과할 때만 Wide와 Current+Future를
  같은 설정으로 추가한다. 최종 조건당 seed는 3개 이상이며 세 조건에 같은 seed 집합을 쓴다.
  최종 성능은 지형 종류 × 난이도 × seed를 고정한 평가 세트에서 비교한다.
- Oracle 스모크는 학습 연결 검증일 뿐 등반 성능 검증이 아니다. 1시간 이상 본 학습은
  실행 명령과 예상 시간을 먼저 제시하고 사용자 확인을 받는다.

### 2026-10-07 OracleCurrent 스윙 발 높이 단일 변수 실험

- 고정 9 cm 계단에서 기존 OracleCurrent seed 43은 0/16, 학습 중 첫 단 통과율도 0이었다.
  기존 정책의 Heightscan을 평탄화하면 발을 더 들고 전진하지만 등반하지 못했으므로,
  새 보상 하나의 효과만 같은 seed·샘플 예산으로 검증한다. 다른 보상, 지형 시작 높이
  8 cm, 명령, PPO, 경로 길이+장애물 통과 게이트는 유지한다.
- OracleCurrent에만 `swing_foot_clearance`를 추가한다. 발바닥 z는 발 링크 중심 z에서
  발 구 반지름을 뺀 값이다. 발 XY 반경 안의 유효 Heightscan hit 중 가장 높은 z를
  지면으로 삼는다. 유효 hit가 없는 발은 제외한다. 접촉력 1 N 미만이며 air time이
  0.05 s를 넘는 발만 스윙으로 본다. 발별 점수는 지형 기준 clearance를 목표 높이로
  나눈 뒤 [0,1]로 자르고 네 발 평균을 쓴다. 명령 XY 속도 노름이 0.1 m/s를 넘을
  때만 `clip((v_body_xy · c_xy)/||c_xy||², 0, 1)`을 곱한다. 두 속도 벡터 모두
  **몸통 좌표계**이며, 정지·역방향·직교 이동은 보상 0이다.
- 기본 목표 clearance 0.10 m, 발 반지름 0.022 m, hit 반경 0.10 m,
  **가중치 1.0**. 값은 모두 `OracleSwingClearanceCfg`에서 바꿀 수 있다.
  20 s·1000스텝·dt 0.02 s에서 트롯처럼 한 번에 두 발이 스윙한다는 조건부
  에피소드 상한은 `0.5 × weight × 0.02 × 1000 = 10 × weight`로, 가중치 1은
  선속도 추종 최대 30의 1/3이다. 네 발 모두 스윙 가능한 수학적 절대 상한은
  `20 × weight`이므로 `10 × weight`를 보편적인 상한이라고 해석하지 않는다.
  기존 정상 평지 보행 checkpoint의 새 게이트·가중치 1 측정값은 +0.368/에피소드다.
  초기 보상 신호가 약할 수 있지만 본 학습 전 가중치는 더 올리지 않는다.
- 에피소드 보상 합과 flat/stairs_up별 스윙 clearance 평균·최댓값을 기록한다.
  이 런은 기존 seed 43과 새 보상 하나만 다른 원인 분리 실험이다. 고정 9/19 cm
  성공률·첫 단 통과율·평지 play 보행을 확인한다. 9 cm 성공 > 0,
  학습 중 첫 단 통과율 > 0, 평지 정상 보행을 모두 만족해야 통과한다.
  새 항 에피소드 합이 선속도 추종 합에 근접하거나 이를 넘으면서 추종이 떨어지면
  보상 착취로 보고 중단을 검토한다. 발은 들지만 첫 단 앞 정지가 지속되거나
  기형 보행이 나타나면 다음 변경은 사용자 확인 후 결정한다. Current가 통과하여 Wide와
  Current+Future를 학습하게 되면, 세 oracle의 공정 비교에서는 동일한 확정 보상을
  세 조건에 적용한다. Student에 적용할지는 oracle 결과 후 결정한다.

### 2026-10-07 OracleCurrent 기본 자세 페널티 분리 실험

- 스윙 높이 보상 단독 런은 iteration 800 부근에 평지 스윙 clearance 약 0.030 m,
  stairs_up 약 0.023 m, 첫 단 통과율 0으로 조기 중단했다. 발 들기를 막는 기본
  자세 페널티가 원인인지 검증하기 위해 이 항 **하나만 분리·완화**한다. 비교 대상은
  동일 seed 43의 swing 런 iteration 0~800이며, 스윙 보상·지형·명령·커리큘럼·
  다른 보상·PPO는 유지한다. 기존 checkpoint에서는 이어 학습하지 않는다.
- OracleCurrent에서 기존 전체 관절 `joint_pos`를 끄고 두 항으로 나눈다.
  `nominal_hip`은 4개 hip 관절 편차의 L2 노름에 기존 가중치 **−0.7**을
  유지한다. `nominal_thigh_calf`는 8개 thigh/calf 관절의
  `Σ(q−q_default)²/max(||v_cmd,xy||²,1)`에 가중치 **−0.15**를 적용한다.
  두 항 모두 기존 함수와 같은 정지 판정(명령 전체가 0이고 몸통 XY 속도
  ≤0.3 m/s)에서만 ×5를 적용한다. 관절 정규식, 계수, 정지 배율·임계값,
  정규화 방식·분모 바닥값은 `OracleNominalPoseCfg`에 노출한다.
- thigh/calf 가중치는 중단된 swing 런 `model_800.pt`를 평지 0.6 m/s,
  32 env·20 s로 재생해 정했다. 단위 가중치 −1의 thigh/calf 항 합은
  −1.953/에피소드로 측정되어, 가중치 −0.15에서 **−0.293/에피소드**다.
  목표는 선속도 추종 최대 30/에피소드의 약 1%인 −0.3이다. hip 항은
  같은 평가에서 −1.984/에피소드였다. APT-RL Table S3의 명목 자세 항은
  제곱합/명령 속도 정규화이지만 우리 기존 항은 노름이므로 가중치를 직접
  비교하지 않는다. hip 약화 시 다리 벌림이 나타났다는 이전 연구 관찰에
  따라 hip 가중치는 유지한다. 이 실험은 우선 Current에만 적용하고,
  이후 세 oracle 공정 비교에서는 확정 보상을 동일하게 적용한다.
- iteration 800에 평지 스윙 clearance 평균 >0.06 m가 아니면 중단을
  검토하고, play 영상 또는 hip 편차로 다리 벌림을 확인해 있으면 중단한다.
  iteration 1500에 stairs_up 첫 단 통과율이 0이면 5 cm 시작 계단과
  평지 워밍업 레벨을 사용자와 논의한다. 최종 통과는 고정 9 cm 성공 >0,
  첫 단 통과율 >0, 평지 play 정상 보행을 모두 만족할 때만 인정한다.

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
| Depth 지연 | 1~3 제어 스텝 랜덤 (실제 지연). IsaacLab 고유 지연 d0 = 0 (실측 2026-10-01: 렌더 직후 순간이동이 다음 렌더에 바로 반영) → 추가 지연 d_add = d − d0 |

## 4. 아키텍처

모든 시퀀스 모듈은 공통 인터페이스(`SequenceBackbone`) 뒤에 두어 GRU ↔ Mamba 교체가 config 한 줄로 가능해야 한다. hidden state는 환경 reset 시 해당 env만 초기화한다.

1. **Depth 전처리** (`jepa_loco/sensors/depth_proc.py`, 시뮬·실기 공통): 해상도 **112×64** (종횡비 1.75, 크롭 없음), **2채널 (정규화 depth, validity mask)**
   - [0.28, 2.0] m: 유효, `(d − 0.28) / (2.0 − 0.28)` ∈ [0, 1]
   - 2.0 m 초과: 1.0으로 clamp, 유효
   - 0 / NaN / 0.28 m 미만: depth 0, mask 0
   - 시뮬레이터: D435i intrinsic을 112×64로 스케일해 **직접 렌더링** (Omniverse는 비정사각 픽셀 미지원 → fx·fy 평균, 차이 0.95%)
   - 실기: 848×480 → 112×64를 **mask-aware nearest 또는 median**으로 변환. **bilinear·area 금지** (계단 모서리에 가짜 중간 depth 생성)
   - 실기 변환과 시뮬 출력의 통계(유효 픽셀 비율, depth 히스토그램)를 `scripts/jepa_loco/compare_depth_stats.py`로 Phase 2 전에 비교한다
2. **Context 인코더** (backbone 교체 가능): 2채널 입력 프레임 1장 [N,2,64,112] → **CNN** → `f_t` [N,128] → 시간축 backbone → `z_t` [N,128] (dim config)
   - 프레임 스택 없음. depth 는 10 Hz 로 1장씩 CNN 에 넣고, backbone 은 새 프레임이 온 스텝에만 갱신한다(`depth_fresh` 게이트)
   - nominal context 2 s 는 truncated BPTT 길이로 보장: PPO rollout `num_steps_per_env = 100` 제어 스텝 = depth 20 프레임
   - **CNN 과 입력 형식은 GRU·Mamba 비교에서 동일하게 유지**한다 (시간축 backbone 만 교체). 8×8 패치 토큰 인코더는 이번 비교 범위에서 제외
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
- **명령 랜덤화** (2026-10-06 갱신): vx ∈ [−0.3, 2.0], vy ∈ [−0.4, 0.4], ωz ∈ [−1.0, 1.0]
- **Phase 2 재학습 명령 스케줄 (2026-10-02 결정):** 세 축 모두 [−0.1, 0.1]에서 시작해 전역 학습 iteration 0~500 동안 위 최종 범위까지 선형 확장한다. 성능 적응형 명령 커리큘럼은 사용하지 않는다. 500 iteration 이후에는 위 명세 범위를 유지한다. depth·blind가 동일한 전역 스케줄을 사용한다.
- **Phase 2 재학습 종료 보상 (2026-10-02 결정):** `is_terminated` weight −200 (제어 dt 0.02 s 적용 후 종료 시 −4). 시간 초과는 제외한다. 기존 entropy 계수 0.01과 학습률은 유지한다.
- **Phase 2 대조군 (2026-10-02 갱신):** 두 blind 모두 카메라 센서를 생성하지 않고, depth 실험과 보상·명령 스케줄·지형·critic·PPO 설정·rollout 길이를 공유한다. 3000 iteration으로 비교하고 학습 곡선·play 영상을 확인한다.
  - **첫 비교군 = 기억 없는 proprio-only MLP** (`Unitree-Go2-JepaLoco-BlindMLP`): depth 정책의 MLP head(512-256-128) 그대로, 입력에서 z_t 만 뺀 [proprio 45] → 12. **depth 경로(CNN+GRU) 하나만 다른** 대조군이라 "depth 효과"는 이것과 비교한다.
  - **강한 대조군 = proprio GRU** (`Unitree-Go2-JepaLoco-BlindGRU`): proprio 를 50 Hz GRU 에 넣어 기억을 준다. depth 정책에는 proprio 기억이 없으므로 depth 와의 차이에는 proprio 기억 효과가 섞인다 — "proprio 기억만으로 어디까지 되는가"의 참고선으로만 쓴다.
- **명령 스케줄 일관성:** 스케줄의 `steps_per_iteration` 은 PPO `num_steps_per_env` 와 같아야 하며, 다르면 학습 시작 시 오류. `--resume` 은 checkpoint iteration × rollout 길이로 `common_step_counter` 를 복원하고 명령을 재샘플한다(2026-10-02 수정 전 코드로는 resume 금지).
- **Oracle 커리큘럼** (2026-10-06 갱신): 지형 종류를 열로 혼합하고 난이도 행을 성능에 따라
  승급·강등한다. 승급 판정은 에피소드 XY 경로 길이와 장애물 통과 기준이며 위 우선 명세의 규칙을 따른다.
  종류는 평지·비정형·오르는 계단 8~20 cm·내려가는 계단 8~20 cm·gap 10~30 cm.
- **OracleCurrent A안 학습용 지형 (2026-10-07):** nominal 비교용 지형과 별도 task
  `Unitree-Go2-JepaLoco-OracleCurrent-EasyStart`를 쓴다. 10행·20열과 지형 종류 비율은
  유지한다. 장애물 열(stairs_up, stairs_down, gap)의 레벨 0·1은 평지 워밍업이다.
  레벨 2~9의 오르기/내리기 계단 높이는 0.05~0.20 m, 단 폭은 0.3 m이며,
  gap 폭은 0.10~0.30 m다. 행 2=0.05 m, 행 9=0.20 m로 행별 난이도를 고정한다.
  레벨 0·1은 경로 길이만으로 승급하고, 레벨 2 이상은 장애물 통과 게이트도
  적용한다. 학습용 높이는 `EasyStartTerrainCfg`에서 설정한다. 평가는 변경 없이
  고정 9 cm와 19 cm 오르는 계단을 쓴다. 보상·명령·관측·PPO는 nominal과 동일하다.
  첫 단 접촉 시도율, 첫 단 ±0.4 m 안에서 종료한 에피소드의 사유별 횟수,
  장애물 통과율, 접근 정지/전진 시도 보상 합을 지형 종류·레벨별로 기록한다.
  iteration 500 부근 레벨 2 오르기 계단 시도율>0·통과율=0이면 중단하고
  접촉 body와 종료 사유를 조사한다. 시도율=0이면 정지 해의 보상 항별 합을
  조사한다. 통과율>0이면 1500까지 레벨 진행을 본다. 최종 통과에는 고정 9 cm
  성공>0, 학습 중 첫 단 통과율>0, 평지 정상 보행이 모두 필요하다.
- depth 렌더링 비용 때문에 env 수는 GPU 메모리를 보고 결정하고, 결정 근거를 기록할 것.

## 7. 구현 단계와 완료 기준

각 단계는 완료 기준을 만족하고 사용자 확인을 받은 뒤 다음으로 넘어간다.

- **Phase 0 — 환경 점검**: 설치된 IsaacLab 버전과 Go2 관련 기존 task/asset, 카메라(TiledCamera 등) API를 **설치된 소스에서 직접 확인**. 기억에 의존해 API를 추측하지 말 것. 카메라를 위 extrinsic으로 붙여 depth 샘플 이미지를 저장.
- **Phase 1 — Teacher**: heightscan 기반 PPO teacher. **보류 (2026-10-01 결정)** — Phase 2 depth PPO 가 명확히 학습되지 않을 때만 teacher + DAgger 를 fallback 으로 추가한다. 반사실 평가(Phase 4)의 분기 실행기가 필요해지면 다시 검토. critic 관측 그룹(heightscan 포함)은 teacher 용으로 재사용 가능하게 유지.
- **Phase 2 — Depth student (GRU, JEPA 없음)**: depth + proprio PPO 기준선을 **먼저** 진행. rough·stairs 학습 curve 와 평가 결과 확보. task `Unitree-Go2-JepaLoco-DepthGRU`.
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
