# JEPA-Locomotion: 제어가능한 잠재공간 기반 Go2 보행 정책

석사 졸업논문 구현 저장소. 이 문서는 확정된 설계 명세다. 명세에 없는 결정이 필요하면 임의로 정하지 말고 선택지와 근거를 제시한 뒤 확인을 받을 것.

## 2026-10-06 연구 구조 전환 — 현재 우선 명세

사용자가 제시한 Mamba + JEPA + privileged future terrain distillation 구조와 구현 요청에 따라
이 절이 아래의 기존 §1, §4, §5, §7 중 **서로 충돌하는 부분을 대체**한다. 기존 절은 GRU
기준선과 종전 반사실 연구안의 기록이다. §2 센서·전처리, §3의 50/10 Hz와 2초 context,
§6의 명령 범위·종료 보상, §8 작업 규칙은 계속 적용한다. Oracle 지형 커리큘럼의
승급·강등 판정은 아래 2026-10-06 결정이 §6의 기존 판정을 대체한다.

### 목표 및 모델

2026-10-08 사용자 결정이 아래 4~7번을 갱신한다. 2번의 공유 미래
terrain head와 그 증류 경로는 당시 설계 이력이며 **현재 기본 경로에서는
꺼진 선택 ablation**이다. 3번의 JEPA 미래 target은 미래 *Depth 표현*을
뜻하며 특권 미래 heightmap 증류를 뜻하지 않는다.

1. Depth 한 프레임을 기존 2채널 CNN에 통과시키고, 10 Hz 새 프레임에서만 갱신하는
   시간축 backbone으로 context `h_t`를 만든다. 기존 GRU는 기준선으로 유지하고 Mamba를
   같은 CNN·입력·평가 조건에서 비교한다. **2026-10-15 결정:** 첫 Mamba 비교는
   depth-only 증류 `--method gru`의 시간축 backbone만 FLA Mamba-1로 바꾼다.
2. **현재/미래 공유 terrain head** `g`를 두어 `z_current = g(h_t)`와
   `z_future = g(P(h_t, p_t, c_t, Δ))`를 얻는다. 초기는 벡터 입력 MLP predictor.
   이 설계에서 `c_t`는 현재 velocity command이며, 종전 K=8 primitive와 attention pooling은
   새 모델의 주 경로에서 사용하지 않는다. 단일 command 조건 결과에 대해 종전의
   행동별 반사실 제어가능성 주장은 하지 않는다.
3. 실제 미래 Depth context를 EMA target encoder에 넣고 stop-gradient한
   `h_target(t+Δ)`와 예측 `h_future`의 MSE를 `L_JEPA`로 쓴다.
4. **2026-10-08 변경:** 고정 teacher는 `Current-EasyStart` seed 42의
   `model_2999.pt`이다. 현재 187점 heightmap을 `187→128→32` encoder로
   압축한 `z_current^T`를 주 목표로 한다. Current+Future는 고정 평가에서
   기준에 미달했으므로 비교 기록으로만 보관한다. 연구 주장은 명령 외삽
   미래 지형 예견에서 **카메라 사각지대의 현재 지형을 과거 Depth로 기억·예측해
   보행**하는 것으로 전환한다. 계단 직전에는 수직면이 보이지만 발을 디딜
   윗면이 시야에서 사라지는 상황이 대표 사례다.
5. Student는 10 Hz Depth→기존 CNN→GRU context `h_t`→terrain head `g`로
   `ẑ_current`를 만든다. 주 손실은 고정 teacher `z_current^T`의 latent 증류와
   teacher 행동 증류다. `z_future` 증류는 기본 꺼짐의 선택 ablation이다.
6. JEPA predictor는 과거 context와 명령(또는 실제 이동량)을 조건으로
   Δ초 뒤 Depth의 EMA target 임베딩을 예측하는 보조 손실이다. 같은
   에피소드·명령 유지 mask를 적용하고 유효 표본 수로 정규화한다. 역할은
   사각지대를 채우는 기억 형성이며 copy predictor 비교를 유지한다.
   초기 계수 `λ_z=1, λ_a=1, λ_J=0.1`, Δ=0.5 s는 configclass에 노출한다.
   **2026-10-09 JEPA 진단 수정:** pilab 초기 56 iteration에서 원시 context MSE가
   latent 증류에 비해 매우 작았다. JEPA와 copy 손실은 유효한 새 목표 depth
   프레임의 EMA target 차원별 배치 분산으로 정규화하고, copy 대비 예측 손실 및
   target 분산을 로깅한다. target은 EMA GRU `context`(현재 기본값) 또는
   EMA CNN `frame_embedding` 중 선택한다. 후자는 새 프레임 시점만 사용한다.
   `frame_embedding` 기본값 전환과 λ_J 조정은 256 env·20 iteration 비교
   결과를 보고 사용자가 확정한다.
   **2026-10-09 Phase 3 보정:** 학습 env 생성 직후 `episode_length_buf`를
   `[0,max_episode_length)` 균등 정수로 초기화한다
   (`init_at_random_ep_len=True`, 평가에는 적용하지 않음). 에피소드 1000스텝과
   rollout 100스텝의 동기 timeout을 분산시키기 위함이다. JEPA의 유효 계수는
   첫 50 iteration 동안 0→설정 λ_J로 선형 증가시키고 `Loss/lambda_j`로 기록한다.
   `jepa_copy_loss`는 **같은 EMA 인코더**의 현재/미래 표현
   `h̄_t`/`h̄_{t+Δ}`의 정규화 MSE다. `frame_embedding`이면 양쪽 모두
   EMA CNN(+projection) 표현이다. online `h_t`와 EMA 미래 target 간 MSE는
   `jepa_copy_online_loss`로 별도 기록한다. **진단용 copy(EMA→EMA)와
   `gru_copy` 학습 방법(online identity)은 별개**다. `gru_copy`는 online
   context `h_t`를 항등 예측값으로 사용하고 EMA 미래 target과의 손실
   기울기를 online CNN·GRU로 보낸다. predictor MLP는 사용하지 않는다.
   이 방법의 JEPA 손실은 `jepa_copy_online_loss`와 같으며, 진단용
   `jepa_copy_loss`와 같을 필요는 없다.
   2026-10-09부터 JEPA 목표 시점(`future`/`present_from_past`)과 predictor 조건
   (`command`/`realized_displacement`)도 설정으로 선택한다. 기본 조건은 `command`다.
   실현 변위는 시작 몸통 좌표계의 SE(2) 변위이며 yaw는 ±π에서 wrap한다.
   고정 Δ·같은 rollout에서는 두 목표 시점 표기가 같은 (시작, 도착) 쌍을 사용하므로
   목표 시점 이름만 바꾼 실험을 독립적인 ablation으로 해석하지 않는다.
7. 기본 student 정책은 고정한 teacher head `π_T`에 `[proprio, ẑ_current]`를
   넣는다. 새 head 학습은 config 옵션으로만 준비한다. teacher head 고정,
   초기 손실 계수와 teacher head 고정은 사용자 최종 확인 대기 항목이다.
   2026-10-09 사용자 결정으로 DAgger teacher 행동 혼합은 β=1에서 시작해
   500 iteration 동안 0까지 선형 감소한다. 100스텝 truncated BPTT를
   유지하며 rollout마다 env 축 4개 미니배치 × 2 epoch, 총 8번 업데이트한다.
   EMA target은 각 optimizer step 직후 갱신한다. 기억 없는 비교군도 10 Hz
   새 프레임에서만 CNN을 실행하고 프레임 사이에는 마지막 표현을 유지한다.
   실기 추론에는 Depth,
   proprio, command만 사용하고 EMA target과 특권 heightmap은 제외한다.

### 2026-10-12 졸업논문 평가 범위 결정

사용자 결정: 졸업논문의 주장은 **계단과 비정형 지형 보행**이다. gap은 평가와 주장에서
제외한다.

- **gap 제외 근거:** teacher가 고정 gap 평가에서 한 번도 통과하지 못했다.
  - 대상: Current-EasyStart s42의 2999~14000 checkpoint.
  - 결과: 모두 안쪽 가장자리 약 0.3 m 앞에서 정지했다.
- **학습 지형은 바꾸지 않는다.** gap 열을 유지해 teacher를 재학습하지 않는다.
  논문에는 "학습 지형에는 포함, 평가 범위에서는 제외"라고 적는다.
- **gap 평가 코드는 삭제하지 않고 보관한다.** 임베디드공학회 원고 수치를 재현하는 데 필요하다.
- **졸업논문 고정 평가 구성:**
  - 오르는 계단 9~15 cm: 핵심 지표. 사각지대 기억 효과를 본다.
  - 내려가는 계단 9~15 cm: 핵심 보조 지표.
  - 비정형 지형 여러 단계: 기억을 추가해도 일반 보행 성능이 유지되는지 본다.
  - 평지: 기본 동작 확인.
- **세부 판정 규칙:** 내려가는 계단·비정형 지형의 판정 규칙과 단계 값은 구현 시
  `docs/experiment_log.md`에 기록한다.
- **연속 코스 보조 평가(2026-10-17):** 경사로→오르기→내리기→비정형을 직진으로 잇고,
  지형 전환 적응을 별도 평가한다. 경사로는 학습 밖 지형이므로 일반화 지표로 분리한다.

### 2026-10-14 teacher v2 추종 std 단일 변수 실험

고정 teacher `Current-EasyStart s42 model_17000.pt`의 고정 평가(seed 43, 높이당 128 env,
0.6 m/s)에서 오르기 14/15 cm 성공률은 69.5/43.8%, 내리기 19/20 cm는
29.7/4.7%였다. 실패는 첫 단/내리막 앞 x≈1.0 m의 정지(time_out)이며 낙상은
0~1개다. 정지한 student의 첫 단 근처 teacher 행동 오차도 0.09~0.13으로 작다.
가설은 선속도 추종 보상 `1.5 exp(-||v_cmd-v_body||²/std²)`의 std=0.5가
0.6 m/s 명령에서 정지해도 최대치의 약 24%를 지급한다는 것이다.

- 기존 `OracleCurrent-EasyStart`의 std=0.5와 나머지 설정은 유지한다.
  신규 `OracleCurrent-EasyStart-TightTrack`만 std=0.25로 둔다(정지 시 약 0.3%).
  다른 보상·지형·명령·커리큘럼·PPO 및 네트워크는 동일하다.
- seed 42, 2048 env로 두 조건 모두 **같은 `model_17000.pt`의 actor·critic·optimizer**에서
  3000 iteration 이어 학습한다. 직접 checkpoint 경로 resume는 저장 iteration 다음인
  17001부터 20000까지 수행하고, 명령 스케줄 전역 스텝은 기존 규칙대로
  `17000×100`에서 복원한다. 500 iteration마다 저장한다. 두 run은 student 학습 뒤
  pilab에서 동시에 시작한다.
- 두 조건 공통 진단: 지형 종류별 종료 에피소드의 `track_lin_vel_xy` 보상 합 평균,
  몸통 좌표계 xy 속도 오차의 에피소드 평균, 명령 xy 속도 >0.3 m/s이고 몸통 xy 속도
  <0.1 m/s인 제어 스텝의 비율. 기존 레벨별 통과율·종료 원인 진단도 유지한다.
- **평가를 보고 teacher를 교체할지 결정한다.** 기존 평가 task로 18000/18500/
  19000/19500/20000 checkpoint를 오르기 9~15 cm, 내리기 15~20 cm, 비정형
  6/8/10/12 cm에서 seed 43·높이당 128 env로 비교한다. TightTrack 채택에는
  (1) 오르기 14·15 cm와 내리기 19·20 cm 성공률이 같은 checkpoint의 std=0.5
  대조군 평균보다 모두 높고, (2) 정지 timeout이 줄되 높이별 낙상이 17000보다
  5%p 이상 증가하지 않고, (3) 비정형 10·12 cm 성공률과 속도 오차가 17000보다
  각각 5%p·20% 이상 악화되지 않으며 평지 성공률 100%, (4) 상위 checkpoint의
  seed 44 재평가와 사용자 play 영상 정상 보행 확인이 모두 필요하다. 실패하면
  `model_17000.pt`를 유지하고 음성 결과로 기록한다.

### 2026-10-17 teacher v3 진행 보상 단일 추가 실험

v2의 TightTrack(std 0.25)은 사전 채택 기준을 통과하지 못했다. 내리기 20 cm는
두 조건 모두 0~2%였고 정지 비율·평지 속도 오차도 대조군보다 나빴다. 좁은
Gaussian은 정지와 느린 전진을 모두 거의 0으로 만들어 구분 신호가 약하다는
가설에 따라, 기존 Gaussian(std 0.5, weight 1.5)은 유지하고 진행 보상 항
`track_lin_vel_progress` 하나만 새 Progress task에 추가한다. 기존 EasyStart
task에는 보상 항 자체가 없다.

- 몸통 좌표계 명령 xy `c`와 속도 xy `v`로 계산한다. `||c||≥0.1`이면
  `clip((v·c)/||c||², −1, 1)`이다. 명령보다 빠른 전진은 1, 정지는 0,
  역방향은 최소 −1이다. 작은 명령은 `exp(−||c−v||²/0.5²)`다. 새 항
  weight 1.0의 20초 에피소드 최대는 +20이며 기존 Gaussian 최대 +30보다 작다.
  가중치·임계값·작은 명령 std·하한은 기존 이름 `OracleProgressCfg`에 노출한다
  (그 class는 이미 지형 통과 진단 설정에도 쓰인다).
- `Unitree-Go2-JepaLoco-OracleCurrent-EasyStart-Progress`는 관측·보상 중 기존 항·
  지형·명령·커리큘럼·PPO·네트워크를 그대로 두고 새 보상만 사용한다. 지형별
  `track_lin_vel_progress` 에피소드 합, 기존 속도 오차·정지 비율과 레벨별 진단을 기록한다.
- seed 42·2048 env에서 `model_17000.pt`의 actor·critic·optimizer를 직접 경로로
  불러와 iteration 17001~20000을 학습하고 500 iteration마다 저장한다.
  `common_step_counter=17000×100`으로 복원한다. 대조군은 v2의 같은 seed·
  resume 방식으로 만든 std 0.5 run의 18000/18500/19000/19500/20000을 재사용한다.
- 고정 평가는 기존 EasyStart task, seed 43, 높이당 128 env로 오르기 9~15 cm,
  내리기 15~20 cm, 비정형 6/8/10/12 cm, 평지를 본다. 채택에는 (1) 오르기
  14·15와 내리기 19·20 cm 성공률이 같은 checkpoint의 대조군 평균보다 높고,
  (2) 개선이 timeout 감소이며 높이별 낙상은 17000 대비 +5%p 미만이고,
  (3) 비정형 10·12 cm 성공률/속도 오차가 17000 대비 각각 5%p/20% 이상
  악화되지 않으며 평지 성공률 100%, (4) 상위 checkpoint의 seed 44 재평가와
  사용자 play 영상 정상 보행 확인이 모두 필요하다. 연속 코스는 보조 지표다.
  조건을 만족하지 못하면 17000을 유지하고 음성 결과로 기록한다.

teacher 교체는 결과 확인 뒤 결정한다.

### 2026-10-18 졸업논문 고정 teacher 교체: v3 model_19000

사용자 결정: 위 v3 실험의 `model_19000.pt`로 고정 teacher를 바꾼다. 위 "model_17000" 절을 대체한다.
경로(pilab): `logs/rsl_rl/jepa_loco_oracle_current_progress/2026-10-17_22-09-28_progress_s42_from17000/model_19000.pt`.
Z790 사본: `logs/rsl_rl/teacher_v2/progress_model_19000.pt`.

- **판정:** 18000~20000 중 사전 기준 (1)~(4)를 모두 통과한 checkpoint는 19000 하나다.
  상세는 `results/jepa_loco/teacher_v3/eval_summary.md`.
  - seed 43 (17000 → 19000): 오르기 14/15 cm 70/47 → 98/92, 내리기 19/20 cm 27/3 → 100/100,
    비정형 10/12 cm 92.2/66.4 → 87.5/61.7, 계단 낙상 0, 평지 100, 연속 코스 14 cm 53 → 100.
  - seed 44에서도 계단 개선이 재현됐다(오르기 15 cm 82, 내리기 20 cm 100).
    비정형 12 cm는 같은 seed 17000 대비 −5.5%p다. 낙상은 줄고 정지가 늘었다.
  - 사용자가 play 영상(오르기 15, 내리기 20, 비정형 12, 평지, 코스 14 cm)으로 정상 보행을 확인했다.
  - 대조군처럼 checkpoint 간 변동이 크다. 17000과 같이 **평가로 고른 checkpoint**다.
- **latent 정규화 통계:** 19000으로 다시 계산한다(`results/jepa_loco/student/teacher_latent_stats_t19000.pt`,
  `calibrate_student_teacher.py`의 기존 설정). 17000 통계는 재사용하지 않는다.
- **student 재증류:** teacher 17000 기반 student 결과(depth-only, propw5, Mamba, 연속 코스)는 기록으로 보관한다.
  졸업논문 비교는 19000 기반으로 다시 학습한다. 학습 설정은 기존 그대로다.
  17000 기반 복원 기준선 run은 시작 직후 중단했다.

### 2026-10-17 복원 기준선

졸업논문의 사각지대 기억 주장에서 잠재공간 예측과 복원의 차이를 검증한다.
같은 teacher 17000, CNN, 5스텝 고유수용감각 창, seed 43에서 `gru_propw5`
(보조 손실 없음), `gru_jepa_propw5`(미래 EMA depth 임베딩 예측)에 두 기준선을
추가한다. 학습용 decoder는 추론 정책·고정 teacher head에 포함하지 않는다.

- `gru_recon_depth`: JEPA와 같은 0.5초 미래 새 depth 프레임, 같은 에피소드·
  명령 유지 마스크 및 `[h_t,p̄_t,condition]`을 쓴다. 목표는 학습 관측 그대로의
  2채널 64×112 depth와 validity mask다. MLP와 transposed convolution으로
  복원하며 유효 depth 픽셀의 차원별 분산 정규화 MSE와 mask BCE를 더한다.
  목표 공간만 EMA 임베딩에서 픽셀로 바뀐다.
- `gru_recon_height`: `[h_t,p̄_t]`에서 **현재** teacher 입력의 187점 높이맵을
  매 제어 스텝 MLP로 복원한다. 이는 현재 높이맵을 teacher latent로 증류하는
  기존 주 손실에 명시적인 전체 해상도 복원을 더한 기준선이다. 카메라 투영·
  렌더 depth 일치로 시야 안/밖 점의 정규화 복원 오차를 따로 기록한다.
- 두 방법은 기존 latent/행동 증류, DAgger, 100스텝 TBPTT, 4 미니배치×2 epoch를
  유지한다. `λ_R=0.1`을 첫 50 iteration 동안 0에서 선형 증가시키고 목표
  차원별 유효 표본 분산을 하한으로 제한한다. 미래 depth는 0.05(JEPA와 같음), 현재 높이맵은 1e-4다. 높이맵 목표는 EMA가 아닌 실제 지형이라 붕괴가 없고, 학습 초기 평지에서 batch 분산이 약 1e-4라서 0.05 하한을 쓰면 손실이 수백 배 작아져 평균 예측 기준 1이 깨진다(2026-10-17 스모크 확인). 계수·하한·decoder 크기는
  `StudentTrainCfg`에 노출한다. GRU·Mamba backbone 양쪽을 허용한다.
- 고정 평가에서 checkpoint 3000은 오르기·내리기·비정형·평지, 2500/2800은
  오르기를 평가한다. 주 지표는 오르기 14/15 cm의 세 checkpoint 평균이다.
  내리기 17~20 cm, 비정형 10/12 cm, 연속 코스 14 cm는 보조 지표다.
  `gru_jepa_propw5`가 depth 복원보다 주 지표에서 높을 때만 "잠재 예측이
  픽셀 예측보다 낫다", height 복원보다 높을 때만 "명시적 복원보다 낫다"고
  쓴다. 차이가 5%p 미만이면 차이 없음으로 기록하고, 모든 결론에
  **seed 43 단일**이라는 제한을 붙인다. 성능 결과는 평가 전까지 주장하지 않는다.

### 2026-10-14 student 고유수용감각 입력 (처음 구조도 복원)

처음 구조도는 Depth 시퀀스 context와 고유수용감각 시퀀스 임베딩을 함께
predictor에 넣는다. 기존 Phase 3 student는 Depth만 GRU로 기억하고
predictor도 context와 명령 3차원만 받았다. 이동·발 접촉 이력이 사각지대
지형의 상대 위치 추정에 필요할 수 있으므로 아래 연결을 선택 옵션으로 추가한다.
**기본값은 꺼짐**이며 기존 checkpoint와 depth-only run의 구조·수치를 유지한다.

1. `obs["policy"]`의 원시 45차원 값을 **고정 teacher의** `actor.obs_normalizer`로
   정규화한 뒤 학습 가능한 ELU MLP(기본 45→64→64)에 통과시켜 `p_t`를 만든다.
   층 크기와 사용 여부는 student 설정에 노출한다. teacher normalizer와
   정책 head `π_T`는 계속 고정한다.
2. JEPA predictor 입력을 `[h_t,p_t,명령 또는 실현 변위]`로 확장한다.
   `gru_copy`의 online identity에는 predictor가 없어 영향을 받지 않는다.
3. GRU 계열의 입력을 `[f_t,p_t]`로, `no_memory`의 현재 프레임 projection
   입력도 `[f_t,p_t]`로 확장한다. 네 비교군의 관측 입력을 맞춘다.
   초기 그림의 Mamba predictor는 시간 기억을 갖지만 현재 predictor는 MLP다.
   따라서 고유수용감각의 과거는 backbone에 저장한다. 나중에 Mamba backbone도
   동일한 결합 입력을 쓴다.

**2026-10-14 사용자 결정 — B안 채택:** `--proprio_input`은 계속 기본 꺼짐이다.
켜면 최근 K개 50 Hz 제어 스텝의 원시 policy 관측을 오래된 것부터 최신 순으로
모아 각 스텝을 공유 teacher 정규화기·고유수용감각 MLP에 통과시킨다.
임베딩 K개와 각 스텝의 valid mask K개를 이어 Linear→ELU로 64차원
`p̄_t`를 만든다. 기본 `proprio_window=5`(depth 주기)이며 K=1이면
요약 Linear를 만들지 않아 기존 A안을 비트 단위로 재현한다. K<1은 오류,
depth 주기와 다른 K는 경고 후 허용한다. 에피소드 이전 값은 원시값과
임베딩 모두 0, valid=0으로 채운다. reset 직후 첫 값만 valid=1이다.

GRU·no-memory projection은 여전히 `depth_fresh`에서만 `p̄_t`를 받아
갱신하고 프레임 사이에는 context를 유지한다. JEPA predictor는 시작 시점
window를 받는다. 수집과 업데이트가 같은 창을 보도록 rollout 시작의
`[K−1,N,45]` prefix와 valid를 저장하고 env 미니배치와 함께 자른다.
중간 reset에서는 이전 에피소드 prefix를 비운다. EMA target은 스텝별
MLP와 요약 Linear 모두 이동평균 사본을 갖고 target 시점에서 끝나는
실제 policy window를 받는다. `frame_embedding` target은 기존 EMA CNN
경로를 유지한다. 추론에서도 env별 window를 유지하고 done에서 초기화한다.

초기 그림과 다른 부분: 정책은 고정 teacher head에 `[proprio,ẑ_current]`를
입력한다. Phase 2의 Depth PPO가 9 cm 첫 단을 넘지 못했고 predictor 출력은
teacher의 `z_T` 공간과 다르기 때문이다. 행동 후보 K개 대신 현재의 단일
명령을 조건으로 쓴다(2026-10-06 결정). 새 연결의 우열은 결과 전까지 주장하지 않는다.

### 2026-10-13 졸업논문 고정 teacher 교체: model_17000

사용자 결정: 졸업논문의 고정 teacher를 Current-EasyStart seed 42를 이어 학습한
`model_17000.pt`로 바꾼다. 경로는
`logs/rsl_rl/jepa_loco_oracle_current/2026-10-11_14-25-27_oracle_current_easystart_s42_cont/`다.

- 위 목표·모델 4번의 `model_2999.pt`는 임베디드공학회 원고(2026-10-12 제출)의 teacher로만 남긴다.
- **선택 근거:** 2999~19998 checkpoint를 평가 seed 43으로 비교하고,
  상위 3개(16000~18000)를 평가 seed 44로 재확인했다.
  - 17000의 성공률(seed 43 기준): 오르기 13 cm 96%, 14 cm 70%, 15 cm 47%,
    내리기 18 cm 65%, 19 cm 27%. 오르기와 내리기를 함께 볼 때 가장 균형이 좋다.
  - 비교로 2999는 오르기 13 cm 4%였다.
  - 10000, 12000, 19000은 성능이 무너진 checkpoint다. 마지막 checkpoint(19998)도 13 cm 3%였다.
  - 그래서 **마지막 checkpoint가 아니라 평가로 고른 checkpoint**를 쓴다.
  - 사용자가 play 영상으로 보행을 확인했다.
  - 상세 표는 `results/jepa_loco/teacher_cont/summary.md`에 있다.
- **latent 정규화 통계:** 17000 기준으로 다시 계산한다. 2999용
  `teacher_latent_stats_s42.pt`는 재사용하지 않는다.
- **student 재증류:** 4개 비교군(no_memory, gru, gru_jepa, gru_copy)을 다시 학습한다.
  학습 설정은 기존 Phase 3 설정을 그대로 쓴다.
- **2026-10-13 JEPA 정규화 분산 하한 0.05 (사용자 결정):**
  - **증상:** 17000 teacher로 학습한 gru_jepa seed 43이 iteration 300에서 무너졌다.
    - EMA target 분산이 0.0017까지 줄었다.
    - 진단 copy 손실이 2021이었다.
    - latent 손실이 2.46으로, gru의 0.90보다 크게 나빴다.
  - **결정:** 그 run은 중단했다. gru_jepa와 gru_copy는 `--jepa_variance_floor 0.05`로 다시 학습한다.
  - **근거:** 원고 run의 정상 구간 target 분산은 0.115~0.185였다.
    0.05는 그 구간에서 작동하지 않고, 분산이 무너질 때만 증폭을 최대 20배로 제한한다.
  - **범위:** config 기본값은 기존 run 재현을 위해 1e-6으로 둔다.
    gru와 no_memory는 JEPA 손실이 없어서 이 설정의 영향을 받지 않는다.

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

### 2026-10-08 Phase 2 teacher 우선순위 및 공정 비교

**2026-10-08 결과에 따른 대체:** Current+Future는 11~13 cm 고정 계단에서
Current보다 낮아 미리 정한 teacher 기준을 통과하지 못했다. Phase 3
teacher는 Current-EasyStart seed 42 `model_2999.pt`로 고정한다. Wide와
Current+Future는 선택 비교 기록으로 남긴다. 아래의 당시 우선순위와
비교 설계 및 위의 연구 실험 순서 2~3번은 결정 이력이며 현재 student
구현의 목표가 아니다. 현재 Phase 3 기본 비교는 기억 없음/GRU/
GRU+JEPA/GRU+copy의 고정 teacher head 증류다.

- Phase 2의 주목표는 Phase 3 depth student가 증류받을 **Current+Future teacher**
  checkpoint를 만드는 것이다. 학생의 공유 terrain encoder는 `z_current`, `z_future`를
  각각 모방하므로 Current 단독 checkpoint는 대조군이다. Wide는 선택 비교군으로
  뒤로 미룬다. 이 결정이 위의 세 조건 동시 비교 순서를 대체한다.
- 새 task `Unitree-Go2-JepaLoco-OracleCurrentFuture-EasyStart`는
  Current-EasyStart의 보상(swing clearance 1.0, nominal hip −0.7,
  thigh/calf −0.15), EasyStart 지형(워밍업 2행, 계단 0.05~0.20 m,
  gap 0.10~0.30 m), 경로 길이+장애물 통과 게이트, 명령 스케줄,
  종료 조건·이벤트·critic·PPO·100스텝 rollout·레벨별 진단을 상속한다.
  보상 계산에 쓰는 `height_scanner`는 두 조건에서 같다. 관측만 Current의
  현재 187점 패치에서 Current+Future의 현재/미래 각 187점 패치로 확장한다.
  후자는 명령을 0.5 s SE(2) 적분한 pose를 중심으로 한다. 같은
  `187→128→32` terrain encoder를 두 패치에 적용해 policy 입력 terrain
  latent는 Current 32차원, Current+Future 64차원이다.
- 미래 패치는 4.0×3.0 m wide 스캐너(0.1 m 간격)의 값을 보간한다.
  최종 명령 범위 `vx∈[−0.3,2.0]`, `vy∈[−0.4,0.4]`,
  `ωz∈[−1,1]`에서 0.5 s 외삽한 패치의 최대 절댓값은
  x 약 1.950 m < 2.0 m, y 약 1.259 m < 1.5 m다. x 여유는 약
  0.050 m이므로 범위 검사를 테스트로 유지한다. 현재 구현의 범위 밖
  `grid_sample(padding_mode="border")`는 경계값으로 채우지만 확정 명령
  범위에서는 사용되지 않는다. wide 크기와 horizon은 바꾸지 않는다.
- **2026-10-08 gap 광선 보정:** RayCaster가 gap에서 `inf` hit를 반환할 수
  있으므로, 미래 패치 bilinear 보간 **전에** wide height 값을
  `terrain_future` 관측의 clip 범위(현재 −1~1)에 맞춰 유한값으로 만든다.
  `−inf→하한`, `+inf→상한`, `NaN→하한`이며 유한값도 같은 범위로
  clip한다. 보간 전후 clip 범위는 하나의 `terrain_future.scan.clip`
  설정을 공유하고 불일치하면 오류를 낸다. 이 규칙은 기존
  `OracleCurrentFuture`와 새 EasyStart task에 모두 적용한다.
- 본 학습 후보는 seed 42(pilab), 43(Z790), 각 2048 env·3000 iteration이다.
  이 2개 seed 비교는 **teacher 전제 확인**이며 최종 통계 비교가 아니다.
  동일 seed의 Current-EasyStart를 9·10·11·12·13·14·15 cm,
  높이당 128 env, 평가 seed 43, 중심 출발, 0.6 m/s의 고정 오르기
  계단에서 다시 평가한다. 모든 평가에서 지형 origin 검사를 통과해야 한다.
  11~13 cm에서 Current+Future가 Current와 같거나 높은지 확인하고,
  학습 곡선의 레벨별 첫 단 통과율과 최초 통과 iteration 및 play 영상을
  함께 본다. 최종 비교에는 위의 조건당 seed 3개 이상 규칙을 적용한다.

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
   - **2026-10-15 구현:** `flash-linear-attention==0.5.2`와 필수 `fla-core==0.5.2`의
     `fla.layers.mamba.Mamba` 한 층을 쓴다. `mamba-ssm` CUDA kernel은 쓰지 않고
     FLA `slow_forward`를 1토큰 순수 PyTorch decode로 직접 호출한다. 입력
     `u = Linear(x)`(입력 128이면 항등), `y=u+Mamba(RMSNorm(u))`,
     `z=RMSNorm(y)`다. hidden 128, intermediate 256, SSM state 16,
     conv kernel 4, dt_rank auto. 상태는 `[z_last(128), conv(1024), ssm(4096)]`
     총 5248차원이며 reset 시 0, depth 새 프레임에서만 갱신한다.
     GRU는 99,072개, FLA mixer는 116,480개 파라미터다(전체 Mamba 블록
     norm 포함 116,736개, 입력 128 기준). 기본 backbone은 GRU로 유지한다.
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
- **고정 계단 평가 생성 (2026-10-08 점검):** EasyStart 학습용 계단 생성 함수는
  레벨 0·1을 평지로 바꾸므로 1행짜리 고정 높이 평가에 재사용하지 않는다.
  평가는 IsaacLab 원본 역피라미드 계단 생성 함수로 각 열을 지정 높이의
  계단으로 만들고, 시작면 origin z가 `−(단 수+1)×단 높이`인지 실행 전에
  검사한다. 이전 height sweep 중 평지로 생성된 열의 성공률은 계단 성능으로
  해석하지 않는다. 성공 판정 자체는 유지한다.
- depth 렌더링 비용 때문에 env 수는 GPU 메모리를 보고 결정하고, 결정 근거를 기록할 것.

## 7. 구현 단계와 완료 기준

각 단계는 완료 기준을 만족하고 사용자 확인을 받은 뒤 다음으로 넘어간다.

- **Phase 0 — 환경 점검**: 설치된 IsaacLab 버전과 Go2 관련 기존 task/asset, 카메라(TiledCamera 등) API를 **설치된 소스에서 직접 확인**. 기억에 의존해 API를 추측하지 말 것. 카메라를 위 extrinsic으로 붙여 depth 샘플 이미지를 저장.
- **Phase 1 — Teacher**: heightscan 기반 PPO teacher. **보류 (2026-10-01 결정)** — Phase 2 depth PPO 가 명확히 학습되지 않을 때만 teacher + DAgger 를 fallback 으로 추가한다. 반사실 평가(Phase 4)의 분기 실행기가 필요해지면 다시 검토. critic 관측 그룹(heightscan 포함)은 teacher 용으로 재사용 가능하게 유지.
- **Phase 2 — Depth student (GRU, JEPA 없음)**: depth + proprio PPO 기준선을 **먼저** 진행. rough·stairs 학습 curve 와 평가 결과 확보. task `Unitree-Go2-JepaLoco-DepthGRU`.
- **Phase 3 — JEPA 보조손실 (GRU, 온라인)**: Predictor + EMA Target 추가. 조건 c 계산 로직은 단위 테스트 필수(몸통 좌표계 변환, yaw wrap-around). 실현 변위 (Δx, Δy, Δyaw) 히스토그램을 지형별로 출력해 primitive 주변 샘플 공백 확인.
- **Phase 4 — 반사실 평가 도구**: 지형별(비정형, 오르는 계단, 내려가는 계단, gap) 저장 상태 500개 × K개 primitive를 teacher로 H 동안 분기 실행 → (ẑ, z̄) 쌍 생성. 지표: retrieval top-1 정확도(우연 1/K), 예측/실제 거리 행렬 Spearman 상관, 조건 셔플 시 오차 증가량. 발 주변 heightmap linear probe(속도 구간별 보고: v ≥ 0.76 m/s는 현재 관측 기반, 미만은 메모리 기반 예견).
- **Phase 5 — 후보 입력 정책**: K개 예측 latent attention pooling을 정책 입력으로. 비교군: (a) 예측기 없음, (b) 무조건부 latent 1개, (c) K개 조건부 latent. 지표: 지형별 성공률, 계단 모서리 발끝 충돌 횟수.
- **Phase 6 — Mamba 교체 (2026-10-15 갱신)**: `mamba-ssm` 설치 계획을 대체한다.
  `pip install flash-linear-attention==0.5.2 fla-core==0.5.2 --no-deps`로 설치하고
  torch 2.7.0+cu128 버전이 보존되는지 확인한다. FLA Mamba-1의 순수 PyTorch
  1토큰 경로를 사용한다. 첫 비교는 같은 teacher 17000, depth-only 증류,
  같은 CNN·평가 조건의 GRU 대 Mamba다. `proprio_input`은 이 첫 run에서 끈다.
- **Phase 7 — 학습 절차 비교**: 오프라인 사전학습 + freeze vs 온라인 보조손실.
- **Phase 8 — 배포 준비**: Orin 기준 K ∈ {1, 4, 8, 16}별 추론 지연을 GRU / Transformer / Mamba로 측정. ONNX/TensorRT 내보내기.

## 8. 작업 규칙

- 설정값은 IsaacLab 관례(configclass)를 따르고, 이 문서의 숫자는 하드코딩하지 말고 config로 노출.
- 새 모듈마다 shape 테스트와 최소 단위 테스트 작성.
- 1시간 이상 걸리는 학습은 실행 전에 명령과 예상 소요 시간을 보여주고 확인받을 것.
- 로깅: TensorBoard(또는 wandb). PPO 지표, `L_pred`, `L_ctr`, latent 표준편차, 커리큘럼 단계를 항상 기록.
- 커밋은 작게, 단계별로. 실험 결과와 결정 사항은 `docs/experiment_log.md`에 날짜와 함께 기록.
- 이 문서의 설계와 충돌하는 변경이 필요하면 먼저 이유를 설명하고 이 문서를 갱신할 것.
