# 실험 로그 — JEPA-Locomotion (CLAUDE.md 명세)

## 2026-10-01 — Phase 0 환경 점검 (Z790)

**설치 확인 (소스에서 직접):**
- IsaacLab `0.54.4` (저장소 `~/graduation/IsaacLab` VERSION 2.3.2, 커밋 858234d06e), Isaac Sim `5.1.0`,
  rsl-rl-lib `5.0.1`, torch `2.7.0+cu128`. 로컬 GPU RTX 5070 Ti 16 GB.
- Go2 자산: `UNITREE_GO2_CFG` (`assets/robots/unitree.py`, USD `unitree_model/Go2/usd/go2.usd`),
  기존 task `Unitree-Go2-Velocity`. URDF `unitree_ros/robots/go2_description` 에 D435i 링크 없음.
- 카메라: `TiledCameraCfg` — depth 타입 `distance_to_image_plane`(`depth` 별칭),
  `depth_clipping_behavior ∈ {max, zero, none}`, `OffsetCfg(convention="world")` = 전방 +X·상방 +Z,
  `PinholeCameraCfg.from_intrinsic_matrix(...)`. 카메라 월드 포즈 로그는 `update_latest_camera_pose=True` 필요.

**URDF FK (기본 관절각 hip ±0.1 / thigh 0.8·1.0 / calf −1.5):**
앞발 base 기준 (0.178, ±0.173, −0.300), 뒷발 (−0.271, ±0.172, −0.291), 발 구 반지름 0.022
→ 명목 base 높이 0.322 m. 카메라–앞발 0.15 m(가정) 적용 시 카메라 base x = 0.328.

**카메라 모델:** 정사각 픽셀, HFOV 87° → 848×480 에서 VFOV 56.5° (명목 58°보다 1.5° 좁음).
지면 시작 거리 0.235 m (명세 0.23 m 와 일치).

**depth 샘플** (`scripts/jepa_loco/phase0_depth_sample.py`, `results/jepa_loco/phase0/`):
평지·오르는 계단(14 cm)·내려가는 계단(14 cm)·gap(20 cm) 4종 렌더 정상.
- ⚠️ PD(Kp 25) 기본 자세 유지 시 base 실측 높이 **0.280 m**(FK 0.322 보다 4.2 cm 처짐)
  → 카메라 실측 높이 **0.339 m** (명세 0.38). pitch 실측 29.75°.
- 하단 중앙 depth 0.3528 m = 높이 0.339 기하 예측 0.351 과 일치 → extrinsic 적용 확인.

**버그 수정 이력:** (1) env 없이 InteractiveScene 만 쓰면 로봇이 지형 원점으로 이동하지 않음 → 직접 root state 기록,
(2) far 초과를 0 으로 두면 `clamp(0.28, 2.0)` 이 0 을 최근접 0.28 로 바꿈 → 렌더링은 `clipping="max"`, 0 은 결손 전용.

**미결 (사용자 확인 필요):** 카메라 마운트 기준 높이(명목 0.322 vs 시뮬 실측 0.280), 카메라–앞발 거리 실측,
결손 픽셀(0)의 정규화 표현과 다운샘플 방식(848×480 → 96×64 는 종횡비 1.77 → 1.5).

## 2026-10-01 — 카메라 장착·전처리 결정 반영

**결정 (사용자):** 카메라 = base 고정 오프셋(지면 0.38 m 명세 삭제, hip 축 기준 실측 → `camera_mount_from_hip.py`),
depth 원점 = 왼쪽 IR, 랜덤화 위치 ±2 cm·pitch ±3°. 입력 = 112×64 2채널(정규화 depth, mask), 크롭 없음,
시뮬은 intrinsic 스케일 직접 렌더링, 실기는 mask-aware nearest/median(bilinear 금지). 왼쪽 결측 띠 = baseline·fx/Z.

**검증:** 112×64 직접 렌더링 평지 depth = 해석해(h 0.3392, pitch 29.75°, f 평균 59.29) 와 0.1 mm 이내 일치.
단위 테스트 14개 통과(2채널 규칙, 계단 모서리에서 중간값 미생성, 결측 처리).

**기록만 (해결하지 않음):**
- PD Kp 25 / Kd 0.5, 지연 0. 정지 자세 base 0.279 m vs FK 0.322 m (−4.3 cm), pitch −0.64°.
  중력·tracking error·compliance 로 인한 정상적 차이로 보고 넘어간다.
- **렌더링 지연:** 렌더 depth 가 물리 상태보다 몇 프레임 늦고 고해상도일수록 더 늦다(정성 관찰).
  → IsaacLab 고유 지연 d0 를 측정해 인위 지연은 d_add = d − d0 로 줘야 한다(그냥 1~3 스텝을 더하면 실제 2~4 스텝).
- **KNOWN ISSUE (진단 중단, time box):** 848×480 검증 카메라의 런타임 월드 pose 가 112×64 카메라와 다르다.
  - projection 은 원인 아님: 런타임 K = 59.293 / 446.803 (fx=fy, cx·cy 중앙), HFOV 86.73° / 87.00°, VFOV 56.71° / 56.49°
    (종횡비 1.75 vs 1.767 로 인한 VFOV 차 0.23° — 관측 편차를 설명 못 함). 역산 코드도 각 해상도 K 를 썼다.
  - 848×480 영상은 **자기 보고 pose 와는 일치**(해석해 오차 ≤ 4.6 mm) → 렌더링 자체는 정상, pose 가 다름.
  - 오프셋은 t=0 부터 상수: z +0.0097 m, pitch −1.675° = base 원점 기준 1.675° 추가 회전(0.33·sin1.675° = 0.0097).
    시간 추종은 정상(오래된 프레임 아님), 단독으로 띄워도 동일, 렌더 오류 로그 없음.
  - reset 전 USD xform 스택(translate/orient/scale, local/world)은 두 카메라 완전히 동일 → 런타임에 갈라짐. 원인 미상.
  - 영향: 정책 경로(112×64 직접 렌더링)는 해석해와 일치하므로 무관. 실기 변환 함수의 **시뮬 내** 검증만 막힘
    (단위 테스트로는 검증됨). 실기 데이터로 `compare_depth_stats.py` 를 돌릴 때 재검토.

## 2026-10-01 — Phase 2 depth GRU PPO 구현 (Z790 스모크)

**결정 (사용자):** CNN 프레임 인코더(GRU·Mamba 공통), 프레임 단위 GRU(스택 없음), unroll 20 depth 프레임,
teacher·DAgger 보류(fallback), JEPA 는 기준선 curve 확보 후.

**구현:** `Unitree-Go2-JepaLoco-DepthGRU`. actor = `DepthRecurrentActor` (CNN 2→16→32→64 → 128, GRUCell 128,
proprio 정규화 45 + z 128 → MLP 512-256-128 → 12). critic = MLP (proprio + 선속도 + 토크 + heightscan).
`depth_fresh` 플래그로 GRU 를 10 Hz 에만 갱신. rollout 100 스텝 = BPTT 길이.
렌더링은 `render_interval` = 20 물리 스텝(10 Hz), 카메라 `update_period=0` (env reset 으로 인한 갱신 위상 어긋남 방지).
reset 이전에 찍힌 렌더는 버린다. latent std 를 `Loss/latent_std_*` 로 기록.

**검증:**
- 단위 테스트 20개. rollout 스텝별 latent 와 update 의 trajectory 재계산이 중간 reset 포함 atol 1e-5 일치.
- 스모크 학습(32 env, 3 iter) 정상 완주, 752 steps/s (5070 Ti).
- **d0 = 0**: 렌더 직후 base 0.15 m 순간이동 → 다음 렌더의 중앙 depth 변화 0.285 m (기하 예측 0.30 m).
- fresh 간격 3–7 스텝(주기 5 ± 추가 지연 1–3), reset 직후 프레임 전까지 전부 mask 0.
- 경계 결측을 절대 차(0.15 m)로 했더니 먼 지면이 원근 때문에 점무늬 결측 → clip 된 depth 의 상대 차(0.2)로 변경.
- 8 env 관측 영상 정상(계단 모서리·낙차·왼쪽 띠·hole). 첫 점검에서 위아래가 뒤집혀 보인 env 0 영상은 재현 안 됨
  (넘어진 로봇으로 추정, 당시 자세 미기록 — 이후 점검 그림에 자세·지형 표기).

**명세와 다른 점 (확인 필요):** 지형 커리큘럼을 IsaacLab 표준 `terrain_levels_vel`(종류 혼합, 난이도 행 진행)로 구현.
명세는 종류별 단계 진행(평지 → 비정형 → 오르는 계단 → 내려가는 계단 → gap).

## 2026-10-01 — pilab 처리량 측정 · env 수 결정

pilab (RTX PRO 6000, 97 GB), 2048 env, rollout 100 스텝: **16,666 steps/s**, iteration 12.3 s
(수집 9.4 s / 학습 2.9 s — 렌더링·물리가 지배). → 1000 iter ≈ 3.4 h, 1500 ≈ 5.1 h, 3000 ≈ 10.2 h.
env 수 2048 근거: rollout depth 저장 ≈ 2048×100×57 KB ≈ 12 GB (+ 미니배치 패딩 복사본) 로 메모리 여유,
iteration 당 20만 스텝. 4096 은 처리량 이득 미측정.

## 2026-10-01 — OOM 수정 (gru_baseline_s42, iter 22)

pilab 2048 env 본 학습이 iter 22 에서 48.85 GiB 단일 할당 OOM. 원인: rsl_rl `recurrent_mini_batch_generator` 가
rollout 전체 trajectory 를 한 번에 rollout 길이(100)로 패딩. trajectory 수 = env 수 + 에피소드 종료 수라
(추정 ~8,500) env 수 기준으로 잡은 메모리 추정(12 GB)이 틀렸다.
1차 수정(미니배치별 패딩 + depth fp16)도 iter 30 에서 17.5 GiB OOM — 512 env 미니배치에 trajectory 약 6,500
(env 당 100 스텝에 12회 이상 reset, 스폰 직후 넘어짐 반복 env 존재 추정). **패딩 방식 자체가 depth 에 부적합.**
최종 수정: `ResetMaskRolloutStorage` — 패딩·복사 없이 env 시퀀스 [100, B] 를 그대로 쓰고 `traj_start` 로
에피소드 경계에서 GRU 상태 0. 메모리는 넘어짐 횟수와 무관. rollout latent 재현 테스트(중간 reset, 이월 상태) 통과.
depth fp16 저장은 유지.
**확인할 것:** 스폰 직후 반복 넘어짐 env 가 있는지(지형별 종료 통계) — 학습 curve 를 볼 때 같이 본다.

## 2026-10-02 — gru_baseline_s42 결과 (3000 iter, pilab, 약 8.3 h)

그림: `results/jepa_loco/figs/gru_baseline_s42_curves.png`. play 영상: `results/jepa_loco/videos/gru_baseline_s42_play.mp4`.

- **쉬운 지형에서는 걷는다**: 최고점(iter ~1300) 에피소드 길이 809/1000, lin vel 추종 1.00/1.5, play 에서 네 발 보행.
- **지형 커리큘럼 진행 실패**: terrain level 최대 0.52 / 9 (마지막 200 iter 평균 0.40). 넘어짐 종료 비율 바닥도 약 30%.
- **붕괴 3회**: iter 0–120 (넘어짐 99%, 에피소드 길이 20), ~1600, 2500–2750 (보상 −150 스파이크).
- **원인 1 (초기 붕괴) — 자살 균형**: iter 5–15 에 추종 보상 0.30 vs 페널티 합 약 1.3 → 에피소드 누적 보상이 음수라
  빨리 넘어질수록 손해가 적다. 이 보상 설정에는 종료 페널티가 없고, 원래 task 의 명령 커리큘럼(±0.1 에서 시작)을
  내가 고정 범위로 바꿔 초기 추종 보상이 낮아진 것이 겹쳤다. 앞선 OOM(trajectory 폭증)도 이 구간에서 났다.
- **원인 2 (후반 붕괴) — value 발산**: iter ~1400 부터 value loss 1e-2 → 1e3, 이어 action std 0.32 → 0.62,
  action_rate 페널티 3배, 추종 절반. value 발산의 근본 원인은 미확인(보상 이상치 −150 관찰).
- latent z std 0.72 로 증가, 죽은 차원 0 — 표현 붕괴는 없음.

## 2026-10-02 — Phase 2 재학습안 확정 및 구현

**결정 (사용자):** 넘어짐 종료 보상 `is_terminated` weight −200 (dt 0.02 적용 시 −4, 시간 초과 제외),
명령 범위는 세 축 ±0.1에서 시작해 전역 iteration 500까지 명세 범위로 선형 확장한다.
성능 적응형 스케줄은 depth/blind 비교를 오염시키므로 사용하지 않는다. entropy 계수 0.01과 학습률은 유지.
depth와 카메라가 없는 proprio-only GRU 대조군을 같은 보상·명령·지형·PPO 설정으로 각각 3000 iteration 학습한다.
보상 항목별 스텝 최솟값·최댓값과 rollout 중 critic value 최솟값·최댓값을 TensorBoard에 기록해 −150 보상 스파이크를 진단한다.

**구현:** `ScheduledVelocityCommand`는 전역 제어 스텝으로 명령 범위를 계산하고 재샘플 직전에 적용한다.
`JepaDiagnosticEnv`가 항목별 스텝 보상 극값을 PPO로 보내며, `LatentLoggingPPO`가 iteration 내 극값을 축적한다.
blind task는 depth 카메라·depth 관측을 만들지 않고 proprio GRU를 제어 주기마다 갱신한다.
단위 테스트 27개 통과(신규 5개 포함), Python compileall 통과. 일반 작업 샌드박스에서는 GPU가 보이지
않았으나, 승인된 GPU 접근으로 Z790의 RTX 5070 Ti와 `env_test`를 사용해 두 task를 각각 8 env·1 iteration
스모크 실행했다. 둘 다 정상 종료했다. 저장된 env 설정에서 blind는 `d435i: null`, `depth: null`,
`depth_fresh: null`이며 depth task에는 카메라와 depth 관측이 존재한다. TensorBoard에 보상 항목별 극값과
value 범위가 기록됐다. blind 스모크의 `reward_min/termination`은 −4.0으로 설정한 종료 페널티가 실제
적용됨을 확인했다. 3000 iteration 본 학습은 pilab에서 실행 예정이다.
학습 결과와 play 판정은 본 학습 후 추가 기록한다.

## 2026-10-02 — 기억 없는 blind 대조군 추가, 스케줄 검사, resume 수정 (Z790)

**배경 (사용자 결정):** pilab 에서 depth GRU 와 proprio-GRU blind 를 각 3000 iteration 학습 중(중단하지 않음).
현재 blind 는 proprio 를 50 Hz GRU 에 넣지만 depth 정책은 proprio 를 MLP 에 직접 넣고 GRU 에는 depth 특징만 넣는다.
→ 두 런의 차이에 "depth 유무"와 "proprio 기억 유무"가 섞인다. BlindGRU 결과는 **proprio 기억이 있는 강한 대조군**으로 보관.

**추가:** `Unitree-Go2-JepaLoco-BlindMLP` — depth 정책의 MLP head(512-256-128)에서 입력 z_t 만 뺀 [proprio 45] → 12.
env 는 BlindGRU 와 같은 카메라 없는 `JepaBlindEnvCfg`, PPO·critic·rollout(100)은 depth 와 동일. 기존 BlindGRU 는 변경 없음.
CLAUDE.md §6 대조군 설명 갱신.

**검사·수정:**
- 명령 스케줄 `steps_per_iteration` ≠ PPO `num_steps_per_env` 이면 `LatentLoggingPPO.construct_algorithm` 에서 ValueError.
  스모크: `agent.num_steps_per_env=24` override → 첫 iteration 전에 오류 확인.
- `--resume` 시 `common_step_counter` 가 0 으로 돌아가 명령 스케줄이 ±0.1 부터 재시작하던 문제:
  `train.py` 가 load 후 `common_step_counter = checkpoint iteration × num_steps_per_env` 로 복원하고 전 env 명령을 재샘플.
  IsaacLab 0.54.4 `command_manager.reset(None)` 은 slice 에 len() 을 호출해 실패하므로 env id 를 명시.
  스모크: model_2.pt 에서 resume → `common_step_counter = 200 (iteration 2)`, iteration 2→3 정상 진행.
  **이 수정 이전 코드(현재 pilab 학습 프로세스 포함)로는 resume 하지 않는다.**

**테스트 재현 명령 (Z790, env_test):**
- 전체 32개 (Isaac 앱 필요): `python scripts/jepa_loco/run_tests_in_app.py --headless` → `32 passed`, `PYTEST_EXIT_CODE=0`
- 앞선 기록의 "27개 통과" 재현 (신규 2파일 제외):
  `python scripts/jepa_loco/run_tests_in_app.py --headless --ignore=$PWD/source/unitree_rl_lab/unitree_rl_lab/jepa_loco/tests/test_checks_blind_mlp.py --ignore=$PWD/source/unitree_rl_lab/unitree_rl_lab/jepa_loco/tests/test_blind_mlp_cfg.py` → `27 passed`
- 앱 없이 `python -m pytest` 로는 `test_baseline_v2.py`, `test_blind_mlp_cfg.py` 가 isaaclab(pxr) import 로 수집 실패한다. 나머지 25개는 앱 없이 통과.

**스모크 (Z790, 64 env):** BlindMLP 3 iteration 정상 완주. actor 입력 45 → 512-256-128 → 12, `reward_min/termination` −4.0.

## 2026-10-06 — v2 결과: depth GRU vs blind proprio-GRU (각 3000 iter, seed 42, pilab)

런: `jepa_loco_depth_gru/2026-10-02_10-42-41_gru_depth_v2_s42_tmux`, `jepa_loco_blind_gru/2026-10-02_10-43-10_gru_blind_v2_s42_tmux`
(같은 이름의 10:17/10:18 런은 재시작 전 런 — 비교에 쓰지 않음). 수정 그림: `results/jepa_loco/figs/v2_depth_vs_blindgru_curves_corrected.png`.
기존 `v2_depth_vs_blindgru_curves.png`의 "fall share"는 `bad_orientation`만 세어 `base_contact`를 누락했다.

| 마지막 200 iter 평균 | v1 depth | v2 depth | v2 blind GRU |
|---|---|---|---|
| 에피소드 길이 (/1000) | 771 | 972 | 959 |
| 비시간초과 종료 비율 (`1 − time_out`) | 0.318 | 0.062 | 0.070 |
| terrain level (/9) | 0.40 | 1.06 | 1.08 |
| lin vel 추종 (/1.5) | 1.00 | 1.17 | 1.16 |
| value loss | 1.97 (발산) | 0.024 | 0.029 |

- **v2 수정은 효과가 있었다**: 초기 붕괴·value 발산 모두 사라짐(value loss > 5 인 iter: v1 440 → v2 0).
- **depth와 blind 모두 terrain level ≈ 1/9에서 정체.** 마지막 200 iter에서 depth 평균 보상 15.72, blind 12.39,
  비시간초과 종료 6.2% vs 7.0%로 depth가 약간 앞서지만 단일 seed·성능 의존 지형 커리큘럼이라 depth 효과로 단정할 수 없다.
  `base_contact`는 depth 1.7%, blind 5.6%; `bad_orientation`은 depth 4.6%, blind 1.5%로 실패 유형이 다르다.
- **유력한 제약 — 커리큘럼 구조:** `terrain_levels_vel` 은 env 원점으로부터의 **직선 변위**가 4 m 를 넘어야 승급, 명령 속도 × 20 s × 0.5
  보다 짧으면 강등한다. 그런데 명령이 yaw rate(ωz ∈ ±0.8, heading_command=False)라 로봇이 원을 그리며 걷는다
  (vx 0.5, ωz 0.5 → 반경 1 m, 변위 ≤ 2 m). 잘 걸어도 승급 조건을 거의 못 채운다 → 두 정책 모두 쉬운 지형(계단 ~9 cm, gap ~12 cm)에
  머문다. yaw 명령 때문에 이 기준을 만족할 수 없는 궤적이 존재한다는 것은 코드·기하로 확인되지만,
  이것이 정체의 유일한 원인인지는 궤적·지형별 평가 없이 확정할 수 없다. **이 데이터는 depth의 유용성 또는 무용성을
  입증하지 못한다.** 이 분석 당시 play 영상이 없어 보행 통과 판정은 보류했다.

## 2026-10-06 — v2 depth play 육안 점검 (Z790)

영상: `logs/rsl_rl/jepa_loco_depth_gru/2026-10-02_10-42-41_gru_depth_v2_s42_tmux/videos/play/rl-video-step-0.mp4`
(model_2999.pt, 약 20 s, play cfg 32 env). 사용자 관찰: 내려가는 계단은 일부 통과하지만 올라가는 계단을
오르지 못하며 평지 보행도 부자연스럽다. 영상의 중앙 계단 구간에서 여러 로봇이 아래쪽에 머무는 모습도 확인.
**Phase 2 보행 통과 판정 보류.**

play cfg 는 5개 지형 난이도 행 전체에 스폰하며, 계단 높이는 약 8–20 cm를 포함한다. 반면 학습의 마지막
200 iter 평균 terrain level 은 1.06/9에 머물렀다. 따라서 이 영상만으로 최저 난이도 계단도 실패하는지,
고난도 계단의 분포 밖 실패인지 구분할 수 없다. 계단 오르기/내리기를 낮은 높이부터 동일 명령으로 따로
평가해야 한다. 평지 보행의 자연스러움은 현재 보상 함수의 명시적 목표가 아니며, 높이·pitch·스윙 발 동작의
보상 조정은 원인 분리 평가 후 사용자와 설계를 확정한다.

## 2026-10-06 — v2 고정 오르는 계단 평가 (Z790)

`scripts/jepa_loco/eval_stairs_fixed.py`로 depth GRU와 proprio-GRU blind의 `model_2999.pt`를 평가했다.
각 정책에 9 cm와 19 cm 오르는 계단 16개 환경씩, seed 42, 직진 명령 0.6 m/s,
20 s(1000 제어 스텝)를 적용했다. 중앙 플랫폼의 첫 단(x=1.2 m) 앞 x=1.0 m에서 출발하고,
상단 진입(x≥2.9 m와 몸체 상승 ≥ 전체 단 높이의 90%)을 25 스텝 유지하면 성공으로 센다.
실제 명령 텐서는 `[0.6, 0, 0]`으로 확인했다. 이 성공 규칙은 보행 영상 판정을 보조하는
위치·높이 대리 지표다.

| 정책 | 단 높이 | 성공 | 최대 전방 위치 평균 (원점 기준) | 비시간초과 종료 |
|---|---:|---:|---:|---:|
| depth GRU | 9 cm | 0/16 | 1.03 m | 0/16 |
| depth GRU | 19 cm | 0/16 | 1.07 m | 0/16 |
| blind GRU | 9 cm | 0/16 | 1.09 m | 0/16 |
| blind GRU | 19 cm | 0/16 | 1.09 m | 0/16 |

모든 조건에서 출발 이후 몸체 높이 증가가 없었다. 최대 전방 위치도 첫 단(x=1.2 m)보다
0.1 m 이상 앞에서 멈춰, 이번 실패는 계단 중간에서 넘어진 것이 아니라 **첫 단 접근·진입 실패**로
분류한다. 이는 사용자의 play 영상 관찰(오르는 계단 실패)과 일치한다. 몸체가 한 번도 상승하지
않았으므로 19 cm가 물리적으로 너무 높은지, 보상 중 수직 속도·자세 페널티가 발 들기를
억제하는지 등의 원인은 이 평가만으로 분리할 수 없다. 두 정책 모두 실패했으므로 이 실험은
depth 유효성 비교 근거가 아니다. 평지 보행과 첫 단 앞 정지의 원인은 별도 궤적·영상 점검이 필요하다.

원시 결과: `results/jepa_loco/stair_eval/depth_gru_v2_s42.json`,
`results/jepa_loco/stair_eval/blind_gru_v2_s42.json`.
Isaac 앱 내부 단위 테스트 34개 통과(`PYTEST_EXIT_CODE=0`).

### 계단 정지 원인 진단

기하 정정: IsaacLab `inverted_pyramid_stairs_terrain`은 단 수를
`floor((size − 2×border − platform_width)/(2×step_width)) + 1`로 정한다.
8 m 패치, border 1 m, platform_width 3 m, step_width 0.3 m에서 6단이며,
**실제 중앙 평지 반폭은 1.2 m**이다. 위 초기 평가의 x=1.0 m 스폰은 앞발 중심이
x≈1.18 m에 놓여 첫 수직 단 바로 앞에서 시작했다. `platform_width=3 m`를
실제 평지 폭으로 해석해 첫 단을 x≈1.5 m로 적었던 설명을 위에서 정정했다.

같은 depth 체크포인트·0.6 m/s 명령으로 평지에서는 20 s 동안 최대 전방 위치 평균
12.54 m(출발 x=1 m), 901스텝의 전진 속도 약 0.59 m/s였다.
따라서 체크포인트 로딩·명령 전달 오류는 아니다. 중앙 x=0 m에서 출발하는 계단 진단은 다음과 같다.
앞발 높이는 스폰 직후 안정화 50스텝을 제외한 **각 환경의 에피소드 최고치**이며,
표의 괄호 값은 16개 환경 중 최고치다. 높이는 중앙 평지 바닥 기준 앞발 링크 중심 높이다.

| 정책 | 단 높이 | 최대 몸체 x 평균 | 앞발 중심 최고 높이 평균 (환경 최대) | 상단 성공 |
|---|---:|---:|---:|---:|
| depth GRU | 9 cm | 0.99 m | 5.2 cm (6.3 cm) | 0/16 |
| depth GRU | 19 cm | 0.68 m | 4.2 cm (5.7 cm) | 0/16 |
| blind GRU | 9 cm | 1.06 m | 4.8 cm (5.7 cm) | 0/16 |
| blind GRU | 19 cm | 1.06 m | 4.9 cm (5.7 cm) | 0/16 |

blind는 첫 단(x=1.2 m)까지 걷다가 앞발 중심이 x≈1.18 m에서 멈춘다.
depth도 9 cm에서는 유사하게 접근하고, 19 cm에서는 더 일찍 감속한다.
평지 depth 보행의 앞발 중심 최고 높이는 평균 6.1 cm(환경 최대 7.2 cm)로,
계단 앞에서도 이 높이를 늘리지 않는다. **관측된 직접 실패 기전은 앞발이 첫 단 높이에
못 미쳐 단에 막힌 뒤 정지하는 것**이다. 몸체가 오르지 못해 상단 성공률은 0이다.

depth 9 cm 정지 상태(901스텝)의 보상은 스텝당 평균 +0.0195다.
속도 추종 항은 약 +0.38/s로 낮아졌으나 yaw 추종 항이 약 +0.75/s이고
관절·행동 페널티를 뺀 전체 보상이 여전히 양수다. 평지 보행은 스텝당 약 +0.0356이다.
따라서 정지는 최적 행동은 아니지만, 장애물 앞에서 발을 높이 들도록 직접 유도하는 항이
없는 현재 보상 아래에서 안정적인 지역 해가 될 수 있다. 계단 학습 노출 부족, 수직 속도·자세
페널티의 상대적 영향은 이 진단만으로 분리되지 않았다. depth와 blind 모두 실패하므로
이번 직접 실패 기전은 depth 센서만의 문제가 아니다.

진단 원시 결과: `results/jepa_loco/stair_eval/depth_gru_v2_s42_flat_diag.json`,
`results/jepa_loco/stair_eval/depth_gru_v2_s42_center_diag.json`,
`results/jepa_loco/stair_eval/blind_gru_v2_s42_center_diag.json`.

## 2026-10-06 — Privileged oracle 경로 구현 및 짧은 검증 (Z790)

**설계 결정:** 사용자가 Mamba context, EMA JEPA, 공유 terrain head, command-extrapolated
future heightmap teacher, real Depth 적응을 포함하는 새 연구 구조를 제시하고 구현을 요청했다.
이에 따라 `CLAUDE.md` 맨 앞에 새 우선 명세를 추가했다. K=8 primitive와 InfoNCE는 종전
연구안으로 보관하고 새 모델의 주 경로에서는 쓰지 않는다. 성능 기반 terrain curriculum을
고정 스케줄로 바꾸는 변경은 아직 결정되지 않아 적용하지 않았다.

계단 첫 단을 오르지 못하는 GRU 기준선 결과를 바탕으로, 제안된 새 연구 구조의 선행 조건인
`Current`, `Wide`, `Current+Future` heightmap oracle PPO task를 구현했다. 세 task 모두
카메라 없이 기존 proprio·보상·명령·지형·critic을 공유한다. `Current+Future`는 현재
명령을 0.5초 동안 SE(2) 적분한 예상 pose 중심의 future patch를 현재 위치의 wide
RayCaster heightmap에서 읽는다. Current/Future는 동일한 187→128→32 terrain encoder를
공유하며, Wide는 1271→128→64로 policy에 들어가는 총 terrain latent 64차원을 맞춘다.
스캐너 범위, 예측 horizon, latent 차원은 configclass에 노출했다.

별도 순수 함수로 command pose 외삽, 실제 미래 pose·명령 변화·에피소드·가시성을 확인하는
future sample mask, 유효 표본 수로 나누는 latent MSE를 구현했다. 아직 이 함수들을 학습에
연결한 student/JEPA는 없다. `eval_stairs_fixed.py`가 세 oracle task를 허용하도록 확장했다.

Isaac 앱 내부 테스트 **45개 통과**(회전된 미래 heightmap 기하 테스트 포함). Z790 RTX 5070 Ti에서 각 oracle task 8 env·1 iteration
PPO 스모크가 종료 코드 0으로 완주했다. 이것은 관측 생성과 최적화 연결만 확인한 것이며,
계단 등반 성능은 아직 검증되지 않았다. 단일 명령 predictor, JEPA MSE, teacher-first
단계는 새 우선 명세에 반영했다. 고정 terrain schedule은 여전히 별도 결정이 필요하다.
현재 oracle 코드는 검증용 구현으로 두고 1시간 이상 본 학습은 실행하지 않았다.

## 2026-10-06 — Oracle 경로 길이 커리큘럼 결정·구현 (Z790)

**사용자 결정:** 성능 기반 승급은 유지한다. 다만 yaw-rate 명령에서 원 운동을 해도
승급할 수 있도록 시작점 직선거리 대신 **에피소드 XY 경로 길이**로 판정한다.
Current를 먼저 본 학습해 고정 9 cm 오르는 계단의 평가·play 영상을 확인한다. 통과 시
Wide와 Current+Future를 같은 규칙·샘플 예산·seed 집합으로 추가하며 조건당 3개 이상
seed를 쓴다. 최종 비교는 지형 종류 × 난이도 × seed 고정 평가 세트로 한다.

**구현:** `OracleCurriculumEnv`가 매 제어 스텝 이동량을 누적하고 reset 직전 마지막
이동을 포함한다. reset으로 인한 위치 순간이동은 누적하지 않는다. 기존 IsaacLab의
승급 `> 패치 길이/2`, 강등 `< 명령 선속도 × 최대 에피소드 시간 × 0.5` 및 승급 우선 규칙은
동일하게 유지한다. 세 oracle task가 이 환경과 동일 curriculum term을 사용한다.
TerrainGenerator seed를 환경 seed로 명시하여 같은 seed의 세 조건이 동일한 지형을
생성한다. TensorBoard에 종류별 평균 terrain level 다섯 개와 평균 누적 경로 길이를 기록한다.

**검증:** Isaac 앱 내부 테스트 **52개 통과**. Current oracle 8 env·1 iteration 스모크
완주. 해당 스모크에서 평균 누적 경로 길이 0.4063 m, 종류별 terrain level 5개가
로그에 기록됐다. Wide와 Current+Future도 각각 8 env·1 iteration 스모크 완주했고
경로 길이 및 종류별 terrain level 5개가 모두 기록됐다. 고정 계단 평가에서는 curriculum을 꺼서 동일 지형을 유지하고,
선택적으로 env 0 영상을 저장할 수 있게 했다. 아직 장시간 학습이나 등반 판정은 하지 않았다.
스모크 체크포인트(`model_0.pt`)로 2 env·100스텝 고정 계단 평가를 실행해 JSON과 MP4 저장을
확인했다. 영상 1초 프레임에서 로봇과 계단이 함께 보인다. 이 checkpoint는 학습 전이므로
해당 평가 수치는 정책 성능으로 해석하지 않는다.

## 2026-10-06 — OracleCurrent 본 학습 (경로 길이 커리큘럼), seed 42 (pilab) · seed 43 (Z790)

처리량: Z790 2048 env 11.5–11.9 s/iter(약 17,500 steps/s, 안정). pilab은 다른 사용자 작업과 GPU 경합으로 8–75 s/iter 변동.

**seed 43 (Z790) 중간 결과 — iter 약 900 (2:41 경과):** terrain level stairs_up **5.52**(단 약 15 cm),
stairs_down 4.67, gap 5.60, rough 6.01, flat 6.12. bad_orientation 2.9%, 에피소드 길이 981.
1:04 경과 시점(stairs_up 0.70, stairs_down 0.17)에서 크게 상승 — v2(전체 평균 레벨 약 1에서 정체)와 달리 계단 커리큘럼이 진행된다.
커리큘럼상 진행일 뿐이며 고정 9 cm 평가·play 영상 판정 전에는 등반 통과로 보지 않는다.

**중단·재개:** iter 901에서 사용자가 실수로 Ctrl+C. `model_900.pt` 에서 `--resume` (2100 iter 추가, 같은 run_name·seed).
`[INFO] resume: common_step_counter = 90000 (iteration 900)` 확인 — 명령 스케줄 복원 정상.
**지형 레벨은 env 상태라 checkpoint 에 없어 0 부터 다시 시작**(재개 직후 stairs_up 0.02) → 레벨 곡선에 재개 구간 하락이 생긴다.
분석 시 원 런(`2026-10-06_14-32-11_oracle_current_path_s43`)과 재개 런을 이어 붙이고 재개 지점을 표시한다.

**seed 43 완료 및 고정 계단 평가 (Z790, 2026-10-06 23:32):** 재개 런이
`model_2999.pt`까지 완료됐다. 마지막 200 iteration 평균은 stairs_up terrain level
5.55, stairs_down 5.62, rough 5.80, gap 5.70, 에피소드 길이 992/1000,
time_out 97.9%, value loss 0.014. 재개 시 레벨 초기화가 있었으므로 연속 3000 iteration
학습과 동일한 지형 노출 이력으로 해석하지 않는다.

`eval_stairs_fixed.py`로 중심 x=0 m에서 시작하는 고정 0.6 m/s 직진 명령,
seed 43, 높이별 16 env, 1000 제어 스텝 평가와 env 0 영상 기록을 수행했다.
첫 단은 x=1.2 m, 단 높이는 각각 9 cm·19 cm다.

| 고정 평가 | 9 cm 오르는 계단 | 19 cm 오르는 계단 |
|---|---:|---:|
| 상단 도달 | 0/16 | 0/16 |
| 몸체 최대 x 평균 | 1.04 m | 0.97 m |
| 안정화 50스텝 이후 앞발 중심 최고 높이 평균 | 4.6 cm | 4.3 cm |
| 안정화 이후 앞발 중심 최고 높이 환경 최대 | 5.1 cm | 4.7 cm |
| 비시간초과 종료 | 0/16 | 0/16 |

**실패 원인 조사 (23:49):** 같은 seed 43 checkpoint로 2 env 진단 평가를 재실행했다.
중심 출발 시 actor의 `terrain_current`는 거의 일정한 −0.10(표준편차 1.1e−6)이었다.
첫 단 직전(101스텝, 몸체 x≈0.97 m)에는 최솟값 −0.554, 최댓값 −0.171,
표준편차 0.120으로 변했다. 따라서 heightscan이 clip 때문에 상수가 된다는 가설은
기각된다. 센서 ray 시작점의 20 m 오프셋은 `sensor.data.pos_w`에는 반영되지 않았고,
관측값은 해당 위치의 지형 높이를 실제로 구별했다.

고정 평가에서 몸체는 첫 단 x=1.2 m 앞의 약 x=0.97 m에 멈춰 남은 에피소드를
거의 정지 상태로 보냈다(201스텝 vx≈−0.035 m/s, 901스텝 vx≈0).
정지 중에도 총 보상은 약 +0.0196/스텝, 시간초과 종료이며 넘어짐은 없었다.
앞발 중심 최고 높이는 첫 단 높이 9 cm보다 낮다. 즉 계단을 보았지만 오르는
발동작은 학습되지 않았다.

현재 승급은 에피소드 **XY 경로 길이 >4 m**만 검사한다. 오르는 계단의 출발
중앙 플랫폼 폭은 3 m이므로, 그 안에서 회전하거나 오가며 4 m를 이동해도
첫 단을 밟지 않고 승급할 수 있다. `stairs_up` 평균 레벨 5.55는 실제 등반
성공률의 대리 지표가 아니었다. 이 메커니즘은 코드와 지형 기하로 확인됐지만,
학습 중 개별 궤적을 저장하지 않아 각 승급이 실제로 이런 경로였는지는 알 수 없다.
실패의 직접 관측은 첫 단 앞 정지이며, 주된 설계 원인은 승급 판정이 등반을
요구하지 않는 점이다. 속도 추종만으로는 단을 올라야 할 유인이 충분하지
않을 수 있으나 그 상대적 기여는 별도 실험이 필요하다.

영상 2초·8초 프레임에서 env 0 로봇이 첫 단 앞에 서 있으며, 발을 단 위로 올리지 못한 채
정지한다. 수치상 몸체 x≈1.04 m, 앞발 x≈1.18 m로 첫 단 x=1.2 m 앞에서 멈춘 것과 일치한다.
**OracleCurrent는 고정 9 cm 통과 조건에 실패했다.** 따라서 사용자 결정에 따라
Wide·Current+Future 본 학습은 시작하지 않는다. 높은 학습 중 terrain level은 고정
계단 통과의 증거가 아니다. 경로 길이 판정은 yaw-rate 원 운동에도 승급을 허용하므로,
계단을 실제로 넘지 않고 승급했을 가능성이 있다. 이 가능성은 학습 궤적이 없어 확정할 수 없다.

원시 결과: `results/jepa_loco/stair_eval/oracle_current_path_s43_fixed_stairs.json`,
영상: `results/jepa_loco/stair_eval/oracle_current_path_s43_fixed_stairs_video/rl-video-step-0.mp4`.

## 2026-10-07 — Oracle 장애물 통과 커리큘럼과 원인 진단 로그

사용자 결정: 순변위만으로 Rudin 판정으로 돌아가는 것은 독립적인 yaw-rate 명령 때문에
원형 보행을 잘못 실패로 처리한다. 따라서 경로 길이 조건은 유지하고 **모든 장애물
지형**(오르는 계단, 내려가는 계단, gap)에 동일한 원칙의 실제 통과 게이트를 추가한다.
로봇 몸체가 지형 설정으로 계산한 첫 장애물 너머까지 나가서 10 제어 스텝
연속 머물러야 승급한다. 계단은 첫 단 너머, gap은 틈 너머다.
여유 거리는 0.1 m이며 두 값 모두 configclass 설정이다.
평지와 rough는 기존 경로 길이만으로 승급한다. 경로 길이만 충족했지만 장애물은
넘지 못한 에피소드의 비율과 실제 장애물 통과율을 종류별로 기록한다.

오르는 계단은 첫 단 접근 후 50스텝 연속 저속(≤0.1 m/s)으로 머문 에피소드와,
첫 단 근처에서 바깥 방향 속도 ≥0.2 m/s 또는 첫 단 너머 진행을 보인 에피소드를
분리한다. 두 집단의 **보상 항별 에피소드 합**과 첫 단 근처 swing 발 중심 최고 높이를
기록한다. `forward_attempt`는 속도에 근거한 관측 가능한 대리 분류이며 의도 자체의
직접 측정은 아니다. 보상·최저 계단 높이는 이번 수정에서 바꾸지 않았다.

명령 최종 범위는 사용자 요청대로 vx ∈ [−0.3, 2.0] m/s, vy ∈ [−0.4, 0.4] m/s,
yaw rate ∈ [−1.0, 1.0] rad/s로 갱신했다. 초기 ±0.1 및 500 iteration 선형 스케줄은
유지한다. 세 oracle 조건은 동일 명령·승급 설정을 공유한다. 기존 체크포인트는
재사용하지 않고 Current부터 새 seed 런으로 학습한다.

검증: 순수 함수 테스트 8개 통과, Isaac 앱 내부 전체 54개 통과. OracleCurrent
8 env·1 iteration 학습 스모크와 curriculum을 끈 고정 계단 2 env·2스텝 평가가
정상 종료했다. 장시간 본 학습과 수정 후 9 cm 영상 평가는 아직 진행하지 않았다.
첫 구현의 계단 게이트가 전체 계단 바깥을 요구하는 것을 확인해, 사용자 결정 A에 맞게
**첫 단 너머**로 바로잡았다. 수정 후 순수 함수 8개와 `oracle_first_step_smoke`
(8 env·1 iteration)를 다시 통과했다. 최종 코드로 Isaac 앱 내부 전체 54개도 재통과했다.

**Z790 본 학습 시작 (00:09, 사용자 승인):** tmux 세션
`oracle_current_clearance_s43`에서 OracleCurrent 2048 env·3000 iteration·seed 43을
새로 시작했다. 로그 디렉터리는
`logs/rsl_rl/jepa_loco_oracle_current/2026-10-07_00-09-25_oracle_current_clearance_s43`.
첫 iteration은 15,930 steps/s, 12.86 s/iter, ETA 약 10시간 43분이다.
기존 체크포인트에서 resume하지 않았다. 첫 iteration 보상·지형 레벨은 초기 정책
상태이므로 성능 판단에 사용하지 않는다. 종료 후 고정 9 cm 평가와 play 영상을 확인한다.

**pilab 구버전 OracleCurrent seed 42 결과 반입·고정 평가 (08:59):**
`2026-10-06_13-06-27_oracle_current_path_s42/model_2999.pt`가 반입됐다.
이 런의 최종 명령 범위는 vx 최대 1.0 m/s·yaw 최대 0.8 rad/s이고,
경로 길이만으로 승급하는 구버전이다. 마지막 200 iteration의
`stairs_up` 평균 레벨은 5.56, 시간초과율은 97.8%였다.
seed 42, 중심 출발, 고정 0.6 m/s 직진 명령, 높이별 16 env·1000스텝
평가에서 9 cm와 19 cm 모두 **0/16 성공**했다. 9 cm의 몸체 최대 x 평균은
1.03 m(첫 단 x=1.2 m), 안정화 후 앞발 중심 최고 높이 평균은 3.8 cm였다.
2초·8초 영상에서도 로봇이 첫 단 앞에 머문다.
결과: `results/jepa_loco/stair_eval/oracle_current_path_s42_fixed_stairs.json`,
영상: `results/jepa_loco/stair_eval/oracle_current_path_s42_fixed_stairs_video/rl-video-step-0.mp4`.
구버전 seed 43 결과와 같은 실패 형태다. 새 Z790 게이트 런과는 명령·승급
설정이 달라 학습 곡선 수치를 직접 비교하지 않는다.

**영상·반사실 원인 조사 (09:25):** pilab s42와 Z790 새 게이트 s43 모두 평지
고정 0.6 m/s 명령에서는 20초 동안 10.9 m / 11.1 m 전진했다. 계단에서는
2초 안팎에 첫 단 x=1.2 m 앞에서 정지하고 나머지 시간에는 자세만 조금 흔든다.
Z790 최종 checkpoint도 고정 9 cm / 19 cm 계단 모두 0/16 성공이었다.

| 정책·9 cm 평가 | 정상 Heightscan | Heightscan 평탄화 반사실 |
|---|---:|---:|
| pilab 구버전 s42: 몸체 최대 x 평균 | 1.03 m | 1.16 m |
| pilab 구버전 s42: 앞발 중심 최고 z 평균 | 3.8 cm | 9.9 cm |
| pilab 구버전 s42: 비시간초과 종료 | 0/16 | 5/16 |
| Z790 새 게이트 s43: 몸체 최대 x 평균 | 0.96 m | 1.17 m |
| Z790 새 게이트 s43: 앞발 중심 최고 z 평균 | 4.0 cm | 14.3 cm |
| Z790 새 게이트 s43: 비시간초과 종료 | 0/16 | 1/16 |

모든 경우 상단 성공은 0/16이다. Heightscan을 평탄화하면 오히려 전진과 발 들기가
늘지만 안정적인 등반은 나타나지 않는다. 같은 자세에서 Heightscan만 평탄화하면
평균 절대 관절 명령 변화는 pilab s42의 계단 앞에서 0.184(원래 행동 절댓값 평균
0.511), Z790 s43에서 약 0.23–0.25였다. Heightscan은 계단 형상을 구별하며
정책 행동에 영향을 준다. '높이를 못 봐서 멈춤'은 근거가 약하다.

Z790 새 게이트는 최종 200 iteration에서 `stairs_up` 레벨 0, 첫 단 통과율 0,
경로 길이만 충족한 가짜 승급 후보 비율 평균 0.62였다. 이전의 평균 레벨 5.5는
실제 계단 통과의 증거가 아니었음이 확인됐다. gate는 거짓 승급을 차단했지만
등반 동작 자체를 만들지는 않았다. 첫 난이도는 여전히 8 cm이고, 보상에는
발끝 높이·계단 위 발 착지·첫 단 통과를 직접 보상하는 항이 없다. 평지에서
안정화 후 앞발 높이는 평균 약 4.4 cm이며 정상 계단 평가에서는 4 cm 이하다.
계단 앞 정지 보상은 약 +0.0195/스텝, 넘어짐은 −4이므로 위험한 첫 단 시도를
피할 유인은 있다. 다만 이는 인과 효과의 정량 분해가 아니라 구조와 반사실
평가에서 나온 해석이다. 발을 9 cm 이상 드는 것만으로는 충분하지 않으며
평탄화 반사실에서 더 높이 들고도 몸체는 첫 단을 넘지 못했다.

구버전 pilab s42를 3 cm·5 cm 단으로 평가해도 상단 성공은 각각 0/16이었다.
3 cm에서는 몸체가 첫 단 x=1.2 m 근처까지 진행했으나 일부 비시간초과 종료가
발생했다. 초기 단 높이 완화만으로 등반이 해결된다는 증거는 없다.
`approach_stall` 학습 진단 분류는 새 Z790 정책이 x≈0.86 m에서 멈춘 반면
분류 구간이 x≥0.95 m라 마지막까지 샘플이 사실상 1회였다. 따라서 그
집단의 보상 항별 평균은 원인 판단에 사용하지 않는다. 다음 런용 분류 설정은
접근 여유 0.25→0.40 m(첫 단 x=1.2 m 기준 진단 시작 x=0.8 m)로 수정했다.
이 변경은 진단 로깅에만 적용되며 완료된 학습의 보상·승급에는 영향이 없다.
원인 조사용 평가 옵션(`--probe_terrain`, `--flatten_terrain_input`, 계단 높이 CLI)을
추가했고 최종 수정 후 Isaac 앱 내부 테스트 54개가 통과했다.

주요 원시 결과:
`results/jepa_loco/stair_eval/oracle_current_clearance_s43_fixed_stairs.json`,
`results/jepa_loco/stair_eval/oracle_current_clearance_s43_fixed_stairs_video/rl-video-step-0.mp4`,
`results/jepa_loco/stair_eval/oracle_current_path_s42_flattened_scan.json`,
`results/jepa_loco/stair_eval/oracle_current_clearance_s43_flattened_scan.json`.

**OracleCurrent play 재생 오류 수정 (2026-10-07):** `scripts/rsl_rl/play.py`가
재생 전에 무조건 ONNX export를 호출해, Oracle의 분리된 actor 관측에서
`mat1 1x45` 대 `77x512` 입력 차원 오류로 종료됐다. 정책 재생에는 export가
필요 없으므로 기본 경로에서 건너뛰고 `--export`를 명시할 때만 시도하도록
바꿨다. Z790에서 동일한 최종 checkpoint를 `--headless --video
--video_length 10`으로 로드해 10스텝 재생·MP4 생성까지 확인했다.

**OracleCurrent 지형 기준 스윙 발 높이 보상 준비 (2026-10-07):** 사용자가
기존 조건의 다른 항목은 그대로 두고 스윙 높이 보상 하나만 추가하는 실험을
확정했다. `jepa_loco/envs/rewards.py`에 발밑 유효 ray의 최고 지형 높이를
사용하는 순수 텐서 계산과 Isaac reward-manager 연결을 구현했다. 네 발 중
비접촉·air time 0.05 s 초과인 발만 점수를 얻고, 명령과 실제 몸통 이동
속도 게이트를 곱한다. OracleCurrent에만 켰으며 Wide·Current+Future는
아직 기존 보상이다. Current 통과 후 세 oracle 비교를 할 때 보상을 맞춘다.

가중치 보정에는 이전 seed 43 최종 checkpoint를 **평지·0.6 m/s·32 env·20 s**로
재생해 정상 보행 정책의 값을 썼다. 새 항의 가중치 1 에피소드 합은 +0.3948,
기존 `joint_pos`는 −3.6271, 시간초과율 100%, 평균 전진 거리 12.13 m였다.
따라서 configclass 가중치를 10으로 정했다. 고정 정책에서 예상 새 항 합은
+3.95로, joint_pos 크기와 비슷하다. 원시 측정은
`results/jepa_loco/stair_eval/oracle_current_swing_calibration_flat_w1.json`.
Isaac 앱 내부 테스트 57개, OracleCurrent 8 env·1 iteration 스모크가 통과했다.
추가로 seed 43·128 env·30 iteration을 처음부터 실행해 보상과 지형별
clearance 로깅을 확인했다. 마지막 로그의 스윙 clearance 평균은 flat
0.0445 m, stairs_up 0.0440 m였다. 이는 초기 학습 확인이며 등반 성공
판정에 쓰지 않는다. 본 학습은 아직 시작하지 않았다.

**본 학습 전 보상 게이트·가중치 재보정 (2026-10-07, 이전 가중치 10 결정을 대체):**
사용자가 발을 충분히 들게 된 뒤 가중치 10의 보상 규모가 과도할 수 있음을
지적했다. 기존 정책에서의 작은 보상 합만으로 가중치를 정한 것은 미래 정책의
상한을 반영하지 못했다. 게이트를 `clip((v_body_xy · c_xy)/||c_xy||²,0,1)`로
바꿔 **몸통 좌표계 속도**를 명령에 투영한다. 명령 XY 노름 ≤0.1 m/s이면 0이다.
기존 0.2 m/s 포화 게이트와 `v_gate` 설정은 제거했다. 지형 기준·스윙 판정·
명령 게이트를 포함한 보상의 다른 부분은 유지했다.

가중치는 1.0으로 되돌렸다. dt 0.02 s·1000스텝에서 한 번에 두 발이 스윙하는
트롯 가정의 조건부 에피소드 상한은 `0.5×1×0.02×1000 = +10`이다.
선속도 추종 최대 `1.5×20 = +30`의 1/3이다. 네 발이 동시에 스윙하는 경우
절대 상한은 +20이므로 +10은 무조건적인 수학적 상한이 아니다.
이전 seed 43 최종 checkpoint를 동일한 평지 0.6 m/s·32 env·20 s에서
새 게이트로 다시 평가한 에피소드 평균은 새 항 **+0.3682**,
`track_lin_vel_xy` **+29.5745**, `joint_pos` **−3.6271**이었다.
초기 신호가 작을 수 있다. 원시 결과는
`results/jepa_loco/stair_eval/oracle_current_swing_gate_flat_w1.json`.
Isaac 앱 내부 테스트 58개와 OracleCurrent 8 env·1 iteration 스모크가 통과했다.
기존 clearance 게이트 seed 43과 새 본 학습의 차이는 이 보상 항 하나로
유지한다. 본 학습은 아직 시작하지 않았다.

본 학습 후 통과 조건은 고정 9 cm 성공 >0, 학습 중 stairs_up 첫 단 통과율
>0, 평지 play 정상 보행의 동시 충족이다. 새 항 에피소드 합이 선속도 추종
합에 근접하거나 넘으면서 속도 추종이 떨어지면 보상 착취 가능성으로 중단을
검토한다. 발을 들어도 첫 단 앞 정지가 계속되면 장애물 통과 보상 또는 정지
페널티를 다음 후보로 두되, 구현 전에 사용자 확인을 받는다.

**본 학습 조기 중단 기준 추가 (2026-10-07):** iteration 800 무렵
`Diagnosis/swing_clearance_mean_m/flat`이 **0.06 m를 넘지 못하면**
(기존 정책 약 0.044 m), 현재 보상 신호가 약한 것으로 보고 학습을 중단한다.
가중치 상향 여부와 새 값은 사용자와 결정한 뒤 별도 실행으로 검증한다.

**스윙 높이 보상 단독 런 조기 중단 및 기본 자세 가설 (2026-10-07):**
사용자 결정으로 Z790 `oracle_current_swing_s43`을 iteration 813 무렵 중단했다.
마지막 checkpoint는 `2026-10-07_10-07-30_oracle_current_swing_s43/model_800.pt`.
미리 정한 평지 clearance 기준 0.06 m에 못 미쳐 약 0.030 m였고,
stairs_up은 iteration 300 이후 약 0.023 m로 정체했다. stairs_up 첫 단
통과율은 0이었다. 발 높이 보상은 약 +0.0185/s(20초당 약 +0.4)였으나
에피소드 길이 987, 넘어짐 약 2.9%로 보행 자체는 안정적이었다.

새 가설은 기본 자세 `joint_pos` 페널티가 발 들기를 억제한다는 것이다.
우리 기존 함수는 12관절 편차의 **L2 노름**에 −0.7을 곱하고 정지 시
×5한다. 중단된 정책에서 약 −6/에피소드로 선속도 추종 최대 +30의
약 20%다. [APT-RL Table S3](https://arxiv.org/html/2607.13579)는
제곱합을 명령 XY 속도의 제곱으로 정규화한 명목 자세 항에 −0.005를
쓴다. APT-RL의 추종 대비 규모를 대략 0.1~1%로 보는 것은 보행
상태에 따른 추정이며 논문의 가중치만으로 확정되는 값은 아니다.
함수 형태가 달라 계수 비율만으로 페널티 세기를 비교하지 않는다.
이전 연구에서 `joint_pos`를 −0.7에서 −0.3으로 완화했을 때 다리를
바깥으로 벌리는 거미 자세가 나타났다. 따라서 hip 항은 기존 형태와
계수 −0.7을 유지하고 thigh/calf 항만 완화한다. 별도의 hip roll
제한 보상은 이번 실험에 추가하지 않는다.

OracleCurrent의 기존 `joint_pos` 하나를 `nominal_hip`과
`nominal_thigh_calf` 두 항으로 대체했다. 전자는 hip 편차 L2 노름,
후자는 thigh/calf 편차 제곱합을 `max(||v_cmd,xy||²,1)`로 나눈다.
두 항 모두 기존과 같은 정지 판정과 ×5 배율을 쓴다. 관절 정규식·계수·
정지 판정·정규화 방식은 configclass에 노출했다. 다른 보상과 학습 설정은
swing 단독 런과 동일하다. Wide와 Current+Future는 아직 기존 보상이다.

가중치 보정: `model_800.pt`를 **평지·0.6 m/s·32 env·20 s**로 재생했다.
thigh/calf 단위 가중치 −1의 항 합은 −1.9534/에피소드였다. 목표
`0.015/s × 20 s = 0.3/에피소드`에 맞춰 가중치 **−0.15**를 정했고,
동일 평가에서 **−0.2930/에피소드**를 재확인했다. hip 항은 기존 계수
−0.7에서 **−1.9839/에피소드**, 선속도 추종은 +29.6005/에피소드였다.
두 평가 모두 시간초과율 100%, 평균 전진 거리 12.29 m였다.
원시 결과: `results/jepa_loco/stair_eval/oracle_current_nominal_unit_flat_s43.json`,
`results/jepa_loco/stair_eval/oracle_current_nominal_calibrated_flat_s43.json`.
Isaac 앱 내부 테스트 61개와 OracleCurrent 8 env·1 iteration 스모크가 통과했다.

새 런은 seed 43·2048 env·3000 iteration, 이름 `oracle_current_nominal_s43`으로
처음부터 학습한다. 비교는 swing 단독 런의 iteration 0~800 곡선과
동일한 seed·예산에서 한다. iteration 800에 평지 스윙 clearance 평균이
0.06 m를 넘지 못하면 중단을 검토한다. play 영상 또는 hip 편차에서
거미 자세가 확인되면 중단한다. iteration 1500에도 stairs_up 첫 단
통과율이 0이면 학습 계단 5 cm 시작과 평지 워밍업 레벨을 사용자와
결정한다. 최종 통과는 고정 9 cm 성공 >0, 첫 단 통과율 >0, 평지
정상 보행을 모두 요구한다. 예상 소요는 Z790 약 9.5시간이며,
본 학습은 사용자 확인 전까지 시작하지 않는다.

**A안 전 사전 점검 (2026-10-07, 지형 변경 전):** Z790 nominal 런은 점검 당시
진행 중이어서 로컬 최신 `2026-10-07_15-07-23_oracle_current_nominal_s43/model_700.pt`를
사용했다. 고정 계단 평가는 중심 출발·직진 0.6 m/s·높이별 16 env·20 s였다.

- **승급 게이트와 로그:** 실제 학습 설정 10행의 `stairs_up`, `stairs_down`, `gap`
  각각에서 로봇을 첫 장애물 앞/첫 단 윗면/통과 경계 +0.05 m로 옮겨 11회
  갱신했다. 앞과 윗면은 모두 `episode_obstacle_cleared=False`, reset 로그
  `obstacle_clear_rate=0`이었다. 경계 너머는 세 지형 모두 `True`와
  `obstacle_clear_rate=1`이었다. 첫 계단 riser x=1.20 m, 첫 단 바깥 x=1.50 m,
  게이트 x=1.60 m, gap 게이트 x=1.90 m였다. 게이트·로깅 불능은 확인되지 않았다.
- **종료 설정:** `time_out`, base 접촉력 >1 N인 `base_contact`, 몸통 기울기
  >0.8 rad인 `bad_orientation` 세 항이다. `undesired_contacts` 보상 센서 대상은
  Head, hip, thigh, calf이고 foot은 포함하지 않는다. 3/5/9 cm 평가의
  비시간초과 종료는 각각 **0/16**이었다. 종료 시 첫 riser ±0.4 m 범위의
  기록은 3 cm 14건, 5 cm 16건, 9 cm 16건으로 모두 `time_out`이었다.
  19 cm도 비시간초과 종료 0/16이었다.
- **첫 단 접촉 보상:** 3 cm와 5 cm에서 앞발 중심이 riser 2 cm 전까지
  도달한 최초 접촉 이후 4스텝 기록을 각 16 env에서 얻었다. 3 cm에는
  `undesired_contacts`와 `termination`이 모두 0이었다. 5 cm에는
  `Head_lower` 접촉 이력으로 한 스텝 `undesired_contacts=-0.02`가 있었고
  `termination=0`이었다. 9/19 cm는 앞발이 이 접촉 기록 조건에 도달하지
  못해 접촉 직후 보상 표본이 없다. 큰 접촉·종료 페널티가 등반을 막는다는
  직접 증거는 이번 점검에서 나오지 않았다.
- **메시:** 학습 레벨 0의 ray hit는 중앙 평지에서 첫 단 윗면으로
  stairs_up +0.0878 m, stairs_down −0.0915 m 변했다. 앞발 하나가
  stairs_up 첫 단 범위 x≈1.27 m에서 약 51 N 접촉력을 보여 충돌 표면이
  실제로 발을 지지한다. 로봇을 기본 관절 목표·zero action으로 첫 단에
  옮기면 20스텝 뒤 몸통이 x≈1.16 m로 뒤로 밀렸으므로 이 임의 자세에서의
  지속 정지는 확인되지 않았다. 단 폭 0.3 m보다 앞뒤 발 간격이 길어
  중앙/첫 단/둘째 단을 동시에 딛는 시작 자세이며, 이 결과만으로 메시
  결함이라 판정하지 않는다. 첫 단 위치와 충돌 높이는 기하 계산과 맞는다.

원시 결과:
`results/jepa_loco/stair_eval/oracle_obstacle_gate_precheck_s43.json`,
`results/jepa_loco/stair_eval/oracle_current_nominal_s43_precheck_3cm_5cm_contact.json`,
`results/jepa_loco/stair_eval/oracle_current_nominal_s43_precheck_9cm_19cm_contact.json`.
게이트, 종료, 페널티, 메시의 확인 가능한 항목에서 A안을 막는 결함은
발견되지 않았다. 물리적으로 임의 첫 단 자세에서 정지하지 않은 한계는
본 학습 중 접촉·시도율 로그와 play에서 다시 확인한다.

**A안 구현 및 본 학습 전 검증 (2026-10-07):** nominal 설정을 보존하기 위해
`Unitree-Go2-JepaLoco-OracleCurrent-EasyStart` 별도 task를 만들었다. 지형만
변경한다. stairs_up/down/gap 열의 행 0·1은 평지, 행 2~9는 장애물이다.
계단 폭 0.3 m와 지형 비율·20열은 그대로 두고, 계단 높이는 행 2에서
0.05 m, 행 9에서 0.20 m까지 선형으로 증가한다. gap은 0.10~0.30 m다.
IsaacLab이 행별 difficulty에 작은 난수를 넣으므로 생성 함수가 행을 복원해
해당 행의 높이/폭을 고정한다. 워밍업 행은 경로 길이만으로 승급한다.
행 수·워밍업 수·범위는 `EasyStartTerrainCfg`에 노출한다.

학습 중 `Diagnosis/stairs_up|stairs_down/level_{i}`에 실제 장애물 행의 첫 단
앞발 접촉 시도율, 통과율, 첫 riser ±0.4 m에서 종료한 에피소드의
`time_out`/`base_contact`/`bad_orientation`/기타 횟수, 종료 에피소드
수를 남긴다. 워밍업 행은 `warmup_path_only`로 표시하며 실제 장애물
통과율의 분모에 넣지 않는다. 기존 stairs_up 접근 정지·전진 시도 그룹에
레벨별 표본 수와 보상 항별 에피소드 합을 추가했다. 시도율은 앞발 하나라도
첫 단 윗면 XY 범위에서 접촉력 ≥1 N이 된 에피소드 비율이다.

사전 점검 모델은 진행 중이던 nominal seed 43의 `model_700.pt`다.
3/5/9/19 cm 고정 계단 평가 모두 성공 0/16, 비시간초과 종료 0/16이었다.
따라서 이번 A안은 5 cm부터 **학습**하는 효과를 보는 실험이며, 이 결과를
5 cm 학습 실패로 해석하지 않는다. 비교 대상은 같은 seed의 nominal 런이다.
Isaac 앱 내부 전체 테스트 **67개 통과**와 신규 task 8 env·1 iteration
스모크 **통과**를 확인했다. 스모크 TensorBoard에서 레벨 0 stairs_up
`episode_count=2`, `warmup_path_only=1`, `first_tread_attempt_rate=0`이고
실제 장애물 통과율과 첫 단 전진율은 기록되지 않았다. 생성 메시 단위
테스트는 행 0·1의 z 범위가 0이고, 행 9와 행 2의 계단 높이 비가 4임을
확인했다. 아직 도달하지 않은 레벨 2는 `episode_count=0`으로 기록하고
시도율은 기록하지 않으므로, 미노출과 시도 실패를 구분할 수 있다.
8 env·1 iteration은 레벨 2 성능을 검증하지 않는다.

**사전 조기 판정:** iteration 500 부근 레벨 2(5 cm) stairs_up의 시도율>0,
통과율=0이면 즉시 중단하고 종료 사유와 접촉 body를 조사한다. 시도율=0이면
접근 회피로 보고 정지/전진 시도 그룹의 보상 항별 합을 조사한다.
통과율>0이면 계속해 iteration 1500의 레벨 진행을 본다. 최종 통과는
고정 9 cm 성공>0, 학습 중 첫 단 통과율>0, 평지 정상 보행이다.
본 학습 예정값은 pilab seed 42, Z790 seed 43, 각 2048 env·3000 iteration,
run_name `oracle_current_easystart_s42`/`s43`이다. 사용자 확인 전 시작하지 않는다.

**비교 자료 상태 (2026-10-07 17:50 KST):** 로컬 nominal seed 43 런의
TensorBoard 마지막 기록은 iteration 704, 마지막 checkpoint는 `model_700.pt`다.
현재 해당 학습 프로세스와 tmux 세션은 보이지 않는다. 종료 원인은 이
점검에서 확인하지 않았다. 따라서 seed 43의 3000 iteration 전체 곡선은
아직 없고, A안과 nominal의 동일 예산 비교는 현재 0~704 구간까지만
가능하다. pilab nominal seed 42 상태는 이 머신에서 확인되지 않았다.

## 2026-10-08 — EasyStart 고정 계단 평가 생성 결함 점검

진행 중인 128 env/높이 평가를 중단하지 않고 기존 JSON과 IsaacLab 0.54.4
소스를 대조했다. `eval_stairs_fixed.py`는 평가 지형을 1행으로 만들면서
EasyStart의 학습용 계단 config `function`을 그대로 복사했다. IsaacLab은
1행에서도 각 열의 difficulty를 0~1로 무작위 생성하고, EasyStart 함수는
이를 학습용 10행 레벨로 해석한다. 따라서 레벨 0·1로 해석된 열은 계단이
아닌 **평지**였다. JSON `step_height_m`는 실제 메시가 아니라 요청 높이였다.

현재 저장된 seed 42 per-env 결과에서 5/9 cm의 각 low 그룹은 17/128개
열이 평지(`terrain_origin_z=0`)였고, 실패 ID 17개와 정확히 같았다.
7/11 cm high 그룹은 32/128개 열이 평지였고, 실패 ID 32개와 정확히
같았다. 이 때문에 5·9 cm 111/128, 7·11 cm 96/128이라는 성공 수가
정확히 반복되고, 7 cm가 9 cm보다 낮아 보였다. 이전 두 정책의 동일한
성공 수 역시 같은 평가 seed가 고른 평지 열 집합에 크게 좌우되었을
가능성이 높다. **기존 height sweep의 성공률은 고정 높이 계단 성공률로
사용하지 않는다.**

9 cm seed 42 per-env 결과의 비시간초과 종료 35건은 모두 이미 성공한
에피소드에서 발생했다. 종료 당시 마지막 활성 위치 x의 중앙값은
약 4.08 m로 평가 타일 경계 x=4.0 m 부근이다. 현재 평가는 성공 후에도
1000스텝까지 진행하므로 `success_rate`와 `non_timeout_rate`가 겹친다.
비시간초과 종료율을 등반 실패율로 해석하지 않는다. 이 판정 정의는 이번
수정에서 바꾸지 않았다.

평가 생성은 IsaacLab 원본 `MeshInvertedPyramidStairsTerrainCfg`를 사용해
고정 높이 계단을 만들도록 수정했다. 평가 시작 전에 모든 env의 지형
origin z가 `−(단 수+1)×요청 높이`인지 검사한다. per-env의 종료 위치는
reset 직전 `_reset_idx` hook에서 캡처하도록 고쳤다. 기존 `last_x/y/yaw`는
종료 스텝이 아닌 한 스텝 전 값이었다. 성공 판정은 바꾸지 않았다.
별도로 `height_ok_ever`와 `position_ok_ever`가 서로 다른 시점에만
성립했으면 `simultaneous_conditions`, 동시에 성립했지만 유지 시간이
부족하면 `hold_steps`로 분류한다. 기존 JSON의 이 두 경우는 합쳐져 있었다.

진행 중인 shell sweep은 높이 쌍마다 새 Python 프로세스를 띄운다. 수정 전
시작한 `height_sweep_n128_perenv/easystart_s43_0.05_0.07.json`에는 여전히
평지 열이 low 17개/high 32개 있지만, 수정 후 시작한
`easystart_s43_0.09_0.11.json`은 두 그룹 모두 평지 열 0개이고
origin z가 각각 −0.63/−0.77 m로 요청 높이와 일치한다. 따라서 같은
폴더의 JSON도 **파일별로** origin z 불변식을 검사한 뒤 분석에 포함한다.
이전 파일을 덮어쓰거나 진행 중 평가를 중단하지 않았다.

### 2026-10-08 — 평가 타일 실물 감사 및 고정 계단 전량 재평가

기존 `height_sweep_n128_perenv/`의 8개 파일을 다시 감사했다. seed 42의
네 파일과 seed 43의 5/7 cm 파일은 각각 256개 중 **49개 평지 열**
(low 17, high 32)을 포함한다. seed 43의 나머지 세 파일은 앞서 적용한
평가 생성 수정 후 실행되어 평지 열이 0개다. 같은 폴더에 수정 전후
결과가 섞여 있었다. 예전 16 env 평가의 9 cm 13/16을 비롯해 이
오염을 받은 성공률은 계단 등반 성공률로 인용하지 않는다.

원인은 평가에서 학습용 `EasyUpCfg.function`을 1행 generator에 그대로
가져오던 경로였다. IsaacLab은 1행에서도 열마다 difficulty를 뽑고,
이 함수는 이를 10행 학습 커리큘럼의 레벨로 복원한다. 그중 0·1행에
해당하는 열은 평지 워밍업 타일이 된다. 현재 평가 생성은 원본
`inverted_pyramid_stairs_terrain`을 가진 별도 고정 높이 cfg를 사용한다.
`OracleCurrentEasyStartEnvCfg_PLAY.__post_init__`은 평가 cfg 교체 **전에**
실행되고, `OracleCurriculumEnv.__init__`은 generator의 seed만 덮어쓴다.
`TerrainGenerator`의 지형 종류 배정·difficulty 경로에는 flat 대체가
없으며 `TerrainImporter`는 이 설정의 256개 열에 env ID 0~255를
각각 대응시킨다. 따라서 재파싱이나 이웃 타일로의 초기 배치는 이
49개 평지 열의 원인이 아니었다.

평가 시작 시 `validate_stair_origins`가 모든 env의 실제 origin z를
검사한다. 예상값은 사용자 제안 `−단 수×단 높이`가 아니라 IsaacLab
inverted-pyramid 구현의 **`−(단 수+1)×단 높이`**다. 6단의 13/15 cm
경우 실제 origin은 각각 −0.91/−1.05 m다. 평지 열의 origin z=0은
즉시 오류로 거부된다. `--audit_terrain_only`로 실제 importer의
`terrain_origins`와 동일 cfg·seed에서 재생성한 각 열의 메시를
대조했다. 13/15 cm, 256열에서 메시 z 범위는 최소 1.04 m / 최대
1.20 m, origin은 low −0.91/high −1.05 m, 평지 메시 0개였고
열 ID와 env ID가 일치했다. 동일 감사를 두 번 실행한 JSON은 완전히
같다 (`height_sweep_verified/audit_13_15_run{1,2}.json`).

`scripts/jepa_loco/recheck_fixed_stairs.sh`로 평가 seed 43, 중심 출발,
0.6 m/s, 높이당 128 env, 두 정책 seed × 네 높이 쌍을 **두 번**
재평가했다. 모든 16개 파일이 origin 검사를 통과했고 평지 열은 0개였다.
아래 값은 각 반복에서 동일했다 (`height_sweep_verified/run{1,2}/`).

| 고정 계단 높이 | 정책 seed 42 | 정책 seed 43 |
|---|---:|---:|
| 5 cm | 128/128 | 128/128 |
| 7 cm | 128/128 | 128/128 |
| 9 cm | 128/128 | 128/128 |
| 11 cm | 128/128 | 79/128 (61.7%) |
| 13 cm | 4/128 (3.1%) | 5/128 (3.9%) |
| 15 cm | 0/128 | 0/128 |
| 17 cm | 0/128 | 0/128 |
| 19 cm | 0/128 | 0/128 |

두 반복의 env별 `per_env` 레코드, 지형 origin 배열, 그룹 요약은
8개 조건 모두 정확히 일치했다 (총 2048개 env 판정 일치). 이전 오염
파일에서 평지 열만 제외한 결과와 추세가 같다. seed 42의 5/7/9/11 cm는
각각 유효 열 111/111 또는 96/96에서 이번 128/128로, 13 cm는
4/111에서 4/128로 바뀌었다. seed 43의 수정 후 9/11/13/15 cm
파일은 이번 결과와 같다. 평지 열로 인한 7 cm < 9 cm 역전은 사라졌고,
11 cm 정책 차이는 평지 타일을 제거해도 남는다. 이번 작업에서 학습은
실행하지 않았다.

## 2026-10-08 — Current+Future-EasyStart teacher 학습 준비

Phase 2의 목표를 Phase 3 depth student가 `z_current`와 `z_future`를
함께 증류받을 **Current+Future teacher 생성**과, 미래 정보가 Current보다
등반에 도움이 되는지에 대한 전제 확인으로 정리했다. Wide는 선택 비교군으로
미룬다. 기존 `OracleCurrentFuture` task는 유지하고 새
`Unitree-Go2-JepaLoco-OracleCurrentFuture-EasyStart` task와 PLAY/runner
cfg를 추가했다. 새 env cfg는 `OracleCurrentEasyStartEnvCfg`를 상속하므로
보상, 지형, 명령, 커리큘럼, 종료, 이벤트, critic, 진단 설정을 한 곳에서
공유한다. PLAY의 최종 명령 범위와 이벤트 설정도 공통 함수가 적용한다.
PPO 알고리즘과 100스텝 rollout은 Current runner에서 상속한다. 차이는
관측에 미래 187점 패치와 이를 위한 wide 스캐너가 추가되고, 공유
`187→128→32` encoder가 현재·미래 패치를 각각 처리해 총 64차원
terrain latent를 policy에 준다는 점이다. Current는 32차원이다.

**미래 패치 가시성:** 현재 wide 스캐너는 4.0×3.0 m, 0.1 m 간격이고
현재/미래 패치는 1.6×1.0 m, 187점이다. 최종 명령의 vx 양끝
(−0.3, 2.0 m/s), vy 양끝(±0.4 m/s) 및 yaw-rate −1~1 rad/s를
촘촘히 훑어 0.5 s SE(2) 외삽 패치의 최대 |x|≈1.950 m,
최대 |y|≈1.259 m를 얻었다. 스캐너 한계는 |x|≤2.0 m,
|y|≤1.5 m이므로 범위 밖 점은 없고, x 여유가 약 5 cm다.
현재 샘플러는 범위 밖일 때 `grid_sample`의 `padding_mode="border"`로
경계값을 복제하고 visible=False를 반환한다. 이번 확정 명령 범위에는
해당하지 않으므로 wide 크기, horizon, 처리 방식을 바꾸지 않았다.
최대 명령 조합과 중간 yaw-rate에서 전체 187점 가시성을 테스트한다.

Isaac 앱 내부 전체 단위 테스트 **73개 통과**. 새 task의 8 env·1 iteration
스모크는 완료했고, TensorBoard `Perf/collection_time=3.647 s`,
`Perf/learning_time=0.112 s`, 합계 **3.760 s/iteration**이었다.
이는 초기 8 env 연결 검사이며 2048 env 처리량의 추정치는 아니다.
스모크 `model_0.pt`로 새 task의 고정 계단 평가 4 env·20스텝을 실행해
origin 검사 통과를 확인했다. 9/11 cm의 origin z는 각각
−0.63/−0.77 m(6단)이고 요청 높이와 일치한다. 이 스모크의 성공률은
학습 전 checkpoint이므로 성능으로 해석하지 않는다.

**본 학습은 시작하지 않음.** 예정 조건은 Current+Future-EasyStart
seed 42(pilab), seed 43(Z790), 각 2048 env·3000 iteration,
run_name `oracle_future_easystart_s42`/`oracle_future_easystart_s43`이다.
비교 대상은 같은 seed의 Current-EasyStart 3000 iteration이다. 기존
Current-EasyStart의 마지막 100 iteration 평균은 pilab 약 7.83 s/iter,
Z790 약 11.05 s/iter였지만 새 조건의 추가 wide 스캐너는 2048 env에서
실측하지 않았다. 실행 전 예상 시간은 pilab **약 7~18 h**,
Z790 **약 10~24 h**의 보수적 범위로 제시하고 사용자 확인을 받는다.

**평가 프로토콜:** 두 정책과 두 seed를 고정 오르기 계단
9/10/11/12/13/14/15 cm, 높이당 128 env, 평가 seed 43,
중심 출발·0.6 m/s로 맞춘다. 모든 env의 origin 검사를 통과한 JSON만
분석한다. 11~13 cm 변별 구간에서 Current+Future의 성공률이 Current와
같거나 높으면 teacher 후보로 확정한다. 조건당 2개 seed는 통계적 우월성
입증이 아니라 이 전제의 초기 확인이다. 학습 중 레벨별 첫 단 통과율,
첫 통과 iteration, play 영상도 함께 확인한다. 최종 비교에는 조건당
3개 이상 seed 규칙을 유지한다.

## 2026-10-08 — Current+Future gap 관측 NaN 수정

Z790의 `oracle_future_easystart_s43` 본 학습이 **iteration 22**에서
`terrain_future` 관측 NaN으로 중단됐다 (`~/oracle_future_easystart_s43.log`).
pilab의 `oracle_future_easystart_s42`도 같은 결함 가능성 때문에 사용자에
의해 중단됐다. **두 런은 성능 비교에 사용하지 않고, 수정 후 seed별로
처음부터 재시작한다.**

원인은 gap의 ray miss에서 `wide_height_scanner.ray_hits_w[...,2]=inf`가
나오고 `pos_z−hit_z−offset=−inf`가 된 데 있다. Current heightscan은
보간이 없어 관측 clip이 이를 −1로 만들지만, Future는 bilinear
`grid_sample`에 `−inf`를 먼저 넣어 NaN을 만들었다. 이후 관측 clip은
NaN을 제거하지 못했다. `future_height_scan`에서 보간 전에 wide 값을
관측 clip 범위로 정리한다: `−inf→lo`, `+inf→hi`, `NaN→lo`, 유한값도
`[lo,hi]`로 clip한다. `terrain_future.scan.clip`을 보간 전
`clip_range`에도 전달하고 런타임에 불일치하면 오류를 낸다. 같은 함수를
쓰는 기존 `OracleCurrentFuture` task에도 이 수정이 적용된다.

순수 함수/관측 테스트에서는 `−inf/+inf/NaN`이 섞인 187점 미래 패치가
전부 유한하고 [−1,1] 안에 있음을 확인했다. horizon 0·명령 0에서
미래 패치가 같은 위치의 clip 후 Current 패치와 일치하며, 두 clip
설정 불일치는 실패한다. Isaac 앱 내부 전체 테스트 **75개 통과**.

gap만 생성하도록 다섯 지형 종류의 비율을 `gap=1`, 나머지 0으로
오버라이드하고 `max_init_terrain_level=2`에서 **64 env·30 iteration**
학습 스모크를 실행했다 (Z790 seed 43, `oracle_future_gap_smoke`).
30/30 iteration이 NaN 없이 끝났고 `model_29.pt`가 저장됐다.
동일 조건의 실제 RayCaster 프로브에서는 64 env 중 gap 레벨 2가
**15개**, 이들의 원시 wide hit 중 비유한 ray가 **1,237개**였다.
최대 전진 명령 2.0 m/s로 미래 패치를 gap 쪽으로 옮기면 수정 전
보간은 그 15개 env에서 **NaN 334개**를 만들고, 수정 후 값은 모두
유한하며 범위 [−1,−0.10] 안이었다. 따라서 스모크가 실제 gap
미충돌 광선 조건을 포함했고, 이전 실패의 보간 경로도 재현했다.

2048 env·3000 iteration 본 학습은 시작하지 않았다. 수정 뒤 신규
run_name은 seed 43(Z790) `oracle_future_easystart_s43`, seed 42(pilab)
`oracle_future_easystart_s42`를 사용할 수 있으나, 기존 중단 런과
구분되도록 새 타임스탬프의 로그 디렉터리에서 처음부터 실행한다.

## 2026-10-08 — 고정 gap 비교 평가 프로토콜 (결과 기록 전)

Current-EasyStart와 Current+Future-EasyStart의 gap 통과 성능을 동일한
고정 평가로 확인한다. 각 정책의 seed 42·43 최종 checkpoint를 사용하고,
평가 seed 43, 원점 중심 출발(x=0, yaw=0), 기존 계단 평가와 같은 좌우
출발 오프셋, 0.6 m/s 직진 명령, 폭당 128 env를 적용한다. 학습 곡선의
gap 통과율 차이(Current+Future 0.068, Current 0.002; seed 43,
iteration 약 600)는 이 고정 평가 전에 성능 결론으로 취급하지 않는다.

평가 지형은 학습용 `EasyGapCfg`를 복사하지 않고 IsaacLab 원본
`MeshGapTerrainCfg(function=gap_terrain)`로 새로 만든다. 두 고정 폭을
각각 열 절반에 배치하며 `platform_width`는 학습 설정에서 읽는다.
중앙 플랫폼의 +x 안쪽 가장자리는 `platform_width/2`, 바깥 지면의
시작은 여기에 gap 폭을 더한 위치다. 시작 전 모든 타일의 origin z=0을
검사하고, **시뮬레이터에 실제로 올린 USD 메시**를 수직 ray로 검사한다:
안쪽 플랫폼과 바깥 지면은 hit, gap 중앙은 miss여야 한다. 어느 타일이든
다르면 평가를 중단한다. `--audit_terrain_only`도 같은 검사를 수행한다.
이는 과거 EasyStart 계단 평가에 평지 타일이 섞인 문제를 방지하기 위한
검사다.

성공은 종료 전 몸체의 원점 상대 x가 `바깥 가장자리 + 0.10 m` 이상인
상태를 25 연속 제어 스텝 유지하는 것이다. 몸체 높이 조건은 없다.
성공 판정은 한 번 켜지면 이후 타일 경계에서 종료돼도 유지한다.
`per_env`에는 폭, 안쪽 가장자리 대비 최대 x 진행량(음수면 미도달),
최대 몸체 낙하량, 실패 사유를 남긴다. 실패 사유는 운영상 분류로,
gap 주변에서 비시간초과 종료하고 몸체가 출발 높이에서 0.15 m 이상
하강한 경우 `gap_fall_termination`, 그 외 비시간초과 종료 없이 최대
x가 안쪽 가장자리 미만이면 `gap_front_stall`, 나머지는 `other`다.
이 분류는 접촉 위치를 직접 측정한 물리적 원인 판정이 아니다.

폭 쌍은 **0.10/0.15, 0.20/0.25, 0.30/0.35 m**로 한다. 0.10~0.30 m는
학습 범위이며 0.35 m는 범위 밖 참고값으로 별도 표기한다. 네 정책 ×
세 폭 쌍을 동일한 평가 seed로 두 번 실행해 생성 지형 검사와 성공
결과의 결정성을 확인한다. 본 비교 평가는 현재 실행하지 않는다.

구현 후 Isaac 앱 내부 전체 단위 테스트 **79개 통과**. GPU 장치로
첫 4 env 스모크를 시도했으나 병행 학습으로 GPU 메모리 약 0.6 GiB만
남아 PhysX의 256 MiB 할당 단계에서 중단됐다. 학습은 유지한 채
CPU 장치로 4 env 지형 감사를 재실행했다. 네 타일 모두 원본
`gap_terrain`이었고, 원점 z=0 및 플랫폼 hit·gap miss·바깥 지면 hit
검사가 통과했다. 이는 지형 생성 검증이며 정책 성능 결과는 아니다.
같은 Current-EasyStart seed 43 최종 checkpoint로 CPU에서
**4 env·50스텝 정책 재생**도 완료했다. JSON에 low/high 각 2개,
`gap_tile_checks` 4개와 `per_env` 4개가 기록됐고, 시작 전 기하 검사가
통과했다. 50스텝은 gap 가장자리에 닿기에도 짧으므로 성공률은
평가하지 않았다.

## 2026-10-09 — Phase 2 teacher 고정 평가와 Phase 3 방향 변경

원본 결과는 `results/jepa_loco/stair_eval/teacher_compare/{stairs,gap}/`.
계단은 높이당 128 env, 평가 seed 43, 중심 출발, 명령 0.6 m/s이며
아래 값은 **학습 seed 42와 43의 성공 수 합계/256**이다.

| 단 높이 | Current | Current+Future |
|---|---:|---:|
| 9 cm | 256/256 (100%) | 256/256 (100%) |
| 10 cm | 248/256 (96.9%) | 206/256 (80.5%) |
| 11 cm | 216/256 (84.4%) | 73/256 (28.5%) |
| 12 cm | 117/256 (45.7%) | 0/256 |
| 13 cm | 10/256 (3.9%) | 0/256 |
| 14~16 cm | 각 0/256 | 각 0/256 |

11 cm seed별 성공 수는 Current 127/128·89/128, Current+Future
37/128·36/128이다. Future 관측과 지형 origin 검사는 정상이다.
Current+Future는 첫 단(x≈1.2 m) 도착 후 속도가 약 0.3→0.08 m/s로
줄고, 11 cm 계단에서 6단 중 평균 약 2단 올라간 뒤 멈췄다. 실패는
대부분 시간초과였다. 학습 중 첫 단 통과 지표는 이 차이를 예고하지
못했다. 11.4 cm 최초 통과 iteration은 Current+Future 1454/1437,
Current 1790/2063으로 오히려 전자가 빨랐지만, 고정 평가는 **6단
꼭대기 도달**을 요구한다. 이후 조건 비교의 판정은 고정 평가를 쓴다.

gap 고정 평가(폭 0.10~0.35 m, 폭당 128 env, 한 번)는 네 정책 모두
모든 폭에서 **0/128 성공**이다. 모든 결과의 최대 몸체 x가 안쪽
가장자리 x=1.5 m 미만이다. 다만 “전부 앞에서 정지”는 정확하지 않다.
Current seed 42의 0.20/0.25 m 쌍에서 80/256, 0.30/0.35 m 쌍에서
165/256 env는 몸체가 시작 높이보다 0.15 m 이상 내려갔다. 모두
안쪽 가장자리 전에서 발생했고, 별도 gap 낙하 입증은 아니다.
바닥 없는 곳에서는 종료 없이 시간초과될 수 있으므로 현재의
`failure_reason`(비시간초과 종료에 기대는 분류)을 낙하 진단으로
해석하지 않는다. 이후 분석에는 `max_body_drop_m`를 우선 함께 보고,
낙하 실패 분류를 갱신한다. 실제 gap을 건넌 로봇이 없어 성공 판정의
실제 양성 사례도 아직 검증되지 않았다.

미리 정한 11~13 cm의 `Current+Future ≥ Current` 기준을 충족하지
못했다. **Phase 3 고정 teacher는 Current-EasyStart seed 42
`model_2999.pt`**(11 cm 127/128)다. Current+Future는 비교 기록으로
보관하고 future terrain 증류는 선택 ablation, 기본 꺼짐으로 둔다.
연구 주장은 미래 지형 외삽에서 카메라 사각지대의 **현재 지형을 과거
Depth로 기억·예측**하는 보행으로 바꾼다. 교사 head 고정 재사용,
손실 초기 계수(1/1/0.1), DAgger 초기 혼합 β=0은 구현 기본값이나
사용자 최종 확인을 기다리는 설계 선택이다.

### Phase 3 동기 그림: D435i 현재 프레임의 teacher 패치 사각지대

`scripts/jepa_loco/blind_spot_analysis.py`는 teacher의 1.6×1.0 m,
0.1 m 간격 높이맵 **187점**을 D435i 임시 위치·30° pitch와
112×64 intrinsic으로 투영한다. 평지는 base 높이 0.279 m에서 기하
가시성을 계산한다. 고정 9/13 cm 계단은 첫 단까지 지정한 거리로
로봇을 배치하고, 실제 height scanner hit를 현재 렌더 depth와
0.06 m 이내로 대조한다. 유효 범위는 0.28~2.0 m다. JSON에는
점별 기하 및 렌더 가시 mask를 저장해 후속 linear probe의 사각지대
분류에도 사용한다. 출력은 `results/jepa_loco/blind_spot/`의
`visibility.json`, `visibility.csv`, `visibility_maps.png`다.

평지 기하에서는 **27/187점(14.4%)**이 현재 프레임에 들어왔다.
계단 접근의 렌더 가시점 수는 아래와 같다(분모는 모두 187점).

| 첫 단까지 거리 | 9 cm | 13 cm |
|---|---:|---:|
| 1.5 m | 22 (11.8%) | 22 (11.8%) |
| 1.0 m | 22 (11.8%) | 22 (11.8%) |
| 0.6 m | 24 (12.8%) | 22 (11.8%) |
| 0.3 m | 28 (15.0%) | 22 (11.8%) |
| 0.0 m | 29 (15.5%) | 31 (16.6%) |

이번 점별 표본에서는 기하상 보이는 점과 렌더 depth가 일치한 점의
수가 같았다. 따라서 이 결과가 직접 보여 주는 것은 **패치의 약
83~88%가 현 시점 FOV·거리 밖**이라는 사실이다. 추가 가림으로
숨은 점은 이 표본에서 확인되지 않았으므로, “계단 수직면이 단 윗면을
가린다”는 별도 현상은 이 그림만으로 주장하지 않는다. 카메라 위치는
실측 전 임시값이므로 실기 장착값이 확정되면 다시 계산해야 한다.
