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
20 s(1000 제어 스텝)를 적용했다. 중앙 플랫폼의 첫 단(x≈1.5 m) 앞 x=1.0 m에서 출발하고,
상단 진입(x≥2.9 m와 몸체 상승 ≥ 전체 단 높이의 90%)을 25 스텝 유지하면 성공으로 센다.
실제 명령 텐서는 `[0.6, 0, 0]`으로 확인했다. 이 성공 규칙은 보행 영상 판정을 보조하는
위치·높이 대리 지표다.

| 정책 | 단 높이 | 성공 | 최대 전방 위치 평균 (원점 기준) | 비시간초과 종료 |
|---|---:|---:|---:|---:|
| depth GRU | 9 cm | 0/16 | 1.03 m | 0/16 |
| depth GRU | 19 cm | 0/16 | 1.07 m | 0/16 |
| blind GRU | 9 cm | 0/16 | 1.09 m | 0/16 |
| blind GRU | 19 cm | 0/16 | 1.09 m | 0/16 |

모든 조건에서 출발 이후 몸체 높이 증가가 없었다. 최대 전방 위치도 첫 단(x≈1.5 m)보다
0.4 m 이상 앞에서 멈춰, 이번 실패는 계단 중간에서 넘어진 것이 아니라 **첫 단 접근·진입 실패**로
분류한다. 이는 사용자의 play 영상 관찰(오르는 계단 실패)과 일치한다. 몸체가 한 번도 상승하지
않았으므로 19 cm가 물리적으로 너무 높은지, 보상 중 수직 속도·자세 페널티가 발 들기를
억제하는지 등의 원인은 이 평가만으로 분리할 수 없다. 두 정책 모두 실패했으므로 이 실험은
depth 유효성 비교 근거가 아니다. 평지 보행과 첫 단 앞 정지의 원인은 별도 궤적·영상 점검이 필요하다.

원시 결과: `results/jepa_loco/stair_eval/depth_gru_v2_s42.json`,
`results/jepa_loco/stair_eval/blind_gru_v2_s42.json`.
Isaac 앱 내부 단위 테스트 34개 통과(`PYTEST_EXIT_CODE=0`).
