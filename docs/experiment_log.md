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
