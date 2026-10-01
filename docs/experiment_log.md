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
