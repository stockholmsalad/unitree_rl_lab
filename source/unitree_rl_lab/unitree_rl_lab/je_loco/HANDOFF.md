# JE-Loco 인수인계 (다른 세션/사람이 이어받기용)

> 최종 갱신: 2026-08-04. 이 문서 + `EXPERIMENTS.md`(상세 실험 로그) + `~/.claude/.../memory/`(자동 로드)
> 를 읽으면 현황 파악 가능. 새 세션 시작 멘트 예: "je_loco 이어서 할 거야. HANDOFF.md·EXPERIMENTS.md
> 읽고 현황 파악해줘."

## 한 줄 상태 (2026-08-04)
JEPA predictor conditioning ablation 학습 중(pilab). **핵심 반전 발견 완료**: ICCAS "예측>재구성
결손강건" 주장은 sentinel 주입 아티팩트였고, 진짜 마스킹에서 사라짐. 지금은 "JEPA 를 실제로
작동시키는 레시피" 방향(conditioning)으로 B 개선 시도 중.

## 연구 서사 (여기까지의 결론 — 뒤집으면 안 됨)
1. **ICCAS 원본 주장**: 백본 고정, head 만 재구성(A)↔예측(B) 스왑 → depth 결손 시 B 가 우아하게 저하.
2. **통제 확립**(EXPERIMENTS 교란1·2): 적응형 커리큘럼(→scripted)·VICReg z_e 백화(→projector) 제거.
   둘 다 필수, ablation 검증. [[je-loco-controls-validated]]
3. **아티팩트 발견**(교란3, 측정④): 결손을 sentinel 좌표 주입으로 흉내내면 mount↔origin(동급 상수)에서
   A>B 가 반전 = PointNet max-pool hijack. 방법론 실패. [[je-loco-sentinel-artifact]]
4. **진짜 마스킹**([D]4): 제거 점을 max-pool 에서 제외(valid 채널, 좌표 무주입). 재학습 필수.
5. **최종 반전**(2026-08-03): 마스킹에서 쉬운·어려운 지형(lv9) 둘 다 **A 안 무너짐**. ICCAS A>B =
   아티팩트 확정. 오히려 재구성이 정밀·강건. "예측>재구성" 폐기. [[je-loco-masking-reversal]]
6. **현재**: JEPA 를 살리려 predictor conditioning. **action 이 예측을 살림, command 는 z_p 에 이미
   있어 미미**(2800 iter 1차 판정). act·both 를 12000 까지 학습 중.

## 지금 돌고 있는 것 (pilab, env_isaaclab)
- `run_cond_matrix.sh` 로 실행. **살아있는 4런**: `Bcondact_s{1,2}`, `Bcondboth_s{1,2}` (12000 목표).
  cmd 단독(`Bcondcmd_s*`)은 예측 개선 미미로 kill 함.
- 로그: `~/mask_logs/Bcond*.log` · 체크포인트: `logs/rsl_rl/je_loco_pc/2026-08-03_18-*_Bcond*/`
- 4 concurrent(총처리량 상한 ~11k steps/s 6등분→4등분). 12000 도달 ≈ 하루.

## 다음 할 일 (12000 도달 시)
로그 Z790 으로 rsync → 판정:
1. **clean err_xy 가 기존 maskB(0.15~0.25)보다 낮아졌나** = B 정밀도 향상(사용자 1차 목표)
2. **마스킹 dropout 강건성** (`eval_pc.py --degradation dropout`, 필요시 `--terrain_level 9`)
3. **예측 R² 확실히 양수인가** (jepa loss / z_e_std² 정규화해서 볼 것 — raw 는 스케일 오염)

## [D] 코드 수정 진행표
- ✅ 1 scripted curriculum (mdp/curriculums.py `scripted_terrain_levels`, env_cfg 에서 교체)
- ✅ 2 VICReg projector (model `vic_projector`, `use_projector` 플래그)
- ✅ 3 height_scanner 전방정렬 (x∈[0.5,2.0], 144셀)
- ✅ 4 진짜 마스킹 (pointcloud 768=192×4 [x,y,z,valid], 인코더 masked max-pool)
- ✅ (신규) JEPA conditioning (`jepa_cond_command`/`jepa_cond_action`, actor 단일소스, runner 가 actor 에서 읽음)
- ⬜ 5 Head C (미래 height map 예측 = recon 타깃 t→t+k, 시점축 분리) — 미착수
- ⬜ 6 blind 모드 (obs_groups 에서 pointcloud 제거, 외수용 차단 baseline) — 미착수

## 환경 (반드시 지킬 것)
- **Z790**(RTX 5070 Ti 16GB): 코드 개발·평가. conda `env_test`. IsaacLab `/home/user/graduation/IsaacLab`(2.3.2).
- **pilab / 9960x**(RTX PRO 6000 96GB): 학습. conda `env_isaaclab`. 경로 `~/workspaces/unitree_rl_lab`.
- 워크플로: Z790 에서 수정→commit→push(origin/pilab, fork `stockholmsalad`) / pilab pull→학습→ckpt·log Z790 rsync.
- **conda env 섞지 말 것**(IsaacLab 체크아웃이 바뀜). 환경 대조: `python scripts/check_env.py --compare scripts/env_reference.json`.
- **긴 train 명령 붙여넣기 금지**(잘림 사고 반복). 반드시 런처(`run_*.sh`) 사용.
- 로그 stdout 은 IsaacLab 하드종료로 버퍼 유실 → **`python -u` 필수**, 또는 텐서보드 이벤트로 판정.

## 실행 레시피
```bash
# 학습(pilab): 마스킹 A/B
bash scripts/je_loco/run_mask_matrix.sh                 # A(recon)·B(jepa) × seed 1,2
# 학습(pilab): JEPA conditioning ablation
bash scripts/je_loco/run_cond_matrix.sh                 # B-cmd/act/both × seed 1,2
# 평가(Z790): 마스킹 결손 스윕
python -u scripts/je_loco/eval_pc.py --checkpoint <ckpt> --num_envs 256 \
  --dropout_levels 0,0.2,0.4,0.6,0.8,1.0 --degradation dropout --headless   # --terrain_level 9 로 어려운지형
```

## 함정 메모
- 헤드 확정은 agent.yaml 이 아니라 **체크포인트 키**로(`jepa_predictor` vs `recon_decoder`). agent.yaml 은 기본값 저장.
- eval 은 default repr_head 로 모델 생성 후 strict=False 로드 → 백본만 로드(repr head 는 추론에 안 씀). ICCAS eval 방식과 동일.
- terrain_levels 는 scripted 라 성능 신호 아님 → 성능은 reward/err_xy/생존율로.
- ICCAS 원본 ckpt(sentinel·96셀 height)는 `/home/user/icros/unitree_rl_lab` 에 보존. 재현·측정④ 용.
