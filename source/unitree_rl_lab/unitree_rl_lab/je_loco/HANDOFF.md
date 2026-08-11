# JE-Loco 인수인계 (다른 세션/사람이 이어받기용)

> 최종 갱신: 2026-08-04. 이 문서 + `EXPERIMENTS.md`(상세 실험 로그) + `~/.claude/.../memory/`(자동 로드)
> 를 읽으면 현황 파악 가능. 새 세션 시작 멘트 예: "je_loco 이어서 할 거야. HANDOFF.md·EXPERIMENTS.md
> 읽고 현황 파악해줘."

## 한 줄 상태 (2026-08-11)
**핵심 반전**: ICCAS "예측>재구성 결손강건" = sentinel 아티팩트, 마스킹에서 사라짐. JEPA 살리기
(conditioning·지평확대) 무신호 — 그냥걷기는 proprioception 포화. → **Stage 2 foothold 로 전환.**
foothold 1차 학습(계단만)은 정지 편법(foothold_safety 를 서서 챙김)으로 실패. **2차 = 보상 재설계
(정지게이트·대각선trot·명령0.5~1.5) + 지형 다양화(계단·gap·디딤돌·구멍·블록) 완료, A/B 재학습 착수 대기.**

## 프로젝트 우선순위 (사용자 확정 2026-08-11)
1. **먼저**: 예측(B) vs 재구성(A) 표현 비교 결론 — **raycasting 입력 고정**(입력 바꾸면 표현비교 오염).
   Stage 2 foothold 다양지형에서 A/B 재학습 → 결손 하 foothold 정밀도 비교.
2. **그다음**: **D435i(depth 이미지) 입력으로 JEPA 비정형지형 실기.** = 입력표현 축 별도 실험.
   → **depth 전환은 표현결론 후 2단계 카드로 확정.** 지금 depth 로 바꾸지 말 것(결손실험 재설계+
   PointNet→CNN 교체 큰 비용, 표현비교 축 흔들림). raycasting proxy 유지가 현 단계 정답.

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

## conditioning 실험 결과 (완료, 2026-08-05)
`Bcondact_s{1,2}`·`Bcondboth_s{1,2}`(11400 iter) 마스킹 dropout eval:
clean err 평균 none 0.168 < act 0.207 < both 0.237, 완전실명 생존 both s2=3%. **conditioning 이
정밀도·생존 둘 다 악화 → k=5 에선 B 향상 실패.** 상세: EXPERIMENTS.md. 체크포인트:
`logs/rsl_rl/je_loco_pc/2026-08-03_18-*_Bcond*/`. 결론: 재설계(지평 확대)로.

## 다음 할 일 — **여기부터 재개: JEPA 재설계**
conditioning 실패로 재설계가 확정됨. 아래 "JEPA 재설계" 순서대로 ④+② 부터 착수.
(사용자 의사: "학습 끝나면 재설계 시작하자" → 지금이 그 시점, 단 사용자 확인 후 착수.)

## 그다음 — JEPA 재설계 (진단 완료, 착수 대기)
**진단**: 현 JEPA 는 "너무 쉬운 문제"를 풂. k=5(0.1초)면 로봇 5cm 이동 → z_e(t+k)≈z_e(t) →
최적 predictor≈identity → shaping 신호 빈약 → 예측이 아무것도 안 가르침. conditioning 이
미미했던 것도 "이미 쉬운 문제"라서. **프레이밍(사용자 동의)**: "B 가 A 를 이긴다"가 아니라
**"긴 지평 예측 = 결손/가림 하 지형 선행(anticipation) = 재구성 불가능한 니치"**. 마스킹 반전
(현재프레임 A 우위)과 공존하는 정직한 B 서사.

**의존성 순서(한 번에 하나씩):**
- **④ done 경계 마스킹** (버그수정): `_jepa_loss_step` 이 valid_t+k 를 done 무시하고 인덱싱 →
  에피소드 리셋 넘는 쌍은 순간이동 타깃. storage.dones 로 마스킹. 몇 줄. ②의 전제.
- **② copy-baseline skill score + residual 타깃**: metric = `1 − MSE(pred,tgt)/MSE(copy=z(t),tgt)`,
  양수여야 예측 성공. 타깃을 Δz=z̄(t+k)−z(t) 로 → 잘 정렬. raw loss 스케일오염 근본 해결.
- **① 다중지평** (핵심): **T=32 하드제약** → k=50 불가. 먼저 **C안: k∈{5,15,27}(0.1/0.3/0.54초)**,
  T=32 유지·무료(k=27 은 valid_t 5개×1024env). horizon 임베딩으로 한 predictor 학습. skill score
  살아나면 지평 레버 확인. 더 필요하면 B안(JEPA 전용 긴 버퍼). A안(num_steps 확대)은 PPO 건드려
  A/정책 비교 끊기니 최후.
- **③ ego-motion 조건**: action평균 → v̂·kΔt(변위추정, 기존 vel_decoder 재사용). 지평 길어야 의미.
- **⑤ grounded 프로브**: frozen height 디코더를 ẑ_e(t+k) 에 붙여 "1초 뒤 지형 몇 cm 오차" 측정 =
  논문 그림. [D]5 Head C(타깃을 미래 height map)와 합류.
- **⑥ 추론 페이오프**: 결손 시 예측으로 롤아웃 메꾸기. 예측 작동해야 의미. 마지막.

착수 단위: **④+② 먼저**(runner.py 한 파일, pilab 학습과 충돌 없음) → ① C안.

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
