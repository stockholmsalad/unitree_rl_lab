# JE-Loco 실험 로그 — 통제 확립 (2026-07)

ICCAS 이식 후 매트릭스 본실험 **전에**, A/B 비교의 정당성을 깨는 교란을 하나씩 찾아 제거한 기록.
각 항목은 검증에 쓴 정확한 run 이름·조건을 남긴다(재현·논문 방법론 章 자산).

플랫폼: pilab(RTX PRO 6000 Blackwell Max-Q 96GB). IsaacLab 2.3.2 / rsl-rl-lib 5.0.1 / torch 2.7.0+cu128.
공통 조건: `--num_envs 1024`, 3개 동시 실행(총처리량 ~11k steps/s 상한, collection 병목).

---

## 교란 1 — 적응형 지형 커리큘럼 (제거됨)

**증상.** 표준 `terrain_levels_vel` 은 "로봇이 걸은 거리"로 지형 레벨을 올린다(성능 의존).
헤드/seed 가 달라져 정책이 달라지면 **훈련 지형 분포까지 달라진다** = 내생적 교란.

**증거 (2026-07-24, 적응형 커리큘럼, 20000 iter).**
| | terrain 최종 | 거동 |
|---|---|---|
| Head A s1/s2/s3 | 6.20 / 5.86 / 6.18 | 안정, 좁게 수렴 |
| Head B s1/s2/s3 | **0.00 / 4.96 / 4.75** | 요동: s1 완전 실패(terrain 0 고착), s2 정상, s3 지연 |

Head B 를 seed 1 만 봤으면 "실패", seed 2 만 봤으면 "정상" — **정반대 결론**. n=1 판정 불가.

**수정.** `scripted_terrain_levels`(mdp/curriculums.py): 난이도 상한 C(t) 를 iteration 함수로
고정, reset env 를 [0, C(t)] 균등 배정. 모든 run 이 동일 지형 분포. je_loco env_cfg 에서
`curriculum.terrain_levels` 를 이걸로 교체.
- **주의:** 이제 `Curriculum/terrain_levels` 는 성능 신호가 아니라 스크립트 평균(iter 12000+ 에서
  frac=1.0 → randint(0,10) 평균 4.5)이다. 정책 성능은 reward/tracking/survival 로만 읽어야 한다.
- 커밋: `scripted terrain curriculum`.

---

## 교란 2 — VICReg 을 z_e 에 직접 부과 → projector 로 이동 (유지 확정)

**증상.** Head B 의 VICReg(var/cov)이 정책 입력 latent `z_e` 에 직접 걸려 z_e 를 등방·std=1 로
백화. Step-0 진단(2026-07, icros ckpt): z_e 가 1 컨트롤 스텝에 79% 탈상관, 예측 R²=−0.13(chance 이하).

**수정.** SSL 표준대로 VICReg 을 `projector(z_e)` 출력에 부과, z_e 자유화(model.py `vic_projector`,
runner.py `_jepa_loss_step`). `use_projector` 플래그(기본 True)로 on/off ablation.

**Ablation (스크립트 커리큘럼 위, 오직 `use_projector` 만 다름, 12000 iter 목표).**
- ON  : `2026-07-27_18-06-01_fixB_s{1,2,3}`
- OFF : `2026-07-30_10-28-21_noprojB_s{1,2,3}` (~8800 iter 에서 조기 판정)

| | reward (s1/s2/s3) | seed std | err_xy std |
|---|---|---|---|
| projector ON  | 43.1 / 46.1 / 41.9 | **1.78** | **0.052** |
| projector OFF | 24.4 / 42.8 / 24.6 | 8.62 | 0.157 |

projector OFF 는 커리큘럼을 고정했는데도 seed 요동(std 3~5배). **off s1 은 iter 8000 에서
reward 3.04 로 붕괴.** → projector 는 커리큘럼과 **별개 교란**을 잡는다. **유지 확정.**

**z_e_std 반전 (중요).** 최종 z_e_std: ON 1.3~1.8(드리프트 큼) vs OFF 1.0~1.1(안정).
즉 z_e 가 **더 흔들리는 ON 이 학습은 더 안정적**이다.
- 원래 가설 "z_e 드리프트가 학습 불안정의 원인"은 **기각**. 드리프트는 건강한 신호였다.
- projector 의 이득 메커니즘은 "드리프트 억제"가 아니라 **z_e 자유화**(std=1 백화 해제)다.
- "VICReg 이 z_e 백화로 학습을 해친다"는 결론은 맞았으나, 메커니즘은 애초 생각과 반대.

---

## 매트릭스 확정 설정

**스크립트 커리큘럼 + projector ON.** 둘 다 검증됨. 각자 다른 교란(terrain 분포 / 학습 안정성)을 잡음.
multi-seed 필수(최소 3, 권장 5): Head B 는 통제 후에도 A 보다 seed 분산이 크다(reward std 1.78 vs A ~0.5대).

## 미결(별도 결정)
- **명령속도 커리큘럼** `lin_vel_cmd_levels` 는 여전히 적응형. 매트릭스 첫 run 에서 A/B 간
  `Curriculum/lin_vel_cmd_levels` 곡선이 갈리는지 로그로 확인 후 판단(옵션 1, 사용자 선택).
- **height_scanner 타깃**([D]3): Head A 재구성 타깃의 ~92%가 카메라 시야 밖(x∈[−0.55,0.55] vs
  카메라 전방 [0.48,2.3]). 정렬 시 "ICCAS 재현 → 개선 baseline" 으로 실험 성격 변화 — 미결.
- **결손 = sentinel 주입**([D]4): 상수 `_HOLE_VALUE` 치환이 max-pool 을 지배(A 가 severity 0.2 에서
  포화). 마스킹(−inf) 기반 재측정 필요(측정 ④). ICCAS 핵심 결과의 sentinel-아티팩트 여부 판정.
