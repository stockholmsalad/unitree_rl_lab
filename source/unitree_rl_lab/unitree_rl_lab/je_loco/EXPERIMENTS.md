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

## 교란 3 — 결손 sentinel 주입이 max-pool 아티팩트를 만듦 (측정 ④, 2026-07-31)

**증상.** 결손(dropout) 실험은 제거된 점을 sentinel 상수로 치환(injection)한다. sentinel 값을
바꿔가며 ICCAS 원본 ckpt(icros: A=`2026-07-11_17-59-32`, B=`2026-07-10_17-32-17`)를 평가하니
**A>B 결과가 sentinel 에 좌우됨** — sentinel 주입은 결손 강건성 측정 도구로 부적합.

dropout err_xy (0%→100%) / 100% 생존율:
| sentinel | A(recon) 80% | A 100%생존 | B(jepa) 80% | B 100%생존 | 판정 |
|---|---|---|---|---|---|
| **mount** (0.325,0,0.045) 학습일치 | 0.326 | 57% | 0.215 | 90% | A급락·B완만 = **ICCAS 주장** |
| **origin** (0,0,0) | 0.115 | 45% | 0.169 | 30% | **A·B 둘 다 평탄, 격차 소멸** |
| mean (유효점 평균=imputation) | 0.081 | 41% | 0.169 | 41% | 둘 다 평탄(결손 약함) |
| below (0,0,−5) 극단 | 0.588 | 1% | 0.688 | 3% | 둘 다 즉시 붕괴 |

**결정적 증거 = mount vs origin.** 둘 다 로봇 근처(33cm 차) 고정 상수로 **동등하게 자의적**인데
(mean 처럼 imputation 도, below 처럼 극단도 아님), A 가 mount 에선 급락(0.079→0.53)·origin 에선
평탄(0.079→0.12). **동급의 두 상수가 반대 결론** → "재구성이 결손에 취약"이 아니라 **특정 좌표가
max-pool 을 hijack**(PointNet max-pool 이 지배적 per-point feature 하나에 장악됨). train-consistent
방어(반론: mount 만 정답)도 무력 — 학습 안 한 origin 에서 A 가 *더* 강건하므로.

**판정(사용자 확정).** mount↔origin 반전 = 방법론 실패. sentinel 주입으로는 "A 가 실제로 결손에
강건한가"를 원리적으로 답할 수 없음(hijack↔imputation 사이 요동). **진짜 마스킹 필수**([D]4):
제거 점을 max-pool 에서 아예 제외(masked_fill −inf, 어떤 값도 주입 안 함) → 인코더 mask 입력 +
학습 시 무효 hit 도 마스킹(train/eval 일관) + **재학습**. 기존 sentinel-학습 ckpt 는 재사용 불가.

**논문 함의.** ICCAS A>B 결과가 mount 아티팩트일 가능성 큼. 진짜 마스킹 재학습에서 A>B 가
남는지가 논문 운명을 정함(남으면 진짜 표현 성질, 사라지면 전면 재구성). sentinel 옵션은
`eval_pc.py --sentinel` 로 보존(mount 기본=ICCAS 재현, 검증됨). 도구: [[je-loco-sentinel-artifact]].

## JEPA 지평 확대(①)도 무신호 — 이 과제가 예측을 요구 안 함 (2026-08-07)

④done마스킹+②skill+residual 위에서 지평 스윕 k∈{5,15,27}(residual=true, seed 1,2, 7000 iter):
- **skill score 양수(~0.5)로 예측 작동 시작**(residual 덕). 하지만 **지평 무관 평평**(k5·15·27 다 ~0.5).
  skill 은 과제난이도 정규화 + ego-motion 얽힘이라 "지형 선행" 판정 못 함.
- 결손 강건성(dropout·occlusion 둘 다, model_7000): **지평 효과 없음, seed 분산이 지배**.
  | | clean err | 배율 | 실명생존(개별) |
  |---|---|---|---|
  | k5  | 0.210 | 2.4x | 95%(100/91) |
  | k15 | 0.211 | 2.1x | 95%(90/100) |
  | k27 | 0.168 | 2.2x | 67%(64/71) |
  같은 k 안 seed 분산(예: occ 60%생존 k5=92/18)이 k 간 차이를 압도. k27 clean 만 약간 우위(미미).
  유보: k27 은 T=32 제약상 valid_t=5(JEPA 신호 굶주림) → "k27 나쁨"에 신호부족 교란. 공정한 k=50 은 T 확대 필요.

**메타 결론(강함):** conditioning 실패 + 지평 무신호가 함께 가리킴 — **이 과제(중간지형·전진보행·마스킹학습)는
결손 강건성이 proprioception 으로 이미 포화**(실명 생존 높음)라 예측이 기여할 여지 없음. 정밀도는
재구성(A) 우위(마스킹 반전). **즉 이 과제엔 예측 표현이 이길 축이 없음.** 예측의 가치를 보이려면 **시각이
load-bearing 한 과제**(정밀 foothold·gap = 로드맵 Stage 2 / [D]5 Head C) 필요. → **전략 전환 후보.**

## JEPA conditioning(k=5)은 B 를 향상 못 시킴 — 재설계 필요 확정 (2026-08-05)

predictor 에 command/action 조건 추가 ablation(마스킹 위, jepa, k=5, ~11400 iter, seed 1,2):
- cmd 단독: 2800 iter 에서 예측 개선 미미(z_p 에 명령 이미 있어 중복) → kill.
- act·both 를 11400 까지 → 마스킹 dropout eval. 기존 none(maskB, 17800):

| | clean err 평균 | 100% 결손 생존(s1/s2) |
|---|---|---|
| none | **0.168** | ~100% / ~100% |
| act  | 0.207 | 97% / **49%** |
| both | 0.237 | 46% / **3%** |

**conditioning 이 clean 정밀도를 악화(0.17→0.21~0.24)시키고, 완전 실명 시 생존을 붕괴시킴**(both s2
=3%). 학습 정규화 예측오차(jepa/z_e_std²)도 셋 다 ~0.3 로 동일(2800→11400 수렴하며 초기 이점 소멸).
(유보: none 17800 vs cond 11400 iter 불일치. 단 방향은 학습지표+eval 일치.)

**원인=진단 입증:** k=5(0.1초)면 0.5m/s 에서 5cm 이동 → z_e(t+k)≈z_e(t) → 최적 predictor≈identity →
conditioning 신호가 흡수됨. **"예측이 의미를 가지려면 지평을 늘려야 한다"가 실험으로 확정.** 다음=
JEPA 재설계(HANDOFF "JEPA 재설계": ④done마스킹+②skill score → ①다중지평 k∈{5,15,27}). [[je-loco-masking-reversal]]

## 핵심 결과 — 진짜 마스킹에서 ICCAS A>B 는 성립하지 않음 (2026-08-03)

마스킹 재학습(A=recon·B=jepa × seed 1,2, 17800 iter) 체크포인트를 **마스킹 dropout 스윕**으로
평가. 쉬운 지형(레벨 2.5)·어려운 지형(레벨 9, 계단 12cm) **둘 다**:

**완전 실명(100% dropout) 시:**
| | ICCAS mount sentinel | 마스킹·쉬운 | 마스킹·어려운(lv9) |
|---|---|---|---|
| A(recon) err 배율 | 6.7x | 1.3x | 1.3x |
| A 실명 생존 | **57%(붕괴)** | 99.6% | 95.5% |
| B(jepa) 실명 생존 | 90% | ~100% | ~100% |

**어느 헤드도 안 무너짐** — 어려운 지형에서 완전 실명해도 생존 95~100%. ICCAS 의 A 붕괴(생존 57%)가
sentinel 제거만으로 사라짐 = **A>B 는 sentinel max-pool hijack 아티팩트로 최종 확정.** 오히려
A s1(좋은 seed)이 clean·실명 모두 err 최저(0.13/0.18), 실명 시 유일하게 제대로 걸음(속도 0.59).

**seed 트레이드오프 발견:** A 는 분산 큼(s1 우수 err 0.13 / s2 나쁨 err 0.45, 실명 시 얼어붙음),
B 는 일관되나 평범(두 seed ~0.25). "예측이 결손강건"이 아니라 "재구성=정밀하나 학습불안정 /
예측=안정적이나 평범"이라는 다른 축. 결손 강건성 자체는 A(최소 s1) 우위. n=2 라 seed≥5 로 확정 필요.

**논문 방향:** ICCAS "예측>재구성 결손강건" 주장 폐기. 재프레이밍 후보 — (a) sentinel 주입이
max-pool 을 hijack 해 허위 차이를 만든다(방법론 규명, mount↔origin 반전 + 마스킹 소거로 입증),
(b) 마스킹 일관 학습 → 표현 무관 depth 강건성, (c) 재구성 정밀도 우위는 강건성 손해 없이 유지.
평가 도구: `eval_pc.py --terrain_level N`(어려운 지형), 마스킹 결손(sentinel 없음). [[je-loco-masking-reversal]]

## 매트릭스 확정 설정

**스크립트 커리큘럼 + projector ON + 진짜 마스킹 결손.** 전부 검증됨. sentinel 주입 결손 금지(아티팩트).
multi-seed 필수(최소 3, **A 분산 커서 권장 5**).

## 완료·미결
- ✅ **height_scanner 타깃**([D]3): 카메라 footprint 로 정렬(x∈[0.5,2.0], 144셀). recon loss
  0.083→0.0035 검증. "ICCAS 재현 → 개선 baseline" 으로 성격 변화(논문에 명시).
- ⏳ **진짜 마스킹**([D]4): 위 교란 3 판정으로 필수 확정. 인코더 mask + 재학습.
- **명령속도 커리큘럼** `lin_vel_cmd_levels` 는 여전히 적응형. 매트릭스 첫 run 에서 A/B 간
  곡선이 갈리는지 로그로 확인 후 판단(옵션 1, 사용자 선택).
- **결손 = sentinel 주입**([D]4): 상수 `_HOLE_VALUE` 치환이 max-pool 을 지배(A 가 severity 0.2 에서
  포화). 마스킹(−inf) 기반 재측정 필요(측정 ④). ICCAS 핵심 결과의 sentinel-아티팩트 여부 판정.
