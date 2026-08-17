# JE-Loco 문헌 조사 (2026-08-17)

계기: "온라인 JEPA 를 RL 과 동시에 학습시킨다는 전제부터 틀린 것 아닌가, JEPA 가 실제로
어떻게 성공했는지를 먼저 파악했어야 하지 않나"는 자기점검. 3개 축으로 병렬 조사.

1. locomotion 에서 시각 인코더 **사전학습**이 통하는가
2. **JEPA/latent-predictive** 가 제어에서 검증된 적 있는가 (예측 vs 재구성)
3. vision teacher-student 의 **표준 레시피**는 무엇인가

> 조사 방식: 웹 검색·논문 본문(ar5iv/HTML) 자동 판독. 손실 가중치·샘플 수 등 세부 수치는
> 인용 전 PDF 원문 재확인 필요. 정성적 주장(행동-BC 지배, DAgger 보편성 등)은 복수 출처 교차확인됨.

---

## 축 1 — 시각 인코더 사전학습

### 핵심: "사전학습 → 동결 → RL" 은 legged 분야에서 **비표준**이고 효과 증거가 희박

**유일한 직접 ablation (DeFM 2026, ANYmal 사다리 오르기):**

| 인코더 | 성공률 |
|---|---|
| 동결 DeFM (depth 6040만 장 SSL 사전학습) | 90.14% |
| scratch CNN (정책과 함께 학습) | **90.45%** |

6천만 장 foundation model 이 scratch 를 못 이김. 저자 주장도 "더 낫다"가 아니라
**"동등하되 연산이 훨씬 적다"** — 이득은 성능이 아니라 **효율**.

**동결 vs fine-tune (DeFM, OOD = 학습노이즈 → Kinect 노이즈):**

| 인코더 | 학습 노이즈 | OOD |
|---|---|---|
| DeFM frozen | 0.809 | 0.486 |
| ImageNet frozen | 0.658 | **0.004** (붕괴) |
| scratch finetuned | 0.777 | 0.725 |
| DeFM finetuned | 0.894 | **0.876** |

**fine-tune 한 모든 변형이 동결한 모든 변형을 OOD 에서 이김.** VC-1(NeurIPS 2023)도 동일 결론.

**SOTA locomotion 은 아무도 시각 인코더를 사전학습하지 않음.** Extreme Parkour(ICRA 2024)는
명시: *"the depth encoding network is **not pre-trained**"*. Miki(Sci.Rob. 2022), Agarwal(CoRL 2022),
Robot Parkour Learning(CoRL 2023), NVM(CVPR 2023) 전부 동일. PIE(RA-L 2024)는 자기 동기를
*"복잡한 사전학습 지형 재구성 모듈을 피하는 것"*이라 서술.

**분야의 설명:** locomotion 은 시각 병목이 아니다. 지형 신호는 저차원(발 주변 heightmap)이라
큰 사전학습 표현이 작은 task-specific 표현보다 나을 여지가 적다.

**조사자 결론:** *"이 ablation 을 돌리는 그룹이 거의 없어 '도움된다는 증거 없음'은 '분야가 안 봤다'에
가깝다. 방어 가능한 주장은 **입증 책임이 사전학습 쪽에 있다**는 것이며, 그런 주장엔 같은 구조·
같은 예산의 joint-training 대조군 + frozen vs finetuned 대조군이 필수."*

---

## 축 2 — JEPA / latent-predictive 가 제어에서 검증됐는가

### 핵심: **"예측 > 재구성"은 확립된 결과가 아니다.** legged 증거는 **0건**

**JEPA 원본 계열은 제어를 거의 측정한 적 없음:**

| 논문 | 평가 |
|---|---|
| I-JEPA (CVPR 2023) | ImageNet linear probe, 전이. **제어 0** |
| V-JEPA (2024) | 비디오 분류(frozen). **제어 0** |
| LeJEPA (2025, LeCun) | **이미지 분류만** |
| V-JEPA 2-AC (2025) | 이 계열 최초 실로봇 제어 |

→ I-JEPA/V-JEPA 를 "예측 표현이 제어에 좋다"의 근거로 인용하는 것은 **외삽**.

**V-JEPA 2-AC 로봇 결과 (Franka, image-goal MPC, task 당 n=10):**

| Task | V-JEPA 2-AC | Octo(BC) | Cosmos(픽셀 확산 WM = 재구성 베이스라인) |
|---|---|---|---|
| Grasp cup | 65% | 15% | 60% |
| Pick-place cup | 80% | 15% | 0% |
| Pick-place box | 65% | 10% | 0% |

n=10 → 65 vs 60 은 95% CI ±30pp 로 **무의미**. pick-place 격차만 살아남으나 Cosmos 0% 는
공정 비교보다 베이스라인 파손에 가까움. **from-scratch 대조군 없음.** 액션당 16초,
카메라 위치 민감(저자들이 "여러 위치를 수동 시도"라 명시).

**RL 에서 예측 vs 재구성 — 정면 모순:**

*Voelcker et al., "When does Self-Prediction help?" (RLC 2024)* — 가장 엄밀한 통제 실험
(MinAtar + DMC-15, matched setup):
- TD 와 함께 **auxiliary** 로 쓸 때: latent self-prediction 이 5개 중 **3개**에서 재구성보다 나음(미미)
- **단독** 표현학습으로 쓸 때: **재구성이 유의하게 나음**. self-prediction 은 *"여러 경우에
  유의미한 특징을 전혀 학습하지 못함"*
- 메커니즘: 예측 목적은 정보를 **더하는** 게 아니라 nuisance 차원을 **버리는(invariance)** 효과.
  그래서 **task signal 이 옆에 없으면 작동 안 함**

*Crafter (Dreamer-CDP 2026):*

| 방법 | 점수 |
|---|---|
| DreamerV3 (**재구성**) | 14.5 ± 1.6 |
| DreamerPro (재구성-free) | 4.7 ± 0.5 |
| MuDreamer (재구성-free) | 7.3 ± 2.6 |
| Dreamer-CDP (JEPA 식, 수리 후) | 16.2 ± 2.1 |

재구성-free 가 2~3배 참패하다 대규모 엔지니어링 후 겨우 역전. ablation: **JEPA predictor loss 만
제거하면 3.2** (전체 16.2) → JEPA 항은 **단독으로 작동하지 않음**. Voelcker 와 동일 결론.

*TD-MPC2(ICLR 2024)* 는 latent 예측 쪽 최강 데이터포인트(DreamerV3 대비 104 task 우세)이나
**재구성을 ablate 하지 않으며**, 붕괴를 막는 건 reward·value 예측(= task 감독)이지 SSL 정규화가 아님.

### ★ 예측 지평 (horizon) — 문헌에서 가장 확실하고 실행 가능한 부분

*"What Can Latent World Models Know?" (2026):*

| 예측 설정 | 물체 위치 복원 R² | 다운스트림 MPC |
|---|---|---|
| **단일 스텝** | **0.04** (버려짐) | 20% |
| 다중 지평 {1, 4, 16} | **0.89** | 38~44% |
| + cross-modal target | 0.98 | 57% |

**메커니즘("lazy equilibrium"):** 짧은 지평에선 정적/느린 상태가 예측 손실에 거의 기여하지 않아
인코더가 **버린다**. 그리고 결정적으로 — *"anti-collapse regularizer 로는 못 고친다. 정규화는
임베딩의 **분포**를 제약할 뿐 **내용**을 제약하지 않는다."* 또한 직접 16-step 예측이
autoregressive 합성보다 나음(0.10 vs 0.19).

동일 방향 타 논문: RC-aux(2026, 다중지평 K-step 롤아웃 감독), Ground-JEPA(사족, **보행 주기
전체 H=12** 를 덮어야 작동한다 주장, Research Square **preprint·미심사·sim-only**),
SkyJEPA(쿼드로터, T=20 = 1.0s@20Hz, **재구성 latent 베이스라인 명시하고 이김** —
실기 원추종 RMSE 0.39→0.24 m, preprint).

### locomotion 증거 = 없음

조사자: *"legged robot 에서 JEPA 식 latent-predictive 목적을 재구성 베이스라인과 ablation 한
peer-reviewed 논문을 하나도 찾지 못했다."*

- **SLR (CoRL 2024)**: 가장 근접(사족, latent self-learning, 실기 계단/바위). 예측 vs 재구성 미분리
- manipulation 은 일부 있으나(DINO-WM ICML 2025 등) **DINO-WM 자체 ablation 이 승리 원인을
  "DINOv2 patch feature vs global feature"** 로 지목 — 인코더 선택이지 목적함수가 아님

---

## 축 3 — Teacher-Student 표준 레시피

### 지배적 레시피

```
Phase 1  scandots/heightmap + privileged 물리 → PPO → teacher   (1~15B samples)
Phase 2  DAgger — 환경을 [student 행동]으로 굴리고, 방문 상태마다 teacher 에게 a_t 질의 →
         ‖â−a‖² 최소화.  ★ CNN 인코더는 GRU 정책 헤드와 **함께** 학습 (따로도, 동결도 아님)
        (+ 선택) λ‖ẑ−z‖² latent 회귀, λ_rec·L_recon (λ_rec 0.01~0.5)
Phase 3  (선택) RL fine-tune 으로 잔여 격차 회수
```

**전이 매체 = 행동.** 서베이 원문: *"**Actions are the primary transfer medium** … the student
learns to mimic the teacher's action outputs."* 우리가 쓰는 **RSL-RL 자체가 distillation
알고리즘을 딱 하나** 구현: *"DAgger 유사 BC — student 정책을 롤아웃해 데이터를 모으고,
expert 행동으로 relabel 하고, 그것으로 student 를 학습."*

**직접 비교:** 「Now You See That」(2026) — depth end-to-end RL **54.0%** vs privileged
distillation **98.9%**.

### 분포 이동 대처 (분야가 만장일치인 유일 항목)

- **on-policy student 롤아웃 + teacher relabel(DAgger)** — Lee, Miki, Agarwal(24-step 언롤),
  Cheng, Zhuang, Parkour in the Wild, RSL-RL 기본값. **"teacher 는 완벽히 걷고 student 는
  비틀거린다"를 막는 핵심 부품**
- **노이즈 주입** — Miki 의 3-체제 exteroception 노이즈(nominal 60% / large offset 30% /
  large noise 10%), ANYmal Parkour 는 elevation map 을 최대 7.5 cm 이동
- **선택적 teacher override** — Extreme Parkour 의 Mixture-of-Teacher-Student(heading 채널,
  오라클과 0.6 rad 이내일 때만 student 예측 사용)
- **BC/DAgger 혼합** — VIRAL ablation: 순수 BC(α=1.0)는 *"손실은 빨리 줄지만 정책이
  부서지기 쉬움"*, α=0.5 가 *"배포 성공률을 크게 개선"*
- **알려진 파손** — 「Distilling Realizable Students from Unrealizable Teachers」(2025):
  state aliasing(teacher 다중 상태 → student 단일 관측 → 라벨 충돌) 하에서 DAgger 자체가
  깨지며, 질의를 늘릴수록 realizability error 가 단조 증가

### RMA 가 latent 만으로 되는 이유 (혼동 주의)

RMA 는 latent 회귀만 쓴다. 그러나 **정책 헤드가 teacher·student 간 동결·공유**된다.
같은 헤드를 쓰므로 **latent 를 맞추는 것 = 행동을 맞추는 것**이 동치. 새 정책 헤드를 RL 로
다시 배우는 경우엔 이 동치가 성립하지 않음.

### "인코더 따로 학습" 선례 — 있으나 결정적으로 다름

ANYmal Parkour(Sci.Rob. 2024), Duan(ICRA 2024), Hoeller(RA-L 2022)는 행동 증류 없이 지각을
따로 학습한다. 그러나:

1. 인코더 출력이 **명시적·의미 규정된 표현(heightmap/occupancy)** — 임의 SSL latent 아님.
   인터페이스가 규정돼 있어 따로 학습해도 조립됨
2. RL 정책은 **그 표현의 ground-truth 에, 예측기 오차통계에 맞춘 노이즈를 얹어** 학습.
   인코더 출력으로 학습하지 않음. 지각은 **배포 시점에 교체**
3. 따라서 **동결 latent 위에서 RL 을 다시 돌리는 단계가 아예 없음**

### 비용

distillation 은 teacher 학습의 **⅓~½ 이하**: Lee 12h→4h, RMA 24h→3h(샘플 15배 적음),
Agarwal 15B→훨씬 적음, Extreme Parkour 8~10h→5~10h. 단 wall-clock 은 **depth 렌더링**이
지배해 샘플 수 대비 시간 이득은 작음.

---

## 우리 파이프라인에 대한 판정

**우리가 한 것:**
```
Phase 2  teacher 롤아웃 데이터로 인코더만 오프라인 SSL 사전학습
Phase 3  인코더 동결 → 정책은 PPO 로 처음부터 재학습 (행동 모방 없음)
```

**빠진 부품 2개:**

1. **행동 수준 감독** — 이 분야 depth 논문 전부가 주 손실로 사용
2. **on-policy 보정** — 인코더는 teacher 상태분포에서 학습됐는데, 새 RL 정책은 초반에
   인코더가 본 적 없는 상태를 돌아다니고, **동결이라 적응 불가**. 표준 DAgger 는 인코더가
   루프 안에서 student 방문 상태로 갱신됨 — 우리가 제거한 게 정확히 그 수리 장치

**조사자 결론:** *"불투명한 인코더를 동결하고 그 위에서 새로 RL 을 돌리는 논문을 legged
분야에서 하나도 찾지 못했다. 이 부재 자체가 발견 — 검증된 대안이 아니라 미개척 영역."*

**권고(개입 비용 순):**
1. **Distillation-PPO**: `L = α·L_DAgger + β·L_PPO` — 문헌에 "둘 중 하나만보다 엄밀히 낫다"
2. **동결 해제**(warm-up 후 낮은 LR) — VIRAL 방식
3. **명시적 heightmap 타깃으로 latent 를 접지**하고 RL 은 노이즈 얹은 GT heightmap 으로 학습
   (ANYmal Parkour/Duan 방식, Science Robotics 수준 실기 검증됨)

---

## 우리 실측과의 대조 — 문헌이 우리 관측을 전부 설명함

| 우리 관측 | 문헌 설명 |
|---|---|
| 온라인 JEPA 가 끝내 예측 못 함(skill ≤ 0) | k=5 = lazy equilibrium. 단일/근사-단일 스텝은 **적극적으로 해로움**(R² 0.04). 문서화된 실패 모드 |
| VICReg projector 없으면 붕괴 | 순수 예측 목적을 동반항 없이 작동시킨 사례가 문헌에 없음 |
| 오프라인 skill 이 지평 단조 증가 (k5 .35 < k15 .46 < k25 .48 < k50 .52) | 문헌이 예측한 방향과 정확히 일치. 독립 재현 |
| S_* 에서 jepa ≈ recon ≤ scratch (최종 성능) | DeFM 무승부(90.14 vs 90.45)와 동일. 동결이 천장 |
| 실명(마스킹)시켜도 생존 95~100% | "locomotion 은 시각 병목이 아니다" |
| 사전학습이 **커리큘럼 통과를 2.8× 가속**(EXPERIMENTS.md 곡선 재분석) | DeFM: 사전학습 이득 = 성능 아닌 **효율** |

---

## 논문 프레이밍에 대한 함의

"JEPA 가 낫다"를 증명하려 하지 말고:

> **"legged locomotion 에서 예측 vs 재구성 표현을 통제 비교한 최초 연구"**

근거:
- legged 에서 예측 vs 재구성 ablation = **peer-reviewed 0건**
- legged 에서 사전학습 vs scratch ablation = **1건뿐, 그것도 무승부**
- 우리는 scratch·frozen 대조군 + 지평 스윕을 이미 보유

무승부여도 결과다(DeFM 도 무승부를 그대로 실었다). 단 **입증 책임은 사전학습 쪽**이므로
같은 구조·같은 예산의 대조군을 반드시 붙일 것.

---

## 주요 출처

**사전학습·teacher-student**
- [Lee et al. 2020](https://arxiv.org/abs/2010.11251) · [Miki et al. 2022](https://arxiv.org/abs/2201.08117) · [RMA](https://arxiv.org/abs/2107.04034)
- [Agarwal CoRL 2022](https://arxiv.org/abs/2211.07638) · [Extreme Parkour](https://ar5iv.labs.arxiv.org/html/2309.14341) ([repo](https://github.com/chengxuxin/extreme-parkour)) · [Robot Parkour Learning](https://arxiv.org/abs/2309.05665)
- [Neural Volumetric Memory](https://arxiv.org/pdf/2304.01201) · [ANYmal Parkour](https://arxiv.org/abs/2306.14874) · [Duan ICRA 2024](https://arxiv.org/abs/2309.14594)
- [SoloParkour](https://gepetto.github.io/SoloParkour/) · [WMP](https://arxiv.org/html/2409.16784) · [Parkour in the Wild](https://arxiv.org/html/2505.11164v1)
- [Distilling Realizable Students](https://arxiv.org/html/2505.09546v1) · [Distillation-PPO](https://arxiv.org/html/2503.08299) · [VIRAL](https://arxiv.org/html/2511.15200)
- [Now You See That](https://arxiv.org/html/2602.06382) · [DeFM](https://arxiv.org/html/2601.18923v1) · [RSL-RL](https://arxiv.org/html/2509.10771v1) · [survey](https://arxiv.org/html/2406.01152v1)

**JEPA·표현학습**
- [I-JEPA](https://arxiv.org/pdf/2301.08243) · [V-JEPA 2](https://arxiv.org/abs/2506.09985) · [LeJEPA](https://arxiv.org/abs/2511.08544)
- [When does Self-Prediction help? (RLC 2024)](https://arxiv.org/abs/2406.17718) · [SPR](https://openreview.net/forum?id=uCQfPZwRaUu) · [BYOL-Explore](https://proceedings.neurips.cc/paper_files/paper/2022/hash/ced0d3b92bb83b15c43ee32c7f57d867-Abstract-Conference.html)
- [TD-MPC2](https://arxiv.org/pdf/2310.16828) · [DINO-WM](https://arxiv.org/abs/2411.04983) · [SLR (CoRL 2024)](https://proceedings.mlr.press/v270/chen25e.html)
- [Dreamer-CDP](https://arxiv.org/html/2603.07083) · [RC-aux](https://arxiv.org/html/2605.07278v1) · [What Can Latent World Models Know?](https://arxiv.org/html/2607.27017v2) · [Reconstruction or Semantics?](https://arxiv.org/abs/2605.06388)
- [SkyJEPA](https://arxiv.org/abs/2606.23444) · [Ground-JEPA (preprint, 미심사)](https://www.researchsquare.com/article/rs-10511499/v1)
