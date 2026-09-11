# ICRA 2027 · II. 관련 연구

> 모든 인용은 2026-09-11 원문 대조 완료. 인용문은 원문 그대로다.

## II. 관련 연구

### A. 외수용을 시간에 걸쳐 축적하는 설계

시각 사족 보행에서 외수용 센서는 제어 주파수보다 느리게 갱신된다. DreamWaQ++ 는 정책을
50 Hz 로 돌리면서 외수용 표본화율을 10 Hz 로 두고, 이 불일치를 문제로 명시한다 —
*"This asynchrony introduces non-negligible delays into the control loop, yielding degraded
performance."* 즉 낮은 외수용 갱신률은 우리가 가정한 조건이 아니라 이 분야가 이미 문제로
지목한 운용 조건이다.

그들의 해법은 **기하학적 기억**이다. 최근 K 개 측정의 점을 이어 붙여 로봇 주위에 더 조밀한
점군을 만들고 — *"a memory structure that generates a denser point cloud … by concatenating
points from the last K measurements"* — 과거 점군을 현재 몸통 프레임으로 SE(3) 변환해
정렬한다. 이 변환은 IMU 자세 측정과 **상태 추정 네트워크가 예측한 몸통 선속도의 적분**을
결합해 매 제어 루프마다 갱신된다.

여기에 두 가지 의존이 생긴다. 하나는 자세 추정이며, 그것도 추정된 속도를 적분하므로 오차가
누적된다. 다른 하나는 무엇을 축적할지의 형식으로, 이 경우 3D 점군 버퍼다.

TRANS 는 명시적 지형 지도 대신 LiDAR SLAM 에 기대는데, 바로 그 자세 추정이 사족 보행에서
흔들린다고 실기 결과로 보고한다 — *"we observed that high-frequency vibrations from
quadrupedal locomotion often induced localization drift in the LiDAR SLAM algorithm, degrading
real-world navigation accuracy."* 실험 전반에서 관측된 문제라고도 적는다 — *"A recurring
challenge observed across all experimental scenarios was localization drift, primarily induced
by high-frequency structural vibrations."*

두 논문을 함께 놓으면 축적 기반 설계의 구조가 드러난다. 사각지대를 건너려면 과거 관측을
현재 프레임으로 옮겨야 하고, 그러려면 자세가 필요하며, 그 자세는 로봇이 걸을 때 가장
흔들린다. 본 연구는 이 사슬을 끊는다. 축적하지 않으므로 정합이 필요 없고, 정합하지 않으므로
자세 추정이 필요 없다.

### B. 지형 표현의 형식

축적되는 것이 무엇이든 그 형식은 사람이 고른다. START 는 높이맵을 지각과 행동 사이의
**명시적 중간 표현**으로 삼는다 — *"we introduce local terrain heightmap as an explicit
intermediate representation between perception and output actions."* 이 선택이 무엇을 잃는지는
저자들 자신이 향후 과제로 적는다. 가장자리 보존 기법을 도입하고 — *"finer edge-preservation
mechanisms in heightmap refinement to better retain critical terrain features"* — 더 표현력
있는 중간 표현으로 나아가겠다는 것이다 — *"more expressive intermediate representations, such
as voxels, that encode terrain geometry with higher fidelity and completeness."*

DreamWaQ++ 도 다른 경로로 같은 기준에 묶인다. 대조 손실의 양성 앵커가 **GT 높이 스캔을
인코딩한 값**이므로 — *"the encoded ground-truth height scan, which is used as the positive
anchor for the contrastive loss"* — 잠재 벡터가 사람이 정한 지형 기술 쪽으로 끌려간다.

본 연구의 인코더는 어떤 지형 기술도 정답으로 쓰지 않는다. 표현은 자신의 미래를 예측하도록만
학습되므로, 무엇을 담을지 정하는 형식이 없다. 다만 이 논문의 실험 지형은 모두 높이장이므로
형식을 규정하지 않아 얻는 이득은 여기서 측정되지 않는다. 우리가 측정하는 것은 B 절이 아니라
A 절의 사슬을 끊은 결과다.

### C. 고유수용 이력과 잠재 상태

DreamWaQ 는 고유수용 이력 H = 5 를 CENet 으로 압축해 몸통 속도와 지형 특성을 추정한다.
본 연구의 `z_p` 경로와 속도 디코더는 이 설계를 따른다. 다만 DreamWaQ 는 외수용 센서를 쓰지
않으므로 사각지대 문제 자체가 발생하지 않는다.

### D. 본 연구의 위치

| | 사각지대를 메우는 방법 | 자세 추정 | 기억 형식 |
|---|---|---|---|
| DreamWaQ++ | K 프레임 점군 SE(3) 정합 | 필요 (IMU + 속도 적분) | 3D 점군 버퍼 |
| TRANS | LiDAR SLAM 지도 | 필요 (드리프트 보고됨) | SLAM 지도 |
| START | 높이맵 복원 | — | 높이맵 |
| **본 연구** | **정책의 잠재 순환** | **불필요** | **없음(학습됨)** |

우리는 이들과 직접 비교하지 않는다. 주장하는 것은 이들이 틀렸다는 것이 아니라, 정합 없는
단일 프레임 설계가 이 과제 난이도에서 도달하는 수준이며, 그 수준이 축적 기구가 요구하는
갱신률보다 낮은 5 Hz 에서도 유지된다는 것이다.
