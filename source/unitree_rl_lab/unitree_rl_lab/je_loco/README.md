# JE-Loco 백본 스켈레톤 (단계 1~2)

**논문(잠정)**: Predictive vs. Reconstructive Representations for Resilient Quadrupedal
Locomotion under Depth Degradation · 로봇: Unitree Go2 · 센서: RealSense D435i depth 단독.

이 패키지는 CLAUDE.md 의 확정 방향을 코드로 옮긴 **뼈대**다. 핵심 규율:
> 백본(depth encoder + proprioception VAE + Mixer + implicit terrain-aware RL)은 **고정**,
> 표현 학습 **헤드만 교체**(A 재구성 / B 예측·JEPA)하여 통제 비교한다.

이 규율(절대 규율 1)이 A/B 비교의 정당성을 만든다 → 코드에서 헤드는 인터페이스(`ReprHead`)로
완전히 분리되어 있고, 학습 루프는 어떤 헤드인지 **모른다**.

## 디렉터리

```
je_loco/
├── obs_spec.py            # ★ 45차원 proprioception 명세 (단일 진리원, 인덱스 범위 포함)
├── models/
│   ├── encoders.py        # DepthCNNEncoder(depth→z_e), ProprioVAE(45×H→z_p + v̂ 추정, CENet)
│   ├── mixer.py           # implicit fusion [z_p, z_e, ẑ_e] → context (mlp | attention)
│   ├── heads/
│   │   ├── base.py        # ReprHead ABC (compute_loss, update_target) + EMA target 유틸
│   │   ├── recon.py       # 헤드 A: z_e→ĥ, L_recon=MSE(h_GT, ĥ)   [DreamWaQ++ 재현]
│   │   └── jepa.py        # 헤드 B: [z_e,z_p]→ẑ_e(t+k) predictor 골격 (compute_loss=stub, 단계 3)
│   ├── policy.py          # asymmetric actor-critic (critic 만 GT v_t 특권)
│   └── backbone.py        # JELocoBackbone: 위를 조립 + 헤드 주입
├── envs/
│   ├── je_loco_env_cfg.py # Go2 + D435i TiledCamera depth + GT heightmap(21×21) env cfg
│   └── mdp_depth.py       # depth 관측 함수 (image-like, 정규화)
├── train/
│   ├── builder.py         # config → 백본 (HEAD_REGISTRY 로 헤드 주입)
│   ├── rollout.py         # RolloutStorage(GAE) + ProprioHistory 히스토리 버퍼
│   ├── ppo.py             # ★ JELocoPPO: 헤드 불가지론 복합손실 PPO 업데이트
│   ├── train.py           # JELocoLearner(정적 배치 손실 스모크/진단) + --smoke
│   └── config/{head_a,head_b}.yaml   # PPO 하이퍼파라미터 = BasePPORunnerCfg 정합
└── tests/test_shapes.py   # Isaac 없이 pure-torch 스모크 (shape·역전파 경로)
```

## 확정 설계 결정 (근거)

- **proprioception = 45차원**: `[ω(3), g(3), cmd(3), θ(12), θ̇(12), a_{t-1}(12)]`. base 선속도 v_t 는
  **넣지 않는다**(실기 드리프트). 대신 (a) critic privileged 로 GT v_t 제공, (b) ProprioVAE 가
  z_p 와 함께 v̂_t 추정(CENet, `L_est=MSE(v̂,v)`). 47차원 아님. → `obs_spec.py`.
- **depth = image-like (TiledCamera)**: point cloud 아님, depth CNN 이 기본(절대 규율 2). PointNeXt 는 부록.
- **헤드 인터페이스**: `ReprHead.compute_loss(z_e, z_p, batch)` + `update_target()`.
  헤드 B(EMA target encoder, cross-modal 예측)를 염두에 두고 설계 → 헤드 A 는 `update_target` no-op.
- **재구성 vs 예측 비교는 z_e 에만**(절대 규율 6). z_p 는 두 헤드 공통 유지.
- **explicit foothold 없음**(절대 규율 3): Mixer 융합 latent 를 그대로 쓰는 implicit 정책.

## 실행

```bash
# (1) pure-torch 스모크 — Isaac 불필요
cd source/unitree_rl_lab
python -m unitree_rl_lab.je_loco.tests.test_shapes
python -m unitree_rl_lab.je_loco.obs_spec          # 45차원·인덱스 출력

# (2) 복합손실 역전파 스모크 (헤드 A, 합성 배치)
python -m unitree_rl_lab.je_loco.train.train --config unitree_rl_lab/je_loco/train/config/head_a.yaml --smoke

# (3) 단계 1 depth 렌더 처리량 벤치마크 (Isaac, GPU 필요)
cd ../../
pkill -f benchmark_depth 2>/dev/null   # 잔여 프로세스 정리(자원 경합 방지)
bash scripts/je_loco/benchmark_depth_sweep.sh 256 1024 2048 4096
#  → logs/je_loco/benchmark_depth.json  (depth on/off env-steps/s, slowdown, GPU mem)
```

### 벤치마크 결과 (2026-07-07, RTX 5070 Ti 16GB) → `BENCHMARK_depth.md`
depth ON = OFF 대비 **일관되게 ~2× 저하**(규모 무관), 4096 병렬환경 + depth ON 이 **13.2GB 로 16GB 에 적합**.
→ **depth CNN encoder 기본 확정**(point cloud 변환은 처리량 때문에 강제되지 않음, 부록 옵션). 상세: `BENCHMARK_depth.md`.

## 아직 안 된 것 (다음 단계)

- **헤드 B(JEPA) 구현** (단계 3): `jepa.py::compute_loss` — target encoder(EMA+stop-grad)로
  `z_e(t+k)` 뽑아 predictor 출력과 예측손실. collapse 방어(분산·rank 모니터: `base.latent_stats`).
- **Isaac rollout 어댑터** (단계 2, 게이트 0): `train.py::main` 의 TODO — `gym.make("Unitree-Go2-JELoco")`
  rollout → rsl_rl PPO surrogate 를 `batch["l_rl"]` 로 넣어 헤드 A 로 DreamWaQ++ 재현.
- **depth 결손 augmentation** (단계 4/§6): 홀·occlusion·거리클리핑·저조도 dropout.
- **probe decoder / controllability 지표** (단계 4, §2.1): latent 축 개입 정량화.
- **미확정(교수 확인 필요)**: control-relevant vs manipulable latent 중 무엇을 1차 축으로.
