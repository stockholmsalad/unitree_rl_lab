# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""JE-Loco — Predictive vs. Reconstructive Representations for Resilient Quadrupedal Locomotion.

CLAUDE.md 가 확정 방향. 백본(encoder + proprio VAE + Mixer + implicit AC)은 고정하고
표현 학습 **헤드만 교체**(A: 재구성 / B: 예측·JEPA)하여 통제 비교한다.

서브패키지:
  obs_spec : 45차원 proprioception 관측 명세 (단일 진리원)
  models/  : encoders, mixer, heads/(base, recon, jepa), policy, backbone
  envs/    : Isaac Lab 환경 (Go2 + D435i TiledCamera depth + GT height map)
  train/   : 공통 학습 루프 + head_a/head_b config
"""

from .obs_spec import DEFAULT_SPEC, JELocoObsSpec

__all__ = ["JELocoObsSpec", "DEFAULT_SPEC"]
