# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""JELocoDistillRunner — DAgger 증류 + 표현 보조손실 + 학습 중 관측 노후화 주입.

왜 이게 필요한가 (2026-08-31, n=5 판정 후):
  기존 3b 는 비교축이 **사전학습 인코더 초기화 하나**뿐이었다. 그런데 8000 iter 동안 teacher
  행동이라는 강한 지도 신호가 인코더를 다시 빚어, 초기화 차이가 씻겨나간다. 실제로 5400 iter
  에 있던 분리가 8000 iter 에서 사라졌고, n=5 최종 판정에서 세 조건(jepa/recon/scratch)의
  무결손 성능·공간 결손·시간 결손이 모두 시드 노이즈 안에서 구별되지 않았다.

  더 결정적으로, 관측이 항상 신선한 설정에서는 **예측 능력을 쓸 자리가 없다.** Phase 2 로그는
  JEPA 가 예측을 실제로 배웠음을 보여준다(copy-baseline 대비 skill k=100 에서 0.584). 배운
  능력이 정책에 도달하지 못한 것이다.

바뀐 것 셋:
  1. 예측기가 **정책 안에** 있다 — 정책 입력 [z_e(o), z_p, ẑ_e(o+k)]. 배포 시에도 살아 있고
     행동손실이 직접 통과하므로 씻겨나갈 수 없다. (model.predictor_in_policy)
  2. 보조손실이 증류 **내내** 걸린다 — λ·L_jepa(k=100). 사전학습 초기화가 아니라 지속 개입.
  3. 학습 중 관측 노후화 d ~ U[0, d_max] 주입(에피소드 상수) — 예측기에 실제 할 일을 준다.
     근거: 어려운 지형 freeze 1.0 = −35pp 로, blind(−26pp)보다 9pp 더 해로운 최악의 고장.

통제: 세 조건 모두 predictor_in_policy=True 로 **구조·파라미터 수를 맞춘다.** 다른 것은
이 예측기 MLP 를 무엇이 학습시키느냐뿐이다 —
  jepa   : 행동손실 + L_jepa(자기지도)          ← 실기 데이터로 학습 가능
  recon  : 행동손실 + L_recon(특권 GT 높이맵)   ← 시뮬 전용
  none   : 행동손실만
teacher 는 노후화되지 않은 특권 관측을 보므로, DAgger 의 감독 신호는 항상 참 상태 기준이다.
"""

from __future__ import annotations

import os
import time

import torch

from rsl_rl.runners import DistillationRunner
from rsl_rl.utils import check_nan

from .mdp_pc import TrainStaleness
from .repr_aux import ReprAuxMixin


class JELocoDistillRunner(ReprAuxMixin, DistillationRunner):
    """DAgger 증류 + (velocity / recon / jepa) 보조손실 + 관측 노후화."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        self._init_repr_aux(train_cfg)

        d_max = int(train_cfg.get("train_staleness_max", 0))
        self._stale = TrainStaleness(d_max) if d_max > 0 else None
        print(f"[JELoco] 학습 중 관측 노후화: "
              f"{'d ~ U[0, %d] 스텝 (에피소드 상수)' % d_max if self._stale else 'OFF'}")

        k = getattr(self, "_jepa_k", 0)
        if self._jepa_optimizer is not None and self.cfg["num_steps_per_env"] <= k:
            raise ValueError(
                f"num_steps_per_env={self.cfg['num_steps_per_env']} 가 jepa_k={k} 이하다. "
                f"(t, t+k) 쌍이 하나도 안 생겨 JEPA 손실이 조용히 0 이 된다.")

    # ── ReprAuxMixin 훅 ─────────────────────────────────────────────────────
    def _repr_model(self):
        return self.alg.student

    def _repr_storage(self):
        return self.alg.storage

    # ── 롤아웃 + 학습 ───────────────────────────────────────────────────────
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if not self.alg.teacher_loaded:
            raise ValueError("teacher 파라미터 미로드 — 증류 불가.")
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length))

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    # 노후화는 **학생이 보는 점군에만** 주입한다. teacher 는 별도 관측 그룹
                    # ("teacher", 특권)을 쓰므로 영향받지 않는다 → 감독 신호는 참 상태 기준.
                    # storage 에도 노후화된 관측이 들어가는데, 지연이 에피소드 상수라 저장열은
                    # 진짜 관측열의 시간 이동본이고 (t, t+k) 간격은 정확히 k 로 보존된다.
                    if self._stale is not None:
                        obs["pointcloud"] = self._stale(obs["pointcloud"])
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device),
                                           dones.to(self.device))
                    if self._stale is not None:
                        # 리셋된 env 는 이전 지형 프레임을 버리고 지연을 다시 뽑는다.
                        self._stale.notify_done(dones)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)
                stop = time.time(); collect_time = stop - start; start = stop

            loss_dict = self.alg.update()
            if self._vel_optimizer is not None:
                loss_dict["vel_estimation"] = self._vel_loss_step()
            if self._recon_optimizer is not None:
                loss_dict["height_recon"] = self._recon_loss_step()
            if self._jepa_optimizer is not None:
                loss_dict.update(self._jepa_loss_step())

            stop = time.time(); learn_time = stop - start
            self.current_learning_iteration = it
            self.logger.log(
                it=it, start_it=start_it, total_it=total_it,
                collect_time=collect_time, learn_time=learn_time, loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=None,
            )
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    # ── 체크포인트: aux optimizer 상태 포함 (EMA target 은 student 서브모듈이라 자동) ──
    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        extra = self._aux_state_dict()
        if not extra:
            return
        saved = torch.load(path, weights_only=False)
        saved.update(extra)
        torch.save(saved, path)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True,
             map_location: str | None = None):
        infos = super().load(path, load_cfg, strict, map_location)
        self._load_aux_state(torch.load(path, weights_only=False, map_location=map_location))
        return infos
