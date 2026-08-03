# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""JELocoOnPolicyRunner — stock rsl_rl OnPolicyRunner + 논문 aux 손실(별도 optimizer).

두 헤드 공통(백본):
  velocity 추정  v̂ = vel_decoder(proprio_encoder(proprio_hist)) = g(z_p), L_est = MSE(v̂, v).
                별도 optimizer(proprio_encoder + vel_decoder). **recon·jepa 모두 동일** → 통제.
표현 헤드(z_e 에만, 교체 = 유일한 변수):
  recon(A): recon_decoder(z_e) vs GT height. optimizer(pc_encoder + recon_decoder).
  jepa(B) : predictor([z_e(t), sg z_p(t)]) vs sg·EMA_target(t+k) + VICReg.
            optimizer(pc_encoder + predictor). z_p 는 detach → proprio_encoder 안 건드림(rule 6).

관측 시퀀스는 alg.storage.observations(T,N,·)에서 획득(update 후에도 유효). GT 선속도 =
critic 그룹 첫 3차원. proprio 는 이미 H=5 히스토리 관측(obs["policy"]=45×5)이라 별도 윈도우 불필요.
"""

from __future__ import annotations

import os
import time
import torch
import torch.nn.functional as F

from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import check_nan


class JELocoOnPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner + velocity(z_p) aux + 교체 표현 헤드(recon/jepa)."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        actor = self.alg.actor

        # ── velocity 추정 (두 헤드 공통) : proprio_encoder + vel_decoder ──
        self._lambda_vel = float(train_cfg.get("lambda_vel", 0.5))
        self._vel_chunks = max(1, int(train_cfg.get("vel_num_chunks", 4)))
        vel_lr = float(train_cfg.get("vel_learning_rate", 1.0e-3))
        if self._lambda_vel > 0.0 and hasattr(actor, "vel_decoder") and hasattr(actor, "proprio_encoder"):
            self._vel_params = list(actor.proprio_encoder.parameters()) + list(actor.vel_decoder.parameters())
            self._vel_optimizer: torch.optim.Optimizer | None = torch.optim.Adam(self._vel_params, lr=vel_lr)
            print(f"[JELoco] velocity aux ON (lambda={self._lambda_vel}, chunks={self._vel_chunks}, lr={vel_lr})")
        else:
            self._vel_params = []
            self._vel_optimizer = None
            print("[JELoco] velocity aux OFF")

        # ── Head A(recon) : pc_encoder + recon_decoder ──
        self._lambda_recon = float(train_cfg.get("lambda_recon", 1.0))
        self._recon_chunks = max(1, int(train_cfg.get("recon_num_chunks", 4)))
        recon_lr = float(train_cfg.get("recon_learning_rate", 1.0e-3))
        if self._lambda_recon > 0.0 and hasattr(actor, "recon_decoder"):
            self._recon_params = list(actor.pc_encoder.parameters()) + list(actor.recon_decoder.parameters())
            self._recon_optimizer: torch.optim.Optimizer | None = torch.optim.Adam(self._recon_params, lr=recon_lr)
            print(f"[JELoco] Head A recon ON (lambda={self._lambda_recon}, chunks={self._recon_chunks}, lr={recon_lr})")
        else:
            self._recon_params = []
            self._recon_optimizer = None

        # ── Head B(jepa) : pc_encoder + predictor (proprio_encoder 제외 — z_p detach) ──
        self._lambda_jepa = float(train_cfg.get("lambda_jepa", 1.0))
        self._jepa_chunks = max(1, int(train_cfg.get("jepa_num_chunks", 4)))
        self._jepa_k = int(train_cfg.get("jepa_k", 5))
        self._ema_tau = float(train_cfg.get("ema_tau", 0.996))
        jepa_lr = float(train_cfg.get("jepa_learning_rate", 1.0e-3))
        self._lambda_var = float(train_cfg.get("lambda_var", 1.0))
        self._lambda_cov = float(train_cfg.get("lambda_cov", 0.04))
        self._var_gamma = float(train_cfg.get("var_gamma", 1.0))
        # predictor conditioning — actor 에서 읽음(단일 소스, model cfg 가 진리원). 불일치 불가능.
        self._cond_action = bool(getattr(actor, "_cond_action", False))    # 지평 평균 행동(12)
        self._cond_command = bool(getattr(actor, "_cond_command", False))  # 명령 c_t(3)
        if hasattr(actor, "jepa_predictor"):
            self._jepa_params = list(actor.pc_encoder.parameters()) + list(actor.jepa_predictor.parameters())
            # projector 도 jepa optimizer 가 함께 학습(VICReg 이 이 출력에 걸리므로).
            _use_proj = getattr(actor, "use_projector", False)
            if _use_proj:
                self._jepa_params += list(actor.vic_projector.parameters())
            self._jepa_optimizer: torch.optim.Optimizer | None = torch.optim.Adam(self._jepa_params, lr=jepa_lr)
            print(f"[JELoco] Head B JEPA ON (lambda={self._lambda_jepa}, k={self._jepa_k}, "
                  f"tau={self._ema_tau}, chunks={self._jepa_chunks}, lr={jepa_lr}, projector={_use_proj}, "
                  f"cond_action={self._cond_action}, cond_command={self._cond_command})")
        else:
            self._jepa_params = []
            self._jepa_optimizer = None

    # ── velocity: v̂ = g(z_p), 두 헤드 동일 ──────────────────────────────────
    def _vel_loss_step(self) -> float:
        if self._vel_optimizer is None:
            return 0.0
        actor = self.alg.actor
        obs_seq = self.alg.storage.observations
        prop = obs_seq["policy"].reshape(-1, obs_seq["policy"].shape[-1])   # (T*N, 45*H)
        gt = obs_seq["critic"][:, :, :3].reshape(-1, 3)                     # (T*N, 3)
        m = prop.shape[0]
        chunks = torch.chunk(torch.arange(m, device=prop.device), self._vel_chunks)
        self._vel_optimizer.zero_grad()
        total = 0.0
        for ch in chunks:
            loss = F.mse_loss(actor.predict_vel(prop[ch]), gt[ch])
            (loss * self._lambda_vel / len(chunks)).backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(self._vel_params, self.alg.max_grad_norm)
        self._vel_optimizer.step()
        return total / len(chunks)

    # ── Head A: z_e → GT height 재구성 ───────────────────────────────────────
    def _recon_loss_step(self) -> float:
        if self._recon_optimizer is None:
            return 0.0
        actor = self.alg.actor
        obs_seq = self.alg.storage.observations
        pc = obs_seq["pointcloud"].reshape(-1, obs_seq["pointcloud"].shape[-1])   # (T*N, P*3)
        gt = obs_seq["height_map"].reshape(-1, obs_seq["height_map"].shape[-1])   # (T*N, hm)
        m = pc.shape[0]
        chunks = torch.chunk(torch.arange(m, device=pc.device), self._recon_chunks)
        self._recon_optimizer.zero_grad()
        total = 0.0
        for ch in chunks:
            loss = F.mse_loss(actor.reconstruct_height(pc[ch]), gt[ch])
            (loss * self._lambda_recon / len(chunks)).backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(self._recon_params, self.alg.max_grad_norm)
        self._recon_optimizer.step()
        return total / len(chunks)

    # ── Head B: [z_e(t), sg z_p(t)] → ẑ_e(t+k) 예측 + VICReg + EMA ────────────
    def _jepa_loss_step(self) -> dict[str, float]:
        if self._jepa_optimizer is None:
            return {}
        actor = self.alg.actor
        obs_seq = self.alg.storage.observations
        pc = obs_seq["pointcloud"]                        # (T, N, P*4)
        prop = obs_seq["policy"]                          # (T, N, 45*H) — 이미 히스토리
        T, N = pc.shape[0], pc.shape[1]
        k = self._jepa_k
        valid_t = torch.arange(0, T - k, device=pc.device)   # t+k<T (히스토리는 obs manager 가 pad)
        if valid_t.numel() == 0:
            return {"jepa": 0.0, "jepa_varcov": 0.0, "z_e_std": 0.0}

        # ── predictor conditioning (미래 행동/명령, storage 에서. grad 없음 = 자동 detach) ──
        cmd_seq = obs_seq["critic"][:, :, 9:12] if self._cond_command else None      # (T,N,3) 명령
        act_seq = self.alg.storage.actions if self._cond_action else None            # (T,N,12) 행동

        chunks = torch.chunk(torch.arange(N, device=pc.device), self._jepa_chunks)
        self._jepa_optimizer.zero_grad()
        tot_j = tot_vc = z_std = 0.0
        for ch in chunks:
            Nc = len(ch)
            pc_c = pc[:, ch].reshape(T * Nc, -1)
            z_e = actor.encode_exteroception(pc_c).reshape(T, Nc, -1)               # online, grad
            z_p = actor.encode_proprio(prop[:, ch].reshape(T * Nc, -1)).reshape(T, Nc, -1).detach()  # 조건, 안 shaping
            z_e_tgt = actor.target_encode(pc_c).reshape(T, Nc, -1)                  # EMA, stop-grad

            # conditioning: command c_t (3) ⊕ 지평 [t,t+k) 평균 행동 (12). cond_dim 은 k 무관 고정.
            cond_parts = []
            if cmd_seq is not None:
                cond_parts.append(cmd_seq[valid_t][:, ch])                          # (nt, Nc, 3)
            if act_seq is not None:
                aw = torch.stack([act_seq[valid_t + j][:, ch] for j in range(k)], 0).mean(0)  # (nt,Nc,12)
                cond_parts.append(aw)
            cond = torch.cat(cond_parts, dim=-1) if cond_parts else None

            pred = actor.jepa_predict(z_e[valid_t], z_p[valid_t], cond)             # (nt, Nc, 64)
            loss_j = F.mse_loss(pred, z_e_tgt[valid_t + k])

            # VICReg — z_e 가 아니라 projector(z_e) 출력에 부과(SSL 표준). z_e 백화 방지.
            # projector 없으면(ablation) h=z_e 로 구 동작. 분산 hinge + 공분산 off-diag.
            zf = z_e.reshape(-1, z_e.shape[-1])
            h = actor.vic_project(zf)
            std = torch.sqrt(h.var(dim=0) + 1e-4)
            loss_var = torch.relu(self._var_gamma - std).mean()
            hc = h - h.mean(dim=0, keepdim=True)
            cov = (hc.T @ hc) / (h.shape[0] - 1)
            off = cov - torch.diag(torch.diagonal(cov))
            loss_cov = off.pow(2).sum() / h.shape[-1]

            ((self._lambda_jepa * loss_j + self._lambda_var * loss_var
              + self._lambda_cov * loss_cov) / len(chunks)).backward()
            tot_j += loss_j.item(); tot_vc += (loss_var + loss_cov).item()
            # z_e_std 는 raw z_e 의 std(드리프트 진단용) — projection std 아님.
            z_std += torch.sqrt(zf.var(dim=0) + 1e-4).mean().item()

        torch.nn.utils.clip_grad_norm_(self._jepa_params, self.alg.max_grad_norm)
        self._jepa_optimizer.step()
        actor.ema_update_target(self._ema_tau)
        n = len(chunks)
        return {"jepa": tot_j / n, "jepa_varcov": tot_vc / n, "z_e_std": z_std / n}

    # stock learn() + (update 직후) velocity(공통) + 표현 헤드(recon 또는 jepa).
    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                stop = time.time(); collect_time = stop - start; start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            loss_dict["vel_estimation"] = self._vel_loss_step()           # 두 헤드 공통
            if self._recon_optimizer is not None:
                loss_dict["height_recon"] = self._recon_loss_step()       # Head A
            if self._jepa_optimizer is not None:
                loss_dict.update(self._jepa_loss_step())                  # Head B: jepa, jepa_varcov, z_e_std

            stop = time.time(); learn_time = stop - start
            self.current_learning_iteration = it
            self.logger.log(
                it=it, start_it=start_it, total_it=total_it,
                collect_time=collect_time, learn_time=learn_time, loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    # aux optimizer 상태도 체크포인트에 포함. target encoder(EMA)는 actor 서브모듈이라 자동 포함.
    def save(self, path: str, infos: dict | None = None) -> None:
        super().save(path, infos)
        extra = {}
        if self._vel_optimizer is not None:
            extra["vel_optimizer_state_dict"] = self._vel_optimizer.state_dict()
        if self._recon_optimizer is not None:
            extra["recon_optimizer_state_dict"] = self._recon_optimizer.state_dict()
        if self._jepa_optimizer is not None:
            extra["jepa_optimizer_state_dict"] = self._jepa_optimizer.state_dict()
        if not extra:
            return
        saved = torch.load(path, weights_only=False)
        saved.update(extra)
        torch.save(saved, path)

    def load(self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None):
        infos = super().load(path, load_cfg, strict, map_location)
        loaded = torch.load(path, weights_only=False, map_location=map_location)
        if self._vel_optimizer is not None and "vel_optimizer_state_dict" in loaded:
            self._vel_optimizer.load_state_dict(loaded["vel_optimizer_state_dict"])
        if self._recon_optimizer is not None and "recon_optimizer_state_dict" in loaded:
            self._recon_optimizer.load_state_dict(loaded["recon_optimizer_state_dict"])
        if self._jepa_optimizer is not None and "jepa_optimizer_state_dict" in loaded:
            self._jepa_optimizer.load_state_dict(loaded["jepa_optimizer_state_dict"])
        return infos
