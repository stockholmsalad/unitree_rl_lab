# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""표현 보조손실(velocity / Head A recon / Head B jepa) — PPO·증류 러너 공용.

runner.py(JELocoOnPolicyRunner, Phase 3 PPO)에만 있던 수식을 여기로 옮겼다. Phase 3b 증류에도
같은 손실을 걸어야 하는데(2026-08-31 결정), 수식이 두 벌이면 한쪽만 고쳐져 조용히 갈라진다.
러너는 두 훅만 제공하면 된다:
  _repr_model()   → 손실을 걸 모델 (PPO: alg.actor · 증류: alg.student)
  _repr_storage() → (T,N,·) 관측 시퀀스를 든 storage
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ReprAuxMixin:
    """velocity(z_p) + 교체 표현헤드(recon/jepa) 보조손실. 각각 **별도 optimizer**."""

    # ── 러너가 구현하는 훅 ───────────────────────────────────────────────────
    def _repr_model(self):
        raise NotImplementedError

    def _repr_storage(self):
        raise NotImplementedError

    # ── 설정 ────────────────────────────────────────────────────────────────
    def _init_repr_aux(self, train_cfg: dict) -> None:
        model = self._repr_model()

        # velocity 추정 (조건 무관 공통) : proprio_encoder + vel_decoder
        self._lambda_vel = float(train_cfg.get("lambda_vel", 0.0))
        self._vel_chunks = max(1, int(train_cfg.get("vel_num_chunks", 4)))
        self._has_critic_obs = "critic" in self._repr_storage().observations.keys()
        if self._lambda_vel > 0.0 and not self._has_critic_obs:
            print("[JELoco] velocity aux 요청됐으나 critic 관측 그룹이 없다 → OFF (GT 선속도 불가)")
            self._lambda_vel = 0.0
        if self._lambda_vel > 0.0 and hasattr(model, "vel_decoder") and hasattr(model, "proprio_encoder"):
            self._vel_params = list(model.proprio_encoder.parameters()) + list(model.vel_decoder.parameters())
            self._vel_optimizer = torch.optim.Adam(
                self._vel_params, lr=float(train_cfg.get("vel_learning_rate", 1.0e-3)))
            print(f"[JELoco] velocity aux ON (lambda={self._lambda_vel})")
        else:
            self._vel_params, self._vel_optimizer = [], None

        # Head A(recon) : pc_encoder + recon_decoder — 감독이 **특권**(시뮬 GT 높이맵)
        self._lambda_recon = float(train_cfg.get("lambda_recon", 0.0))
        self._recon_chunks = max(1, int(train_cfg.get("recon_num_chunks", 4)))
        if self._lambda_recon > 0.0 and hasattr(model, "recon_decoder"):
            self._recon_params = list(model.pc_encoder.parameters()) + list(model.recon_decoder.parameters())
            self._recon_optimizer = torch.optim.Adam(
                self._recon_params, lr=float(train_cfg.get("recon_learning_rate", 1.0e-3)))
            print(f"[JELoco] Head A recon ON (lambda={self._lambda_recon})")
        else:
            self._recon_params, self._recon_optimizer = [], None

        # Head B(jepa) : pc_encoder + predictor (+ projector). z_p 는 detach → proprio 경로 불변.
        self._lambda_jepa = float(train_cfg.get("lambda_jepa", 0.0))
        self._jepa_chunks = max(1, int(train_cfg.get("jepa_num_chunks", 4)))
        self._jepa_k = int(train_cfg.get("jepa_k", 100))
        self._ema_tau = float(train_cfg.get("ema_tau", 0.996))
        self._lambda_var = float(train_cfg.get("lambda_var", 1.0))
        self._lambda_cov = float(train_cfg.get("lambda_cov", 0.04))
        self._var_gamma = float(train_cfg.get("var_gamma", 1.0))
        self._jepa_residual = bool(train_cfg.get("jepa_residual", True))
        self._cond_action = bool(getattr(model, "_cond_action", False))
        self._cond_command = bool(getattr(model, "_cond_command", False))
        if self._lambda_jepa > 0.0 and hasattr(model, "jepa_predictor") and hasattr(model, "target_pc_encoder"):
            self._jepa_params = list(model.pc_encoder.parameters()) + list(model.jepa_predictor.parameters())
            if getattr(model, "use_projector", False):
                self._jepa_params += list(model.vic_projector.parameters())
            self._jepa_optimizer = torch.optim.Adam(
                self._jepa_params, lr=float(train_cfg.get("jepa_learning_rate", 1.0e-3)))
            print(f"[JELoco] Head B JEPA ON (lambda={self._lambda_jepa}, k={self._jepa_k}, "
                  f"tau={self._ema_tau}, residual={self._jepa_residual})")
        else:
            self._jepa_params, self._jepa_optimizer = [], None

    # ── velocity ────────────────────────────────────────────────────────────
    def _vel_loss_step(self) -> float:
        if self._vel_optimizer is None:
            return 0.0
        model, obs_seq = self._repr_model(), self._repr_storage().observations
        prop = obs_seq["policy"].reshape(-1, obs_seq["policy"].shape[-1])
        # GT 선속도 = critic 그룹 첫 3차원. teacher 그룹은 proprio 가 각속도로 시작하므로 못 쓴다.
        gt = obs_seq["critic"][:, :, :3].reshape(-1, 3)
        chunks = torch.chunk(torch.arange(prop.shape[0], device=prop.device), self._vel_chunks)
        self._vel_optimizer.zero_grad()
        total = 0.0
        for ch in chunks:
            loss = F.mse_loss(model.predict_vel(prop[ch]), gt[ch])
            (loss * self._lambda_vel / len(chunks)).backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(self._vel_params, 1.0)
        self._vel_optimizer.step()
        return total / len(chunks)

    # ── Head A: z_e → GT 높이맵 ─────────────────────────────────────────────
    def _recon_loss_step(self) -> float:
        if self._recon_optimizer is None:
            return 0.0
        model, obs_seq = self._repr_model(), self._repr_storage().observations
        pc = obs_seq["pointcloud"].reshape(-1, obs_seq["pointcloud"].shape[-1])
        gt = obs_seq["height_map"].reshape(-1, obs_seq["height_map"].shape[-1])
        chunks = torch.chunk(torch.arange(pc.shape[0], device=pc.device), self._recon_chunks)
        self._recon_optimizer.zero_grad()
        total = 0.0
        for ch in chunks:
            loss = F.mse_loss(model.reconstruct_height(pc[ch]), gt[ch])
            (loss * self._lambda_recon / len(chunks)).backward()
            total += loss.item()
        torch.nn.utils.clip_grad_norm_(self._recon_params, 1.0)
        self._recon_optimizer.step()
        return total / len(chunks)

    # ── Head B: [z_e(t), sg z_p(t)] → ẑ_e(t+k) + VICReg + EMA ───────────────
    def _jepa_loss_step(self) -> dict[str, float]:
        if self._jepa_optimizer is None:
            return {}
        model, storage = self._repr_model(), self._repr_storage()
        obs_seq = storage.observations
        pc, prop = obs_seq["pointcloud"], obs_seq["policy"]
        T, N = pc.shape[0], pc.shape[1]
        k = self._jepa_k
        if T - k <= 0:
            # 지평이 롤아웃 길이보다 길면 쌍이 하나도 없다. 조용히 0 을 내면 "JEPA 를 켰는데
            # 아무 일도 안 일어나는" 상태가 되므로 명시적으로 알린다.
            raise ValueError(
                f"jepa_k={k} ≥ 롤아웃 길이 T={T}. num_steps_per_env 를 k 보다 크게 잡아야 한다.")
        valid_t = torch.arange(0, T - k, device=pc.device)

        # done 경계 마스킹: t~t+k 사이 리셋이 있으면 t+k 는 새 지형(순간이동) → 쌍에서 제외
        dones = storage.dones.squeeze(-1)
        win_reset = torch.stack([dones[valid_t + j] for j in range(k)], 0).sum(0)
        pair_ok = (win_reset == 0)

        chunks = torch.chunk(torch.arange(N, device=pc.device), self._jepa_chunks)
        self._jepa_optimizer.zero_grad()
        tot_j = tot_vc = z_std = tot_skill = 0.0
        for ch in chunks:
            Nc = len(ch)
            pc_c = pc[:, ch].reshape(T * Nc, -1)
            z_e = model.encode_exteroception(pc_c).reshape(T, Nc, -1)
            z_p = model.encode_proprio(prop[:, ch].reshape(T * Nc, -1)).reshape(T, Nc, -1).detach()
            z_e_tgt = model.target_encode(pc_c).reshape(T, Nc, -1)

            pred = model.jepa_predict(z_e[valid_t], z_p[valid_t], None)

            tgt_now, tgt_next = z_e_tgt[valid_t], z_e_tgt[valid_t + k]
            if self._jepa_residual:
                target, copy_ref = tgt_next - tgt_now, torch.zeros_like(tgt_next)
            else:
                target, copy_ref = tgt_next, tgt_now
            m = pair_ok[:, ch].float()
            denom = m.sum().clamp(min=1.0)
            loss_j = (((pred - target).pow(2).mean(-1)) * m).sum() / denom
            copy_mse = (((copy_ref - target).pow(2).mean(-1)) * m).sum() / denom
            tot_skill += float(1.0 - (loss_j.detach() / copy_mse.clamp(min=1e-8)))

            # VICReg — z_e 가 아니라 projector(z_e) 출력에 부과(z_e 백화 방지, SSL 표준)
            zf = z_e.reshape(-1, z_e.shape[-1])
            h = model.vic_project(zf)
            std = torch.sqrt(h.var(dim=0) + 1e-4)
            loss_var = torch.relu(self._var_gamma - std).mean()
            hc = h - h.mean(dim=0, keepdim=True)
            cov = (hc.T @ hc) / (h.shape[0] - 1)
            loss_cov = (cov - torch.diag(torch.diagonal(cov))).pow(2).sum() / h.shape[-1]

            ((self._lambda_jepa * loss_j + self._lambda_var * loss_var
              + self._lambda_cov * loss_cov) / len(chunks)).backward()
            tot_j += loss_j.item(); tot_vc += (loss_var + loss_cov).item()
            z_std += torch.sqrt(zf.var(dim=0) + 1e-4).mean().item()

        torch.nn.utils.clip_grad_norm_(self._jepa_params, 1.0)
        self._jepa_optimizer.step()
        model.ema_update_target(self._ema_tau)
        n = len(chunks)
        # jepa_skill: copy-baseline 대비 skill(>0=예측 성공). 예측이 실제로 되는지의 주 지표.
        return {"jepa": tot_j / n, "jepa_varcov": tot_vc / n,
                "z_e_std": z_std / n, "jepa_skill": tot_skill / n}

    # ── 체크포인트 ──────────────────────────────────────────────────────────
    def _aux_state_dict(self) -> dict:
        out = {}
        for name in ("vel", "recon", "jepa"):
            opt = getattr(self, f"_{name}_optimizer", None)
            if opt is not None:
                out[f"{name}_optimizer_state_dict"] = opt.state_dict()
        return out

    def _load_aux_state(self, loaded: dict) -> None:
        for name in ("vel", "recon", "jepa"):
            opt = getattr(self, f"_{name}_optimizer", None)
            key = f"{name}_optimizer_state_dict"
            if opt is not None and key in loaded:
                opt.load_state_dict(loaded[key])
