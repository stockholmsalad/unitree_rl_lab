# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""Phase 2-b — 오프라인 표현 사전학습 (JEPA vs recon 대조, 같은 teacher 데이터).

순수 PyTorch — Isaac 불필요, Z790 에서 실행 가능. 인코더는 rsl_rl_pc.model.PointCloudEncoder
그대로 (마스크드 max-pool) → 저장된 state_dict 를 Phase 3 에서 pc_encoder 에 바로 로드.

목표 (--objective):
  jepa : action-conditioned 다중지평 JEPA.
         pred([z_t, act↓, k_emb]) → Δz = sg[Ē(pc_{t+k})] − sg[Ē(pc_t)]   (residual, ② 교훈)
         + VICReg(projector(z_t)) (붕괴방지, z_e 자유 — projector 교훈)
         + EMA target encoder. 지평 k 는 {5,15,25,50} 샘플 — T=32 제약 없음(① 해소).
         지표 = copy-baseline skill (>0 이어야 예측 성공. 온라인에서 검증된 자).
  recon: 같은 인코더로 heightmap(187, privileged GT) 회귀 — S-recon 대조군 (사전학습 통제).

done-crossing 쌍 제외(④ 교훈 오프라인 적용). --mask_aug 로 결손(valid→0) 증강(결손 하 표현 유지).

사용:
  python scripts/je_loco/pretrain_repr.py --data datasets/teacher_v1 --objective jepa  --out pretrained/jepa_v1
  python scripts/je_loco/pretrain_repr.py --data datasets/teacher_v1 --objective recon --out pretrained/recon_v1
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from unitree_rl_lab.je_loco.rsl_rl_pc.model import PointCloudEncoder

P = argparse.ArgumentParser(description="오프라인 표현 사전학습 (Phase 2-b)")
P.add_argument("--data", type=str, required=True, help="collect_teacher_data 출력 디렉터리")
P.add_argument("--objective", type=str, required=True, choices=["jepa", "recon"])
P.add_argument("--out", type=str, required=True)
P.add_argument("--epochs", type=int, default=20)
P.add_argument("--steps_per_epoch", type=int, default=2000)
P.add_argument("--batch", type=int, default=1024)
P.add_argument("--lr", type=float, default=1e-3)
P.add_argument("--ks", type=str, default="5,15,25,50", help="jepa 예측 지평 후보(쉼표)")
P.add_argument("--act_ds", type=int, default=8, help="행동열 다운샘플 스텝수(가변 k 대응)")
P.add_argument("--mask_aug", type=float, default=0.3, help="context 결손 증강 최대비율(0=끔). U(0,p) 샘플")
P.add_argument("--ema_tau", type=float, default=0.996)
P.add_argument("--lambda_var", type=float, default=1.0)
P.add_argument("--lambda_cov", type=float, default=0.04)
P.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
P.add_argument("--seed", type=int, default=0)
args = P.parse_args()

torch.manual_seed(args.seed); np.random.seed(args.seed)
KS = [int(x) for x in args.ks.split(",")]
K_MAX = max(KS)
PROPRIO_DIM, ACT_DIM = 45, 12


class ShardDataset:
    """npz 샤드 로드(fp16, RAM). done-safe (t, t+k) 쌍 샘플."""

    def __init__(self, data_dir: str):
        paths = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
        assert paths, f"샤드 없음: {data_dir}"
        self.pc, self.pol, self.act, self.done = [], [], [], []
        for p in paths:
            z = np.load(p)
            self.pc.append(z["pc"]); self.pol.append(z["policy"])
            self.act.append(z["act"]); self.done.append(z["done"])
        self.T = self.pc[0].shape[0]; self.N = self.pc[0].shape[1]
        self.pc_dim = self.pc[0].shape[-1]
        self.hmap_dim = self.pol[0].shape[-1] - PROPRIO_DIM
        total = sum(a.shape[0] * a.shape[1] for a in self.pc)
        print(f"[data] {len(paths)} shards, {total:,} env-steps, pc={self.pc_dim}, hmap={self.hmap_dim}")

    def sample(self, batch: int, k: int, device: str):
        """(pc_t, hmap_t, act_seq[t:t+k], pc_{t+k}) — 창 내 done 있으면 재샘플."""
        s_idx = np.random.randint(len(self.pc), size=batch)
        out_t = np.empty(batch, dtype=np.int64); out_n = np.empty(batch, dtype=np.int64)
        for i in range(batch):
            s = s_idx[i]
            for _ in range(50):                                   # done-free 창 재시도
                t = np.random.randint(0, self.T - k); n = np.random.randint(self.N)
                if not self.done[s][t:t + k, n].any():
                    break
            out_t[i], out_n[i] = t, n
        def gather(arrs, t_off=0):
            return np.stack([arrs[s_idx[i]][out_t[i] + t_off, out_n[i]] for i in range(batch)])
        pc_t = torch.from_numpy(gather(self.pc).astype(np.float32)).to(device)
        pc_k = torch.from_numpy(gather(self.pc, k).astype(np.float32)).to(device)
        hmap = torch.from_numpy(gather(self.pol).astype(np.float32)[:, PROPRIO_DIM:]).to(device)
        # 행동열 t..t+k-1 → act_ds 균등 다운샘플 (가변 k 를 고정 차원으로)
        sel = np.linspace(0, k - 1, args.act_ds).round().astype(int)
        acts = np.stack([
            np.stack([self.act[s_idx[i]][out_t[i] + j, out_n[i]] for j in sel]) for i in range(batch)
        ]).astype(np.float32)                                     # (B, act_ds, 12)
        act_seq = torch.from_numpy(acts.reshape(batch, -1)).to(device)
        return pc_t, hmap, act_seq, pc_k


def mask_augment(pc: torch.Tensor, max_p: float) -> torch.Tensor:
    """valid 채널 dropout (결손 증강). 좌표 무주입 — 마스킹 규율 유지."""
    if max_p <= 0:
        return pc
    B = pc.shape[0]; Pn = pc.shape[-1] // 4
    x = pc.view(B, Pn, 4).clone()
    p = torch.rand(B, 1, device=pc.device) * max_p                # 샘플별 결손율 U(0, max_p)
    drop = torch.rand(B, Pn, device=pc.device) < p
    x[..., 3] = x[..., 3] * (~drop).float()
    return x.view(B, -1)


def main():
    os.makedirs(args.out, exist_ok=True)
    ds = ShardDataset(args.data)
    num_points = ds.pc_dim // 4
    dev = args.device

    enc = PointCloudEncoder(num_points, 64).to(dev)
    params = list(enc.parameters())

    if args.objective == "recon":
        head = nn.Sequential(nn.Linear(64, 256), nn.ELU(), nn.Linear(256, ds.hmap_dim)).to(dev)
        params += list(head.parameters())
    else:
        import copy as _copy
        tgt_enc = _copy.deepcopy(enc)
        for q in tgt_enc.parameters():
            q.requires_grad_(False)
        k_emb = nn.Embedding(len(KS), 8).to(dev)
        predictor = nn.Sequential(
            nn.Linear(64 + args.act_ds * ACT_DIM + 8, 256), nn.ELU(),
            nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 64),
        ).to(dev)
        projector = nn.Sequential(nn.Linear(64, 128), nn.ELU(), nn.Linear(128, 128)).to(dev)
        params += list(predictor.parameters()) + list(projector.parameters()) + list(k_emb.parameters())

    opt = torch.optim.Adam(params, lr=args.lr)
    print(f"[pretrain] objective={args.objective} ks={KS} mask_aug≤{args.mask_aug} device={dev}")

    for ep in range(args.epochs):
        logs = {}
        for _ in range(args.steps_per_epoch):
            if args.objective == "recon":
                pc_t, hmap, _, _ = ds.sample(args.batch, 1, dev)
                z = enc(mask_augment(pc_t, args.mask_aug))
                loss = F.mse_loss(head(z), hmap)
                logs.setdefault("recon", []).append(loss.item())
            else:
                ki = np.random.randint(len(KS)); k = KS[ki]
                pc_t, _, act_seq, pc_k = ds.sample(args.batch, k, dev)
                z = enc(mask_augment(pc_t, args.mask_aug))         # online (증강 context)
                with torch.no_grad():
                    zt_now = tgt_enc(pc_t)                         # clean targets (EMA enc)
                    zt_fut = tgt_enc(pc_k)
                    target = zt_fut - zt_now                       # residual Δz (② 교훈)
                kv = k_emb(torch.full((z.shape[0],), ki, device=dev, dtype=torch.long))
                pred = predictor(torch.cat([z, act_seq, kv], dim=-1))
                loss_pred = F.mse_loss(pred, target)
                # copy-baseline skill: copy=Δ0. >0 = 예측 성공 (검증된 자)
                copy_mse = target.pow(2).mean()
                skill = 1.0 - loss_pred.detach() / copy_mse.clamp(min=1e-8)
                # VICReg on projector(z) — z_e 자유 (projector 교훈)
                h = projector(z)
                std = torch.sqrt(h.var(dim=0) + 1e-4)
                loss_var = torch.relu(1.0 - std).mean()
                hc = h - h.mean(dim=0, keepdim=True)
                cov = (hc.T @ hc) / (h.shape[0] - 1)
                loss_cov = (cov - torch.diag(torch.diagonal(cov))).pow(2).sum() / h.shape[-1]
                loss = loss_pred + args.lambda_var * loss_var + args.lambda_cov * loss_cov
                logs.setdefault("pred", []).append(loss_pred.item())
                logs.setdefault(f"skill_k{k}", []).append(skill.item())
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if args.objective == "jepa":                           # EMA target 갱신
                with torch.no_grad():
                    for tp, op_ in zip(tgt_enc.parameters(), enc.parameters()):
                        tp.mul_(args.ema_tau).add_(op_, alpha=1.0 - args.ema_tau)
        msg = "  ".join(f"{k_}={np.mean(v):.4f}" for k_, v in sorted(logs.items()))
        print(f"[ep {ep+1:3d}/{args.epochs}] {msg}", flush=True)

    # 저장 — Phase 3 에서 model.pc_encoder.load_state_dict 로 직접 로드 가능
    torch.save(enc.state_dict(), os.path.join(args.out, "pc_encoder.pt"))
    if args.objective == "jepa":
        torch.save({"predictor": predictor.state_dict(), "projector": projector.state_dict(),
                    "k_emb": k_emb.state_dict()}, os.path.join(args.out, "jepa_heads.pt"))
    else:
        torch.save(head.state_dict(), os.path.join(args.out, "recon_head.pt"))
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"objective": args.objective, "ks": KS, "mask_aug": args.mask_aug,
                   "epochs": args.epochs, "num_points": num_points, "data": args.data}, f, indent=2)
    print(f"[pretrain] 저장 완료 → {args.out}/pc_encoder.pt")


if __name__ == "__main__":
    main()
