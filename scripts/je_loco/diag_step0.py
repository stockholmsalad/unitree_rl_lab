# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""Step 0 오프라인 진단 — 재학습 없이 기존 체크포인트로 두 가지를 판정한다.

(1) **A 가 실제로 재구성을 했는가**
    Loss/height_recon = 9e-4 라는 절대값만으로는 판정 불가. height map 의 분산을 몰라서다.
    정규화 재구성 오차 = MSE / var(h) = 1 - R^2 로 환산한다.
      ~1.0 → 평균만 맞히는 수준(사실상 미학습)   ~0.0 → 완전 복원

(2) **z_e 스케일 교란 가설**  (A 는 비정규화, B 는 VICReg 로 std→1 강제)
    obs_normalization=False 라 z_e 가 raw 로 GRU 에 들어간다. A·B 의 z_e/z_p 실제 스케일을
    같은 축에서 잰다. A 도 크게 드리프트했는데 커리큘럼 정체가 없었다면 스케일 가설은 약해진다.
    B 는 정규화 예측오차(1-R^2)도 함께 산출 → A 의 (1) 과 같은 축에서 비교 가능.

체크포인트의 repr_head 는 state_dict 키로 **자동 판별**한다 (agent_cfg 하드코딩 무시).
평가 분포는 eval_pc.py 와 동일(seed 42 + terrain 커리큘럼 정지) → A·B 동일 지형.

사용:
  python -u scripts/je_loco/diag_step0.py --headless --num_envs 256 --steps 300 \
      --load_run 2026-07-11_17-59-32 --checkpoint model_24000.pt
"""

import argparse
import json
import os
from importlib.metadata import version

from isaaclab.app import AppLauncher
import cli_args  # isort: skip

parser = argparse.ArgumentParser(description="JE-Loco Step 0 offline diagnostics (no retraining).")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--steps", type=int, default=300, help="통계 수집 스텝(warmup 이후)")
parser.add_argument("--warmup", type=int, default=120, help="측정 전 안정화 스텝")
parser.add_argument("--eval_seed", type=int, default=42, help="eval_pc.py 와 동일 → A·B 동일 지형")
parser.add_argument("--fix_terrain", action="store_true", default=True)
parser.add_argument("--no_fix_terrain", dest="fix_terrain", action="store_false")
parser.add_argument("--out", type=str, default="", help="JSON 저장 경로(기본: diag_step0_<tag>.json)")
parser.add_argument("--task", type=str, default="Unitree-Go2-JELoco-PC")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import gymnasium as gym
import isaaclab_tasks  # noqa: F401
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
import unitree_rl_lab.je_loco.rsl_rl_pc  # noqa: F401
from unitree_rl_lab.je_loco.rsl_rl_pc.mdp_pc import DEGRADATIONS
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg

torch.backends.cudnn.enabled = False   # Blackwell sm_120 우회(의도적) — 되살리지 말 것


def detect_head(ckpt_path: str) -> str:
    """체크포인트 state_dict 키로 repr_head 판별. agent_cfg 하드코딩(recon)에 의존하지 않는다."""
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    keys = set()
    for v in sd.values():
        if isinstance(v, dict):
            keys |= {k for k in v.keys() if isinstance(k, str)}
    has_recon = any("recon_decoder" in k for k in keys)
    has_jepa = any("jepa_predictor" in k for k in keys)
    if has_recon and not has_jepa:
        return "recon"
    if has_jepa and not has_recon:
        return "jepa"
    if has_jepa and has_recon:
        raise RuntimeError("체크포인트에 recon·jepa 헤드가 모두 있음 — 수동 확인 필요")
    return "none"


def stats(x: torch.Tensor) -> dict:
    """(M, D) → 차원별 std 의 평균(= 학습 로그 Loss/z_e_std 와 동일 정의) 등."""
    per_dim_std = x.std(dim=0)
    return {
        "mean_dim_std": per_dim_std.mean().item(),          # TB Loss/z_e_std 와 같은 정의
        "min_dim_std": per_dim_std.min().item(),
        "max_dim_std": per_dim_std.max().item(),
        "mean_abs": x.abs().mean().item(),
        "global_std": x.std().item(),
        "mean_dim_var": per_dim_std.pow(2).mean().item(),
    }


def main():
    env_cfg = parse_env_cfg(args_cli.task, num_envs=args_cli.num_envs, entry_point_key="play_env_cfg_entry_point")
    env_cfg.seed = args_cli.eval_seed
    if args_cli.fix_terrain and getattr(env_cfg.curriculum, "terrain_levels", None) is not None:
        env_cfg.curriculum.terrain_levels = None
        print("[diag] terrain 커리큘럼 정지 + seed 고정 → eval_pc.py 와 동일 분포")

    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    ckpt = args_cli.checkpoint
    if ckpt and ("/" in ckpt or os.path.isfile(ckpt)):
        resume_path = retrieve_file_path(ckpt)
    else:
        run = args_cli.load_run if args_cli.load_run else agent_cfg.load_run
        resume_path = get_checkpoint_path(log_root, run, ckpt if ckpt else agent_cfg.load_checkpoint)

    head = detect_head(resume_path)
    agent_cfg.actor.repr_head = head          # ← 하드코딩 대신 체크포인트 기준으로 모델 구성
    print(f"[diag] checkpoint : {resume_path}")
    print(f"[diag] repr_head  : {head}  (state_dict 키로 자동 판별)")

    env = gym.make(args_cli.task, cfg=env_cfg)
    uenv = env.unwrapped
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, version("rsl-rl-lib"))
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path,
                load_cfg={"actor": True, "critic": True, "optimizer": False, "iteration": False, "rnd": False},
                strict=True)     # 헤드 모듈까지 온전히 로드됐는지 확인(strict)
    policy = runner.get_inference_policy(device=uenv.device)
    actor = runner.alg.actor
    dev = uenv.device
    k = int(agent_cfg.to_dict().get("jepa_k", 5))

    buf = {"z_e": [], "z_p": [], "hm": [], "recon": [], "z_e_tgt": [], "done": [], "pc": []}
    obs = env.get_observations()
    total = args_cli.warmup + args_cli.steps
    print(f"[diag] rollout {total} steps ({uenv.num_envs} envs) …")
    for t in range(total):
        with torch.inference_mode():
            act = policy(obs)
            if t >= args_cli.warmup:
                pc = obs["pointcloud"]
                buf["pc"].append(pc.cpu())                     # 결손 민감도(측정②) 용 원본 점군
                buf["z_e"].append(actor.encode_exteroception(pc).cpu())
                buf["z_p"].append(actor.encode_proprio(obs["policy"]).cpu())
                buf["hm"].append(obs["height_map"].cpu())
                if head == "recon":
                    buf["recon"].append(actor.reconstruct_height(pc).cpu())
                elif head == "jepa":
                    buf["z_e_tgt"].append(actor.target_encode(pc).cpu())
            obs, _, dones, _ = env.step(act)
            if t >= args_cli.warmup:
                buf["done"].append(dones.bool().cpu())

    z_e = torch.cat(buf["z_e"]).float()                     # (T*N, 64)
    z_p = torch.cat(buf["z_p"]).float()                     # (T*N, 32)
    hm = torch.cat(buf["hm"]).float()                       # (T*N, 96)
    T, N = args_cli.steps, uenv.num_envs

    res = {
        "checkpoint": resume_path, "repr_head": head, "num_envs": N, "steps": T,
        "eval_seed": args_cli.eval_seed, "fix_terrain": bool(args_cli.fix_terrain),
        "z_e": stats(z_e), "z_p": stats(z_p), "height_map": stats(hm),
    }

    # ── z_e 의 시간 일관성 (헤드 무관 — 두 헤드 공통 진단) ─────────────────────
    # JEPA 과제가 애초에 풀 수 있는 문제인지 결정한다. 지속(persistence) 오차가 1.0 에 붙으면
    # z_e(t) 와 z_e(t+k) 가 무상관 → 어떤 예측기도 평균 예측을 못 이긴다 = 과제 자체가 ill-posed.
    # z_p 를 대조로 같이 잰다(고유수용은 히스토리라 매끄러워야 정상).
    def persistence_curve(seq: torch.Tensor) -> dict:
        S = seq.reshape(T, N, -1)
        v = seq.var(dim=0).mean().item()
        out = {}
        for kk in (1, 2, 3, 5, 10, 20, 50):
            if kk >= T:
                continue
            a = S[: T - kk].reshape(-1, S.shape[-1])
            b = S[kk:].reshape(-1, S.shape[-1])
            out[str(kk)] = (a - b).pow(2).mean().item() / max(v, 1e-12)
        return out

    res["persistence_vs_k"] = {"z_e": persistence_curve(z_e), "z_p": persistence_curve(z_p),
                               "height_map": persistence_curve(hm)}

    # ── (1) A: 정규화 재구성 오차 = MSE / var(h) = 1 - R^2 ────────────────────
    if head == "recon":
        rec = torch.cat(buf["recon"]).float()
        mse = (rec - hm).pow(2).mean().item()
        var_h = hm.var(dim=0).mean().item()                 # 차원별 분산의 평균
        res["recon"] = {"mse": mse, "var_height_map": var_h,
                        "normalized_error": mse / max(var_h, 1e-12),
                        "r2": 1.0 - mse / max(var_h, 1e-12)}

    # ── (2) B: 정규화 예측오차. done 을 건너뛰는 (t, t+k) 쌍은 제외한 값도 함께 ──
    if head == "jepa":
        ze_s = torch.cat(buf["z_e"]).float().reshape(T, N, -1)
        zp_s = torch.cat(buf["z_p"]).float().reshape(T, N, -1)
        tgt_s = torch.cat(buf["z_e_tgt"]).float().reshape(T, N, -1)
        dn = torch.stack(buf["done"]).reshape(T, N)         # step t 종료 여부
        vt = torch.arange(0, T - k)
        with torch.inference_mode():
            pred = actor.jepa_predict(ze_s[vt].reshape(-1, ze_s.shape[-1]).to(dev),
                                      zp_s[vt].reshape(-1, zp_s.shape[-1]).to(dev)).cpu()
        tgt = tgt_s[vt + k].reshape(-1, tgt_s.shape[-1])
        var_t = tgt.var(dim=0).mean().item()
        mse_all = (pred - tgt).pow(2).mean().item()
        # (t, t+k) 사이에 리셋이 없는 쌍만 — 학습 코드는 이 마스킹을 하지 않는다(기지의 근사)
        crossed = torch.stack([dn[i:i + k].any(dim=0) for i in vt]).reshape(-1)
        keep = ~crossed
        mse_valid = (pred[keep] - tgt[keep]).pow(2).mean().item() if keep.any() else float("nan")
        res["jepa"] = {
            "k": k, "mse_all_pairs": mse_all, "mse_reset_masked": mse_valid,
            "var_target": var_t,
            "normalized_error": mse_all / max(var_t, 1e-12),
            "normalized_error_reset_masked": mse_valid / max(var_t, 1e-12),
            "r2": 1.0 - mse_all / max(var_t, 1e-12),
            "frac_pairs_crossing_reset": crossed.float().mean().item(),
        }

        # ── R^2<0 원인 분해: 예측기가 망가졌나 / EMA 가 어긋났나 / 타깃이 예측불가인가 ──
        tgt_ctx = tgt_s[vt].reshape(-1, tgt_s.shape[-1])          # z̄_e(t)   — 타깃 인코더, 현재
        onl_ctx = ze_s[vt].reshape(-1, ze_s.shape[-1])            # z_e(t)   — 온라인, 현재
        onl_fut = ze_s[vt + k].reshape(-1, ze_s.shape[-1])        # z_e(t+k) — 온라인, 미래

        def nerr(a, b, v):    # 정규화 오차 = MSE(a,b)/v
            return (a - b).pow(2).mean().item() / max(v, 1e-12)

        res["jepa_decomp"] = {
            # (i) 지속(persistence) 기준선: 타깃의 과거로 타깃의 미래를 맞히기. 낮으면 타깃은 예측 가능.
            "persistence_tgt_t_to_tgt_tk": nerr(tgt_ctx, tgt, var_t),
            # (ii) 항등 기준선: 학습된 predictor 대신 온라인 z_e(t) 를 그대로 답으로 냈을 때.
            "identity_onl_t_to_tgt_tk": nerr(onl_ctx, tgt, var_t),
            # (iii) EMA 정합: 같은 시점에서 온라인 vs 타깃 인코더가 얼마나 어긋나 있나. 크면 EMA 문제.
            "ema_misalign_onl_t_vs_tgt_t": nerr(onl_ctx, tgt_ctx, var_t),
            # (iv) 온라인 공간에서의 시간 예측 가능성(타깃 인코더 배제).
            "persistence_onl_t_to_onl_tk": nerr(onl_ctx, onl_fut, ze_s.reshape(-1, ze_s.shape[-1]).var(dim=0).mean().item()),
            # (v) 차원별 R^2 분포 — 소수 고분산 차원이 집계값을 지배하는지 확인.
            "per_dim_r2": (1.0 - (pred - tgt).pow(2).mean(dim=0) / tgt.var(dim=0).clamp_min(1e-12)).tolist(),
        }
        pdr = torch.tensor(res["jepa_decomp"]["per_dim_r2"])
        res["jepa_decomp"]["per_dim_r2_summary"] = {
            "median": pdr.median().item(), "frac_positive": (pdr > 0).float().mean().item(),
            "max": pdr.max().item(), "min": pdr.min().item(),
        }

    # ── 측정① z_e 정보량: frozen z_e → 현재 height map, **새로 적합한** probe ──
    # A 의 학습된 recon_decoder 는 재구성 목적으로 학습돼 유리하므로 비교에 못 쓴다.
    # A·B 에 동일 구조·동일 하이퍼로 fresh probe 를 적합하고 held-out R^2 를 본다.
    # z_e 스케일이 A(0.08) vs B(1.24) 로 15배 다르므로 **입력을 표준화**해야 공정하다.
    def fit_probe(x: torch.Tensor, y: torch.Tensor, kind: str, epochs: int = 4000) -> dict:
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(x.shape[0], generator=g)
        ntr = int(0.8 * x.shape[0])
        itr, ite = perm[:ntr], perm[ntr:]
        xtr, xte, ytr, yte = x[itr].to(dev), x[ite].to(dev), y[itr].to(dev), y[ite].to(dev)
        # x·y 모두 train 통계로 표준화. y 를 표준화하지 않으면 (평균 0.34, 분산 6e-4) bias 가
        # 수렴을 못 해 R^2 가 음수로 나온다. R^2 는 y 의 공통 아핀변환에 불변이므로 값은 동등.
        mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True).clamp_min(1e-6)
        xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
        ymu, ysd = ytr.mean(0, keepdim=True), ytr.std(0, keepdim=True).clamp_min(1e-9)
        ytr, yte = (ytr - ymu) / ysd, (yte - ymu) / ysd
        torch.manual_seed(0)
        net = (torch.nn.Linear(x.shape[1], y.shape[1]) if kind == "linear" else
               torch.nn.Sequential(torch.nn.Linear(x.shape[1], 128), torch.nn.ELU(),
                                   torch.nn.Linear(128, y.shape[1]))).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        best, tr0 = float("inf"), None
        for e in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(net(xtr), ytr)
            loss.backward()
            opt.step()
            sch.step()
            if e == 0:
                tr0 = loss.item()
            if e % 200 == 0 or e == epochs - 1:
                with torch.inference_mode():
                    best = min(best, (net(xte) - yte).pow(2).mean().item())
        with torch.inference_mode():
            mse = (net(xte) - yte).pow(2).mean().item()
            train_mse = torch.nn.functional.mse_loss(net(xtr), ytr).item()
        var = yte.var(dim=0).mean().item()   # 표준화했으므로 ~1.0
        return {"test_mse": mse, "test_var": var, "r2": 1.0 - mse / max(var, 1e-12),
                "r2_best": 1.0 - best / max(var, 1e-12), "train_mse": train_mse,
                "train_mse_init": tr0, "converged": bool(train_mse < 0.9 * tr0)}

    print("[diag] probe 적합 중 (linear · mlp) …")
    res["probe_z_e_to_height"] = {k: fit_probe(z_e, hm, k) for k in ("linear", "mlp")}
    # 대조: z_p 만으로 height map 을 얼마나 맞히나(외수용 없이 얻어지는 하한선).
    # height_map = base_z − terrain_z 라 관절각·자세만으로도 상당 부분 결정된다 → baseline 이 높다.
    res["probe_z_p_to_height"] = {"mlp": fit_probe(z_p, hm, "mlp")}
    # **한계 기여**: z_e 가 z_p 너머로 더하는 정보. 단독 probe 두 개의 차이가 아니라 이것이 정답.
    res["probe_joint_to_height"] = {"mlp": fit_probe(torch.cat([z_e, z_p], dim=1), hm, "mlp")}

    # ── 측정② 입력 결합 강도: 결손 severity 별 z_e 이동량 ──────────────────────
    # Δ_std(s) = MSE(z_e_clean, z_e_corrupt) / var(z_e_clean)  ← 위 지속표와 **동일 정규화**
    #   → "결손 s 가 z_e 를 자기 산포의 몇 배만큼 움직이는가". 1.0 이면 무상관 수준으로 이동.
    # Δ_rel(s) = ‖Δz_e‖/‖z_e‖ (원 제안). z_e 의 DC 성분이 커서 값이 눌리므로 참고용으로만 둔다.
    pc_all = torch.cat(buf["pc"]).float()
    var_ze = z_e.var(dim=0).mean().item()
    norm_clean = z_e.norm(dim=1).mean().item()
    sens = {}
    for dname, dfn in DEGRADATIONS.items():
        rows = {}
        for s in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            torch.manual_seed(1234)                 # A·B 에 **동일한 결손 마스크** → 공정 비교
            d_mse = d_rel = 0.0
            nb = 0
            for ch in torch.chunk(torch.arange(pc_all.shape[0]), 16):
                p = pc_all[ch].to(dev)
                with torch.inference_mode():
                    zc = actor.encode_exteroception(p)
                    zd = actor.encode_exteroception(dfn(p.clone(), s))
                d_mse += (zc - zd).pow(2).mean().item()
                d_rel += ((zc - zd).norm(dim=1) / zc.norm(dim=1).clamp_min(1e-9)).mean().item()
                nb += 1
            rows[str(s)] = {"delta_std": (d_mse / nb) / max(var_ze, 1e-12),
                            "delta_rel": d_rel / nb}
        sens[dname] = rows
    res["degradation_sensitivity"] = {"var_z_e": var_ze, "norm_z_e_clean": norm_clean, "curves": sens}

    # ── 출력 ──────────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print(f"[Step 0 진단]  head={head}  {os.path.basename(os.path.dirname(resume_path))}")
    print("=" * 68)
    for name in ("z_e", "z_p", "height_map"):
        s = res[name]
        print(f"{name:>12}  dim별std 평균 {s['mean_dim_std']:.4f}  "
              f"[{s['min_dim_std']:.4f}, {s['max_dim_std']:.4f}]  |x| {s['mean_abs']:.4f}")
    pk = res["persistence_vs_k"]
    ks = list(pk["z_e"].keys())
    print("\n  [시간 일관성 — 지속 정규화 오차, 1.0 = 무상관(예측 불가)]")
    print("      k        " + "".join(f"{x:>8}" for x in ks))
    for nm in ("z_e", "z_p", "height_map"):
        print(f"      {nm:<10}" + "".join(f"{pk[nm][x]:>8.3f}" for x in ks))

    if "recon" in res:
        r = res["recon"]
        print(f"\n  재구성 MSE        {r['mse']:.6f}")
        print(f"  var(height_map)   {r['var_height_map']:.6f}")
        print(f"  정규화 오차(1-R²) {r['normalized_error']:.4f}   ← 1.0=미학습, 0.0=완전복원")
    if "jepa" in res:
        j = res["jepa"]
        print(f"\n  예측 MSE (k={j['k']})    {j['mse_all_pairs']:.6f}  (리셋제외 {j['mse_reset_masked']:.6f})")
        print(f"  var(target z_e)   {j['var_target']:.6f}")
        print(f"  정규화 오차(1-R²) {j['normalized_error']:.4f}   ← 1.0=미학습, 0.0=완전예측")
        print(f"  리셋 횡단 쌍 비율 {100*j['frac_pairs_crossing_reset']:.1f}%")
    if "jepa_decomp" in res:
        d = res["jepa_decomp"]
        print("\n  [원인 분해 — 모두 정규화 오차, 낮을수록 좋음]")
        print(f"    (i)   지속 z̄(t)→z̄(t+k)      {d['persistence_tgt_t_to_tgt_tk']:.4f}  ← 타깃이 애초에 예측 가능한가")
        print(f"    (ii)  항등 z(t)→z̄(t+k)       {d['identity_onl_t_to_tgt_tk']:.4f}  ← 학습 없이 그냥 복사했다면")
        print(f"    (iii) EMA 정합 z(t) vs z̄(t)  {d['ema_misalign_onl_t_vs_tgt_t']:.4f}  ← 크면 EMA 타깃이 어긋남")
        print(f"    (iv)  온라인 지속 z(t)→z(t+k) {d['persistence_onl_t_to_onl_tk']:.4f}")
        s = d["per_dim_r2_summary"]
        print(f"    (v)   차원별 R²  중앙값 {s['median']:.3f}  양수비율 {100*s['frac_positive']:.0f}%  "
              f"[{s['min']:.2f}, {s['max']:.2f}]")
    p = res["probe_z_e_to_height"]
    print("\n  [측정① z_e 정보량 — fresh probe, held-out R², 높을수록 정보적]")
    print(f"    z_e → height  linear R² {p['linear']['r2']:.4f}   mlp R² {p['mlp']['r2']:.4f}"
          f"   (수렴 {p['linear']['converged']}/{p['mlp']['converged']})")
    r_zp = res["probe_z_p_to_height"]["mlp"]["r2"]
    r_j = res["probe_joint_to_height"]["mlp"]["r2"]
    print(f"    z_p → height  (하한선)             mlp R² {r_zp:.4f}")
    print(f"    [z_e,z_p] → height (결합)          mlp R² {r_j:.4f}")
    print(f"    → z_e 의 **한계 기여** ΔR² = 결합 − z_p단독 = {r_j - r_zp:+.4f}")
    if "recon" in res:
        print(f"    (참고) 학습된 recon_decoder R² {res['recon']['r2']:.4f}")

    print("\n  [측정② 결손 민감도 Δ_std — z_e 가 자기 산포의 몇 배 움직이나, 1.0=무상관 수준]")
    sv = res["degradation_sensitivity"]["curves"]
    lv = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
    print("      severity   " + "".join(f"{x:>8}" for x in lv))
    for dname in sv:
        print(f"      {dname:<11}" + "".join(f"{sv[dname][x]['delta_std']:>8.3f}" for x in lv))
    print("      (참고 Δ_rel = ‖Δz‖/‖z‖)")
    for dname in sv:
        print(f"      {dname:<11}" + "".join(f"{sv[dname][x]['delta_rel']:>8.3f}" for x in lv))
    print("=" * 68)

    tag = os.path.basename(os.path.dirname(resume_path)) + "_" + os.path.basename(resume_path).replace(".pt", "")
    out = args_cli.out or f"diag_step0_{head}_{tag}.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[saved] {out}")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
