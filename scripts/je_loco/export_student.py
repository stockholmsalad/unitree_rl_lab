# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""증류 student(PointNet + proprio enc + predictor + GRU) → ONNX / TorchScript.

rsl_rl 의 표준 as_onnx() 는 쓸 수 없다. 그건 모델이 **평탄한 관측 벡터 하나**를 받아
바로 RNN 에 넣는다고 가정하는데(_OnnxRNNModel), 우리 student 는 get_latent 안에서
그룹별 인코딩(pc→z_e, proprio→z_p)과 예측기(ẑ)를 거친다. 그 경로를 통째로 건너뛴
그래프가 나온다 → 실기에서 조용히 틀린 행동이 나온다.

그래서 배포 경로를 그대로 따라가는 래퍼를 따로 만든다:

    입력  proprio (1, 225)   = 45 × H(5), 학습 때 obs["policy"] 와 같은 순서
          pointcloud (1, 768) = 192 × 4, 각 점 [x, y, z, valid], **base frame**
          h_in (1, 1, 256)    = GRU 은닉 상태 (첫 스텝은 0)
    출력  action (1, 12)      = 관절 목표(스케일 전), 결정론적(분포 평균)
          h_out (1, 1, 256)   = 다음 스텝에 그대로 넣을 은닉 상태

은닉 상태를 버퍼로 숨기지 않고 **입출력으로 노출**한다. 실기 루프가 재시작하거나
프레임을 흘렸을 때 상태를 명시적으로 0 으로 되돌릴 수 있어야 하고, onnxruntime 은
어차피 stateful 버퍼를 갱신해 주지 않는다.

cfg 를 읽지 않고 **체크포인트 텐서 모양에서 구조를 유도**한다. 이유가 둘이다.
cfg 는 isaaclab 을 끌어오는데 배포용 노트북에는 Isaac Sim 이 없고, 무엇보다 저장된
가중치가 그 런의 아키텍처에 대한 유일하게 확실한 근거다. cfg 는 그 뒤로 바뀌었을 수
있지만 체크포인트는 안 바뀐다.

Isaac Sim 이 필요 없다 — torch / tensordict / rsl_rl 만 있으면 돈다.

  # ★ Z790 또는 pilab
  python scripts/je_loco/export_student.py \
      --checkpoint logs/rsl_rl/je_loco_distill/<run>/model_7999.pt
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from tensordict import TensorDict

from unitree_rl_lab.je_loco.rsl_rl_pc.model import PointCloudRNNModel

NUM_POINTS = 192    # frustum 16(yaw) × 12(pitch). 가중치에 안 남는 유일한 값.


def infer_arch(sd: dict) -> dict:
    """student_state_dict 텐서 모양에서 배포에 필요한 구조를 되읽는다."""
    a = {
        "proprio_dim": sd["proprio_encoder.0.weight"].shape[1],
        "proprio_hidden_dim": sd["proprio_encoder.0.weight"].shape[0],
        "zp_dim": sd["proprio_encoder.2.weight"].shape[0],
        "pc_out_dim": sd["pc_encoder.global_mlp.0.weight"].shape[0],
        "vel_hidden_dim": sd["vel_decoder.0.weight"].shape[0],
        "rnn_hidden_dim": sd["rnn.rnn.weight_hh_l0"].shape[1],
        "rnn_num_layers": sum(1 for k in sd if k.startswith("rnn.rnn.weight_hh_l")),
        # 분포는 배포 시 평균만 쓰지만, 있고 없고에 따라 mlp 출력 폭이 달라지므로
        # 모델을 같은 모양으로 세우려면 그대로 재현해야 한다.
        "std_type": ("scalar" if "distribution.std_param" in sd
                     else "log" if "distribution.log_std_param" in sd else None),
    }
    mlp_idx = sorted({int(k.split(".")[1]) for k in sd if k.startswith("mlp.")})
    a["hidden_dims"] = [sd[f"mlp.{i}.weight"].shape[0] for i in mlp_idx[:-1]]
    a["action_dim"] = sd[f"mlp.{mlp_idx[-1]}.weight"].shape[0]

    # 예측기가 정책 입력에 들어가는지는 GRU 입력 폭이 말해 준다. 이걸 틀리면
    # 가중치는 멀쩡히 로드되면서 잠재 순서만 어긋나 조용히 다른 정책이 된다.
    gru_in = sd["rnn.rnn.weight_ih_l0"].shape[1]
    base = a["zp_dim"] + a["pc_out_dim"]
    if gru_in == base + a["pc_out_dim"]:
        a["predictor_in_policy"] = True
    elif gru_in == base:
        a["predictor_in_policy"] = False
    else:
        raise RuntimeError(
            f"GRU 입력 폭 {gru_in} 이 z_p({a['zp_dim']})+z_e({a['pc_out_dim']})"
            f"[+ẑ] 어느 쪽과도 안 맞는다. 배포 경로를 신뢰할 수 없다.")
    if a["predictor_in_policy"] and "jepa_predictor.0.weight" not in sd:
        raise RuntimeError("GRU 는 ẑ 를 기대하는데 체크포인트에 예측기가 없다.")
    a["recon_hidden_dim"] = (
        sd["jepa_predictor.0.weight"].shape[0] if "jepa_predictor.0.weight" in sd else 128)
    return a


class DeployStudent(nn.Module):
    """student 의 배포 경로만 뽑아낸 모듈. 은닉 상태는 인자로 주고받는다."""

    def __init__(self, model: PointCloudRNNModel) -> None:
        super().__init__()
        self.pc_encoder = model.pc_encoder
        self.proprio_encoder = model.proprio_encoder
        self.predictor = getattr(model, "jepa_predictor", None)
        self.obs_normalizer = model.obs_normalizer     # obs_normalization=False → Identity
        self.rnn = model.rnn.rnn                       # nn.GRU (래퍼의 상태 로직 우회)
        self.mlp = model.mlp
        if model.distribution is not None:
            self.head = model.distribution.as_deterministic_output_module()
        else:
            self.head = nn.Identity()
        # 학습 때 latent 를 어떤 순서로 이었는지 그대로 옮긴다. obs_groups 가
        # ["policy", "pointcloud"] 이므로 [z_p, z_e], 그 뒤에 ẑ.
        self.groups = list(model.obs_groups)
        self.use_predictor = bool(getattr(model, "_predictor_in_policy", False))

    def forward(self, proprio: torch.Tensor, pointcloud: torch.Tensor,
                h_in: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_p = self.proprio_encoder(proprio)
        z_e = self.pc_encoder(pointcloud)
        parts = []
        for g in self.groups:
            parts.append(z_p if g == "policy" else z_e)
        if self.use_predictor:
            parts.append(self.predictor(torch.cat([z_e, z_p], dim=-1)))
        latent = self.obs_normalizer(torch.cat(parts, dim=-1))
        out, h_out = self.rnn(latent.unsqueeze(0), h_in)
        return self.head(self.mlp(out.squeeze(0))), h_out


def build_model(a: dict, num_points: int) -> PointCloudRNNModel:
    """유도한 구조로 student 를 빈 껍데기로 만든다(가중치는 이후 로드)."""
    obs = TensorDict(
        {
            "policy": torch.zeros(1, a["proprio_dim"]),
            "pointcloud": torch.zeros(1, num_points * 4),
        },
        batch_size=[1],
    )
    return PointCloudRNNModel(
        obs, {"student": ["policy", "pointcloud"]}, "student", a["action_dim"],
        hidden_dims=a["hidden_dims"], activation="elu", obs_normalization=False,
        distribution_cfg=(None if a["std_type"] is None else {
            "class_name": "rsl_rl.modules.distribution:GaussianDistribution",
            "init_std": 1.0, "std_type": a["std_type"]}),
        rnn_type="gru", rnn_hidden_dim=a["rnn_hidden_dim"],
        rnn_num_layers=a["rnn_num_layers"],
        pc_out_dim=a["pc_out_dim"], pc_group="pointcloud",
        proprio_group="policy", zp_dim=a["zp_dim"],
        proprio_hidden_dim=a["proprio_hidden_dim"], vel_hidden_dim=a["vel_hidden_dim"],
        repr_head="none", recon_hidden_dim=a["recon_hidden_dim"],
        predictor_in_policy=a["predictor_in_policy"],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="증류 체크포인트 model_*.pt")
    ap.add_argument("--outdir", default="", help="기본값 = 체크포인트 폴더/exported")
    ap.add_argument("--num_points", type=int, default=NUM_POINTS)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    key = "student_state_dict" if "student_state_dict" in ckpt else "actor_state_dict"
    sd = ckpt[key]
    a = infer_arch(sd)
    print(f"[export] {key} — 유도한 구조")
    print(f"  proprio {a['proprio_dim']} → z_p {a['zp_dim']}   "
          f"점군 {args.num_points}×4 → z_e {a['pc_out_dim']}")
    print(f"  ẑ 정책입력 {'포함' if a['predictor_in_policy'] else '없음'}   "
          f"GRU {a['rnn_hidden_dim']}×{a['rnn_num_layers']}   "
          f"MLP {a['hidden_dims']} → {a['action_dim']}")

    model = build_model(a, args.num_points)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # 표현 헤드·EMA 타깃은 배포에 안 쓰이므로 unexpected 로 남아도 된다.
    # 반대로 missing 은 배포 경로에 구멍이 뚫린 것이므로 즉시 세운다.
    if missing:
        raise RuntimeError(f"체크포인트에 없는 배포 경로 파라미터: {missing}")
    print(f"[export] 로드 완료 — 미사용 키 {len(unexpected)}개(표현 헤드/EMA 타깃)")

    model.eval()
    deploy = DeployStudent(model).eval()

    outdir = args.outdir or os.path.join(os.path.dirname(args.checkpoint), "exported")
    os.makedirs(outdir, exist_ok=True)

    pc_dim = args.num_points * 4
    proprio = torch.zeros(1, a["proprio_dim"])
    # 더미 점군은 전부 유효로 둔다. 전부 무효(valid=0)면 max-pool 의 −inf 경로가
    # 트레이싱돼서 그래프에 상수로 굳을 위험이 있다.
    pc = torch.rand(1, args.num_points, 4)
    pc[..., 3] = 1.0
    pc = pc.reshape(1, pc_dim)
    h0 = torch.zeros(a["rnn_num_layers"], 1, a["rnn_hidden_dim"])

    onnx_path = os.path.join(outdir, "student.onnx")
    torch.onnx.export(
        deploy, (proprio, pc, h0), onnx_path,
        input_names=["proprio", "pointcloud", "h_in"],
        output_names=["action", "h_out"],
        dynamic_axes={"proprio": {0: "batch"}, "pointcloud": {0: "batch"},
                      "h_in": {1: "batch"}, "action": {0: "batch"},
                      "h_out": {1: "batch"}},
        opset_version=args.opset, do_constant_folding=True,
    )
    jit_path = os.path.join(outdir, "student.pt")
    torch.jit.save(torch.jit.trace(deploy, (proprio, pc, h0)), jit_path)

    # ── 검증 ────────────────────────────────────────────────────────────────
    # 관측이 결손된 경우까지 돌려본다. 실기에서 valid=0 은 예외가 아니라 상시 상태다.
    # 은닉 상태를 20 스텝 굴리므로 GRU 재귀가 어긋나면 오차가 누적돼 드러난다.
    print("\n[export] torch ↔ onnxruntime 대조 (결손·실명 포함, 20 스텝 재귀)")
    import numpy as np
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst = 0.0
    h_t = h0.clone()
    h_o = h0.clone().numpy()
    for step in range(20):
        p = torch.randn(1, a["proprio_dim"]) * 0.3
        c = torch.rand(1, args.num_points, 4)
        c[..., :3] = c[..., :3] * 2.0 - 1.0
        # 스텝마다 결손 비율을 바꾼다. 20 스텝 중 한 번은 전부 무효(카메라 실명).
        keep = 0.0 if step == 13 else float(torch.rand(()))
        c[..., 3] = (torch.rand(1, args.num_points) < keep).float()
        c = c.reshape(1, pc_dim)
        with torch.no_grad():
            a_t, h_t = deploy(p, c, h_t)
        a_o, h_o = sess.run(
            None, {"proprio": p.numpy(), "pointcloud": c.numpy(), "h_in": h_o})
        d = float(np.abs(a_t.numpy() - a_o).max())
        worst = max(worst, d)
        if step == 13:
            print(f"  step {step:2d} (점군 전부 무효): |Δaction| = {d:.2e}")
    print(f"  20 스텝 최대 오차 = {worst:.2e}")
    if worst > 1e-4:
        raise RuntimeError(f"ONNX 와 torch 가 어긋난다 ({worst:.2e}). 배포 금지.")

    print(f"\n[export] ONNX  {onnx_path}")
    print(f"[export] JIT   {jit_path}")
    print(f"[export] 입력  proprio({a['proprio_dim']}) pointcloud({pc_dim}) "
          f"h_in({a['rnn_num_layers']},1,{a['rnn_hidden_dim']})")
    print(f"[export] 출력  action({a['action_dim']}) h_out(같은 모양)")


if __name__ == "__main__":
    main()
