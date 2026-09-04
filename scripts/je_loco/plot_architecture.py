#!/usr/bin/env python3
"""발표용 아키텍처 그림 — 3단계 파이프라인 + 학생 네트워크 + 표현 헤드 대비.

python scripts/je_loco/plot_architecture.py --out docs/figs/architecture.png
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

BLUE, ORANGE, PURPLE = "#2c6fbb", "#d1780a", "#7c4bb8"   # validate_palette 통과 팔레트
INK, MUTED, LINE, BG = "#1a1a1a", "#5c5c5c", "#c8c8c8", "#f4f4f2"


def box(ax, x, y, w, h, text, fc="white", ec=LINE, fs=9, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                fc=fc, ec=ec, lw=1.3, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            color=tc, zorder=3, fontweight="bold" if bold else "normal", linespacing=1.5)


def arrow(ax, p, q, color=MUTED, style="-|>", lw=1.4, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13,
                                 color=color, lw=lw, linestyle=ls, zorder=1,
                                 shrinkA=2, shrinkB=2))


def main():   # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", action="store_true",
                    help="2차 설계(예측기 정책 내재화 · 보조 목적함수 비교축)로 그린다")
    ap.add_argument("--out", default="docs/figs/architecture.png")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    fig = plt.figure(figsize=(15, 10.5))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.16, wspace=0.10)

    # ── (1) 3단계 파이프라인 ────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, :]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("① 전체 파이프라인 — Teacher 학습 → 표현 사전학습 → DAgger 증류",
                 fontsize=13, color=INK, pad=12, loc="left", fontweight="bold")

    box(ax, 0.01, 0.52, 0.29, 0.40,
        "Phase 1 · Teacher (PPO)\n\n"
        "관측 = proprio 45 + privileged\nheight_scan 187 (17×11) = 232\n"
        "MLP actor-critic · 20,000 iter\n\n"
        "→ 잘 걷는다 (ep_len 948~979)\n※ 실기 배포 불가", fc=BG, fs=9)

    box(ax, 0.355, 0.52, 0.29, 0.40,
        "Phase 2 · 표현 사전학습 (오프라인)\n\n"
        "teacher 로 수집한 점군 시퀀스\n\n" +
        ("jepa : z_e(t) → Δz_e(t+k)\n       k∈{25,50,100,150}, EMA τ=.996\n       + VICReg\n"
         "→ skill 0.584 @ k=100 (지평 선택 근거)"
         if a.v2 else
         "jepa_v1 : z_e(t) → Δz_e(t+k)\n         k∈{5,15,25,50}, EMA τ=.996\n         + VICReg\n"
         "recon_v1: z_e → GT 높이맵 (특권)"), fc=BG, fs=9)

    box(ax, 0.70, 0.52, 0.29, 0.40,
        ("Phase 3b · DAgger 증류\n\n" if a.v2 else "Phase 3b · DAgger 증류  ★현재\n\n") +
        "환경을 [학생 행동]으로 굴리고\n방문 상태마다 teacher 질의\n"
        "loss = MSE(student(o), a_teacher)\n\n" +
        ("2,000 iter × 128 steps · 1024 env\n(환경 스텝 예산 1차와 동일)\n"
         "+ 학습 중 노후화 d~U[0,25]" if a.v2 else "8,000 iter · 1024 env"),
        fc="#eef4fb", ec=BLUE, fs=9)

    arrow(ax, (0.30, 0.72), (0.355, 0.72)); ax.text(0.327, 0.755, "데이터\n수집", ha="center", fontsize=7.5, color=MUTED)
    arrow(ax, (0.645, 0.72), (0.70, 0.72)); ax.text(0.672, 0.755, "인코더\n가중치", ha="center", fontsize=7.5, color=MUTED)

    axis_txt = ("★ 비교 축 = 증류 중 보조 목적함수 (jepa / recon / none) — "
                "구조 동일, 배포 파라미터 498,427 로 일치"
                if a.v2 else
                "★ 비교 축 = 학생 인코더 초기화 단 하나 (jepa / recon / scratch) — 나머지 전부 동일")
    ax.text(0.5, 0.40, axis_txt,
            ha="center", fontsize=11, color=INK, fontweight="bold")
    note_txt = ("1차(인코더 초기화)는 세 게이트 전부 격차 < 시드폭 → 개입을 지속적인 것으로 바꾼다: "
                "보조손실을 증류 내내 · 학습 중 관측 노후화 d~U[0,25] 주입"
                if a.v2 else
                "Phase 3(동결+PPO)은 세 조건 모두 걷지 못해 무효 판정 → 행동 감독 + on-policy 보정을 되살린 DAgger 로 전환")
    ax.text(0.5, 0.30, note_txt,
            ha="center", fontsize=8.5, color=MUTED)
    ax.text(0.5, 0.21, "(「Now You See That」: depth end-to-end RL 54.0% vs privileged distillation 98.9%)",
            ha="center", fontsize=8, color=MUTED, style="italic")

    # ── (2) 학생 네트워크 ───────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("② 학생 정책 (배포 대상)", fontsize=13, color=INK, pad=12, loc="left", fontweight="bold")

    box(ax, 0.03, 0.85, 0.42, 0.10, "점군 192점  [x, y, z, valid]\n(D435i 프러스텀 16×12)", fc=BG, fs=8.5)
    box(ax, 0.55, 0.85, 0.42, 0.10, "proprio 45 × H=5\n[w, g, cmd, q, dq, a(t-1)]", fc=BG, fs=8.5)

    box(ax, 0.03, 0.60, 0.42, 0.18,
        "PointNet-lite\n3→32→64→128\n★ masked max-pool (valid=0 제외)\n128→64", fc="white", ec=BLUE, fs=8.5)
    box(ax, 0.55, 0.63, 0.42, 0.12, "proprio encoder\n225→128→32", fc="white", ec=BLUE, fs=8.5)

    arrow(ax, (0.24, 0.85), (0.24, 0.78)); arrow(ax, (0.76, 0.85), (0.76, 0.75))
    ax.text(0.24, 0.555, "z_e (64)", ha="center", fontsize=9, color=BLUE, fontweight="bold")
    ax.text(0.76, 0.585, "z_p (32)", ha="center", fontsize=9, color=BLUE, fontweight="bold")

    box(ax, 0.55, 0.44, 0.42, 0.09,
        "vel_decoder -> v_hat (3)\n※ 증류에선 critic 관측 없어 OFF" if a.v2
        else "vel_decoder -> v_hat (3)\n※ 두 조건 공통", fc="white", ec=LINE, fs=8)
    arrow(ax, (0.76, 0.63), (0.76, 0.53), color=LINE, ls=":")

    if a.v2:
        # 2차의 핵심 구조 변경: 예측기가 배포 경로 안에 있어 행동손실이 직접 통과한다
        box(ax, 0.03, 0.40, 0.42, 0.11,
            "predictor 96→128→64\n→ z_hat_e(o+k),  k=100 (2.0s)", fc="#eef4fb", ec=BLUE, fs=8.5)
        arrow(ax, (0.24, 0.545), (0.24, 0.51))
        arrow(ax, (0.24, 0.40), (0.42, 0.36))
        arrow(ax, (0.70, 0.44), (0.56, 0.36))
    else:
        arrow(ax, (0.24, 0.545), (0.42, 0.36)); arrow(ax, (0.70, 0.44), (0.56, 0.36))
    box(ax, 0.28, 0.24, 0.42, 0.11,
        "GRU  hidden 256 · 1층\n입력 = [z_e, z_p, z_hat_e(o+k)]" if a.v2
        else "GRU  hidden 256 · 1층", fc="white", ec=BLUE, fs=9)
    arrow(ax, (0.49, 0.24), (0.49, 0.17))
    box(ax, 0.28, 0.06, 0.42, 0.10, "MLP 256→128 → action 12", fc="white", ec=BLUE, fs=9)

    # ── (3) 표현 헤드 대비 ──────────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1]); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.set_title("③ 보조 목적함수 — 같은 예측기를 무엇이 학습시키는가" if a.v2
                 else "③ 표현 헤드 — 학습 시에만 존재, 배포 시 폐기",
                 fontsize=13, color=INK, pad=12, loc="left", fontweight="bold")

    box(ax, 0.03, 0.72, 0.94, 0.19,
        ("recon   λ_recon = 1   →  z_e → GT 높이맵(144) 회귀\n\n"
         "감독 = 시뮬 raycaster  →  특권 정보 · 실기 학습 불가\n예측기는 행동손실로만 학습된다"
         if a.v2 else
         "Head A · recon   →  z_e(64) → 128 → GT 높이맵(144)\n\n"
         "감독 = 시뮬 raycaster 의 clean 높이맵  →  특권 정보 · 실기 학습 불가\n26,896 param"),
        fc="#fdf3e7", ec=ORANGE, fs=9)

    box(ax, 0.03, 0.42, 0.94, 0.25,
        ("jepa   λ_jepa = 1   →  z_hat_e(t+100) vs EMA 타깃\n\n"
         "감독 = 자기 자신의 EMA target encoder (τ=.996) + VICReg\n"
         "자기지도 · 무라벨 실기 로그로 학습 가능  →  실기 이식 가능한 유일한 조건"
         if a.v2 else
         "Head B · jepa   →  [z_e(t), z_p(t), cond] → Δz_e(t+k)\n\n"
         "감독 = 자기 자신의 EMA target encoder  →  자기지도 · 무라벨 실기 데이터로 학습 가능\n"
         "predictor 20,672 + EMA target 18,816 + VICReg projector 24,832 = 64,320 param"),
        fc="#eef4fb", ec=BLUE, fs=9)

    box(ax, 0.03, 0.20, 0.94, 0.17,
        ("none   보조손실 없음   →  예측기가 행동손실로만 학습된다\n\n"
         "통제 조건. 예측기라는 구조 자체가 주는 이득과 목적함수가 주는 이득을 분리한다"
         if a.v2 else
         "Head C · pcrecon  (★ 추가 예정)  →  z_e → 입력 점군 복원\n\n"
         "감독 = 관측 자신 → 자기지도.  \"라벨이 필요 없으면 오토인코더는 왜 안 되나\"의 대조군"),
        fc="#f5f0fb", ec=PURPLE, fs=9)

    ax.text(0.5, 0.115,
            "※ 세 조건 모두 predictor_in_policy=True → 배포 경로 파라미터 498,427 로 완전 일치" if a.v2
            else "※ 배포 시엔 세 헤드 모두 버려진다 → 정책 파라미터는 조건과 무관하게 동일",
            ha="center", fontsize=8.5, color=MUTED, fontweight="bold")
    ax.text(0.5, 0.045,
            "유일한 변수 = 그 예측기 MLP 를 무엇이 학습시키느냐" if a.v2
            else "차별점은 \"decoder 유무\"가 아니라 \"복원 대상이 특권 정보인가 관측 자신인가\"",
            ha="center", fontsize=8.5, color=INK)

    fig.suptitle("JE-Loco — 특권 정보 없는 자기지도 표현으로 학습하는 사족보행 정책",
                 fontsize=15, color=INK, y=0.985, fontweight="bold")
    fig.savefig(a.out, dpi=190, bbox_inches="tight", facecolor="white")
    print(f"[saved] {a.out}")


if __name__ == "__main__":
    main()
