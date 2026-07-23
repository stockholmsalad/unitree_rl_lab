# Copyright (c) 2025, JE-Loco.
# SPDX-License-Identifier: BSD-3-Clause
"""환경 대조 — 개발 머신(5070 Ti)과 학습 서버(pilab) 사이의 은닉 불일치 탐지.

배경: 코드는 git 으로 동기화되지만 **IsaacLab 은 별도 저장소**라 따라오지 않는다. 두 머신의
IsaacLab 커밋이나 패키지 버전이 어긋나면 `RslRlRNNModelCfg` 같은 API 가 조용히 달라지고,
그 버그는 학습 중반에야 드러난다(= GPU 하루 낭비). 매 세션 시작 시 양쪽에서 이걸 돌린다.

Isaac Sim 을 기동하지 않는다(`AppLauncher` 안 씀). `isaaclab` 을 import 하지도 않는다 —
`pxr` 의존 때문에 앱 밖에서는 실패하므로, `find_spec` 으로 경로만 얻어 파일을 읽는다.

사용법
------
    python scripts/check_env.py                     # 리포트 출력
    python scripts/check_env.py --save env_ref.json # 기준 스냅샷 저장(개발 머신에서 1회)
    python scripts/check_env.py --compare env_ref.json   # 대조. 불일치 시 exit 1
    python scripts/check_env.py --json              # 기계 판독용

MUST_MATCH 와 INFO 구분이 핵심이다. GPU·드라이버는 두 머신이 **당연히** 다르므로
(5070 Ti 16GB vs RTX PRO 6000 96GB) 불일치로 치면 안 된다. 코드 거동을 바꾸는
라이브러리 버전만 강제 대조한다.
"""

from __future__ import annotations

import argparse
import importlib.metadata as md
import importlib.util
import json
import os
import pathlib
import platform
import subprocess
import sys

# 불일치 시 학습이 조용히 깨지는 항목 — 반드시 같아야 한다.
MUST_MATCH = [
    "python",
    "torch",
    "isaaclab",
    "isaaclab_rl",
    "isaaclab_tasks",
    "rsl_rl_lib",
    "isaaclab_version",
    "isaaclab_commit",
]

# 머신마다 달라도 정상인 항목 — 표시만 한다.
INFO_ONLY = [
    "gpu_name",
    "gpu_capability",
    "gpu_memory_mb",
    "cuda_runtime",
    "cudnn_version",
    "cudnn_gru_ok",
    "repo_commit",
    "repo_dirty",
    "unitree_model_dir",
    "unitree_ros_dir",
    "alloc_conf",
    "hostname",
]

_PKGS = {
    "isaaclab": "isaaclab",
    "isaaclab_rl": "isaaclab-rl",
    "isaaclab_tasks": "isaaclab-tasks",
    "rsl_rl_lib": "rsl-rl-lib",
    "torch": "torch",
}


def _git(repo: pathlib.Path, *args: str) -> str:
    """repo 에서 git 명령 실행. 실패하면 '<unavailable>'."""
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() if out.returncode == 0 else "<unavailable>"
    except Exception:
        return "<unavailable>"


def _find_isaaclab_root() -> pathlib.Path | None:
    """isaaclab 패키지를 **import 하지 않고** 체크아웃 루트를 찾는다.

    editable 설치면 origin 이 `<checkout>/source/isaaclab/isaaclab/__init__.py` 이다.
    깊이를 고정하지 않고 위로 올라가며 VERSION 또는 .git 을 찾는다(레이아웃 변경에 견고).
    """
    spec = importlib.util.find_spec("isaaclab")
    if spec is None or spec.origin is None:
        return None
    for parent in pathlib.Path(spec.origin).resolve().parents:
        if (parent / "VERSION").is_file() or (parent / ".git").exists():
            return parent
    return None


def _cudnn_gru_probe() -> str:
    """cuDNN GRU 커널 동작 여부. Blackwell(sm_120) + 구 cuDNN 조합에서 실패한 이력이 있다.

    실패하면 `train_pc.py` 의 `torch.backends.cudnn.enabled=False` 우회가 필요하고,
    성공하면 그 줄을 지워 GRU BPTT 를 약 3.6배 가속할 수 있다(실측).
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return "no-cuda"
        prev = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = True
        try:
            gru = torch.nn.GRU(97, 256, 1).cuda()
            x = torch.randn(8, 32, 97, device="cuda")
            y, _ = gru(x)
            y.sum().backward()
            torch.cuda.synchronize()
            return "ok"
        finally:
            torch.backends.cudnn.enabled = prev
    except Exception as e:  # noqa: BLE001 — 어떤 실패든 진단 정보로 남긴다
        return f"FAIL: {type(e).__name__}"


def collect() -> dict[str, str]:
    """현재 환경 스냅샷."""
    info: dict[str, str] = {}

    info["hostname"] = platform.node()
    info["python"] = ".".join(map(str, sys.version_info[:3]))

    for key, dist in _PKGS.items():
        try:
            info[key] = md.version(dist)
        except md.PackageNotFoundError:
            info[key] = "<missing>"

    # ── IsaacLab 체크아웃 (git 동기화 안 되는 부분 = 최대 위험) ──
    root = _find_isaaclab_root()
    if root is None:
        info["isaaclab_version"] = "<not found>"
        info["isaaclab_commit"] = "<not found>"
        info["isaaclab_path"] = "<not found>"
    else:
        vf = root / "VERSION"
        info["isaaclab_version"] = vf.read_text().strip() if vf.is_file() else "<no VERSION>"
        info["isaaclab_commit"] = _git(root, "rev-parse", "--short", "HEAD")
        info["isaaclab_path"] = str(root)

    # ── 이 저장소 ──
    repo = pathlib.Path(__file__).resolve().parents[1]
    info["repo_commit"] = _git(repo, "rev-parse", "--short", "HEAD")
    info["repo_dirty"] = "yes" if _git(repo, "status", "--porcelain") else "no"

    # ── 로봇 에셋 경로 (머신마다 다름. env var 미설정 시 fallback 검증) ──
    from_env = "env" if os.environ.get("UNITREE_MODEL_DIR") else "fallback"
    model_dir = os.environ.get("UNITREE_MODEL_DIR", str(repo / "unitree_model"))
    ros_dir = os.environ.get("UNITREE_ROS_DIR", str(repo / "unitree_ros"))
    ok_m = "OK" if pathlib.Path(model_dir).is_dir() else "MISSING"
    ok_r = "OK" if pathlib.Path(ros_dir).is_dir() else "MISSING"
    info["unitree_model_dir"] = f"{model_dir} [{from_env}, {ok_m}]"
    info["unitree_ros_dir"] = f"{ros_dir} [{from_env}, {ok_r}]"

    # ── GPU / CUDA ──
    try:
        import torch

        info["cuda_runtime"] = torch.version.cuda or "<none>"
        cudnn_v = torch.backends.cudnn.version()
        info["cudnn_version"] = str(cudnn_v) if cudnn_v else "<none>"
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_capability"] = "sm_%d%d" % torch.cuda.get_device_capability(0)
            info["gpu_memory_mb"] = str(
                torch.cuda.get_device_properties(0).total_memory // (1024**2)
            )
        else:
            info["gpu_name"] = info["gpu_capability"] = info["gpu_memory_mb"] = "<no cuda>"
    except Exception as e:  # noqa: BLE001
        info["cuda_runtime"] = info["cudnn_version"] = f"<torch error: {e}>"
        info["gpu_name"] = info["gpu_capability"] = info["gpu_memory_mb"] = "<unknown>"

    info["cudnn_gru_ok"] = _cudnn_gru_probe()
    info["alloc_conf"] = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "<unset>")

    return info


def report(info: dict[str, str]) -> None:
    """사람이 읽는 리포트. MUST_MATCH 를 먼저, INFO 를 뒤에."""
    width = max(len(k) for k in info)

    print("=" * 72)
    print(f"  JE-Loco 환경 리포트 — {info['hostname']}")
    print("=" * 72)

    print("\n[ 반드시 일치해야 하는 항목 ]")
    for k in MUST_MATCH:
        print(f"  {k:<{width}} : {info.get(k, '<n/a>')}")
    print(f"  {'isaaclab_path':<{width}} : {info.get('isaaclab_path', '<n/a>')}")

    print("\n[ 머신마다 달라도 되는 항목 ]")
    for k in INFO_ONLY:
        print(f"  {k:<{width}} : {info.get(k, '<n/a>')}")

    # 즉시 조치가 필요한 것만 경고
    print()
    if info.get("cudnn_gru_ok") == "ok":
        print("  NOTE: cuDNN GRU 동작함 → train_pc.py 의 "
              "`torch.backends.cudnn.enabled=False` 를 지우면 GRU 가속(실측 약 3.6배).")
    elif str(info.get("cudnn_gru_ok", "")).startswith("FAIL"):
        print("  WARN: cuDNN GRU 실패 → `cudnn.enabled=False` 우회를 반드시 유지할 것.")
    if "MISSING" in info.get("unitree_model_dir", "") or "MISSING" in info.get("unitree_ros_dir", ""):
        print("  WARN: 로봇 에셋 경로가 존재하지 않음. UNITREE_MODEL_DIR / UNITREE_ROS_DIR 확인.")
    if info.get("repo_dirty") == "yes":
        print("  NOTE: 저장소에 커밋되지 않은 변경이 있음 (양쪽 머신이 같은 코드인지 확인 요망).")


def compare(info: dict[str, str], ref_path: pathlib.Path) -> int:
    """기준 스냅샷과 대조. MUST_MATCH 불일치가 하나라도 있으면 1 을 반환."""
    ref = json.loads(ref_path.read_text())
    print(f"\n[ 대조: {ref_path}  (기준 머신: {ref.get('hostname', '?')}) ]")

    bad = []
    for k in MUST_MATCH:
        cur, exp = info.get(k, "<n/a>"), ref.get(k, "<n/a>")
        if cur == exp:
            print(f"  ok    {k} = {cur}")
        else:
            print(f"  MISMATCH  {k}\n            기준: {exp}\n            현재: {cur}")
            bad.append(k)

    if bad:
        print(f"\n  ❌ {len(bad)}개 불일치: {', '.join(bad)}")
        print("     학습을 시작하기 전에 맞추십시오. 특히 isaaclab_commit 은")
        print("     git 으로 동기화되지 않으므로 IsaacLab 체크아웃에서 직접 checkout 해야 합니다.")
        return 1

    print("\n  ✅ 필수 항목 전부 일치")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="JE-Loco 환경 대조 (Isaac Sim 기동 없음)")
    p.add_argument("--json", action="store_true", help="JSON 만 출력")
    p.add_argument("--save", metavar="PATH", help="현재 환경을 기준 스냅샷으로 저장")
    p.add_argument("--compare", metavar="PATH", help="기준 스냅샷과 대조. 불일치 시 exit 1")
    args = p.parse_args()

    info = collect()

    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
    else:
        report(info)

    if args.save:
        pathlib.Path(args.save).write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")
        print(f"\n  기준 스냅샷 저장: {args.save}")

    if args.compare:
        return compare(info, pathlib.Path(args.compare))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
