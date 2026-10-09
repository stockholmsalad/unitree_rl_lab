"""고정 폭 gap 평가의 메시 검사, 기하 및 성공 판정."""

from __future__ import annotations

import numpy as np
import torch
import trimesh
import isaaclab.terrains as terrain_gen


def fixed_width_gap_cfg(template, width_m: float, proportion: float):
    """학습용 EasyGapCfg.function을 가져오지 않는 평가 전용 원본 gap cfg."""
    if width_m <= 0 or proportion <= 0:
        raise ValueError("평가 gap 폭과 비율은 양수여야 한다")
    return terrain_gen.MeshGapTerrainCfg(
        proportion=proportion, gap_width_range=(width_m, width_m),
        platform_width=template.platform_width,
    )


def gap_geometry(size_xy: tuple[float, float], platform_width: float,
                 gap_width: float) -> tuple[float, float]:
    """중앙 원점 기준 +x 방향 플랫폼 안쪽/바깥 지면 시작 가장자리."""
    if min(*size_xy, platform_width, gap_width) <= 0:
        raise ValueError("타일 크기·플랫폼 폭·gap 폭은 양수여야 한다")
    inner_edge = platform_width / 2
    outer_edge = inner_edge + gap_width
    if outer_edge >= min(size_xy) / 2:
        raise ValueError("gap 바깥쪽에 지면이 남아야 한다")
    return inner_edge, outer_edge


def validate_gap_origins(origin_z: torch.Tensor, tolerance_m: float) -> None:
    """IsaacLab gap_terrain의 중앙 플랫폼 높이는 z=0이다."""
    if origin_z.ndim != 1 or tolerance_m < 0:
        raise ValueError("origin_z는 [N], 허용오차는 0 이상이어야 한다")
    if not torch.isfinite(origin_z).all() or (origin_z.abs() > tolerance_m).any():
        bad = (~torch.isfinite(origin_z) | (origin_z.abs() > tolerance_m)).nonzero(as_tuple=True)[0]
        raise ValueError(f"평가 gap origin z가 0이 아닌 env가 있다: {bad[:10].tolist()}")


def validate_gap_tiles(mesh: trimesh.Trimesh, origins: torch.Tensor, widths: torch.Tensor,
                       size_xy: tuple[float, float], platform_width: float,
                       ray_height_m: float, inner_inset_m: float,
                       outer_outset_m: float, tolerance_m: float) -> list[dict]:
    """가져온 실제 메시의 안쪽/구멍/바깥쪽을 수직 ray로 검사한다."""
    if origins.ndim != 2 or origins.shape[1] != 3 or widths.shape != (origins.shape[0],):
        raise ValueError("origins는 [N,3], widths는 [N]이어야 한다")
    if ray_height_m <= 0 or inner_inset_m <= 0 or outer_outset_m <= 0:
        raise ValueError("ray 높이와 가장자리 회피 거리는 양수여야 한다")
    validate_gap_origins(origins[:, 2], tolerance_m)
    rows = []
    starts = []
    for env_id, (origin, width) in enumerate(zip(origins.tolist(), widths.tolist())):
        inner, outer = gap_geometry(size_xy, platform_width, width)
        if inner_inset_m >= inner or outer + outer_outset_m >= size_xy[0] / 2:
            raise ValueError("gap 검사용 ray가 플랫폼 또는 타일 밖에 있다")
        sample_x = (inner - inner_inset_m, (inner + outer) / 2, outer + outer_outset_m)
        starts.extend((origin[0] + x, origin[1], origin[2] + ray_height_m) for x in sample_x)
        rows.append({"env_id": env_id, "gap_width_m": width, "inner_edge_m": inner,
                     "outer_edge_m": outer, "probe_x_m": list(sample_x)})
    ray_starts = np.asarray(starts, dtype=np.float64)
    ray_dirs = np.tile(np.array([[0.0, 0.0, -1.0]]), (len(starts), 1))
    hits = mesh.ray.intersects_any(ray_starts, ray_dirs).reshape(-1, 3)
    for row, (inner_hit, gap_hit, outer_hit) in zip(rows, hits.tolist()):
        row["inner_ground_hit"] = inner_hit
        row["gap_miss"] = not gap_hit
        row["outer_ground_hit"] = outer_hit
        if not (inner_hit and not gap_hit and outer_hit):
            raise ValueError(f"평가 gap 타일의 실제 ray 기하가 다르다: {row}")
    return rows


def gap_success_step(forward_x: torch.Tensor, outer_edges: torch.Tensor,
                     active: torch.Tensor, consecutive: torch.Tensor,
                     success: torch.Tensor, margin_m: float,
                     hold_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """종료 전 바깥 가장자리+여유에 연속 체류한 env만 성공으로 latch한다."""
    if (forward_x.ndim != 1 or outer_edges.shape != forward_x.shape
            or active.shape != forward_x.shape or consecutive.shape != forward_x.shape
            or success.shape != forward_x.shape):
        raise ValueError("gap 성공 판정 입력은 모두 [N]이어야 한다")
    if margin_m < 0 or hold_steps < 1:
        raise ValueError("위치 여유는 0 이상, 유지 스텝은 양수여야 한다")
    next_consecutive = torch.where(active & (forward_x >= outer_edges + margin_m), consecutive + 1, 0)
    return next_consecutive, success | (next_consecutive >= hold_steps)


def gap_failure_reason(success: bool, non_timeout_done: bool, max_forward_m: float,
                       last_x_m: float, max_body_drop_m: float, inner_edge_m: float,
                       outer_edge_m: float, fall_drop_m: float,
                       near_margin_m: float) -> str:
    """종료 시 gap 주변의 몸체 낙하, 접근 정지, 그 밖의 실패를 분리한다."""
    if success:
        return "success"
    if non_timeout_done and (inner_edge_m - near_margin_m <= last_x_m <= outer_edge_m + near_margin_m) and max_body_drop_m >= fall_drop_m:
        return "gap_fall_termination"
    if max_body_drop_m >= fall_drop_m:
        # 시간초과까지 구덩이에 남는 경우도 있다. x가 gap 전이면 gap 추락으로 단정하지 않는다.
        return "body_drop"
    if not non_timeout_done and max_forward_m < inner_edge_m:
        return "gap_front_stall"
    return "other"
