"""에피소드 경로 길이 기반 지형 승급과 종류별 난이도 집계."""

from __future__ import annotations

import torch


def xy_path_increment(current_xy: torch.Tensor, previous_xy: torch.Tensor,
                      valid: torch.Tensor) -> torch.Tensor:
    """각 제어 스텝의 XY 이동량. reset 직후의 무효 위치 차이는 0이다."""
    if current_xy.ndim != 2 or current_xy.shape[-1] != 2 or previous_xy.shape != current_xy.shape:
        raise ValueError("current_xy와 previous_xy는 [N,2]여야 한다")
    if valid.shape != (current_xy.shape[0],):
        raise ValueError("valid는 [N]이어야 한다")
    return torch.where(valid.bool(), torch.linalg.vector_norm(current_xy - previous_xy, dim=-1), 0.0)


def path_length_decisions(path_length_m: torch.Tensor, command_xy: torch.Tensor,
                          terrain_length_m: float, episode_length_s: float,
                          up_fraction: float, down_command_fraction: float,
                          obstacle_cleared: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """기존 IsaacLab 임계값을 경로 길이에 적용한다. 승급이 강등보다 우선한다."""
    if path_length_m.ndim != 1 or command_xy.shape != (path_length_m.shape[0], 2):
        raise ValueError("path_length_m은 [N], command_xy는 [N,2]여야 한다")
    if terrain_length_m <= 0 or episode_length_s <= 0 or up_fraction <= 0 or down_command_fraction < 0:
        raise ValueError("지형·에피소드 길이와 승급 비율은 양수, 강등 비율은 음수가 아니어야 한다")
    move_up = path_length_m > terrain_length_m * up_fraction
    if obstacle_cleared is not None:
        if obstacle_cleared.shape != path_length_m.shape or obstacle_cleared.dtype != torch.bool:
            raise ValueError("obstacle_cleared는 [N] bool이어야 한다")
        move_up &= obstacle_cleared
    move_down = path_length_m < torch.linalg.vector_norm(command_xy, dim=-1) * episode_length_s * down_command_fraction
    return move_up, move_down & ~move_up


def terrain_column_type_ids(num_cols: int, proportions: list[float] | tuple[float, ...],
                            column_assignment_eps: float) -> torch.Tensor:
    """IsaacLab TerrainGenerator._generate_curriculum_terrains와 동일한 열→종류 할당."""
    weights = torch.as_tensor(proportions, dtype=torch.float64)
    if num_cols < 1 or weights.ndim != 1 or weights.numel() < 1 or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("num_cols와 지형 비율이 올바르지 않다")
    if column_assignment_eps < 0:
        raise ValueError("열 할당 epsilon은 음수가 될 수 없다")
    boundaries = torch.cumsum(weights / weights.sum(), dim=0)
    positions = torch.arange(num_cols, dtype=torch.float64) / num_cols + column_assignment_eps
    return torch.searchsorted(boundaries, positions, right=True).clamp(max=weights.numel() - 1)


def terrain_level_means_by_type(levels: torch.Tensor, columns: torch.Tensor,
                                names: list[str], column_type_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    """각 지형 종류의 현재 env 평균 레벨. env가 없는 종류는 0으로 기록한다."""
    if levels.ndim != 1 or columns.shape != levels.shape or column_type_ids.ndim != 1:
        raise ValueError("levels/columns는 [N], column_type_ids는 [num_cols]여야 한다")
    type_per_env = column_type_ids.to(columns.device)[columns.long()]
    means = {}
    for type_id, name in enumerate(names):
        mask = type_per_env == type_id
        means[name] = (levels.float() * mask).sum() / mask.sum().clamp(min=1)
    return means


def terrain_levels_path(env, env_ids, *, command_name: str,
                        up_fraction: float, down_command_fraction: float) -> torch.Tensor:
    """IsaacLab curriculum term. reset 직전까지 누적된 실제 XY 경로 길이를 사용한다."""
    terrain = env.scene.terrain
    if terrain.cfg.terrain_type != "generator":
        raise ValueError("경로 길이 커리큘럼은 generator 지형이 필요하다")
    command = env.command_manager.get_command(command_name)
    path = env.episode_path_length_m[env_ids]
    move_up, move_down = path_length_decisions(
        path, command[env_ids, :2], terrain.cfg.terrain_generator.size[0], env.max_episode_length_s,
        up_fraction, down_command_fraction, env.episode_obstacle_cleared[env_ids],
    )
    terrain.update_env_origins(env_ids, move_up, move_down)
    return terrain.terrain_levels.float().mean()


def obstacle_outer_edges(sub_terrains: dict, patch_size: tuple[float, float],
                         margin_m: float) -> torch.Tensor:
    """중앙 시작점에서 장애물 바깥의 안전한 지면까지 필요한 축별 거리."""
    if margin_m < 0:
        raise ValueError("통과 여유 거리는 음수가 될 수 없다")
    edges = []
    for name, cfg in sub_terrains.items():
        if name.startswith(("stairs_up", "stairs_down")):
            edges.append(min(patch_size) / 2 - cfg.border_width + margin_m)
        elif name == "gap":
            edges.append(cfg.platform_width / 2 + max(cfg.gap_width_range) + margin_m)
        else:
            edges.append(0.0)
    return torch.tensor(edges, dtype=torch.float32)


def obstacle_clearance_step(root_xy: torch.Tensor, origin_xy: torch.Tensor,
                            outer_edge_m: torch.Tensor, consecutive: torch.Tensor,
                            cleared: torch.Tensor, hold_steps: int) -> tuple[torch.Tensor, torch.Tensor]:
    """외곽 안전 지면에 연속으로 머물렀으면 통과. 0 임계값은 장애물 없는 지형이다."""
    if root_xy.ndim != 2 or root_xy.shape[-1] != 2 or origin_xy.shape != root_xy.shape:
        raise ValueError("root_xy와 origin_xy는 [N,2]여야 한다")
    if outer_edge_m.shape != (root_xy.shape[0],) or consecutive.shape != outer_edge_m.shape or cleared.shape != outer_edge_m.shape:
        raise ValueError("거리와 상태는 [N]이어야 한다")
    if hold_steps < 1:
        raise ValueError("hold_steps는 양수여야 한다")
    outside = (root_xy - origin_xy).abs().amax(dim=1) >= outer_edge_m
    next_consecutive = torch.where(outside, consecutive + 1, 0)
    return next_consecutive, cleared | (outer_edge_m <= 0) | (next_consecutive >= hold_steps)
