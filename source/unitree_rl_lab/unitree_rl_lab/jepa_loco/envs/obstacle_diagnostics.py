"""첫 단 접촉과 근접 종료를 분류하는 Isaac 비의존 함수."""

from __future__ import annotations

import torch


TERMINATION_NAMES = ("time_out", "base_contact", "bad_orientation", "other")


def termination_classes(time_out: torch.Tensor, base_contact: torch.Tensor,
                        bad_orientation: torch.Tensor) -> torch.Tensor:
    if time_out.ndim != 1 or base_contact.shape != time_out.shape or bad_orientation.shape != time_out.shape:
        raise ValueError("종료 플래그는 [N]이어야 한다")
    classes = torch.full_like(time_out, 3, dtype=torch.long)
    classes[bad_orientation.bool()] = 2
    classes[base_contact.bool()] = 1
    classes[time_out.bool()] = 0
    return classes


def first_tread_contact(front_foot_xy: torch.Tensor, origin_xy: torch.Tensor,
                        front_force_n: torch.Tensor, near_edge_m: float,
                        far_edge_m: float, force_threshold_n: float) -> torch.Tensor:
    if (front_foot_xy.ndim != 3 or front_foot_xy.shape[1:] != (2, 2)
            or origin_xy.shape != (front_foot_xy.shape[0], 2)
            or front_force_n.shape != (front_foot_xy.shape[0], 2)):
        raise ValueError("앞발 [N,2,2], 원점 [N,2], 접촉력 [N,2]가 필요하다")
    if near_edge_m < 0 or far_edge_m <= near_edge_m or force_threshold_n < 0:
        raise ValueError("첫 단 범위와 접촉력 임계값이 올바르지 않다")
    local = (front_foot_xy - origin_xy[:, None]).abs()
    along = local.amax(dim=-1)
    on_tread = (along >= near_edge_m) & (along <= far_edge_m)
    return (on_tread & (front_force_n >= force_threshold_n)).any(dim=1)
