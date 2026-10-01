"""시간축 backbone 공통 인터페이스 (CLAUDE.md §4). GRU ↔ Mamba 교체는 ``make_backbone`` 의 이름 하나로 한다.

모든 backbone 은 **게이트된 1스텝 갱신**을 지원한다: depth 는 10 Hz 로만 새 프레임이 오고 제어는 50 Hz 이므로,
``update`` 가 0 인 env 는 상태와 출력을 그대로 유지한다(마지막 값 재사용).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SequenceBackbone(nn.Module):
    """x_t [N, in_dim] 과 상태 h [N, state_dim] 로부터 z_t [N, out_dim] 를 내는 순환 모듈."""

    in_dim: int
    out_dim: int
    state_dim: int

    def initial_state(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(n, self.state_dim, device=device)

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """게이트 없는 1스텝. (z, h_new)."""
        raise NotImplementedError

    def output(self, h: torch.Tensor) -> torch.Tensor:
        """상태에서 출력 z 를 읽는다(갱신 없는 스텝에서 재사용)."""
        raise NotImplementedError

    def gated_step(self, x: torch.Tensor, h: torch.Tensor, update: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """update [N] (bool) 인 env 만 갱신."""
        _, h_new = self.step(x, h)
        h = torch.where(update[:, None], h_new, h)
        return self.output(h), h

    def gated_sequence(
        self, x: torch.Tensor, h0: torch.Tensor, update: torch.Tensor, reset: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """x [T, N, in_dim], update [T, N] → z [T, N, out_dim], 마지막 상태.

        reset [T, N]: 해당 스텝 처리 전에 상태를 0 으로 (시퀀스 중간의 에피소드 시작).
        """
        h = h0
        outs = []
        for t in range(x.shape[0]):
            if reset is not None:
                h = torch.where(reset[t][:, None], torch.zeros_like(h), h)
            z, h = self.gated_step(x[t], h, update[t])
            outs.append(z)
        return torch.stack(outs), h


class GRUBackbone(SequenceBackbone):
    def __init__(self, in_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.in_dim, self.out_dim, self.state_dim = in_dim, hidden_dim, hidden_dim
        self.cell = nn.GRUCell(in_dim, hidden_dim)

    def step(self, x, h):
        h = self.cell(x, h)
        return h, h

    def output(self, h):
        return h


BACKBONES: dict[str, type[SequenceBackbone]] = {"gru": GRUBackbone}


def make_backbone(name: str, in_dim: int, hidden_dim: int) -> SequenceBackbone:
    if name not in BACKBONES:
        raise ValueError(f"backbone={name!r}, 지원: {sorted(BACKBONES)}")
    return BACKBONES[name](in_dim, hidden_dim)
