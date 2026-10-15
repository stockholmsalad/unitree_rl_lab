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


class MambaBackbone(SequenceBackbone):
    """FLA Mamba-1 한 층. 평탄 상태는 [마지막 출력, convolution, SSM] 순서다."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, intermediate_size: int = 256,
                 state_size: int = 16, conv_kernel: int = 4, dt_rank: int | str = "auto"):
        super().__init__()
        if min(in_dim, hidden_dim, intermediate_size, state_size, conv_kernel) < 1:
            raise ValueError("Mamba 차원과 kernel 크기는 양수여야 한다")
        from fla.layers.mamba import Mamba

        self.in_dim = in_dim
        self.out_dim = hidden_dim
        self.intermediate_size = intermediate_size
        self.state_size = state_size
        self.conv_kernel = conv_kernel
        self.state_dim = hidden_dim + intermediate_size * (conv_kernel + state_size)
        self.input_projection = nn.Identity() if in_dim == hidden_dim else nn.Linear(in_dim, hidden_dim)
        self.pre_norm = nn.RMSNorm(hidden_dim)
        self.mixer = Mamba(hidden_size=hidden_dim, intermediate_size=intermediate_size,
                           state_size=state_size, conv_kernel=conv_kernel, dt_rank=dt_rank, layer_idx=0)
        self.post_norm = nn.RMSNorm(hidden_dim)

    def step(self, x: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h.shape != (x.shape[0], self.state_dim):
            raise ValueError("Mamba 상태 shape가 입력 batch와 맞지 않는다")
        _, conv, ssm = torch.split(
            h, (self.out_dim, self.intermediate_size * self.conv_kernel,
                self.intermediate_size * self.state_size), dim=-1,
        )
        conv = conv.reshape(-1, self.intermediate_size, self.conv_kernel)
        ssm = ssm.reshape(-1, self.intermediate_size, self.state_size)
        u = self.input_projection(x)
        # FLA forward의 선택적 CUDA 커널 dispatch를 거치지 않는다. 첫 토큰도
        # 0 상태를 넘겨 항상 slow_forward의 cached 1-token decode를 사용한다.
        mixed, conv_new, ssm_new = self.mixer.slow_forward(
            self.pre_norm(u).unsqueeze(1),
            last_state={"conv_state": conv, "recurrent_state": ssm},
            use_cache=True,
        )
        z = self.post_norm(u + mixed[:, 0])
        state = torch.cat((z, conv_new.flatten(1), ssm_new.flatten(1)), dim=-1)
        return z, state

    def output(self, h: torch.Tensor) -> torch.Tensor:
        return h[:, :self.out_dim]


BACKBONES: dict[str, type[SequenceBackbone]] = {"gru": GRUBackbone, "mamba": MambaBackbone}


def make_backbone(name: str, in_dim: int, hidden_dim: int, **kwargs) -> SequenceBackbone:
    if name not in BACKBONES:
        raise ValueError(f"backbone={name!r}, 지원: {sorted(BACKBONES)}")
    return BACKBONES[name](in_dim, hidden_dim, **kwargs)
