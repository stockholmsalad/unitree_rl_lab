"""고정 teacher 평가의 latent 잡음. 시뮬레이터 RNG와 독립된 난수열을 사용한다."""

from __future__ import annotations

import torch


class TeacherLatentNoise:
    """teacher 표준편차 단위의 Gaussian 잡음을 step 또는 episode마다 만든다."""

    def __init__(self, std: torch.Tensor, sigma: float, mode: str, seed: int):
        if std.ndim != 1 or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("teacher std는 유한한 양수 [D]여야 한다")
        if sigma < 0 or not torch.isfinite(torch.tensor(sigma)):
            raise ValueError("sigma는 유한한 0 이상이어야 한다")
        if mode not in ("step", "episode"):
            raise ValueError("잡음 모드는 step 또는 episode여야 한다")
        self.std = std
        self.sigma = float(sigma)
        self.mode = mode
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
        self.episode_epsilon: torch.Tensor | None = None

    def _draw(self, shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
        return torch.randn(shape, generator=self.generator, device="cpu").to(device)

    def apply(self, teacher_z: torch.Tensor) -> torch.Tensor:
        if teacher_z.ndim != 2 or teacher_z.shape[1] != self.std.numel():
            raise ValueError("teacher latent는 [N,D]여야 한다")
        if self.sigma == 0:
            return teacher_z  # 기존 teacher 행동 경로와 bitwise 동일; RNG도 소비하지 않는다.
        if self.mode == "step":
            epsilon = self._draw(tuple(teacher_z.shape), teacher_z.device)
        else:
            if self.episode_epsilon is None or self.episode_epsilon.shape != teacher_z.shape:
                self.episode_epsilon = self._draw(tuple(teacher_z.shape), teacher_z.device)
            epsilon = self.episode_epsilon
        return teacher_z + self.std.to(teacher_z.device) * self.sigma * epsilon

    def reset(self, dones: torch.Tensor) -> None:
        if self.sigma == 0 or self.mode != "episode" or self.episode_epsilon is None:
            return
        mask = dones.bool().reshape(-1)
        if mask.shape[0] != self.episode_epsilon.shape[0]:
            raise ValueError("done mask 크기가 env 수와 다르다")
        ids = mask.nonzero(as_tuple=True)[0]
        if ids.numel():
            fresh = self._draw((ids.numel(), self.std.numel()), self.episode_epsilon.device)
            self.episode_epsilon = self.episode_epsilon.index_copy(0, ids, fresh)
