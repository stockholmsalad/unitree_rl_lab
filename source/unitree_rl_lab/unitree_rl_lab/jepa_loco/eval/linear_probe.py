"""Student terrain latent의 teacher 높이맵 선형 복원과 사각지대 오차."""

from __future__ import annotations

import torch


def fit_linear_probe(features: torch.Tensor, heightmap: torch.Tensor,
                     ridge_alpha: float) -> torch.Tensor:
    """별도 train 데이터 [N,D]→[N,P] ridge 회귀; 마지막 행은 bias."""
    if features.ndim != 2 or heightmap.ndim != 2 or features.shape[0] != heightmap.shape[0]:
        raise ValueError("probe 입력은 같은 N의 [N,D], [N,P]여야 한다")
    if features.shape[0] < 2 or ridge_alpha < 0:
        raise ValueError("probe 표본은 2개 이상, ridge_alpha는 0 이상이어야 한다")
    x = torch.cat((features.double(), torch.ones(features.shape[0], 1, device=features.device,
                                                dtype=torch.float64)), dim=1)
    y = heightmap.double()
    regularizer = torch.eye(x.shape[1], device=x.device, dtype=x.dtype) * ridge_alpha
    regularizer[-1, -1] = 0
    return torch.linalg.solve(x.T @ x + regularizer, x.T @ y)


def probe_predict(features: torch.Tensor, coefficients: torch.Tensor) -> torch.Tensor:
    if features.ndim != 2 or coefficients.ndim != 2 or coefficients.shape[0] != features.shape[1] + 1:
        raise ValueError("probe feature/계수 shape가 맞지 않는다")
    x = torch.cat((features.double(), torch.ones(features.shape[0], 1, device=features.device,
                                                dtype=torch.float64)), dim=1)
    return (x @ coefficients).to(features.dtype)


def masked_reconstruction_mse(pred: torch.Tensor, target: torch.Tensor,
                              point_mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.ndim != 2 or point_mask.shape != (pred.shape[1],):
        raise ValueError("복원값 [N,P]와 점 mask [P]가 필요하다")
    if not point_mask.any() or pred.shape[0] == 0:
        raise ValueError("가시/사각 점과 평가 표본이 1개 이상 필요하다")
    return (pred[:, point_mask] - target[:, point_mask]).square().mean()
