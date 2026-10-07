"""첫 단 접촉 시도율과 종료 분류 순수 함수의 shape/값 검증."""

import pytest
import torch

from unitree_rl_lab.jepa_loco.envs.obstacle_diagnostics import first_tread_contact, termination_classes


def test_termination_reason_counts():
    categories = termination_classes(
        torch.tensor([1, 0, 0, 0], dtype=torch.bool),
        torch.tensor([0, 1, 0, 0], dtype=torch.bool),
        torch.tensor([0, 0, 1, 0], dtype=torch.bool),
    )
    assert categories.shape == (4,)
    assert torch.bincount(categories, minlength=4).tolist() == [1, 1, 1, 1]


def test_first_tread_attempt_rate_and_shape():
    feet = torch.tensor([
        [[1.30, 0.0], [0.5, 0.0]],
        [[1.30, 0.0], [1.35, 0.0]],
        [[1.10, 0.0], [0.5, 0.0]],
    ])
    forces = torch.tensor([[2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    attempts = first_tread_contact(feet, torch.zeros(3, 2), forces, 1.2, 1.5, 1.0)
    assert attempts.shape == (3,)
    assert attempts.tolist() == [True, False, False]
    assert attempts.float().mean().item() == pytest.approx(1 / 3)
