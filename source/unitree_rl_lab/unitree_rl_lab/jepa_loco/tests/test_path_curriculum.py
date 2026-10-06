"""yaw-rate 원 운동, 강등, 열→지형 종류별 로그의 순수 함수 검증."""

import pytest
import torch

from unitree_rl_lab.jepa_loco.envs.path_curriculum import (
    path_length_decisions,
    terrain_column_type_ids,
    terrain_level_means_by_type,
    xy_path_increment,
)


def test_path_accumulates_circle_and_ignores_initial_teleport():
    positions = [torch.tensor([[0.0, 0.0]]), torch.tensor([[1.0, 0.0]]),
                 torch.tensor([[1.0, 1.0]]), torch.tensor([[0.0, 1.0]]),
                 torch.tensor([[0.0, 0.0]])]
    traveled = sum(xy_path_increment(b, a, torch.tensor([True])) for a, b in zip(positions[:-1], positions[1:]))
    assert traveled.item() == 4.0
    assert torch.equal(xy_path_increment(torch.tensor([[10.0, 10.0]]), positions[0], torch.tensor([False])),
                       torch.zeros(1))


def test_circular_trajectory_progresses_by_walked_path():
    # 시작점으로 돌아온 로봇이어도 4m 넘게 걸었다면 승급한다.
    path = torch.tensor([4.1, 0.2, 3.9])
    command = torch.tensor([[0.5, 0.0], [0.5, 0.0], [0.0, 0.0]])
    up, down = path_length_decisions(path, command, 8.0, 20.0, 0.5, 0.5)
    assert up.tolist() == [True, False, False]
    assert down.tolist() == [False, True, False]


def test_up_wins_when_both_thresholds_match():
    up, down = path_length_decisions(torch.tensor([4.1]), torch.tensor([[1.0, 0.0]]), 8.0, 20.0, 0.5, 0.5)
    assert up.item() and not down.item()
    with pytest.raises(ValueError):
        path_length_decisions(torch.zeros(2, 1), torch.zeros(2, 2), 8.0, 20.0, 0.5, 0.5)


def test_terrain_type_columns_and_per_type_means():
    names = ["flat", "rough", "stairs_up", "stairs_down", "gap"]
    mapping = terrain_column_type_ids(20, [0.1, 0.2, 0.25, 0.25, 0.2], 0.001)
    assert torch.bincount(mapping, minlength=5).tolist() == [2, 4, 5, 5, 4]
    levels = torch.tensor([1, 3, 2, 4, 5], dtype=torch.long)
    columns = torch.tensor([0, 1, 2, 6, 16], dtype=torch.long)
    means = terrain_level_means_by_type(levels, columns, names, mapping)
    assert {name: value.item() for name, value in means.items()} == {
        "flat": 2.0, "rough": 2.0, "stairs_up": 4.0, "stairs_down": 0.0, "gap": 5.0,
    }


def test_type_mapping_rejects_invalid_weights():
    with pytest.raises(ValueError):
        terrain_column_type_ids(20, [0.0, 0.0], 0.001)
