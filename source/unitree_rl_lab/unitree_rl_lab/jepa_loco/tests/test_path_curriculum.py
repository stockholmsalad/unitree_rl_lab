"""yaw-rate 원 운동, 강등, 열→지형 종류별 로그의 순수 함수 검증."""

import pytest
import torch
from types import SimpleNamespace

from unitree_rl_lab.jepa_loco.envs.path_curriculum import (
    effective_clearance_edges,
    obstacle_clearance_step,
    obstacle_clearance_edges,
    path_length_decisions,
    terrain_column_type_ids,
    terrain_level_means_by_type,
    terrain_levels_path,
    stair_first_step_edges,
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


def test_obstacle_gate_blocks_path_only_promotion():
    up, down = path_length_decisions(
        torch.tensor([5.0, 5.0]), torch.tensor([[0.2, 0.0], [0.2, 0.0]]),
        8.0, 20.0, 0.5, 0.5, torch.tensor([False, True]),
    )
    assert up.tolist() == [False, True]
    assert down.tolist() == [False, False]


def test_warmup_uses_path_only_then_obstacle_gate():
    edges = effective_clearance_edges(
        torch.tensor([0.0, 1.6]), torch.tensor([1, 1, 1]),
        torch.tensor([0, 1, 2]), warmup_levels=2,
    )
    assert edges.tolist() == pytest.approx([0.0, 0.0, 1.6])
    consecutive, cleared = obstacle_clearance_step(
        torch.zeros(3, 2), torch.zeros(3, 2), edges,
        torch.zeros(3, dtype=torch.long), torch.zeros(3, dtype=torch.bool), 10,
    )
    up, _ = path_length_decisions(torch.full((3,), 5.0), torch.zeros(3, 2), 8.0, 20.0, 0.5, 0.5, cleared)
    assert up.tolist() == [True, True, False]


def test_obstacle_clearance_needs_sustained_outer_ground():
    sub_terrains = {
        "flat": SimpleNamespace(),
        "stairs_up": SimpleNamespace(border_width=1.0, platform_width=3.0, step_width=0.3),
        "stairs_down": SimpleNamespace(border_width=1.0, platform_width=3.0, step_width=0.3),
        "gap": SimpleNamespace(platform_width=3.0, gap_width_range=(0.1, 0.3)),
    }
    assert stair_first_step_edges(sub_terrains["stairs_up"], (8.0, 8.0)) == pytest.approx((1.2, 1.5))
    assert obstacle_clearance_edges(sub_terrains, (8.0, 8.0), 0.1).tolist() == pytest.approx([0.0, 1.6, 1.6, 1.9])
    root = torch.tensor([[0.0, 0.0], [1.7, 0.0], [0.0, 1.7], [2.0, 0.0]])
    edge = obstacle_clearance_edges(sub_terrains, (8.0, 8.0), 0.1)
    consecutive = torch.zeros(4, dtype=torch.long)
    cleared = torch.zeros(4, dtype=torch.bool)
    for _ in range(2):
        consecutive, cleared = obstacle_clearance_step(root, torch.zeros_like(root), edge, consecutive, cleared, 3)
    assert cleared.tolist() == [True, False, False, False]
    consecutive, cleared = obstacle_clearance_step(root, torch.zeros_like(root), edge, consecutive, cleared, 3)
    assert cleared.tolist() == [True, True, True, True]
    root[1] = 0
    consecutive, cleared = obstacle_clearance_step(root, torch.zeros_like(root), edge, consecutive, cleared, 3)
    assert cleared[1].item()


def test_curriculum_term_passes_path_decisions_to_terrain():
    captured = {}
    terrain = SimpleNamespace(
        cfg=SimpleNamespace(terrain_type="generator", terrain_generator=SimpleNamespace(size=(8.0, 8.0))),
        terrain_levels=torch.tensor([0, 2]),
        update_env_origins=lambda ids, up, down: captured.update(ids=ids, up=up, down=down),
    )
    env = SimpleNamespace(
        scene=SimpleNamespace(terrain=terrain),
        command_manager=SimpleNamespace(get_command=lambda _: torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])),
        episode_path_length_m=torch.tensor([4.1, 0.2]),
        episode_obstacle_cleared=torch.tensor([True, False]),
        max_episode_length_s=20.0,
    )
    ids = torch.tensor([0, 1])
    mean = terrain_levels_path(env, ids, command_name="base_velocity", up_fraction=0.5,
                               down_command_fraction=0.5)
    assert captured["ids"] is ids
    assert captured["up"].tolist() == [True, False]
    assert captured["down"].tolist() == [False, True]
    assert mean.item() == 1.0


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
