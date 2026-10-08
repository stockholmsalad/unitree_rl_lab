"""wide heightscan의 future pose 샘플링 순서·기하·가시성."""

from types import SimpleNamespace

import pytest
import torch

from unitree_rl_lab.jepa_loco.envs.oracle_heightmap import (
    future_height_scan, sample_command_future_patch, sanitize_height_scan,
)


def _linear_wide(size=(4.0, 3.0), resolution=0.1):
    x = torch.arange(-size[0] / 2, size[0] / 2 + 1.0e-9, resolution)
    y = torch.arange(-size[1] / 2, size[1] / 2 + 1.0e-9, resolution)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    return (2 * xx + 3 * yy).reshape(1, -1)


def test_future_patch_shape_order_and_translation():
    wide = _linear_wide().repeat(2, 1)
    command = torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    patch, visible = sample_command_future_patch(wide, command, 1.0, (4.0, 3.0), (1.6, 1.0), 0.1)
    assert patch.shape == (2, 187) and visible.tolist() == [True, True]
    x = torch.arange(-0.8, 0.8 + 1.0e-9, 0.1)
    y = torch.arange(-0.5, 0.5 + 1.0e-9, 0.1)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    expected = (2 * xx + 3 * yy).flatten()
    assert torch.allclose(patch[0], expected, atol=1e-5)
    assert torch.allclose(patch[1], expected + 1.0, atol=1e-5)


def test_future_patch_visibility_and_shape_validation():
    wide = _linear_wide()
    far_command = torch.tensor([[2.0, 0.0, 0.0]])
    patch, visible = sample_command_future_patch(wide, far_command, 1.0, (4.0, 3.0), (1.6, 1.0), 0.1)
    assert patch.shape == (1, 187) and not visible.item()
    with pytest.raises(ValueError):
        sample_command_future_patch(wide[:, :10], far_command, 1.0, (4.0, 3.0), (1.6, 1.0), 0.1)


def test_future_patch_rotates_with_extrapolated_yaw():
    wide = _linear_wide()
    command = torch.tensor([[0.0, 0.0, torch.pi / 2]])
    patch, visible = sample_command_future_patch(wide, command, 1.0, (4.0, 3.0), (1.6, 1.0), 0.1)
    x = torch.arange(-0.8, 0.8 + 1.0e-9, 0.1)
    y = torch.arange(-0.5, 0.5 + 1.0e-9, 0.1)
    xx, yy = torch.meshgrid(x, y, indexing="xy")
    expected = (-2 * yy + 3 * xx).flatten()
    assert visible.item()
    assert torch.allclose(patch[0], expected, atol=1e-5)


def _fake_future_env(raw_values: torch.Tensor, clip_range=(-1.0, 1.0)):
    wide_pattern = SimpleNamespace(size=(4.0, 3.0), resolution=0.1, ordering="yx")
    current_pattern = SimpleNamespace(size=(1.6, 1.0), resolution=0.1, ordering="xy")
    wide_hits = torch.zeros(1, raw_values.numel(), 3)
    wide_hits[..., 2] = 20.0 - raw_values.reshape(1, -1) - 0.5
    wide = SimpleNamespace(cfg=SimpleNamespace(pattern_cfg=wide_pattern),
                           data=SimpleNamespace(pos_w=torch.tensor([[0.0, 0.0, 20.0]]),
                                                ray_hits_w=wide_hits))
    current = SimpleNamespace(cfg=SimpleNamespace(pattern_cfg=current_pattern))
    scan_cfg = SimpleNamespace(clip=clip_range)
    env = SimpleNamespace(
        cfg=SimpleNamespace(observations=SimpleNamespace(terrain_future=SimpleNamespace(scan=scan_cfg))),
        scene=SimpleNamespace(sensors={"wide": wide, "current": current}),
        command_manager=SimpleNamespace(get_command=lambda name: torch.zeros(1, 3)),
    )
    return env, SimpleNamespace(name="wide"), SimpleNamespace(name="current")


def test_future_scan_sanitizes_nonfinite_gap_hits_before_interpolation():
    raw = _linear_wide().flatten()
    raw[17 * 31 + 13] = -float("inf")
    raw[18 * 31 + 14] = float("inf")
    raw[19 * 31 + 15] = float("nan")
    env, wide_cfg, current_cfg = _fake_future_env(raw)
    patch = future_height_scan(env, wide_cfg, current_cfg, "base_velocity", horizon_s=0.05)
    assert patch.shape == (1, 187)
    assert torch.isfinite(patch).all()
    assert (patch >= -1.0).all() and (patch <= 1.0).all()
    assert sanitize_height_scan(torch.tensor([-float("inf"), float("inf"), float("nan")]),
                                (-1.0, 1.0)).tolist() == [-1.0, 1.0, -1.0]


def test_zero_horizon_future_matches_current_clipped_patch_and_clip_cfg():
    raw = _linear_wide().flatten()
    raw[17 * 31 + 13] = -float("inf")
    raw[18 * 31 + 14] = float("inf")
    env, wide_cfg, current_cfg = _fake_future_env(raw)
    future = future_height_scan(env, wide_cfg, current_cfg, "base_velocity", horizon_s=0.0)
    # mdp.height_scan의 현재 관측은 보간 없이 ObservationTermCfg.clip을 적용한다.
    clipped_wide = raw.clamp(-1.0, 1.0).reshape(41, 31)
    current = clipped_wide[12:29, 10:21].transpose(0, 1).reshape(1, -1)
    assert torch.allclose(future, current, atol=1.0e-5)
    env.cfg.observations.terrain_future.scan.clip = (-0.5, 0.5)
    with pytest.raises(ValueError, match="clip_range와 관측 clip이 다르다"):
        future_height_scan(env, wide_cfg, current_cfg, "base_velocity", horizon_s=0.0)
