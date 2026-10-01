"""depth 전처리 shape·의미 테스트."""

import math

import pytest
import torch

from unitree_rl_lab.jepa_loco.sensors.depth_proc import apply_left_band, downsample_real, to_two_channel


def test_two_channel_rules():
    d = torch.tensor([[0.0, math.nan, 0.2, 0.28, 1.14, 2.0, 5.0, math.inf]])
    out = to_two_channel(d)
    assert out.shape == (2, 1, 8)
    depth, mask = out[0, 0], out[1, 0]
    assert mask.tolist() == [0, 0, 0, 1, 1, 1, 1, 0]
    assert depth[:3].tolist() == [0, 0, 0] and depth[7] == 0
    assert depth[3] == pytest.approx(0.0)
    assert depth[4] == pytest.approx(0.5)
    assert depth[5] == pytest.approx(1.0) and depth[6] == pytest.approx(1.0)  # >2 m clamp, 유효


def test_two_channel_batch_shape():
    assert to_two_channel(torch.rand(4, 3, 64, 112)).shape == (4, 3, 2, 64, 112)


def test_left_band():
    d = torch.full((2, 64, 112), 1.0)
    d[1] = 0.5
    out = apply_left_band(d, fx=59.0, baseline=0.05)
    assert (out[0, :, :3] == 0).all() and (out[0, :, 3:] == 1.0).all()  # 0.05·59/1.0 = 2.95 px
    assert (out[1, :, :6] == 0).all() and (out[1, :, 6:] == 0.5).all()  # 5.9 px


@pytest.mark.parametrize("mode", ["nearest", "median"])
def test_downsample_shape_and_constant(mode):
    d = torch.full((2, 480, 848), 1.3)
    out = downsample_real(d, (64, 112), mode=mode)
    assert out.shape == (2, 64, 112)
    assert torch.allclose(out, torch.tensor(1.3))


@pytest.mark.parametrize("mode", ["nearest", "median"])
def test_downsample_no_fake_intermediate_depth_at_edge(mode):
    # 계단 모서리: 왼쪽 0.5 m, 오른쪽 1.5 m. 출력은 둘 중 하나여야 한다(bilinear 이면 중간값이 생김).
    d = torch.full((480, 848), 0.5)
    d[:, 425:] = 1.5
    out = downsample_real(d, (64, 112), mode=mode)
    assert set(out.unique().tolist()) <= {0.5, 1.5}


def test_downsample_holes():
    d = torch.full((480, 848), 1.0)
    d[:, ::2] = 0.0  # 절반 결측
    # nearest: footprint 에 유효 픽셀이 있으면 유효
    assert (downsample_real(d, (64, 112), mode="nearest") == 1.0).all()
    # median: 유효 비율 ≈ 0.5 — 임계 0.6 이면 결측
    assert (downsample_real(d, (64, 112), mode="median", min_valid_frac=0.6) == 0).all()
    d[:] = 0.0
    assert (downsample_real(d, (64, 112), mode="nearest") == 0).all()
