"""별도 데이터 선형 probe와 보이지 않는 점 오차 테스트."""

import torch

from unitree_rl_lab.jepa_loco.eval.linear_probe import (
    fit_linear_probe, masked_reconstruction_mse, probe_predict,
)


def test_linear_probe_shape_and_hidden_error():
    train_x = torch.tensor([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    train_y = torch.cat((train_x, train_x[:, :1] + train_x[:, 1:]), dim=1)
    coefficients = fit_linear_probe(train_x, train_y, ridge_alpha=0.01)
    assert coefficients.shape == (3, 3)
    test_x = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
    test_y = torch.cat((test_x, torch.ones(2, 1)), dim=1)
    pred = probe_predict(test_x, coefficients)
    assert pred.shape == test_y.shape
    hidden = torch.tensor([False, False, True])
    assert masked_reconstruction_mse(pred, test_y, hidden) < 0.001


def test_probe_rejects_wrong_shapes():
    try:
        masked_reconstruction_mse(torch.zeros(2, 3), torch.zeros(2, 4), torch.ones(3, dtype=torch.bool))
    except ValueError:
        pass
    else:
        raise AssertionError("shape mismatch가 거절되지 않았다")
