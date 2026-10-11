"""Teacher latent 잡음의 shape·seed·episode reset·영잡음 동일성."""

import torch

from unitree_rl_lab.jepa_loco.eval.latent_noise import TeacherLatentNoise


def test_step_noise_shape_seed_and_new_draw_every_step():
    z = torch.zeros(3, 32)
    std = torch.linspace(0.5, 1.5, 32)
    a = TeacherLatentNoise(std, sigma=0.5, mode="step", seed=43)
    b = TeacherLatentNoise(std, sigma=0.5, mode="step", seed=43)
    first_a, first_b = a.apply(z), b.apply(z)
    second_a, second_b = a.apply(z), b.apply(z)
    assert first_a.shape == z.shape
    torch.testing.assert_close(first_a, first_b)
    torch.testing.assert_close(second_a, second_b)
    assert not torch.equal(first_a, second_a)


def test_episode_noise_only_resamples_done_envs():
    z = torch.zeros(3, 32)
    a = TeacherLatentNoise(torch.ones(32), sigma=0.5, mode="episode", seed=43)
    b = TeacherLatentNoise(torch.ones(32), sigma=0.5, mode="episode", seed=43)
    initial = a.apply(z).clone()
    torch.testing.assert_close(initial, a.apply(z))
    torch.testing.assert_close(initial, b.apply(z))
    a.reset(torch.tensor([False, True, False]))
    b.reset(torch.tensor([False, True, False]))
    updated = a.apply(z)
    torch.testing.assert_close(updated, b.apply(z))
    torch.testing.assert_close(updated[[0, 2]], initial[[0, 2]])
    assert not torch.equal(updated[1], initial[1])


def test_zero_noise_uses_original_tensor_and_does_not_advance_rng():
    z = torch.randn(2, 32)
    for mode in ("step", "episode"):
        noise = TeacherLatentNoise(torch.ones(32), sigma=0, mode=mode, seed=43)
        rng_before = noise.generator.get_state().clone()
        assert noise.apply(z) is z
        noise.reset(torch.tensor([True, False]))
        assert noise.apply(z) is z
        torch.testing.assert_close(noise.generator.get_state(), rng_before)
