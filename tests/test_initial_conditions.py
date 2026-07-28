import numpy as np
import pytest
from fp1d.grid import Grid1D, finite_domain
from fp1d.initial_conditions import build_initial_density, sample_particles_from_density


def test_presets_integrate_to_one():
    grid = Grid1D(-4, 4, 100)
    for name in ['gaussian', 'uniform', 'bimodal', 'left-block']:
        p = build_initial_density(name, grid)
        assert abs(grid.dx * np.sum(p) - 1.0) < 1e-10


def test_custom_expression_is_normalized():
    grid = Grid1D(-4, 4, 100)
    p = build_initial_density('custom', grid, custom_expr='np.exp(-0.5*(x/0.5)**2)')
    assert abs(grid.dx * np.sum(p) - 1.0) < 1e-10


def test_custom_expression_rejects_negative_values():
    grid = Grid1D(-4, 4, 50)
    with pytest.raises(ValueError):
        build_initial_density('custom', grid, custom_expr='x')  # negative for x < 0


def test_particles_sampled_from_density_match_its_mean():
    grid = Grid1D(-6, 6, 200)
    p = build_initial_density('gaussian', grid)  # mean 0 by construction
    rng = np.random.default_rng(0)
    samples = sample_particles_from_density(grid, p, n_trials=200_000, rng=rng)
    assert abs(samples.mean()) < 0.02


def test_finite_domain_truncates_infinite_bounds():
    left, right, msg = finite_domain('-inf', '+inf', -7.0, 9.0)
    assert left == -7.0 and right == 9.0
    assert msg  # a truncation message must be shown to the user


def test_finite_domain_substitutes_only_the_infinite_side():
    left, right, msg = finite_domain('-2.5', '+inf', -7.0, 9.0)
    assert left == -2.5 and right == 9.0
    assert msg


def test_finite_domain_passes_through_finite_bounds():
    left, right, msg = finite_domain('-2.5', '3.0', -7.0, 9.0)
    assert left == -2.5 and right == 3.0
    assert msg == ''
