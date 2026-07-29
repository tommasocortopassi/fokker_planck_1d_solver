"""The programmatic front end must agree with the solvers it wraps, and
must accept the same inputs the GUI does (expressions, callables, numbers,
infinite bounds).
"""
import numpy as np
import pytest

from fp1d.api import solve
from fp1d.boundary_conditions import BoundaryCondition
from fp1d.finite_volume import backward_euler
from fp1d.grid import make_grid
from fp1d.initial_conditions import build_initial_density


def test_api_matches_calling_the_solver_directly():
    """`solve` is a wrapper, not a second implementation: on a finite
    domain, where adaptive truncation does nothing, it must reproduce a
    hand-assembled call bit for bit.
    """
    save_times = [0.0, 0.5, 1.0]
    run = solve(method='be', drift='-x', diffusion=0.4, left=-4, right=4,
                dx=0.05, total_time=1.0, dt=0.005, save_times=save_times,
                bc='neumann')

    grid, _ = make_grid(-4, 4, 0.05)
    p0 = build_initial_density('gaussian', grid)
    direct = backward_euler(p0, grid, lambda x, t: -x,
                            lambda x, t: 0.4 * np.ones_like(x),
                            0.005, 1.0, save_times, BoundaryCondition('neumann'))

    assert np.allclose(run.result.snapshots, direct.snapshots)


def test_coefficients_accept_expression_callable_and_number():
    """The three spellings of a constant coefficient are equivalent."""
    common = dict(method='be', left=-3, right=3, dx=0.05, total_time=0.5,
                  dt=0.005, bc='neumann')
    by_number = solve(drift=0.0, diffusion=0.25, **common)
    by_string = solve(drift='0.0', diffusion='0.25', **common)
    by_callable = solve(drift=lambda x, t: np.zeros_like(x),
                        diffusion=lambda x, t: 0.25 * np.ones_like(x), **common)

    assert np.allclose(by_number.result.snapshots, by_string.result.snapshots)
    assert np.allclose(by_number.result.snapshots, by_callable.result.snapshots)


def test_infinite_bounds_are_truncated_and_conserve_mass():
    """An unbounded domain is closed off far enough that a reflecting
    wall never sees the density, so mass stays 1 and the domain grows
    past the initial guess.
    """
    run = solve(method='be', drift=0.0, diffusion=0.5, left='-inf',
                right='+inf', dx=0.05, total_time=1.0, dt=0.005, bc='neumann')

    assert run.grid.left < -3.0 and run.grid.right > 3.0
    assert run.parameters['max_mass_deviation_from_1'] < 1e-10


def test_dirichlet_reports_absorbed_fraction_not_conservation_error():
    run = solve(method='be', drift=0.0, diffusion=0.5, left=-1.5, right=1.5,
                dx=0.02, total_time=1.0, dt=0.002, bc='dirichlet')

    assert 'absorbed_mass_fraction' in run.parameters
    assert 'max_mass_deviation_from_1' not in run.parameters
    assert run.parameters['absorbed_mass_fraction'] > 0.1


def test_rejected_inputs():
    with pytest.raises(ValueError):
        solve(method='runge-kutta', total_time=0.1, dt=0.01)
    with pytest.raises(ValueError):
        solve(bc='periodic', left='-inf', total_time=0.1, dt=0.01)
    with pytest.raises(ValueError):
        solve(diffusion=-1.0, total_time=0.1, dt=0.01)
