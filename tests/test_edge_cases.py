"""Regression tests for the edge cases fixed after the correctness review.

Each test names the behaviour it pins down, and each one fails on the
version of the code that preceded the fix.
"""
import numpy as np
import pytest

from fp1d import solve
from fp1d.boundary_conditions import BoundaryCondition
from fp1d.coefficients_io import load_coefficient_set
from fp1d.finite_volume import backward_euler
from fp1d.grid import make_grid, parse_bound
from fp1d.initial_conditions import build_initial_density
from fp1d.results import SolutionRecorder
from fp1d.stochastic_solver import euler_maruyama


@pytest.mark.parametrize('method', ['forward euler', 'backward euler',
                                     'euler-maruyama'])
@pytest.mark.parametrize('total_time', [0.0, -1.0])
def test_non_positive_total_time_is_rejected(method, total_time):
    """A zero-length run used to divide by zero while renormalizing dt,
    and a negative one used to run happily and report a save time before
    the start of the simulation.
    """
    with pytest.raises(ValueError, match='Total time must be positive'):
        solve(method=method, total_time=total_time, dt=0.01, trials=10)


@pytest.mark.parametrize('method', ['backward euler', 'euler-maruyama'])
def test_non_positive_dt_is_rejected(method):
    with pytest.raises(ValueError, match='dt must be positive'):
        solve(method=method, total_time=1.0, dt=0.0, trials=10)


def test_constant_custom_initial_condition_is_broadcast():
    """A custom expression that ignores x is a flat density, exactly as
    the same expression in the drift or diffusivity box is a constant
    field - it used to be rejected as 'one value per grid point'.
    """
    grid, _ = make_grid(-2.0, 2.0, 0.1)
    flat = build_initial_density('custom', grid, '1.0')

    assert flat.shape == grid.centers.shape
    assert np.allclose(flat, flat[0])
    assert np.isclose(grid.dx * flat.sum(), 1.0)
    assert np.allclose(flat, build_initial_density('uniform', grid))


def test_wrongly_shaped_custom_expression_still_rejected():
    """Broadcasting a scalar must not weaken the shape check itself."""
    grid, _ = make_grid(-2.0, 2.0, 0.1)
    with pytest.raises(ValueError, match='one value per grid point'):
        build_initial_density('custom', grid, 'np.zeros(3)')


def test_negative_diffusion_only_at_a_dirichlet_wall_is_rejected():
    """The bulk non-negativity check samples cell centres, which never
    land on x = left or x = right. A diffusivity negative only there used
    to run to completion and report a plausible absorbed fraction.
    """
    with pytest.raises(ValueError, match='Diffusion must be non-negative'):
        solve(method='backward euler', drift=0.0,
              diffusion='np.where(np.abs(x) >= 1.0, -1.0, 0.5)',
              left=-1.0, right=1.0, dx=0.1, total_time=0.1, dt=0.01,
              bc='dirichlet')


def test_non_negative_diffusion_at_the_wall_still_runs():
    """The guard above must not reject a legitimate problem."""
    run = solve(method='backward euler', drift=0.0, diffusion=0.5,
                left=-1.0, right=1.0, dx=0.1, total_time=0.1, dt=0.01,
                bc='dirichlet')
    assert 0.0 < run.parameters['absorbed_mass_fraction'] < 1.0


def test_recorders_agree_on_which_save_times_are_in_scope():
    """`SolutionRecorder` used to keep targets past the end of the run
    (harmlessly, since they were never reached) while `EnsembleRecorder`
    dropped them. The two now apply the same rule.
    """
    grid, _ = make_grid(-1.0, 1.0, 0.1)
    recorder = SolutionRecorder(grid, [-1.0, 0.5, 2.0, 99.0],
                                total_time=1.0, nsteps=10)

    kept = []
    while recorder._next_target is not None:
        kept.append(recorder._next_target)
        recorder._next_target = next(recorder._remaining_targets, None)

    assert kept == [0.5]


def test_solve_drops_save_times_outside_the_run():
    times = solve(method='backward euler', total_time=1.0, dt=0.01,
                  save_times=[-3.0, 0.5, 7.0]).result.save_times
    assert times.tolist() == [0.0, 0.5, 1.0]


@pytest.mark.parametrize('token, expected', [
    ('\u2212inf', -np.inf),      # U+2212 MINUS SIGN
    ('\u22122.5', -2.5),
    ('\u221e', np.inf),          # U+221E INFINITY
    ('-inf', -np.inf),
    ('-2.5', -2.5),
])
def test_parse_bound_accepts_typeset_minus_and_infinity(token, expected):
    """Copying a bound out of a typeset formula yields U+2212, not the
    ASCII hyphen `float()` expects.
    """
    assert parse_bound(token) == expected


def test_t_range_reported_only_when_a_time_axis_is_used(tmp_path):
    """A file may carry a 't' array that neither coefficient indexes; the
    GUI decides whether to print 'static in t' from `t_range`, so
    reporting a range there labelled static coefficients as
    time-dependent.
    """
    unused = tmp_path / 'unused_t.npz'
    np.savez(unused, x=np.linspace(-1, 1, 5), t=np.linspace(0, 1, 3),
             b=np.zeros(5), D=np.ones(5))
    assert load_coefficient_set(unused)[2]['t_range'] is None

    used = tmp_path / 'used_t.npz'
    np.savez(used, x=np.linspace(-1, 1, 5), t=np.linspace(0, 1, 3),
             b=np.zeros((3, 5)), D=np.ones(5))
    assert load_coefficient_set(used)[2]['t_range'] == (0.0, 1.0)


def test_loaded_coefficients_are_clamped_not_extrapolated():
    """Queries outside the saved x range return the edge value. The
    interpolator itself is now configured to produce NaN rather than
    extrapolate, so a failure of the clamping would be visible instead
    of silently plausible.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as handle:
        np.savez(handle, x=np.linspace(0.0, 1.0, 3), t=np.linspace(0.0, 1.0, 3),
                 b=np.array([[0.0, 1.0, 2.0]] * 3), D=np.ones((3, 3)))
        path = handle.name

    drift, _, _ = load_coefficient_set(path)
    far_outside = drift(np.array([-50.0, 0.5, 50.0]), 0.5)

    assert np.all(np.isfinite(far_outside))
    assert far_outside[0] == pytest.approx(0.0)   # clamped to x = 0
    assert far_outside[2] == pytest.approx(2.0)   # clamped to x = 1


def test_direct_solver_calls_are_guarded_too():
    """The guards live in the solvers, not only in `solve`, so calling an
    integrator directly is protected as well.
    """
    grid, _ = make_grid(-1.0, 1.0, 0.1)
    p0 = build_initial_density('gaussian', grid)
    bc = BoundaryCondition('neumann')

    with pytest.raises(ValueError, match='Total time must be positive'):
        backward_euler(p0, grid, lambda x, t: 0 * x, lambda x, t: 0 * x + 1.0,
                       0.01, 0.0, [0.0], bc)
    with pytest.raises(ValueError, match='Total time must be positive'):
        euler_maruyama(grid, lambda x, t: 0 * x, lambda x, t: 0 * x + 1.0,
                       p0, 10, 0.01, 0.0, [0.0], bc, seed=0)
