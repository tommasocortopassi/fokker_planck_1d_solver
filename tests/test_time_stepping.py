"""The time grid and the state advance must use the same dt.

`finite_volume._run` renormalizes the requested dt to
`total_time / ceil(total_time/dt)` so the steps tile [0, T] exactly. That
renormalized value has to be the one the step actually advances by. It
previously wasn't: `forward_euler`/`backward_euler` closed over the
caller's original dt, so a dt that didn't evenly divide T made the solver
integrate to `ceil(T/dt)*dt` while reporting the result as being at T.

Pure advection is the sharpest probe: upwind finite volumes transport the
first moment at exactly the drift velocity, so the mean displacement of
the density after time T must be b*T no matter how coarse the grid is,
and any mismatch is purely a time-integration error.
"""
import numpy as np
import pytest
from fp1d.grid import Grid1D
from fp1d.boundary_conditions import BoundaryCondition
from fp1d.finite_volume import forward_euler, backward_euler
from fp1d.stochastic_solver import euler_maruyama
from fp1d.initial_conditions import build_initial_density

SPEED = 1.0
TOTAL_TIME = 1.0


def drift(x, t):
    return SPEED * np.ones_like(x)


def zero_diffusion(x, t):
    return np.zeros_like(x)


def _mean_displacement(result, grid):
    return grid.dx * np.sum(grid.centers * result.snapshots[-1])


# 0.25 divides T exactly; the other two do not, and are where the bug
# used to show up (0.3 gave 1.2 and 0.7 gave 1.4 instead of 1.0).
@pytest.mark.parametrize('dt', [0.25, 0.3, 0.7])
def test_forward_euler_stops_at_total_time_for_any_dt(dt):
    grid = Grid1D(-10, 10, 400)
    p0 = build_initial_density('gaussian', grid)
    result = forward_euler(p0, grid, drift, zero_diffusion, dt, TOTAL_TIME,
                           save_times=[0, TOTAL_TIME],
                           bc=BoundaryCondition('neumann'))
    assert abs(_mean_displacement(result, grid) - SPEED * TOTAL_TIME) < 1e-9


@pytest.mark.parametrize('dt', [0.25, 0.3, 0.7])
def test_backward_euler_stops_at_total_time_for_any_dt(dt):
    grid = Grid1D(-10, 10, 400)
    p0 = build_initial_density('gaussian', grid)
    result = backward_euler(p0, grid, drift, zero_diffusion, dt, TOTAL_TIME,
                            save_times=[0, TOTAL_TIME],
                            bc=BoundaryCondition('neumann'))
    # Looser than the explicit case: backward Euler couples the whole
    # domain in a single solve, so a trace of mass (O(1e-6) here) reaches
    # the reflecting wall and perturbs the first moment. That is real
    # implicit smearing, not a time-integration error - and it is six
    # orders of magnitude away from the 0.2-0.4 discrepancy this test
    # exists to catch.
    assert abs(_mean_displacement(result, grid) - SPEED * TOTAL_TIME) < 1e-5


def test_euler_maruyama_stops_at_total_time_for_indivisible_dt():
    """The particle solver renormalizes dt in its own scope and always
    used it correctly; this pins that down alongside the PDE solvers.
    """
    grid = Grid1D(-10, 10, 400)
    p0 = build_initial_density('gaussian', grid)
    result = euler_maruyama(grid, drift, zero_diffusion, p0, n_trials=20_000,
                            dt=0.3, total_time=TOTAL_TIME,
                            save_times=[0, TOTAL_TIME],
                            bc=BoundaryCondition('neumann'), seed=1)
    # Binning the ensemble costs a little accuracy that the PDE side
    # doesn't pay, hence the looser tolerance; the point is that it lands
    # at 1.0 rather than at 1.2.
    assert abs(_mean_displacement(result, grid) - SPEED * TOTAL_TIME) < 0.02
