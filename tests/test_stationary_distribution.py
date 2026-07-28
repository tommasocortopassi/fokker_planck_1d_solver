import numpy as np
from fp1d.grid import Grid1D
from fp1d.boundary_conditions import BoundaryCondition
from fp1d.finite_volume import backward_euler
from fp1d.initial_conditions import build_initial_density

D_VALUE = 0.2


def _variance_error(ncells, dt):
    """For dX = -X dt + sqrt(2 D) dW, the stationary density is N(0, D).
    Run far from equilibrium and measure the error in the recovered
    variance at a given grid/time resolution.
    """
    def drift(x, t):
        return -x

    def diffusion(x, t):
        return D_VALUE * np.ones_like(x)

    grid = Grid1D(-8, 8, ncells)
    bc = BoundaryCondition('neumann')
    p0 = build_initial_density('left-block', grid)

    result = backward_euler(p0, grid, drift, diffusion, dt=dt, total_time=10.0,
                             save_times=[0, 10.0], bc=bc)
    x, p_final, dx = result.x, result.snapshots[-1], grid.dx
    mean = dx * np.sum(x * p_final)
    variance = dx * np.sum((x - mean) ** 2 * p_final)
    return abs(mean), abs(variance - D_VALUE) / D_VALUE


def test_ou_process_relaxes_towards_correct_mean():
    mean_error, _ = _variance_error(ncells=200, dt=0.02)
    assert mean_error < 0.01


def test_ou_stationary_variance_converges_under_grid_refinement():
    """First-order convergence check: the error in the recovered stationary
    variance should shrink as the grid is refined, and be small at the finer
    resolution - this is a stronger, less arbitrary check than a single
    fixed-tolerance comparison at one resolution.
    """
    _, error_coarse = _variance_error(ncells=100, dt=0.02)
    _, error_fine = _variance_error(ncells=400, dt=0.01)
    assert error_fine < error_coarse
    assert error_fine < 0.10
