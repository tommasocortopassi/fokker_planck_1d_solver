"""The adaptive-domain verdict must not depend on the requested save times.

`_boundary_leakage` used to scan `result.snapshots`, i.e. only the output
times the user typed into the GUI. A density that crossed the truncated
boundary *between* two widely-spaced save times therefore left no trace,
and `solve_with_adaptive_domain` accepted a domain through which nearly
all the mass had leaked. It now scans `result.frames`, which are recorded
on a fixed step stride regardless of the save times.
"""
import numpy as np
from fp1d.boundary_conditions import BoundaryCondition
from fp1d.domain_truncation import solve_with_adaptive_domain
from fp1d.finite_volume import backward_euler
from fp1d.initial_conditions import build_initial_density


def drift(x, t):
    return 5.0 * np.ones_like(x)


def diffusion(x, t):
    return 0.01 * np.ones_like(x)


def _solve_with(save_times):
    """Strong drift to the right, absorbing walls, and a deliberately
    far-too-small starting domain: the cloud must escape unless the
    domain is grown to follow it out to x ~ 10.
    """
    def solve_fn(grid):
        p0 = build_initial_density('gaussian', grid)
        return backward_euler(p0, grid, drift, diffusion, 0.002, 2.0,
                              save_times, BoundaryCondition('dirichlet'))

    return solve_with_adaptive_domain(solve_fn, -1.0, 1.0, 0.02,
                                      left_is_infinite=True,
                                      right_is_infinite=True)


def test_truncation_verdict_is_independent_of_save_times():
    sparse_times = [0, 2.0]
    dense_times = [0, 0.5, 1.0, 1.5, 2.0]

    _, sparse_result, _, sparse_right, _ = _solve_with(sparse_times)
    _, dense_result, _, dense_right, _ = _solve_with(dense_times)

    # Asking for fewer output times must not buy a smaller domain.
    assert sparse_right >= 0.9 * dense_right
    # And in neither case may the run quietly absorb everything.
    assert sparse_result.masses[-1] > 0.5
    assert dense_result.masses[-1] > 0.5


def test_enlargement_works_for_an_off_center_initial_condition():
    """Growth is measured from the domain width, not from x=0, so an
    initial condition sitting far from the origin still gets a domain
    that follows it rather than one that collapses toward zero.
    """
    def solve_fn(grid):
        p0 = build_initial_density('custom', grid,
                                   custom_expr='np.exp(-0.5*((x-8.0)/0.5)**2)')
        return backward_euler(p0, grid, drift, diffusion, 0.002, 1.0,
                              [0, 0.5, 1.0], BoundaryCondition('neumann'))

    _, result, left, right, _ = solve_with_adaptive_domain(
        solve_fn, 6.0, 10.0, 0.05,
        left_is_infinite=False, right_is_infinite=True)

    assert right > 12.0          # followed the drift out to x ~ 13
    assert left == 6.0           # the finite side was left alone
    assert abs(result.masses[-1] - 1.0) < 1e-9
