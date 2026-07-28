"""Forward and backward Euler must agree on the same Dirichlet problem.

This file used to test *nonzero* Dirichlet boundary values, a feature
that no longer exists (see boundary_conditions.py: only the homogeneous
case of each condition is supported, and the nonzero value fields were
removed because neither solver ever honored them). The check worth
keeping from it is the one that made it a useful regression test in the
first place: the boundary contribution must enter the implicit operator
exactly as it enters the explicit one, so the two integrators - which
share `assemble_operator` but use it very differently - must converge to
the same answer on a problem where the boundary is doing real work.
"""
import numpy as np
from fp1d.grid import Grid1D
from fp1d.boundary_conditions import BoundaryCondition
from fp1d.finite_volume import forward_euler, backward_euler
from fp1d.initial_conditions import build_initial_density


def test_forward_and_backward_euler_agree_under_dirichlet():
    def drift(x, t):
        return np.zeros_like(x)

    def diffusion(x, t):
        return 0.3 * np.ones_like(x)

    grid = Grid1D(-3, 3, 150)
    bc = BoundaryCondition('dirichlet')
    # A uniform start puts density right up against both walls, so mass
    # leaves immediately and the boundary terms dominate the answer.
    p0 = build_initial_density('uniform', grid)

    fe = forward_euler(p0, grid, drift, diffusion, dt=0.0002, total_time=0.3,
                       save_times=[0, 0.3], bc=bc)
    be = backward_euler(p0, grid, drift, diffusion, dt=0.005, total_time=0.3,
                        save_times=[0, 0.3], bc=bc)

    assert fe.masses[-1] < 0.99  # the boundary really is absorbing here
    assert abs(fe.masses[-1] - be.masses[-1]) < 5e-4
    assert np.max(np.abs(fe.snapshots[-1] - be.snapshots[-1])) < 5e-3


def test_dirichlet_boundary_flux_uses_interior_diffusivity():
    """With D varying in x, the boundary flux must difference (D p)
    between the wall and the *cell center*, not evaluate D at the wall
    for both. Getting that wrong leaves an O(dx) error in the boundary
    flux that grid refinement then fails to remove cleanly.
    """
    def drift(x, t):
        return np.zeros_like(x)

    def diffusion(x, t):  # strongly varying right where the wall is
        return 0.2 + 0.6 * np.exp(-0.5 * ((x - 2.0) / 1.0) ** 2)

    masses = []
    for ncells in (200, 400, 800):
        grid = Grid1D(-3, 3, ncells)
        p0 = build_initial_density('gaussian', grid)
        result = backward_euler(p0, grid, drift, diffusion, dt=1e-4,
                                total_time=0.5, save_times=[0, 0.5],
                                bc=BoundaryCondition('dirichlet'))
        masses.append(result.masses[-1])

    coarse_gap = abs(masses[1] - masses[0])
    fine_gap = abs(masses[2] - masses[1])
    assert fine_gap < 0.6 * coarse_gap  # converging, at better than 1st order
