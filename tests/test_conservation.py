import numpy as np
from fp1d.grid import Grid1D
from fp1d.boundary_conditions import BoundaryCondition
from fp1d.finite_volume import forward_euler, backward_euler
from fp1d.initial_conditions import build_initial_density

TOL = 1e-10


def test_neumann_conserves_mass_under_pure_advection():
    """Regression test: a reflecting wall must block advective flux too,
    not only diffusive flux (this was previously broken)."""
    def drift(x, t):
        return np.ones_like(x)

    def diffusion(x, t):
        return np.zeros_like(x)

    grid = Grid1D(-3, 3, 120)
    bc = BoundaryCondition('neumann')
    p0 = build_initial_density('gaussian', grid)
    result = forward_euler(p0, grid, drift, diffusion, dt=0.005, total_time=2.0,
                            save_times=[0, 2.0], bc=bc)
    assert abs(result.masses[-1] - 1.0) < TOL


def test_neumann_conserves_mass_advection_diffusion_backward_euler():
    def drift(x, t):
        return -x

    def diffusion(x, t):
        return 0.15 * np.ones_like(x)

    grid = Grid1D(-6, 6, 150)
    bc = BoundaryCondition('neumann')
    p0 = build_initial_density('gaussian', grid)
    result = backward_euler(p0, grid, drift, diffusion, dt=0.01, total_time=1.0,
                             save_times=[0, 1.0], bc=bc)
    assert abs(result.masses[-1] - 1.0) < TOL


def test_periodic_conserves_mass():
    length = 6.0

    def drift(x, t):
        return 0.5 * np.sin(2 * np.pi * (x + 3.0) / length)

    def diffusion(x, t):
        return 0.2 * np.ones_like(x)

    grid = Grid1D(-3, 3, 150)
    bc = BoundaryCondition('periodic')
    p0 = build_initial_density('bimodal', grid)
    result = backward_euler(p0, grid, drift, diffusion, dt=0.01, total_time=1.0,
                             save_times=[0, 1.0], bc=bc)
    assert abs(result.masses[-1] - 1.0) < TOL


def test_dirichlet_lets_mass_leave():
    """Under an absorbing (Dirichlet=0) boundary, mass should strictly
    decrease once the density has any weight near the edges."""
    def drift(x, t):
        return np.zeros_like(x)

    def diffusion(x, t):
        return 0.3 * np.ones_like(x)

    grid = Grid1D(-3, 3, 150)
    bc = BoundaryCondition('dirichlet')
    p0 = build_initial_density('gaussian', grid)
    result = backward_euler(p0, grid, drift, diffusion, dt=0.01, total_time=1.0,
                             save_times=[0, 0.5, 1.0], bc=bc)
    assert result.masses[0] > result.masses[1] > result.masses[2]
