import numpy as np
from fp1d.grid import Grid1D
from fp1d.boundary_conditions import BoundaryCondition
from fp1d.finite_volume import backward_euler
from fp1d.stochastic_solver import euler_maruyama
from fp1d.initial_conditions import build_initial_density


def test_pde_and_sde_agree_on_absorbing_mass_decay():
    """The PDE and particle solvers share the same initial density and
    boundary semantics, so their mass-loss curves under an absorbing
    (Dirichlet=0) boundary should agree within Monte Carlo noise.
    """
    def drift(x, t):
        return np.zeros_like(x)

    def diffusion(x, t):
        return 0.3 * np.ones_like(x)

    grid = Grid1D(-3, 3, 150)
    bc = BoundaryCondition('dirichlet')
    p0 = build_initial_density('gaussian', grid)

    pde = backward_euler(p0, grid, drift, diffusion, dt=0.005, total_time=1.0,
                          save_times=[0, 1.0], bc=bc)
    sde = euler_maruyama(grid, drift, diffusion, p0, n_trials=200_000, dt=0.002,
                          total_time=1.0, save_times=[0, 1.0], bc=bc, seed=7)

    assert abs(pde.masses[-1] - sde.masses[-1]) < 0.01


def test_pde_and_sde_agree_with_spatially_varying_diffusion():
    """With D depending on x, the two solvers only agree if both use the
    same diffusion convention (d^2/dx^2(Dp), see numerical_methods_notes.md
    section 2.4) - the plain SDE drift b needs no dD/dx correction term
    under that convention. This is the case that would expose a mismatch
    if that convention were ever inconsistent between the two solvers.
    """
    def drift(x, t):
        return np.zeros_like(x)

    def diffusion(x, t):
        return 0.05 + 0.15 * np.exp(-0.5 * (x / 1.5) ** 2)  # peaked at x=0

    grid = Grid1D(-6, 6, 150)
    bc = BoundaryCondition('neumann')
    p0 = build_initial_density('gaussian', grid)

    pde = backward_euler(p0, grid, drift, diffusion, dt=0.01, total_time=2.0,
                          save_times=[0, 2.0], bc=bc)
    sde = euler_maruyama(grid, drift, diffusion, p0, n_trials=300_000, dt=0.005,
                          total_time=2.0, save_times=[0, 2.0], bc=bc, seed=3)

    x = grid.centers
    pde_var = grid.dx * np.sum(x ** 2 * pde.snapshots[-1])
    sde_var = grid.dx * np.sum(x ** 2 * sde.snapshots[-1])
    assert abs(pde_var - sde_var) / pde_var < 0.05
