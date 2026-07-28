import threading
import numpy as np
from fp1d.grid import Grid1D
from fp1d.boundary_conditions import BoundaryCondition
from fp1d.finite_volume import forward_euler, backward_euler
from fp1d.stochastic_solver import euler_maruyama
from fp1d.initial_conditions import build_initial_density


def drift(x, t):
    return -x


def diffusion(x, t):
    return 0.15 * np.ones_like(x)


def _already_set_event():
    ev = threading.Event()
    ev.set()
    return ev


def test_forward_euler_stops_immediately():
    grid = Grid1D(-5, 5, 100)
    bc = BoundaryCondition('neumann')
    p0 = build_initial_density('gaussian', grid)
    result = forward_euler(p0, grid, drift, diffusion, dt=0.001, total_time=1.0,
                            save_times=[0, 1.0], bc=bc, stop_event=_already_set_event())
    assert result.stopped_early
    assert result.final_time == 0.0


def test_backward_euler_stops_immediately():
    grid = Grid1D(-5, 5, 100)
    bc = BoundaryCondition('neumann')
    p0 = build_initial_density('gaussian', grid)
    result = backward_euler(p0, grid, drift, diffusion, dt=0.01, total_time=1.0,
                             save_times=[0, 1.0], bc=bc, stop_event=_already_set_event())
    assert result.stopped_early
    assert result.final_time == 0.0


def test_euler_maruyama_stops_after_a_batch_not_at_a_time():
    """Stopping the particle solver costs *samples*, not simulated time.

    Every trajectory it has finished ran the whole interval, so an
    interrupted run still describes all of [0, T] - only from fewer
    realizations, hence noisier. That is the opposite of the PDE solvers
    above, where stopping truncates the time axis.
    """
    grid = Grid1D(-5, 5, 100)
    bc = BoundaryCondition('neumann')
    p0 = build_initial_density('gaussian', grid)
    n_trials = 5000
    result = euler_maruyama(grid, drift, diffusion, p0, n_trials=n_trials, dt=0.01,
                             total_time=1.0, save_times=[0, 0.5, 1.0], bc=bc,
                             seed=0, stop_event=_already_set_event())

    assert result.stopped_early
    # No estimate can be formed from zero samples, so one batch always runs
    # - but nowhere near all of them.
    assert 0 < result.trials_completed < n_trials
    # Full time coverage: the last output time is still T, not the moment
    # the stop happened.
    assert result.final_time == 1.0
    assert result.save_times[-1] == 1.0
    assert np.allclose(result.save_times, [0.0, 0.5, 1.0])
    # And it is a genuine density at every output time, not a partly-filled
    # array: mass is conserved under reflection whatever the sample count.
    for mass in result.masses:
        assert abs(mass - 1.0) < 1e-12


def test_euler_maruyama_stopped_run_matches_a_short_full_run():
    """The stopped estimate is not merely non-empty - it is the same
    estimator, just with a smaller N. Running the full solver with exactly
    the number of trials that the interrupted one completed must reproduce
    it bit for bit, given the same seed and batch size.
    """
    grid = Grid1D(-5, 5, 100)
    bc = BoundaryCondition('neumann')
    p0 = build_initial_density('gaussian', grid)

    stopped = euler_maruyama(grid, drift, diffusion, p0, n_trials=4000, dt=0.02,
                              total_time=1.0, save_times=[0, 1.0], bc=bc, seed=7,
                              batch_size=500, stop_event=_already_set_event())
    reference = euler_maruyama(grid, drift, diffusion, p0,
                                n_trials=stopped.trials_completed, dt=0.02,
                                total_time=1.0, save_times=[0, 1.0], bc=bc, seed=7,
                                batch_size=500)

    assert stopped.trials_completed == reference.trials_completed
    assert np.array_equal(stopped.snapshots, reference.snapshots)
