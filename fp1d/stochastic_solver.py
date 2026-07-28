"""Euler-Maruyama solver for the Ito SDE associated with the Fokker-Planck
equation dp/dt = -d/dx(b p) + d^2/dx^2(D p):

    dX_t = b(X_t, t) dt + sqrt(2 D(X_t, t)) dW_t.

Each particle takes the step X_{n+1} = X_n + b(X_n, t_n) dt
+ sqrt(2 D(X_n, t_n) dt) * Z, Z ~ N(0, 1) - the direct SDE analogue of
forward Euler. The density p(x, t) is then recovered as a normalized
histogram of the ensemble at each requested time.

Trajectories are simulated in batches, each run from 0 to T before the next
batch starts, rather than advancing one big ensemble step by step. The two
give the same estimator; this ordering is the one that matches what the
method is, and it means an interrupted run still describes the whole
interval - just from fewer samples. See `euler_maruyama` below.

Note: the drift used here is exactly the user-supplied b(x,t), with no
extra dD/dx correction term. That is only correct because the PDE side
(finite_volume.py) uses the d^2/dx^2(Dp) diffusion convention rather than
d/dx(D dp/dx) - see docs/numerical_methods_notes.md, section 2.4, for why
these two conventions differ whenever D depends on x, and why this project
picks the one that keeps the two solvers consistent with no correction.

Why use this instead of the PDE solver
---------------------------------------
The finite-volume scheme differences b and D between neighboring cells, so
it implicitly assumes they vary smoothly at the grid scale. If b or D are
rough, discontinuous, or only known pointwise/empirically, that differencing
has no clean meaning and the PDE solution can be inaccurate or oscillatory
regardless of dt. Euler-Maruyama only ever *evaluates* b and D at particle
positions - it never differentiates them - so it degrades gracefully (its
usual O(sqrt(dt)) statistical error) even when the coefficients are highly
irregular. The price is Monte Carlo noise instead of a numerically exact
mesh solution.

Boundary conditions for particles (all homogeneous, matching the PDE side)
---------------------------------------------------------------------------
periodic  - positions are wrapped back into [left, right].
neumann   - reflecting wall: a particle that crosses an edge is folded
            back in, matching the PDE's zero-flux condition. Folding uses
            an exact closed-form triangle wave, so it is correct even if a
            single step overshoots the domain by more than its width (a
            large dt relative to the domain, or a large diffusive jump).
dirichlet - absorbing wall: a particle that crosses an edge is removed
            from the live ensemble, matching the PDE's Dirichlet condition,
            under which probability mass is free to leave through the
            boundary and never returns.
"""
import numpy as np

from .boundary_conditions import BoundaryCondition
from .diagnostics import check_periodic_consistency, is_time_dependent
from .initial_conditions import sample_particles_from_density
from .results import EnsembleRecorder


def _reflect(X, left, right):
    """Fold positions back into [left, right] as an elastic (mirror)
    boundary, correctly handling *any* amount of overshoot.

    The reflected trajectory is exactly the "unfold" of a triangle wave of
    period 2*(right-left): shift into [0, period), then fold the second
    half back onto the first. This closed form does the same thing as
    repeatedly reflecting off alternating walls, without needing a loop
    for the (rare, but possible) case of a step large enough to bounce off
    more than one wall.
    """
    width = right - left
    period = 2.0 * width
    u = np.mod(X - left, period)
    return left + np.where(u <= width, u, period - u)


def _apply_boundary(X, alive, left, right, kind):
    """Enforce the boundary rule on particles currently marked alive.

    Returns the (possibly modified) `X` and `alive` arrays. `X` is
    modified in place for particles that are no longer alive too (clipped
    to the domain), purely so that a stray absorbed particle can never
    leak into a later histogram through a bookkeeping mistake elsewhere;
    it plays no role in the physics once `alive` is False.
    """
    active = alive.copy()

    if kind == 'periodic':
        X[active] = left + np.mod(X[active] - left, right - left)

    elif kind == 'neumann':
        X[active] = _reflect(X[active], left, right)

    elif kind == 'dirichlet':
        exited = active & ((X < left) | (X > right))
        X[exited] = np.clip(X[exited], left, right)
        alive[exited] = False

    else:
        raise ValueError(f'Unsupported boundary condition: {kind}')

    return X, alive


def _default_batch_size(n_trials, target_batches=25, min_batch=250):
    """Trajectories per batch.

    Two competing pressures. Larger batches vectorize better, since each
    step is one numpy operation over the whole batch and the per-step
    Python overhead is paid once regardless of width. Smaller batches give
    finer progress reporting and a quicker response to Stop, which is only
    honored between batches. ~25 batches is a reasonable compromise, with a
    floor so that a small run stays one wide batch instead of many thin
    inefficient ones.
    """
    if n_trials <= min_batch:
        return n_trials
    return max(min_batch, int(np.ceil(n_trials / target_batches)))


def euler_maruyama(grid, drift, diffusion, p0, n_trials, dt, total_time,
                    save_times, bc: BoundaryCondition, seed=None,
                    progress_callback=None, stop_event=None, max_frames=200,
                    batch_size=None):
    """Simulate n_trials trajectories and reconstruct the density by binning.

    p0: initial cell-average density (same array used by the PDE solver),
    from which the initial ensemble is sampled.

    Trajectories are simulated in batches, each batch carried from t=0 all
    the way to `total_time` before the next one starts. This mirrors what
    the method actually is - an average over independent realizations of
    the SDE, not a state marched forward in time - and has two consequences
    the caller can see:

    - progress is reported as *completed trials*, so `progress_callback`
      receives (trials_done, n_trials, None); the third argument is the
      simulated time for the PDE solvers and is None here, because at any
      instant this solver holds trajectories spanning the whole interval
      rather than one shared current time.

    - stopping early yields a usable answer over all of [0, T] rather than
      a run truncated in time. The estimate is built from however many
      trajectories finished, so it has the right shape everywhere and
      simply larger Monte Carlo fluctuations (~1/sqrt(N)). At least one
      batch always completes, since no estimate can be formed from zero
      samples.

    Note that for a fixed `seed`, changing `batch_size` changes the
    results: the random draws are consumed batch by batch. Reproducibility
    therefore requires fixing both.
    """
    bc.validate()
    if n_trials < 1:
        raise ValueError('Number of trials must be at least 1.')

    rng = np.random.default_rng(seed)
    left, right = grid.left, grid.right
    nsteps = int(np.ceil(total_time / dt))
    dt = total_time / nsteps  # spread any leftover evenly over all steps

    # A periodic domain requires b and D to match at the two edges. If
    # they don't depend on t, checking that once settles it for the whole
    # run; only genuinely time-dependent coefficients need re-checking
    # every step, and then only during the first batch - every later batch
    # walks the identical time grid.
    recheck_periodic_each_step = False
    if bc.kind == 'periodic':
        check_periodic_consistency(drift, diffusion, left, right, 0.0)
        recheck_periodic_each_step = (
            is_time_dependent(drift, grid.centers, total_time)
            or is_time_dependent(diffusion, grid.centers, total_time)
        )

    recorder = EnsembleRecorder(grid, save_times, total_time, nsteps, dt, max_frames)
    if batch_size is None:
        batch_size = _default_batch_size(n_trials)

    stopped_early = False
    trials_done = 0
    while trials_done < n_trials:
        n_batch = int(min(batch_size, n_trials - trials_done))
        X = sample_particles_from_density(grid, p0, n_batch, rng)
        alive = np.ones(n_batch, dtype=bool)
        recorder.accumulate(0, X, alive)

        for n in range(nsteps):
            t_n = n * dt
            if recheck_periodic_each_step and trials_done == 0:
                check_periodic_consistency(drift, diffusion, left, right, t_n)

            idx = np.where(alive)[0]
            b = np.asarray(drift(X[idx], t_n), dtype=float)
            D = np.asarray(diffusion(X[idx], t_n), dtype=float)
            if np.any(D < 0.0):
                raise ValueError('Diffusion must stay non-negative.')
            Z = rng.standard_normal(idx.size)
            X[idx] = X[idx] + b * dt + np.sqrt(2.0 * D * dt) * Z
            X, alive = _apply_boundary(X, alive, left, right, bc.kind)

            recorder.accumulate(n + 1, X, alive)

        recorder.finish_batch(n_batch)
        trials_done += n_batch

        if progress_callback:
            progress_callback(trials_done, n_trials, None)

        # Checked only between batches: a half-stepped batch holds
        # trajectories that exist at early times but not at later ones, and
        # folding those into the counts would bias the estimate toward the
        # start of the interval.
        if (stop_event is not None and stop_event.is_set()
                and trials_done < n_trials):
            stopped_early = True
            break

    return recorder.finalize(stopped_early)
