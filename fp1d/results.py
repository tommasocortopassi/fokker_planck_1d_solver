"""Common return type for every solver, so the GUI and plotting code don't
need to special-case forward Euler, backward Euler, or Euler-Maruyama.

Also holds `discrete_mass` and `SolutionRecorder`, the bookkeeping shared
by every solver's time-stepping loop (which times to keep, the running
mass history, and a denser stream of frames for the animation). Factoring
this out means `finite_volume._run` and `stochastic_solver.euler_maruyama`
implement the same recording logic once, instead of each keeping their own
slightly-diverging copy of it.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class SolverResult:
    x: np.ndarray                # cell centers, shape (ncells,)
    save_times: np.ndarray        # requested output times actually reached
    snapshots: np.ndarray         # density at save_times, shape (n_saved, ncells)
    masses: np.ndarray            # total mass at save_times, shape (n_saved,)
    frame_times: np.ndarray       # denser time series used only for animation
    frames: np.ndarray            # density at frame_times, shape (n_frames, ncells)
    stopped_early: bool = False
    final_time: float = 0.0
    # Monte Carlo only: how many trajectories the estimate is built from.
    # None for the PDE solvers, where the notion doesn't apply. This is
    # what tells the caller that a stopped Euler-Maruyama run is short of
    # *samples* rather than short of simulated time.
    trials_completed: int = None


def discrete_mass(p: np.ndarray, grid) -> float:
    """Total probability mass represented by cell-average density `p`.

    Exact for any uniform grid: each cell contributes its average density
    times its width. Used both as a diagnostic (mass should stay 1 for
    periodic/Neumann boundaries, and decay for Dirichlet/absorbing ones)
    and for the mass-history plot.
    """
    return grid.dx * float(np.sum(p))


class SolutionRecorder:
    """Accumulates the outputs every solver needs to return, from a
    sequence of (step index, current time, density) events.

    Usage: `record_initial` once at t=0, `record_step` once per completed
    time step, `record_final` once after the loop ends (unless it was
    stopped early), then `finalize` to build the `SolverResult`.

    `record_step` takes a *callable* `density_fn` rather than the density
    array itself, and only calls it when a frame or a requested save time
    is actually due. This matters for Euler-Maruyama, where "the density"
    means re-histogramming the live particle ensemble - not free to
    compute on every single step.
    """

    def __init__(self, grid, save_times, total_time, nsteps, max_frames=200):
        self.grid = grid
        self.total_time = total_time
        self.saved = {}
        self.masses = {}
        # Save times strictly after t=0; t=0 itself is always recorded by
        # `record_initial` regardless of whether the user asked for it.
        self._remaining_targets = iter(sorted(t for t in save_times if t > 0.0))
        self._next_target = next(self._remaining_targets, None)
        # Record at most `max_frames` animation frames, evenly spaced in
        # step count (not simulated time, which isn't known in advance).
        self.stride = max(1, nsteps // max_frames)
        self.frame_times = []
        self.frames = []

    def record_initial(self, density: np.ndarray):
        self.saved[0.0] = density.copy()
        self.masses[0.0] = discrete_mass(density, self.grid)
        self.frame_times.append(0.0)
        self.frames.append(density.copy())

    def record_step(self, steps_completed: int, current_time: float,
                     is_last_step: bool, density_fn):
        """Call once per completed step. `density_fn()` must return the
        current density as a cell-average array; it is only evaluated if
        this step turns out to be a frame step or to have reached (or
        passed) the next requested save time.
        """
        is_frame_step = (steps_completed % self.stride == 0) or is_last_step
        target_hit = (self._next_target is not None
                      and current_time >= self._next_target - 1e-12)
        if not (is_frame_step or target_hit):
            return

        density = density_fn()

        if is_frame_step:
            self.frame_times.append(current_time)
            self.frames.append(density.copy())

        # A single (large) step can overshoot more than one requested
        # save time at once, so drain every target that's now behind us.
        while (self._next_target is not None
               and current_time >= self._next_target - 1e-12):
            self.saved[self._next_target] = density.copy()
            self.masses[self._next_target] = discrete_mass(density, self.grid)
            self._next_target = next(self._remaining_targets, None)

    def record_final(self, density_fn):
        """Make sure `total_time` itself is always among the saved
        snapshots, even if it wasn't reached exactly via `record_step`
        (e.g. because of the -1e-12 tolerance or floating-point step
        accumulation). No-op if it's already there.
        """
        if self.total_time not in self.masses:
            density = density_fn()
            self.saved[self.total_time] = density.copy()
            self.masses[self.total_time] = discrete_mass(density, self.grid)

    def finalize(self, stopped_early: bool, final_time: float) -> SolverResult:
        times_sorted = sorted(self.saved.keys())
        return SolverResult(
            x=self.grid.centers,
            save_times=np.array(times_sorted),
            snapshots=np.array([self.saved[t] for t in times_sorted]),
            masses=np.array([self.masses[t] for t in times_sorted]),
            frame_times=np.array(self.frame_times),
            frames=np.array(self.frames),
            stopped_early=stopped_early,
            final_time=final_time,
        )


class EnsembleRecorder:
    """Accumulates a Monte Carlo density estimate from batches of complete
    trajectories.

    `SolutionRecorder` above assumes the run advances in time: one state is
    stepped forward, and recorded as it passes each requested output time.
    Euler-Maruyama traverses the problem the other way round - each
    trajectory is simulated from 0 all the way to T before the next one
    starts - so at no point is there a single "current time" to report, and
    the honest unit of progress is completed trials, not completed steps.

    What that ordering buys is an *anytime* estimate. The quantities
    accumulated here are histogram counts summed over trajectories, so
    after any number of completed batches, dividing by the trials completed
    so far already gives a valid density over the whole interval [0, T].
    Stopping early therefore costs accuracy, not time coverage: the shape
    is right everywhere, just noisier, with fluctuations shrinking like
    1/sqrt(N) in the number of trajectories.

    Counts are indexed by *step number* rather than by time, because every
    batch walks the identical step grid; that is what lets batches be
    summed into the same slots.
    """

    def __init__(self, grid, save_times, total_time, nsteps, dt, max_frames=200):
        self.grid = grid
        self.total_time = total_time
        self.dt = dt
        self.nsteps = nsteps
        self.trials_completed = 0

        # Requested output times, clipped to the run and always including
        # both endpoints - the same convention as SolutionRecorder.
        targets = sorted({0.0, float(total_time)} |
                         {float(t) for t in save_times if 0.0 <= t <= total_time})
        self.save_times = np.array(targets)
        save_steps = [self._step_at_or_after(t) for t in targets]

        # Animation frames: evenly spaced in step count, endpoints included.
        stride = max(1, nsteps // max_frames)
        frame_steps = sorted({0, nsteps} |
                             {n for n in range(1, nsteps + 1) if n % stride == 0})
        self.frame_times = np.array([n * dt for n in frame_steps])

        self._save_counts = np.zeros((len(targets), grid.ncells), dtype=np.int64)
        self._frame_counts = np.zeros((len(frame_steps), grid.ncells), dtype=np.int64)

        # step -> (save slots, frame slots) to be filled at that step. A
        # single step can feed several slots at once (two requested output
        # times inside one dt, or an output time that is also a frame).
        self._targets = {}
        for slot, step in enumerate(save_steps):
            self._targets.setdefault(step, ([], []))[0].append(slot)
        for slot, step in enumerate(frame_steps):
            self._targets.setdefault(step, ([], []))[1].append(slot)

    def _step_at_or_after(self, t):
        """Index of the first completed step whose time is >= t."""
        return int(min(self.nsteps, max(0, np.ceil(t / self.dt - 1e-9))))

    def accumulate(self, step, X, alive):
        """Bin the live particles of the current batch, if this step feeds
        any output slot. Cheap no-op otherwise, so the solver can call it
        unconditionally once per step.

        Only live particles are counted, and the normalization later
        divides by the trials *attempted*, not the survivors - which is
        what makes absorbed mass show up as a decaying total, matching the
        PDE's Dirichlet boundary rather than hiding the decay.
        """
        entry = self._targets.get(step)
        if entry is None:
            return
        counts, _ = np.histogram(X[alive], bins=self.grid.faces)
        save_slots, frame_slots = entry
        for slot in save_slots:
            self._save_counts[slot] += counts
        for slot in frame_slots:
            self._frame_counts[slot] += counts

    def finish_batch(self, n_in_batch):
        """Mark a batch of trajectories as fully simulated. Only completed
        batches count: a partially-stepped batch would bias the estimate,
        since its trajectories exist at early times and not at later ones.
        """
        self.trials_completed += int(n_in_batch)

    def finalize(self, stopped_early):
        if self.trials_completed < 1:
            raise ValueError('No trajectories completed; nothing to estimate from.')
        norm = self.trials_completed * self.grid.dx
        snapshots = self._save_counts / norm
        frames = self._frame_counts / norm
        return SolverResult(
            x=self.grid.centers,
            save_times=self.save_times,
            snapshots=snapshots,
            masses=np.array([discrete_mass(p, self.grid) for p in snapshots]),
            frame_times=self.frame_times,
            frames=frames,
            stopped_early=stopped_early,
            # Every completed trajectory ran the full interval, so the
            # result always describes [0, T] - even when stopped early.
            final_time=self.total_time,
            trials_completed=self.trials_completed,
        )
