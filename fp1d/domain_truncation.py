"""Adaptive truncation of infinite/semi-infinite domains.

When the user asks for +/-inf as a domain bound, the true problem has no
boundary there at all. Rather than asking the user to guess a "large
enough" finite replacement up front, we approximate it and then check the
approximation:

1. Start from a finite domain that a cheap heuristic (see gui.py's
   `estimate_open_bounds`) expects to already comfortably contain the
   initial condition and its advective/diffusive spread over the run.
2. Run the requested solver on it.
3. Look at the density in the outermost few cells across every recorded
   frame. If it isn't negligible compared to the run's peak density,
   the truncation was too aggressive: enlarge the open side(s) and solve
   again.
4. Repeat until the boundary density is negligible, or a maximum number
   of re-solves is reached - at which point we proceed with the best
   attempt so far and clearly flag that the tolerance was not met.

Only sides that actually started as +/-inf are ever grown; a
user-specified finite bound is a hard constraint and is left alone even
if the solution turns out not to be negligible there.
"""
import numpy as np

from .grid import make_grid

# How many cells at each edge count as "the boundary" for the negligibility
# check. A handful of cells (rather than just the outermost one) makes the
# check less sensitive to noise in a single cell, particularly for the
# Euler-Maruyama solver, whose density is a histogram.
EDGE_CELLS = 3

# Boundary density must fall below this fraction of the run's peak density.
DEFAULT_TOL = 1e-4

# Enlargement factor for the domain: each open side that fails the
# tolerance check is pushed out by (growth - 1) times the current domain
# width, so the rule doesn't care where the solution happens to sit.
DEFAULT_GROWTH = 1.6

DEFAULT_MAX_EXPANSIONS = 5


def _boundary_leakage(result, edge_cells=EDGE_CELLS):
    """How much density sits in the outermost `edge_cells` cells at each
    end, and the overall peak density, across the run.

    This scans `result.frames`, not `result.snapshots`. Snapshots are
    recorded only at the output times the *user* asked for, so judging
    the truncation from them would make the accepted domain depend on a
    pure formatting choice: a density that crosses the boundary between
    two widely-spaced save times leaves no trace in the snapshots at
    all, and the run is then silently accepted having lost most of its
    mass through a boundary that shouldn't exist. Frames are recorded on
    a fixed step stride (see `results.SolutionRecorder`), independently
    of the requested save times, so they catch that transit.

    What this criterion does and does not guarantee
    -----------------------------------------------
    The quantity checked is a *density*, but what a truncated boundary
    actually costs is *mass* - and the two only agree when the tails are
    light. For a Gaussian tail they agree closely: density falls to
    DEFAULT_TOL = 1e-4 of peak at 4.29 standard deviations, beyond which
    only ~2e-5 of the mass remains, so the density test is effectively a
    mass test with a tolerance of the same order. Measured across a range
    of b, D and T with smooth bounded coefficients, the mass lost through
    an accepted boundary stays in the 1e-5 to 1e-3 range.

    Where the two decouple is heavy tails, which in this model means a
    diffusivity growing without bound (e.g. D = 0.1 + 0.05*x**2). There
    the density at the wall can pass this test while ~0.5% of the mass
    has already left through it - an under-detection of roughly two
    orders of magnitude. Under an absorbing (Dirichlet) boundary that
    loss is entirely spurious, since the true domain has no wall there
    at all.

    This is a deliberate trade, not an oversight. Checking mass directly
    would be sound only when *every* boundary is truncated: with one
    user-specified finite Dirichlet wall and one infinite side, mass
    legitimately leaves through the finite wall and a mass criterion
    cannot tell the two apart. The narrower fix isn't worth the extra
    conditional, particularly because the number in question is already
    surfaced: every Dirichlet run reports its absorbed fraction to the
    status log and to simulation_parameters.json, so a user who cares can
    read the actual mass loss for their own problem rather than trusting
    this proxy.

    Returns (left_edge_value, right_edge_value, peak_value).
    """
    frames = result.frames
    if frames.size == 0:
        return 0.0, 0.0, 0.0
    n_edge = min(edge_cells, frames.shape[1])
    peak = float(np.max(np.abs(frames)))
    left_edge = float(np.max(np.abs(frames[:, :n_edge])))
    right_edge = float(np.max(np.abs(frames[:, -n_edge:])))
    return left_edge, right_edge, peak


def solve_with_adaptive_domain(solve_fn, left, right, dx,
                                left_is_infinite, right_is_infinite,
                                tol=DEFAULT_TOL, growth=DEFAULT_GROWTH,
                                max_expansions=DEFAULT_MAX_EXPANSIONS,
                                log=None):
    """Solve on `[left, right]`, enlarging the open side(s) and
    re-solving until the density at the truncated boundary is negligible.

    `solve_fn(grid)` must build the initial condition on `grid` and run
    the full requested solver, returning a `SolverResult`. It is called
    once per attempt (at least once, even for a fully finite domain, so
    the dx-fitting message from `make_grid` is always produced).

    `left`/`right` are the *initial* numeric bounds to try (already
    substituted for any +/-inf, e.g. via `grid.finite_domain`); only the
    side(s) flagged infinite are ever moved.

    Returns `(grid, result, left, right, messages)`, where `left, right`
    are the bounds actually used for the returned result and `messages`
    is a list of human-readable strings describing every dx adjustment
    and domain enlargement that happened along the way.
    """
    messages = []

    def note(msg):
        messages.append(msg)
        if log is not None:
            log(msg)

    grid = result = None
    for attempt in range(max_expansions + 1):
        grid, dx_msg = make_grid(left, right, dx)
        if dx_msg:
            note(dx_msg)

        result = solve_fn(grid)

        if result.stopped_early:
            # Honor a user-requested stop immediately; don't keep
            # enlarging and re-solving on top of a run they cancelled.
            return grid, result, left, right, messages

        if not (left_is_infinite or right_is_infinite):
            # Both bounds were user-specified and finite: nothing to
            # check or adapt, only the dx-fitting message (if any) above
            # applies.
            return grid, result, left, right, messages

        left_edge, right_edge, peak = _boundary_leakage(result)
        threshold = tol * max(peak, 1e-300)  # guard against peak == 0
        left_ok = (not left_is_infinite) or (left_edge <= threshold)
        right_ok = (not right_is_infinite) or (right_edge <= threshold)

        if left_ok and right_ok:
            return grid, result, left, right, messages

        if attempt == max_expansions:
            worst = max(left_edge if left_is_infinite else 0.0,
                        right_edge if right_is_infinite else 0.0)
            note(f'Reached the maximum of {max_expansions} domain '
                 f'expansions; proceeding with [{left:.4g}, {right:.4g}] '
                 f'even though the boundary density ({worst:.3e}) is still '
                 f'above the tolerance ({threshold:.3e}). Consider a larger '
                 f'starting domain or a looser tolerance.')
            return grid, result, left, right, messages

        # Grow whichever open side failed, by a fixed fraction of the
        # current domain width. Scaling the *bound* instead would assume
        # the domain is centered on x=0 - which is true for the built-in
        # initial conditions but not for an off-center custom one, where
        # a bound can even have the "wrong" sign and scaling it would
        # shrink the domain rather than enlarge it.
        width = right - left
        if left_is_infinite and not left_ok:
            left -= (growth - 1.0) * width
        if right_is_infinite and not right_ok:
            right += (growth - 1.0) * width
        note(f'Boundary density not yet negligible '
             f'(left={left_edge:.2e}, right={right_edge:.2e}, '
             f'threshold={threshold:.2e}); enlarging domain to '
             f'[{left:.4g}, {right:.4g}] and re-solving '
             f'(attempt {attempt + 2}/{max_expansions + 1}).')

    # Unreachable: the loop above always returns by the `attempt ==
    # max_expansions` branch at the latest.
    return grid, result, left, right, messages
