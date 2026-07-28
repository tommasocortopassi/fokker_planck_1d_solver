"""CFL and Peclet diagnostics for the explicit (forward Euler) scheme.

Stability of the upwind-advection / centered-diffusion forward-Euler
scheme is governed by two dimensionless numbers, both scaling with dt:

    lambda(x,t) = |b(x,t)| * dt / dx        (advective Courant number)
    mu(x,t)     = D(x,t)   * dt / dx**2      (diffusion number)

and the combined stability condition is

    lambda(x,t) + 2 * mu(x,t) <= 1   for every grid point x and every
                                       time t over the run,

(see docs/numerical_methods_notes.md for the von Neumann derivation).
Note that this must hold *pointwise* - at every (x, t), not just for the
worst |b| and worst D considered separately. Combining the separate
maxima, max|b| * dt/dx + 2 * max(D) * dt/dx**2, is a valid but generally
over-conservative upper bound on the true combined number: it implicitly
assumes the worst drift and the worst diffusion happen at the same point
at the same time, which they need not. `max_combined_cfl` below computes
the *exact* pointwise maximum instead, from the actual (time, space)
field of lambda + 2*mu.

The Peclet number Pe = |b| * dx / (2 * D) is a purely spatial quantity -
the ratio of advective to diffusive transport across one cell - and does
not constrain dt at all; a large Pe just means the (unavoidable, from
upwinding) numerical diffusion is significant relative to the true D on
this grid.

Backward Euler is unconditionally stable and does not need any of these
bounds to run, but they are still reported for insight into the
problem's regime.
"""
import numpy as np

# Fractions of total_time used to probe for time-dependence, deliberately
# *irregularly* spaced (rather than e.g. [0, 1/2, 1]) so they are very
# unlikely to all land on equal values of a genuinely time-dependent but
# periodic coefficient - evenly-spaced probes can be fooled by a period
# that happens to divide total_time evenly (e.g. sin(pi*t/2) over
# total_time=4 repeats at t=0, 2, 4). These particular fractions have no
# simple common period with anything a typical expression would produce.
_PROBE_FRACTIONS = (0.0, 0.173205, 0.414214, 0.618034, 0.836660, 1.0)


def is_time_dependent(coefficient, x, total_time):
    """Cheaply test whether `coefficient(x, t)` actually varies with t,
    by evaluating it at a handful of irregularly-spaced times and
    checking for any difference.

    This is what lets a static coefficient (the common case for one
    loaded from a time-independent .npz array - see coefficients_io.py -
    but also just any autonomous drift/diffusion typed by hand) be
    sampled once instead of `n_time_samples` times. `finite_volume` uses
    it for the same reason: a static pair (b, D) means the discrete
    operator is the same at every step and can be assembled once.

    Not foolproof: a coefficient that happens to return identical values
    at every probed time would still be misclassified as static. The
    irregular spacing above is chosen to make that unlikely for
    genuinely periodic coefficients, but it's a heuristic, not a
    guarantee, and needs no cooperation from the caller.
    """
    probe_times = np.asarray(_PROBE_FRACTIONS) * max(total_time, 1e-12)
    reference = np.asarray(coefficient(x, float(probe_times[0])), dtype=float)
    for t in probe_times[1:]:
        value = np.asarray(coefficient(x, float(t)), dtype=float)
        if not np.array_equal(value, reference):
            return True
    return False


def _sample_field(coefficient, x, total_time, n_time_samples):
    """Evaluate `coefficient(x, t)` over a grid of times, returning a
    (n_times_used, ncells) array - using only ONE time sample (t=0) if
    the coefficient turns out not to depend on t at all, instead of the
    full `n_time_samples`.
    """
    if is_time_dependent(coefficient, x, total_time):
        times = np.linspace(0.0, total_time, max(2, n_time_samples))
    else:
        times = np.array([0.0])
    return np.stack([np.asarray(coefficient(x, float(t)), dtype=float) for t in times])


def sample_diagnostics(grid, drift, diffusion, dt, total_time, n_time_samples=25):
    """Sample |b| and D over the run (each only as many times as it
    actually needs - see `_sample_field`) and derive every reported
    stability/regime number from that single sampling.

    `lambda` and `mu` come from the separately-maximized |b| and D (they
    answer "how restrictive would advection/diffusion be on its own");
    `combined_CFL` (and `stable`, `dt_suggested`) instead use the exact
    pointwise maximum of lambda(x,t) + 2*mu(x,t) - see the module
    docstring for why that's tighter than combining the separate maxima.
    """
    x = grid.centers
    dx = grid.dx

    B = _sample_field(drift, x, total_time, n_time_samples)
    D = _sample_field(diffusion, x, total_time, n_time_samples)
    if np.any(D < 0):
        raise ValueError('Diffusion must remain non-negative.')

    max_abs_b = float(np.max(np.abs(B)))
    max_D = float(np.max(D))

    # Pointwise Peclet number, |b|*dx/(2D); undefined (treated as +inf)
    # wherever D = 0 and b != 0, since that's pure advection locally.
    with np.errstate(divide='ignore', invalid='ignore'):
        pe_field = np.abs(B) * dx / (2.0 * D)
    pe_field = np.where(D > 0.0, pe_field, np.where(np.abs(B) > 0.0, np.inf, 0.0))
    # B and D can have different numbers of time samples (one static, one
    # not); broadcasting the elementwise ops above already lines them up,
    # so no explicit reshaping is needed before taking the max.
    max_pe = float(np.max(pe_field))

    lambda_ = max_abs_b * dt / dx
    mu = max_D * dt / dx ** 2

    # Exact pointwise combined number: same broadcasting trick as above.
    combined_field = np.abs(B) * (dt / dx) + 2.0 * D * (dt / dx ** 2)
    combined_cfl = float(np.max(combined_field))

    # The combined field is exactly linear in dt (both terms scale with
    # it), so the largest dt keeping it at or below 1 is just a rescaling
    # of the dt already used - no need to resample the coefficients again
    # to search for it.
    if dt > 0.0 and combined_cfl > 0.0:
        dt_suggested = dt / combined_cfl
    else:
        dt_suggested = float('inf')

    return {
        'max_abs_b': max_abs_b,
        'max_D': max_D,
        'lambda': lambda_,
        'mu': mu,
        'combined_CFL': combined_cfl,
        'Peclet': max_pe,
        'dt_suggested': dt_suggested,
        'stable': combined_cfl <= 1.0,
    }


def max_combined_cfl(grid, drift, diffusion, dt, total_time, n_time_samples=25):
    """Standalone convenience wrapper around the combined-number
    computation in `sample_diagnostics`, for callers that only want this
    one number without the rest of the diagnostics dictionary.
    """
    return sample_diagnostics(grid, drift, diffusion, dt, total_time,
                               n_time_samples)['combined_CFL']


def check_periodic_consistency(drift, diffusion, left, right, t):
    """Raise if drift/diffusion disagree at the two domain edges.

    A periodic boundary condition only makes sense if the physics is
    itself periodic on [left, right]; otherwise the flux at the shared
    seam is ambiguous.
    """
    x_edges = np.array([left, right])
    b = np.asarray(drift(x_edges, t), dtype=float)
    D = np.asarray(diffusion(x_edges, t), dtype=float)
    if not (np.isclose(b[0], b[1], rtol=1e-10, atol=1e-12)
            and np.isclose(D[0], D[1], rtol=1e-10, atol=1e-12)):
        raise ValueError(
            'Periodic boundary condition requires drift and diffusion to '
            'agree at x=left and x=right.'
        )
