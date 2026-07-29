"""One-call programmatic entry point: `solve(...)`.

`gui.py` and this module are the two front ends to the same pipeline, and
they take the same inputs. The GUI reads them from Tkinter widgets; this
module reads them from function arguments, so the package can be driven
from a script, a notebook, or a parameter sweep - anywhere a window
cannot be opened.

The pipeline itself is unchanged:

    coefficients -> adaptive domain truncation -> grid + initial condition
    -> forward_euler / backward_euler / euler_maruyama -> SolverResult

Everything the GUI does *beyond* that (progress bars, live truncation
preview, warning dialogs) is presentation, not computation, and stays in
`gui.py`.

Typical use:

    from fp1d.api import solve

    run = solve(method='backward euler',
                drift='-x', diffusion='0.5',
                left=-6, right=6, dx=0.02,
                total_time=2.0, dt=0.001,
                bc='neumann', initial_condition='gaussian')

    p_final = run.result.snapshots[-1]      # density at t = T
    x = run.result.x                        # cell centers
"""
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .boundary_conditions import BoundaryCondition
from .diagnostics import preflight_warnings
from .domain_truncation import solve_with_adaptive_domain
from .grid import make_grid
from .finite_volume import backward_euler, forward_euler
from .grid import parse_bound
from .initial_conditions import build_initial_density, safe_eval
from .results import SolverResult
from .stochastic_solver import euler_maruyama

# Accepted spellings for each method, so callers can write whichever of
# 'Backward Euler', 'backward euler' or 'be' they find natural - the GUI
# passes its own dropdown labels through unchanged.
METHOD_ALIASES = {
    'forward euler': 'forward euler', 'forward-euler': 'forward euler',
    'fe': 'forward euler', 'explicit': 'forward euler',
    'backward euler': 'backward euler', 'backward-euler': 'backward euler',
    'be': 'backward euler', 'implicit': 'backward euler',
    'euler-maruyama': 'euler-maruyama', 'euler maruyama': 'euler-maruyama',
    'em': 'euler-maruyama', 'sde': 'euler-maruyama', 'particles': 'euler-maruyama',
}


class RunAborted(RuntimeError):
    """Raised when the user declines to continue after a warning.

    A distinct type, rather than `ValueError`, because nothing was
    invalid: the run was well-formed and the user chose not to start it.
    """


def _resolve_warnings(messages, on_warning, log):
    """Apply the `on_warning` policy to the preflight messages.

    Modes:

    'warn'   - emit each message as a `UserWarning` and continue. The
               default, because it is the only mode that behaves
               sensibly in every context this package runs in: scripts,
               notebooks, test suites and worker threads.
    'ask'    - print the messages and ask the user to confirm, aborting
               with `RunAborted` on a negative answer. Falls back to
               'warn' when stdin is not a terminal - see below.
    'raise'  - turn the first message into a `RunAborted`. Useful in a
               parameter sweep, where an unstable point should be
               recorded rather than silently produce garbage.
    'ignore' - proceed silently. What the GUI passes, having already
               shown its own dialog.

    The fallback in 'ask' is not defensive padding. `input()` with no
    terminal attached blocks forever with no visible prompt: under
    pytest it hangs the suite, and in the GUI - where `solve` runs on a
    worker thread while Tk owns the main thread - it would freeze the
    window on a question the user cannot see.
    """
    if on_warning not in ('warn', 'ask', 'raise', 'ignore'):
        raise ValueError(
            f"on_warning must be one of 'warn', 'ask', 'raise', 'ignore'; "
            f"got {on_warning!r}.")

    for message in messages:
        log(message)

    if not messages or on_warning == 'ignore':
        return

    if on_warning == 'raise':
        raise RunAborted(messages[0])

    interactive = on_warning == 'ask' and sys.stdin is not None
    if interactive:
        try:
            interactive = sys.stdin.isatty()
        except (ValueError, AttributeError):
            interactive = False

    if not interactive:
        for message in messages:
            warnings.warn(message, UserWarning, stacklevel=3)
        return

    print()
    for message in messages:
        print(f'WARNING: {message}')
    while True:
        answer = input('Continue anyway? [y/n] ').strip().lower()
        if answer in ('y', 'yes'):
            return
        if answer in ('n', 'no'):
            raise RunAborted('Run cancelled by the user after a warning.')
        print("Please answer 'y' or 'n'.")


def _as_coefficient(spec, name):
    """Turn `spec` into a `(x, t) -> array` callable.

    Accepts three forms, all of which the GUI also accepts in one field
    or another:

    - a callable `f(x, t)`, used as-is (this is what a .npz coefficient
      set loaded through `coefficients_io` already provides);
    - a string, evaluated as a Python expression in `x` and `t` under the
      same restrictions as the GUI's drift/diffusivity fields - numpy as
      `np` and a small whitelist of builtins, so no imports and no
      filesystem access. `name` ('drift' or 'diffusion') is what any
      error message will call the offending field;
    - a number, broadcast to a constant field.

    Whatever the input, the result is broadcast to the shape of `x`, so
    downstream code never has to distinguish a scalar coefficient from a
    varying one.
    """
    if callable(spec):
        return spec

    if isinstance(spec, str):
        expr = spec

        def from_expression(x, t=0.0):
            value = safe_eval(expr, {'x': x, 't': t}, name)
            arr = np.asarray(value, dtype=float)
            if arr.shape == ():
                arr = np.full_like(np.asarray(x, dtype=float), float(arr))
            return arr

        return from_expression

    constant = float(spec)

    def from_constant(x, t=0.0):
        return np.full_like(np.asarray(x, dtype=float), constant)

    return from_constant


def _describe(spec):
    """Short human-readable form of a coefficient, for the saved JSON."""
    if isinstance(spec, str):
        return spec
    if callable(spec):
        return getattr(spec, '__name__', 'callable')
    return repr(spec)


def _open_bound_guess(left, right, total_time, drift, diffusion,
                       initial_condition, custom_expr):
    """First guess for a bound the caller left infinite.

    Deliberately crude: sample the coefficients on a provisional grid and
    allow for advective travel `|b| T` plus several diffusive standard
    deviations `sqrt(2 D T)` beyond the initial condition's support. It
    does not need to be right, only non-tiny - `solve_with_adaptive_domain`
    measures the density that actually reaches the artificial wall and
    enlarges the domain until the leakage is negligible. Erring small
    costs one extra solve; erring large costs cells.
    """
    finite_left = left if np.isfinite(left) else -5.0
    finite_right = right if np.isfinite(right) else 5.0
    probe = np.linspace(finite_left, finite_right, 201)

    b_max = float(np.max(np.abs(drift(probe, 0.0))))
    D_max = float(np.max(np.abs(diffusion(probe, 0.0))))
    reach = b_max * total_time + 4.0 * np.sqrt(2.0 * D_max * total_time)
    # The presets are all supported within a couple of units of the
    # origin; a custom expression could be anything, so leave more room.
    is_custom = (initial_condition == 'custom') and bool(custom_expr)
    support = 5.0 if is_custom else 3.0
    span = max(reach + support, 1.0)

    return (left if np.isfinite(left) else -span,
            right if np.isfinite(right) else span)


@dataclass
class Run:
    """What one `solve(...)` call produced.

    `result` is the same `SolverResult` every integrator returns, so
    anything written against the solvers directly also works here.
    `grid` is the grid actually used, which may be wider than requested
    if a bound was infinite. `messages` collects the notes the GUI would
    have printed to its log (dx or dt renormalized, domain enlarged).
    """
    result: SolverResult
    grid: object
    messages: list = field(default_factory=list)
    parameters: dict = field(default_factory=dict)

    def save(self, directory='output', log=None):
        """Write the same five artifacts a GUI run writes: snapshot plot,
        mass-history plot, raw `.npz`, parameters `.json`, animation.

        `log`, if given, is called with a short note before each step -
        the animation in particular can take a while, and a caller with
        a progress display needs to say so.

        Returns a dict of the paths written, keyed by 'directory',
        'snapshots', 'mass_history', 'solution', 'parameters' and
        'animation'. The animation key is whichever format
        `save_animation` managed to produce (.mp4 when ffmpeg is
        available, .gif otherwise), which is why it is reported back
        rather than assumed.
        """
        from .io_utils import create_run_directory, save_json
        from .visualization import (plot_mass_history, plot_snapshots,
                                    save_animation, save_solution_npz)

        def note(message):
            if log is not None:
                log(message)

        run_dir = create_run_directory(directory)
        r = self.result

        note('Saving snapshot plot...')
        plot_snapshots(r.x, r.save_times, r.snapshots, run_dir / 'snapshots.png')
        note('Saving mass history plot...')
        plot_mass_history(r.save_times, r.masses, run_dir / 'mass_history.png')
        note('Saving solution file...')
        save_solution_npz(r.x, r.save_times, r.snapshots, r.masses,
                          run_dir / 'solution.npz')
        note('Saving simulation parameters...')
        save_json(self.parameters, run_dir / 'simulation_parameters.json')
        note('Saving animation...')
        animation_file = save_animation(r.x, r.frame_times, r.frames,
                                        run_dir / 'animation')

        return {
            'directory': run_dir,
            'snapshots': run_dir / 'snapshots.png',
            'mass_history': run_dir / 'mass_history.png',
            'solution': run_dir / 'solution.npz',
            'parameters': run_dir / 'simulation_parameters.json',
            'animation': Path(animation_file),
        }


def solve(method='backward euler', drift=0.0, diffusion=1.0,
          left=-5.0, right=5.0, dx=0.05,
          total_time=1.0, dt=0.001, save_times=None,
          bc='neumann', initial_condition='gaussian', custom_expr='',
          trials=100_000, seed=0,
          left_is_infinite=None, right_is_infinite=None,
          coefficient_description=None, on_warning='warn',
          progress=None, stop_event=None, log=None) -> Run:
    """Run one simulation and return everything it produced.

    Parameters mirror the GUI's fields one for one:

    method
        'forward euler', 'backward euler' or 'euler-maruyama' (also 'fe',
        'be', 'em'; case-insensitive).
    drift, diffusion
        A callable `(x, t) -> array`, a Python expression string in `x`
        and `t` (e.g. `'-x'`, `'0.2 + 0.1*np.sin(t)'`), or a number.
        `diffusion` must be non-negative; the solvers reject it otherwise.
    left, right
        Domain bounds. Numbers, or the strings `'-inf'` / `'+inf'` for an
        unbounded side, which is then truncated adaptively.
    dx
        Requested cell width. Shrunk to the next value that divides the
        domain length exactly, as in the GUI.
    total_time, dt
        Simulation length and time step. `dt` is likewise shrunk so the
        run ends exactly at `total_time`.
    save_times
        Times at which to record the density. Defaults to six points
        evenly spaced over `[0, total_time]`. `0` and `total_time` are
        always included.
    bc
        'periodic', 'neumann' or 'dirichlet', all homogeneous.
    initial_condition, custom_expr
        One of 'gaussian', 'uniform', 'bimodal', 'left-block', or
        'custom' - in which case `custom_expr` is a non-negative Python
        expression in `x`. Always renormalized to unit mass.
    trials, seed
        Euler-Maruyama only: ensemble size and RNG seed.
    left_is_infinite, right_is_infinite
        Override for whether each side is unbounded. Normally inferred
        from `left`/`right` being infinite, but a caller that has already
        computed a better starting estimate than `_open_bound_guess`
        (the GUI does, from the initial condition's support) passes that
        finite estimate as the bound and flags it as open here, so the
        adaptive loop still knows it may grow that side.
    coefficient_description
        Text recorded in the saved parameters instead of the automatic
        description - for coefficients loaded from a .npz file, where the
        callable's name says nothing useful.
    on_warning
        What to do when the pre-run checks find something: 'warn' (the
        default) emits a `UserWarning` and continues, 'ask' prints the
        warnings and waits for a y/n answer on the terminal, 'raise'
        turns the first one into a `RunAborted`, and 'ignore' proceeds
        silently. 'ask' degrades to 'warn' when stdin is not a terminal,
        since a prompt nobody can answer would hang the caller.
    progress, stop_event, log
        Optional hooks, all no-ops by default. `progress(step, total, t)`
        is called during integration, `stop_event` is a
        `threading.Event`-like object polled for early termination, and
        `log(message)` receives the truncation notes.

    Returns a `Run`. Before integrating, the same checks the GUI performs
    run here too: Forward Euler with `lambda + 2*mu > 1`, a cell Peclet
    number large enough that upwinding's artificial diffusion dominates
    the physical one, and - as an error, not a warning - a periodic run
    whose coefficients disagree at the two edges. `run.messages` records
    whatever was reported, whichever `on_warning` mode was in force, and
    `diagnostics.preflight_warnings` runs the same checks standalone.
    """
    key = METHOD_ALIASES.get(str(method).strip().lower())
    if key is None:
        raise ValueError(
            f'Unknown method {method!r}. Expected one of: forward euler, '
            f'backward euler, euler-maruyama.')

    # Checked here as well as inside the solvers, because everything
    # between this point and the first solver call - the open-bound
    # estimate in particular, which takes sqrt(2*D*total_time) - assumes
    # a forward-going run. Reaching that with a negative total_time
    # produces a NaN and a warning before the real error surfaces.
    if dt <= 0.0:
        raise ValueError('Time step dt must be positive.')
    if total_time <= 0.0:
        raise ValueError('Total time must be positive.')

    drift_fn = _as_coefficient(drift, 'drift')
    diffusion_fn = _as_coefficient(diffusion, 'diffusion')
    boundary = BoundaryCondition(bc) if isinstance(bc, str) else bc
    boundary.validate()

    left_value = parse_bound(left) if isinstance(left, str) else float(left)
    right_value = parse_bound(right) if isinstance(right, str) else float(right)
    if left_is_infinite is None:
        left_is_infinite = not np.isfinite(left_value)
    if right_is_infinite is None:
        right_is_infinite = not np.isfinite(right_value)
    if (left_is_infinite or right_is_infinite) and boundary.kind == 'periodic':
        raise ValueError('An unbounded domain cannot be periodic.')

    left_start, right_start = _open_bound_guess(
        left_value, right_value, total_time, drift_fn, diffusion_fn,
        initial_condition, custom_expr)

    if save_times is None:
        save_times = np.linspace(0.0, total_time, 6).tolist()
    # Both endpoints are always recorded; anything outside the run is
    # dropped rather than silently ignored downstream. Same rule as
    # `gui.App.parse_inputs`, so the two front ends agree.
    save_times = sorted({0.0, float(total_time)} |
                        {float(t) for t in save_times
                         if 0.0 <= float(t) <= total_time})

    messages = []

    def record(message):
        messages.append(message)
        if log is not None:
            log(message)

    # Pre-run checks, on the grid the first attempt will actually use.
    # For an open bound that is the initial estimate rather than the
    # final domain; the numbers that matter here - dt/dx ratios and the
    # spread of the coefficients - are not sensitive to a later
    # enlargement, and waiting for the final grid would mean warning
    # about an unstable dt only after having already paid for a run.
    probe_grid, _ = make_grid(left_start, right_start, dx)
    preflight, _diagnostics = preflight_warnings(
        probe_grid, drift_fn, diffusion_fn, dt, total_time, key,
        boundary.kind)
    _resolve_warnings(preflight, on_warning, record)

    def solve_once(grid):
        """One attempt on one grid. The adaptive-domain loop may call
        this more than once with a wider grid, so the initial condition
        is rebuilt here rather than hoisted out - it is sampled on
        whatever grid this attempt uses.
        """
        p0 = build_initial_density(initial_condition, grid, custom_expr)
        if key == 'euler-maruyama':
            return euler_maruyama(grid, drift_fn, diffusion_fn, p0, trials,
                                  dt, total_time, save_times, boundary,
                                  seed=seed, progress_callback=progress,
                                  stop_event=stop_event)
        integrator = forward_euler if key == 'forward euler' else backward_euler
        return integrator(p0, grid, drift_fn, diffusion_fn, dt, total_time,
                          save_times, boundary, progress_callback=progress,
                          stop_event=stop_event)

    grid, result, _left, _right, _ = solve_with_adaptive_domain(
        solve_once, left_start, right_start, dx,
        left_is_infinite, right_is_infinite, log=record)

    parameters = {
        'method': key,
        'domain_left': grid.left,
        'domain_right': grid.right,
        'dx': grid.dx,
        'ncells': grid.ncells,
        'T': total_time,
        'dt_requested': dt,
        'boundary_condition': boundary.kind,
        'coefficients': coefficient_description if coefficient_description
                        else f'drift = {_describe(drift)}\n'
                             f'diffusivity = {_describe(diffusion)}',
        'initial_condition': initial_condition,
        'custom_expression': custom_expr,
        'trials_maruyama': trials if key == 'euler-maruyama' else None,
        'trials_completed': result.trials_completed,
        'seed': seed,
        'save_times': result.save_times.tolist(),
        'mass_at_t0': float(result.masses[0]),
        'final_mass': float(result.masses[-1]),
        'stopped_early': result.stopped_early,
        'final_time_reached': result.final_time,
    }
    # Mass is *supposed* to decay under an absorbing boundary, so a
    # deviation-from-1 figure there would read like an error on a
    # perfectly correct run. Report the absorbed fraction instead.
    if boundary.kind == 'dirichlet':
        parameters['absorbed_mass_fraction'] = float(
            1.0 - result.masses[-1] / result.masses[0])
    else:
        parameters['max_mass_deviation_from_1'] = float(
            max(abs(m - 1.0) for m in result.masses))

    return Run(result=result, grid=grid, messages=messages,
               parameters=parameters)
