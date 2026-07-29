"""A single initial condition, shared by the PDE and SDE solvers.

The PDE branch uses the density directly as cell averages. The SDE branch
draws particles from that same density, so both solvers always start from
an identical physical state - no separate, independently-tuned "initial
particle distribution" to keep in sync.
"""
import re
import numpy as np


def _gaussian(x, mean=0.0, sigma=0.6):
    """Single bump centered at `mean` with standard deviation `sigma`."""
    return np.exp(-0.5 * ((x - mean) / sigma) ** 2)


def _uniform(x):
    """Flat density (normalized afterwards by `normalize_to_mass_one`)."""
    return np.ones_like(x)


def _bimodal(x, mean1=-1.0, mean2=1.0, sigma1=0.35, sigma2=0.45, w1=0.5):
    """Two Gaussian bumps mixed with weight `w1` on the first."""
    return (w1 * np.exp(-0.5 * ((x - mean1) / sigma1) ** 2)
            + (1.0 - w1) * np.exp(-0.5 * ((x - mean2) / sigma2) ** 2))


def _left_block(x, cutoff=0.0):
    """A step: higher density for x <= cutoff, lower beyond it."""
    return np.where(x <= cutoff, 1.0, 0.15)


PRESETS = {
    'gaussian': _gaussian,
    'uniform': _uniform,
    'bimodal': _bimodal,
    'left-block': _left_block,
}


# The only builtins exposed to user-typed expressions - here and in the
# GUI's drift/diffusivity fields, which evaluate expressions the same way
# (see `gui.App.build_coefficients`). Everything else, `__import__` and
# `open` above all, stays out of reach.
#
# Note that an empty `__builtins__` is NOT the same as omitting the key:
# Python injects the real builtins module into any globals dict that
# lacks it, so the key must be present and must be this whitelist.
SAFE_BUILTINS = {
    'abs': abs, 'float': float, 'int': int, 'len': len,
    'max': max, 'min': min, 'pow': pow, 'round': round, 'sum': sum,
}


def safe_eval(expr: str, variables: dict, what: str = 'expression'):
    """Evaluate one user-typed expression in the restricted namespace.

    `variables` are the names the expression may use besides numpy -
    `{'x': ...}` for an initial condition, `{'x': ..., 't': ...}` for a
    coefficient. `what` names the field in any error message, so the user
    is told *which* box was wrong.

    Every failure comes back as a `ValueError` carrying a message meant
    to be shown verbatim in the GUI. The most common mistake by far is
    writing `cos(t)` rather than `np.cos(t)`, so a `NameError` naming
    something numpy provides is answered with that specific correction
    instead of a bare "not defined".
    """
    try:
        return eval(expr, {'np': np, '__builtins__': SAFE_BUILTINS},
                    dict(variables))
    except NameError as exc:
        # `NameError.name` exists from Python 3.10; fall back to parsing
        # the message so the hint still works on older interpreters.
        name = getattr(exc, 'name', None)
        if name is None:
            match = re.search(r"name '([^']+)' is not defined", str(exc))
            name = match.group(1) if match else '?'
        if hasattr(np, name):
            raise ValueError(
                f"Unknown name '{name}' in the {what}. Mathematical "
                f"functions come from numpy: write 'np.{name}(...)' "
                f"instead of '{name}(...)'.") from None
        available = ', '.join(sorted(variables))
        raise ValueError(
            f"Unknown name '{name}' in the {what}. Available names are "
            f"{available}, and numpy as 'np' - for example "
            f"np.exp(-x**2).") from None
    except SyntaxError as exc:
        raise ValueError(f'The {what} is not valid Python: {exc.msg}.') from None
    except Exception as exc:
        raise ValueError(f'Could not evaluate the {what}: {exc}') from None


def evaluate_custom_expression(expr: str, x: np.ndarray) -> np.ndarray:
    """Evaluate a user-typed density expression like 'np.exp(-x**2)'.

    Restricted to numpy, the variable x, and the small set of builtins
    in `SAFE_BUILTINS`.

    An expression that ignores `x` entirely - a bare constant such as
    '1.0' - evaluates to a scalar, and is broadcast to a flat density
    over the grid. This matches how the drift and diffusivity fields
    treat a constant expression (see `api._as_coefficient` and
    `gui.App.build_coefficients`), so the same input means the same
    thing in every expression box in the program.
    """
    values = np.asarray(safe_eval(expr, {'x': x}, 'custom initial condition'),
                        dtype=float)
    if values.shape == ():
        values = np.full_like(x, float(values), dtype=float)
    if values.shape != x.shape:
        raise ValueError('Custom expression must return one value per grid point.')
    return values


def normalize_to_mass_one(p: np.ndarray, dx: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    mass = dx * np.sum(p)
    if mass <= 0:
        raise ValueError('Initial condition must have positive mass.')
    return p / mass


def build_initial_density(name: str, grid, custom_expr: str = '') -> np.ndarray:
    """Return a normalized array of cell averages on `grid.centers`."""
    x = grid.centers
    if name == 'custom':
        raw = evaluate_custom_expression(custom_expr, x)
        if np.any(raw < 0.0):
            raise ValueError('Custom initial condition must be non-negative.')
    else:
        if name not in PRESETS:
            raise ValueError(f'Unknown initial condition: {name}')
        raw = PRESETS[name](x)
    return normalize_to_mass_one(raw, grid.dx)


def sample_particles_from_density(grid, p: np.ndarray, n_trials: int,
                                   rng: np.random.Generator) -> np.ndarray:
    """Draw n_trials particle positions matching the cell-average density p.

    Each particle is assigned to a cell with probability proportional to
    that cell's share of the total mass (p[i] * dx), then placed uniformly
    at random within the cell so trajectories aren't stacked on centers.
    """
    dx = grid.dx
    weights = p * dx
    weights = weights / weights.sum()
    cell_index = rng.choice(grid.ncells, size=n_trials, p=weights)
    faces = grid.faces
    jitter = rng.uniform(0.0, dx, size=n_trials)
    return faces[cell_index] + jitter
