"""A single initial condition, shared by the PDE and SDE solvers.

The PDE branch uses the density directly as cell averages. The SDE branch
draws particles from that same density, so both solvers always start from
an identical physical state - no separate, independently-tuned "initial
particle distribution" to keep in sync.
"""
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


def evaluate_custom_expression(expr: str, x: np.ndarray) -> np.ndarray:
    """Evaluate a user-typed density expression like 'np.exp(-x**2)'.

    Restricted to numpy, the variable x, and the small set of builtins
    in `SAFE_BUILTINS`.
    """
    safe_globals = {'np': np, '__builtins__': SAFE_BUILTINS}
    try:
        values = eval(expr, safe_globals, {'x': x})
    except Exception as exc:
        raise ValueError(f'Could not evaluate custom expression: {exc}')
    values = np.asarray(values, dtype=float)
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
