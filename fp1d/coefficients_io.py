"""Load drift/diffusion coefficients from a saved .npz file, as an
alternative to typing a Python expression in the GUI.

Expected file format
--------------------
Each .npz file holds exactly one (drift, diffusion) pair:

    x   - 1D array of sample points, shape (nx,). Required.
    b   - drift samples: shape (nx,) if b is static (independent of
          time), or shape (nt, nx) if it varies with time. Required.
    D   - diffusion samples: shape (nx,) or (nt, nx), same rule as b.
          Required; must be non-negative everywhere.
    t   - 1D array of sample times, shape (nt,). OMIT this key entirely
          if both b and D are static (x-only).

b and D don't have to share the same time-dependence: e.g. a constant
diffusivity (`D` of shape (nx,)) can be paired with a time-varying drift
(`b` of shape (nt, nx)) in the same file, as long as `t` is present.

The saved samples don't need to be on the same grid the solver actually
uses (`x` here is just wherever the file happens to have data): both
coefficients are interpolated (linearly, and clamped rather than
extrapolated beyond the saved range) onto whatever points and time they
are actually called with, exactly like a typed expression would be
evaluated fresh at any (x, t).
"""
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def list_coefficient_files(coefficients_dir):
    """Return the sorted list of .npz file names (not full paths) found
    directly inside `coefficients_dir`, or an empty list if that
    directory doesn't exist.
    """
    directory = Path(coefficients_dir)
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.glob('*.npz'))


def _require_increasing(samples, name):
    """Raise unless `samples` is a strictly increasing 1D array.

    Both interpolators below take this for granted: `np.interp` silently
    returns nonsense for unsorted sample points rather than complaining,
    so an out-of-order file would otherwise produce a plausible-looking
    but wrong coefficient with no indication anything went wrong.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 1 or samples.size < 2:
        raise ValueError(f"'{name}' must be a 1D array of at least 2 samples.")
    if np.any(np.diff(samples) <= 0.0):
        raise ValueError(f"'{name}' samples must be strictly increasing.")


def _build_interpolator(x_saved, values, t_saved=None):
    """Return a callable `coefficient(x, t=0.0) -> array`, matching the
    (x, t) signature every drift/diffusion function in this project
    uses, that linearly interpolates `values` - clamping query points to
    the saved range rather than extrapolating beyond it.

    `values` has shape (nx,) if `t_saved` is None (static), or
    (nt, nx) if `t_saved` is given (time-dependent).
    """
    x_saved = np.asarray(x_saved, dtype=float)
    values = np.asarray(values, dtype=float)
    _require_increasing(x_saved, 'x')

    if t_saved is None:
        if values.shape != x_saved.shape:
            raise ValueError(
                f'Expected a static coefficient of shape {x_saved.shape}, '
                f'got {values.shape}.'
            )
        x_min, x_max = float(x_saved.min()), float(x_saved.max())

        def coefficient(x, t=0.0):
            x = np.asarray(x, dtype=float)
            return np.interp(np.clip(x, x_min, x_max), x_saved, values)

        return coefficient

    t_saved = np.asarray(t_saved, dtype=float)
    _require_increasing(t_saved, 't')
    if values.shape != (t_saved.size, x_saved.size):
        raise ValueError(
            f'Expected a time-dependent coefficient of shape '
            f'({t_saved.size}, {x_saved.size}), got {values.shape}.'
        )
    interpolator = RegularGridInterpolator(
        (t_saved, x_saved), values, method='linear',
        bounds_error=False, fill_value=None,
    )
    t_min, t_max = float(t_saved.min()), float(t_saved.max())
    x_min, x_max = float(x_saved.min()), float(x_saved.max())

    def coefficient(x, t=0.0):
        x = np.asarray(x, dtype=float)
        x_clamped = np.clip(x, x_min, x_max)
        t_clamped = np.clip(float(t), t_min, t_max)
        query_points = np.column_stack(
            [np.full(x_clamped.shape, t_clamped), x_clamped]
        )
        return interpolator(query_points)

    return coefficient


def load_coefficient_set(path):
    """Load one .npz file and return `(drift, diffusion, info)`.

    `drift`/`diffusion` are callables with the usual `(x, t=0.0)`
    signature, interpolated from the saved samples. `info` is a small
    dict describing what was loaded, meant for display in the GUI:

        file                       - the file name
        x_range                    - (min, max) of the saved x samples
        t_range                    - (min, max) of the saved t samples,
                                      or None if neither b nor D carries
                                      a time axis
        drift_time_dependent       - bool
        diffusion_time_dependent   - bool
    """
    path = Path(path)
    with np.load(path) as data:
        missing = [key for key in ('x', 'b', 'D') if key not in data]
        if missing:
            raise ValueError(
                f"'{path.name}' is missing required array(s): {', '.join(missing)}."
            )
        x_saved = data['x']
        t_saved = data['t'] if 't' in data else None
        b_raw = np.asarray(data['b'], dtype=float)
        D_raw = np.asarray(data['D'], dtype=float)

    if np.any(D_raw < 0.0):
        raise ValueError(f"'{path.name}': diffusion values must be non-negative.")

    # b and D may independently be static (nx,) or time-dependent
    # (nt, nx); with no 't' key at all, neither can carry a time axis.
    b_time_dep = (t_saved is not None) and (b_raw.ndim == 2)
    D_time_dep = (t_saved is not None) and (D_raw.ndim == 2)

    drift = _build_interpolator(x_saved, b_raw, t_saved if b_time_dep else None)
    diffusion = _build_interpolator(x_saved, D_raw, t_saved if D_time_dep else None)

    info = {
        'file': path.name,
        'x_range': (float(np.min(x_saved)), float(np.max(x_saved))),
        't_range': ((float(np.min(t_saved)), float(np.max(t_saved)))
                    if t_saved is not None else None),
        'drift_time_dependent': b_time_dep,
        'diffusion_time_dependent': D_time_dep,
    }
    return drift, diffusion, info
