# Coefficient files

Drop a `.npz` file here to make it selectable in the GUI as a source for
drift `b(x,t)` and diffusivity `D(x,t)`, instead of typing a Python
expression. Click "Refresh list" in the GUI if you add a file while it's
already running.

## Format

Each file holds exactly one `(drift, diffusion)` pair, as plain NumPy
arrays:

| key | required | shape | meaning |
|---|---|---|---|
| `x` | yes | `(nx,)` | sample points |
| `b` | yes | `(nx,)` or `(nt, nx)` | drift samples |
| `D` | yes | `(nx,)` or `(nt, nx)` | diffusion samples (must be $\ge 0$) |
| `t` | only if `b` or `D` is time-dependent | `(nt,)` | sample times |

- Omit `t` entirely if **both** `b` and `D` are static (shape `(nx,)`).
- `b` and `D` don't need to share the same time-dependence: a static
  `D` of shape `(nx,)` can be paired with a time-dependent `b` of shape
  `(nt, nx)` in the same file, as long as `t` is present.
- The saved `x` (and `t`) grid does **not** need to match the solver's
  own grid or time step - both coefficients are linearly interpolated
  onto whatever point and time they're actually queried at, and clamped
  (not extrapolated) if a query falls outside the saved range.

## Example

```python
import numpy as np

x = np.linspace(-15, 15, 601)          # can be much finer/coarser than
                                        # the solver's own grid
b = -x                                 # static Ornstein-Uhlenbeck drift
D = np.ones_like(x)                    # static, constant diffusivity
np.savez('coefficients/ou_static.npz', x=x, b=b, D=D)
```

For a time-dependent version:

```python
t = np.linspace(0, 20, 41)
b_t = np.array([-x * (1.0 + 0.5 * np.sin(2 * np.pi * tt / 10)) for tt in t])  # (nt, nx)
np.savez('coefficients/ou_breathing.npz', x=x, t=t, b=b_t, D=D)
```

See `make_examples.py` in this folder for a runnable version of both.
