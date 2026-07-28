"""Uniform cell-centered grid for the finite-volume discretization.

Cells, not nodes, hold the unknowns: `p[i]` is the average density over
cell `i`, and faces (cell edges) are where fluxes are evaluated. This is
what makes exact mass conservation possible (see finite_volume.py).

`Grid1D` itself is built from a cell *count*, since that's what the
solvers actually need. The GUI instead lets the user specify a cell
*width* `dx`; `make_grid` bridges the two, and `finite_domain` turns
`+/-inf` domain bounds into finite numeric guesses. Truncating an
unbounded domain and then verifying/enlarging it based on the actual
solution lives in `domain_truncation.py`, one level up from here.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid1D:
    left: float
    right: float
    ncells: int

    def __post_init__(self):
        if self.right <= self.left:
            raise ValueError('Require right > left.')
        if self.ncells < 4:
            raise ValueError('Require ncells >= 4.')

    @property
    def dx(self) -> float:
        return (self.right - self.left) / self.ncells

    @property
    def faces(self) -> np.ndarray:
        """Cell edges, ncells + 1 of them."""
        return np.linspace(self.left, self.right, self.ncells + 1)

    @property
    def centers(self) -> np.ndarray:
        """Cell midpoints, where the unknowns p[i] live."""
        f = self.faces
        return 0.5 * (f[:-1] + f[1:])


def make_grid(left: float, right: float, requested_dx: float, min_cells: int = 4):
    """Build a `Grid1D` covering `[left, right]` whose cell width is the
    largest value not exceeding `requested_dx` for which the domain
    length is exactly divisible - i.e. `Grid1D` always tiles the domain
    exactly, and we round the *number of cells up* (never down) so the
    actual `dx` never exceeds what the user asked for.

    Returns `(grid, message)`. `message` is non-empty whenever the actual
    `dx` differs from `requested_dx` (because the length wasn't an exact
    multiple of it), and should be shown to the user.
    """
    if requested_dx <= 0:
        raise ValueError('dx must be positive.')
    length = right - left
    if length <= 0:
        raise ValueError('Require right > left.')

    # ncells = ceil(length / requested_dx) is the *smallest* cell count
    # whose resulting dx = length/ncells does not exceed requested_dx -
    # equivalently, the largest such dx. The small epsilon subtraction
    # guards against float noise pushing an exact ratio (e.g. 10/0.1)
    # just above an integer, which would otherwise round up to one cell
    # too many.
    ncells = max(min_cells, int(np.ceil(length / requested_dx - 1e-9)))
    grid = Grid1D(left, right, ncells)

    message = ''
    if not np.isclose(grid.dx, requested_dx, rtol=1e-9, atol=1e-12):
        message = (f'Requested dx = {requested_dx:.6g} does not evenly divide '
                    f'the domain length {length:.6g}; using dx = {grid.dx:.6g} '
                    f'({ncells} cells) instead.')
    return grid, message


def parse_bound(token: str) -> float:
    """Parse one domain-bound entry, accepting 'inf'/'-inf'/'+inf' (with
    or without a leading unicode minus/infinity symbol) as well as
    ordinary numbers. Shared by `finite_domain` below and by the GUI's
    live input parsing, so there's exactly one place that defines what
    counts as a valid bound.
    """
    token = token.strip().lower().replace('\u221e', 'inf')
    if token in ('inf', '+inf'):
        return float('inf')
    if token == '-inf':
        return float('-inf')
    return float(token)


def finite_domain(left_text: str, right_text: str,
                   left_guess: float, right_guess: float):
    """Parse domain bounds, turning +/-inf into a finite truncation.

    `left_guess` is the bound to substitute when the left bound is -inf,
    and `right_guess` likewise for +inf; each is ignored when the
    corresponding bound is already finite. Both are ordinary coordinates
    on the x axis, in the same (left, right) order as everything else
    here, and are meant to be pre-computed from the initial condition's
    support plus the advective/diffusive spread over the run time (see
    `gui.App.estimate_open_bounds`).

    Returns (left, right, message). `message` is non-empty whenever a
    truncation happened, and should be shown to the user: the true
    mathematical domain was unbounded, and the computation only ever
    covers [left, right].
    """
    left = parse_bound(left_text)
    right = parse_bound(right_text)
    truncated = not (np.isfinite(left) and np.isfinite(right))
    if left == -np.inf:
        left = left_guess
    if right == np.inf:
        right = right_guess
    if not (np.isfinite(left) and np.isfinite(right)):
        raise ValueError('Bounds must be finite or +/-inf.')

    msg = ''
    if truncated:
        msg = (f'Infinite domain truncated to [{left:.4g}, {right:.4g}] '
               f'for computation.')
    return left, right, msg