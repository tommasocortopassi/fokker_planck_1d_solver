"""Cell-centered finite-volume discretization of

    dp/dt = -d/dx(b p) + d^2/dx^2( D p )  =:  -d/dx J,   J = b p - d/dx(D p).

Unknowns p[i] are cell averages; J is evaluated on the n+1 cell faces.
Integrating the PDE over cell i and dividing by dx gives the exact
semi-discrete balance

    dp[i]/dt = -(J[i+1] - J[i]) / dx.

Face flux model
----------------
On each face, let p_L and p_R denote the cell averages of p on the left and right cells
 respectively. The flux J is approximated as

    J = b_face * p_upwind - (D_R p_R - D_L p_L) / distance,

i.e. upwind for advection (stable for any Peclet number) and a centered
difference of the product (D p) for the diffusive part. `distance` is the
spacing between the two states entering the face: dx between two
neighboring cell centers, or dx/2 between a boundary face and the one
adjacent cell center it touches.

Because both terms are linear in (p_L, p_R), we can write J = cL * p_L + cR * p_R for
some coefficients (cL, cR); `_face_coeffs` returns them. Every face's pair is
computed in one vectorized batch and summed straight into a sparse matrix
(see `assemble_operator`).

Boundary conditions
--------------------
Only the homogeneous case of each condition is supported (no nonzero
Dirichlet/Neumann values to inject):
- periodic: the domain is a ring, so face 0 and face n are the same
  physical face, coupling cell n-1 to cell 0 exactly like an interior face.
- neumann (reflecting / no-flux): J = 0 at the two boundary faces, by
  construction - no advective or diffusive transport crosses the wall, and
  the domain's total mass is exactly conserved regardless of b, D.
- dirichlet: the density is fixed to 0 at the extremal boundary face
  (distance dx/2 from the first/last cell center), with b sampled at that
  exact face location and D sampled at each of the two states entering the
  face (the boundary itself for the ghost side, the adjacent cell center
  for the interior side). Because the fixed value is 0, it never needs
  its own source term - see `assemble_operator` below.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve, splu

from .boundary_conditions import BoundaryCondition
from .diagnostics import check_periodic_consistency, is_time_dependent
from .results import SolutionRecorder


def _face_coeffs(bL, bR, DL, DR, distance):
    """J = cL * p_L + cR * p_R at one *or many* faces; return (cL, cR).

    Vectorized: bL, bR, DL, DR may be scalars (one boundary face) or
    arrays (every interior/periodic face at once) - `np.where` plays the
    role the scalar `if` used to, without a Python-level loop over faces.

    `distance` is the spacing between the two states entering the flux
    (dx for an interior face, dx/2 for a boundary face against a fixed
    ghost value) - see the module docstring's face flux model.
    """
    b_face = 0.5 * (bL + bR)
    # Upwind: use whichever side the flow actually comes from, so the
    # scheme stays stable no matter how advection-dominated the cell is.
    adv_L = np.where(b_face >= 0.0, b_face, 0.0)
    adv_R = np.where(b_face >= 0.0, 0.0, b_face)
    cL = adv_L + DL / distance
    cR = adv_R - DR / distance
    return cL, cR


def _at_point(coefficient, x_value, t):
    """Evaluate a coefficient at the single point `x_value`, as a float.

    The coefficient callables are array-valued, so a lone boundary face
    still has to pass a length-1 array in and take element 0 back out.
    """
    return float(np.asarray(coefficient(np.array([x_value]), t))[0])


def assemble_operator(grid, drift, diffusion, t, bc: BoundaryCondition):
    """Build dp/dt = A p as a sparse matrix A (no source term: every
    boundary condition supported here is homogeneous, so there is never
    a fixed nonzero value entering from outside the domain).

    Every face's contribution is computed as one vectorized batch (rather
    than assembled face-by-face in a Python loop) and handed to
    `scipy.sparse.coo_matrix`, which sums the duplicate (row, col) entries
    that naturally arise where two neighboring faces both touch the same
    cell - this is what turns per-face fluxes into a per-cell balance.
    This matters because `assemble_operator` is called once per time
    step by forward/backward Euler, so a Python-level loop over faces
    here would dominate the whole solver's runtime on a fine grid.
    """
    bc.validate()
    n = grid.ncells
    dx = grid.dx
    x = grid.centers
    b = np.asarray(drift(x, t), dtype=float)
    D = np.asarray(diffusion(x, t), dtype=float)
    if np.any(D < 0.0):
        raise ValueError('Diffusion must be non-negative.')
    if bc.kind == 'periodic':
        check_periodic_consistency(drift, diffusion, grid.left, grid.right, t)

    # Face `f` sits between cells iL=f-1 (left) and iR=f (right); it
    # contributes -J[f]/dx to cell iL's balance (flux leaving through its
    # right face) and +J[f]/dx to cell iR's balance (flux entering
    # through its left face). For a periodic ring there are exactly n
    # faces, one per cell, wrapping cell n-1 to cell 0; otherwise there
    # are n-1 (purely interior) faces and the two domain edges are
    # handled separately below.
    if bc.kind == 'periodic':
        iL = np.arange(n)
        iR = (iL + 1) % n
    else:
        iL = np.arange(0, n - 1)
        iR = np.arange(1, n)

    cL, cR = _face_coeffs(b[iL], b[iR], D[iL], D[iR], dx)
    rows = np.concatenate([iL, iL, iR, iR])
    cols = np.concatenate([iL, iR, iL, iR])
    data = np.concatenate([-cL / dx, -cR / dx, cL / dx, cR / dx])

    # Periodic and Neumann need nothing further: periodic's wrap face is
    # already included above, and Neumann simply sets J = 0 at both
    # boundary faces (i.e. contributes nothing). Only Dirichlet adds the
    # two boundary-face terms below.
    if bc.kind == 'dirichlet':
        # Left boundary face: the fixed ghost value sits dx/2 to the left
        # of cell 0, and is 0 (homogeneous), so it only ever contributes
        # through cR - there is no p_L term (no p_ghost variable), hence
        # no source vector, just a modified coefficient on cell 0 itself.
        # The two states entering this face are the ghost value at
        # x = grid.left and the cell average of cell 0, so (D p) must be
        # sampled at those two *different* places: D at the boundary for
        # the ghost side, D at the cell centre for the interior side.
        # Using the boundary D on both sides would be an O(dx) error in
        # the flux whenever D varies in x.
        b_L = _at_point(drift, grid.left, t)
        D_L = _at_point(diffusion, grid.left, t)

        # Right boundary face: symmetric story, dx/2 to the right of
        # cell n-1.
        b_R = _at_point(drift, grid.right, t)
        D_R = _at_point(diffusion, grid.right, t)

        # D sampled *at* the wall never reaches the operator: it belongs
        # to the ghost side of the face, and the ghost value is 0 under a
        # homogeneous Dirichlet condition, so `_face_coeffs` returns it
        # in the coefficient that is discarded just below. It is still
        # evaluated and checked here because a diffusivity that goes
        # negative anywhere on the closed domain is a broken
        # diffusivity, and the bulk check above only ever saw cell
        # centres - which never coincide with x = left or x = right.
        if D_L < 0.0 or D_R < 0.0:
            raise ValueError('Diffusion must be non-negative.')

        _, cR_left = _face_coeffs(b_L, b_L, D_L, float(D[0]), dx / 2.0)
        cL_right, _ = _face_coeffs(b_R, b_R, float(D[-1]), D_R, dx / 2.0)

        rows = np.concatenate([rows, [0, n - 1]])
        cols = np.concatenate([cols, [0, n - 1]])
        data = np.concatenate([data, [cR_left / dx, -cL_right / dx]])

    return sparse.coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _run(p0, grid, dt, total_time, save_times,
         step_fn, progress_callback=None, stop_event=None, max_frames=200):
    """Shared time-stepping loop for forward and backward Euler.

    `step_fn(p, t_old, t_new, dt) -> p_new` advances the state by one
    step; everything else here (recording snapshots/mass/frames, progress
    reporting, early stop) is common to both integrators.

    `dt` and `total_time` are both required to be strictly positive.
    Without that guard `nsteps` becomes 0 for `total_time = 0` and the
    renormalization below divides by it - a `ZeroDivisionError` from deep
    inside the loop setup, rather than a message naming the bad input.

    `dt` is passed *into* `step_fn` rather than captured by it, because
    the requested dt is renormalized below whenever it doesn't evenly
    divide `total_time`. A `step_fn` closing over the caller's original
    dt would advance the state by a different amount than the time grid
    here assumes, and the run would silently end at
    ceil(total_time/dt)*dt instead of at `total_time`.
    """
    if dt <= 0.0:
        raise ValueError('Time step dt must be positive.')
    if total_time <= 0.0:
        raise ValueError('Total time must be positive.')

    nsteps = int(np.ceil(total_time / dt))
    dt = total_time / nsteps  # spread any leftover evenly over all steps
    p = np.asarray(p0, dtype=float).copy()

    recorder = SolutionRecorder(grid, save_times, total_time, nsteps, max_frames)
    recorder.record_initial(p)

    stopped_early = False
    current_time = 0.0
    for k in range(nsteps):
        if stop_event is not None and stop_event.is_set():
            stopped_early = True
            break

        t_old = k * dt
        t_new = (k + 1) * dt
        p = step_fn(p, t_old, t_new, dt)
        current_time = t_new

        # `p` already *is* the density here, so this closure is trivial -
        # unlike Euler-Maruyama's histogram, there's no extra cost from
        # calling it even when the step turns out not to be recorded.
        recorder.record_step(k + 1, current_time, k == nsteps - 1, lambda: p)

        if progress_callback and ((k + 1) % max(1, nsteps // 100) == 0 or k == nsteps - 1):
            progress_callback(k + 1, nsteps, current_time)

    if not stopped_early:
        recorder.record_final(lambda: p)

    return recorder.finalize(stopped_early, current_time)


def _make_assembler(grid, drift, diffusion, bc, total_time):
    """Return `assemble(t) -> A`, reusing a single matrix whenever both
    coefficients turn out to be independent of t.

    `assemble_operator` costs the same at every step, so for a static
    (b, D) - the common case - reassembling it once per step dominates
    the whole explicit solver's runtime while producing an identical
    matrix each time. `diagnostics.is_time_dependent` decides which
    regime we're in, once, before the loop starts.

    The returned callable carries an `is_static` attribute, so callers
    that can cache more than the matrix itself (backward Euler caches
    the factorization) know whether that's safe.
    """
    x = grid.centers
    is_static = (not is_time_dependent(drift, x, total_time)
                 and not is_time_dependent(diffusion, x, total_time))
    cached = None

    def assemble(t):
        nonlocal cached
        if not is_static:
            return assemble_operator(grid, drift, diffusion, t, bc)
        if cached is None:
            cached = assemble_operator(grid, drift, diffusion, 0.0, bc)
        return cached

    assemble.is_static = is_static
    return assemble


def forward_euler(p0, grid, drift, diffusion, dt, total_time, save_times,
                   bc: BoundaryCondition, progress_callback=None, stop_event=None):
    """Explicit update p^{n+1} = p^n + dt * (A p^n).

    Cheap per step (one sparse matrix-vector product), but only
    conditionally stable: dt must respect the combined CFL bound in
    diagnostics.py (lambda + 2*mu <= 1), or errors can grow instead of
    decay.
    """
    assemble = _make_assembler(grid, drift, diffusion, bc, total_time)

    def step(p, t_old, t_new, step_dt):
        A = assemble(t_old)
        return p + step_dt * (A @ p)

    return _run(p0, grid, dt, total_time, save_times,
                step, progress_callback, stop_event)


def backward_euler(p0, grid, drift, diffusion, dt, total_time, save_times,
                    bc: BoundaryCondition, progress_callback=None, stop_event=None):
    """Implicit update (I - dt A) p^{n+1} = p^n.

    Requires one sparse linear solve per step, but is unconditionally
    stable: no restriction on dt from diffusion or advection. The
    operator is assembled at t_new since the update is implicit (the
    right-hand side is evaluated at the *new* time level).

    When b and D are static, I - dt*A is the *same* matrix at every
    step, so it is factorized once with `splu` and only the (cheap)
    triangular solves are repeated - rather than having `spsolve`
    refactorize from scratch on every step.
    """
    assemble = _make_assembler(grid, drift, diffusion, bc, total_time)
    identity = sparse.identity(grid.ncells, format='csc')
    factorization = None

    def step(p, t_old, t_new, step_dt):
        nonlocal factorization
        A = assemble(t_new)
        if not assemble.is_static:
            return spsolve((identity - step_dt * A).tocsc(), p)
        if factorization is None:
            factorization = splu((identity - step_dt * A).tocsc())
        return factorization.solve(p)

    return _run(p0, grid, dt, total_time, save_times,
                step, progress_callback, stop_event)
