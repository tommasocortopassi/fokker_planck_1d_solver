# 1D Fokker-Planck Solver
*Three independent solvers against the exact solution:*
![Forward Euler, Backward Euler and Euler-Maruyama against the exact solution](docs/img/demo.gif)


- a **finite-volume PDE solver** (forward and backward Euler)
- an **Euler-Maruyama particle solver** for the equivalent stochastic
  differential equation (SDE)

driven by a single GUI that shares the same initial condition, domain, and
boundary condition between both solvers. This README is a practical
overview and usage guide. For the full derivations - the origin of artificial (numerical) diffusion, the von Neumann stability
analysis behind the CFL and Peclet numbers, and the Ito-calculus derivation
that connects the SDE to the PDE - see `docs/numerical_methods_notes.md`.

## The equation

The Fokker-Planck equation describes how a probability density $p(x, t)$
evolves under a deterministic drift $b(x, t)$ and a diffusion coefficient
$D(x, t) \geq 0$:

$$
\partial_t p = -\partial_x( b(x,t) p ) + \partial_{xx}^2( D(x,t) p ) = -\partial_x J(x,t) \qquad (1)
$$

where the flux is $J(x,t) = b(x,t) p - \partial_x[D(x,t) p]$. Writing the
right-hand side as $-\partial_x J$ is what makes the finite-volume
discretization exactly conservative (see below). Physically, (1) is also an
`advection-diffusion equation`: it describes a quantity $p(x,t)$
transported by the velocity field $b(x,t)$ and diffused with coefficient
$D(x,t)$.

Equation (1) is also the density evolution equation for the Ito process

$$
dX_t = b(X_t, t) dt + \sqrt{2 D(X_t, t)} dW_t .\qquad (2)
$$

Both solvers in this repository target the same $p(x, t)$: one by
discretizing the PDE directly, the other by simulating an ensemble of paths
of (2) and estimating $p(x,t)$ empirically as a histogram.
`docs/numerical_methods_notes.md` derives (2) $\to$ (1) from scratch (the
Lagrangian-to-Eulerian route) and proves why the two really do describe the
same physics.

#### Remark on nomenclature
We always use the terms cells and faces as usual in finite-volume schemes,
but since we are in 1D we could replace such terms with intervals and
points.

## Finite-volume discretization (the PDE solver)

`fp1d/finite_volume.py` uses a cell-centered grid (`fp1d/grid.py`): the
domain is split into $n$ cells of width $dx$, the unknown $p_t[i]$ is the
*average* density over cell $i$ at time $t$, and fluxes $J$ are evaluated on the $n+1$
faces between and around cells:

```

         dx
I=    |-----|-----|-----| .... |-----|
      x₁    x₂    x₃    x₄     xₙ    xₙ₊₁
         I₁    I₂    I₃           Iₙ
```

Integrating (1) over each cell gives an *exact* identity for the cell
averages - no approximation is made at this stage. Summing that identity
over every cell telescopes: every interior face flux cancels, and only the
two domain-boundary fluxes survive, so

$$
\frac{d}{dt}\int_I p(x,t)dx = J_t[x_1] - J_t[x_{n+1}].
$$

**Mass conservation is therefore a structural property of the
discretization**, not something that has to be checked - positive flux at
the left edge brings mass in, positive flux at the right edge lets mass
out, and nothing happens to the total in between no matter how the
interior faces are approximated. `docs/numerical_methods_notes.md` §1.1
works through the telescoping argument in full.

### Face flux model

The only approximation the scheme makes is in reconstructing $J$ at each
face from the two neighboring cell averages on the left $p_L$ and on the right $p_R$ (and
$b_L, b_R, D_L, D_R$):

- The **diffusive part** is a centered difference of $Dp$ between the two
  neighboring cells.
- The **advective part** is **upwinded**: it uses $p_L$ if the flow points
  left-to-right at that face, $p_R$ otherwise. Upwinding is what keeps the
  explicit scheme stable at any Peclet number - a centered average of
  $p_L, p_R$ would oscillate once advection dominates diffusion - at the
  cost of quietly adding a numerical diffusion term of size $\sim |b|dx$.
  See `docs/numerical_methods_notes.md` §1.2 for mor details.

Because both terms are linear in $(p_L, p_R)$, the whole face flux is
$J = c_L p_L + c_R p_R$ for coefficients computed once per face
(`_face_coeffs` in `fp1d/finite_volume.py`), and the entire operator
assembles directly into a sparse matrix, face by face.

### Boundary conditions (BC)

- **Periodic**: the domain is a circle. The first and last faces are
  identified, so $J[x_1] = J[x_{n+1}]$ by construction and mass is
  conserved.
- **Homogeneous Neumann (reflecting / no-flux)**: a reflecting wall means
  *no probability/flux crosses it, in either direction* - so we simply fix
  $J[x_1] = J[x_{n+1}] = 0$. Mass is conserved. **Beware** that this condition
  is imposed ** on the flux J** and **NOT** on the solution $p$, as usually done with 
  Neumann BC in PDEs.
- **Homogeneous Dirichlet**: the density *exactly at the boundary face*, which sits at a distance`dx/2` from the adjacent cell center, is
  set to 0. Mass is not conserved.

### Time integration

Writing the assembled system as $dp/dt = A p $, we discretize in time as:

- **Forward Euler**: $p^{m+1} = p^m + dt(A^m p^m)$. One sparse
  matrix-vector product per step. Cheap, but only *conditionally* stable:
  if $dt$ is too large relative to $dx$ and the size of $b$, $D$, small
  errors are amplified instead of damped. The GUI computes the CFL/Peclet
  bounds below and warns before running if $dt$ exceeds them (see
  `fp1d/diagnostics.py`), but still lets you proceed.
- **Backward Euler**: $(I - dtA)p^{m+1} = p^m$. One sparse
  linear solve per step. Because the update is implicit, it is
  **unconditionally stable** in the diffusive sense. There is no $dt$ threshold below 
  which it must stay to avoid blowing up. It still loses
  accuracy for very large $dt$ (truncation error grows), but it will not
  diverge.

`docs/numerical_methods_notes.md` §1.4 derives exactly where these
stability thresholds come from (von Neumann analysis), plus a couple of
more intuitive, non-Fourier ways to see the same bounds. We also estimate the artificial (numerical)
diffusion introduced by the methods, which is also reported in the GUI.

### CFL and Peclet numbers

- **Advective CFL** $= \max_{x,t}\{\|b(x,t)\| dt / dx\}$: how far (in cells)
  information travels by advection in one time step.
- **Diffusive CFL** $= \max \{D(x,t) dt / dx^2\}$: how far diffusion spreads in
  one step. Forward Euler requires `advective CFL + 2diffusive CFL <= 1` to stay
  stable.
- **Peclet number** $\mathrm{Pe} = \max\{| b(x,t)|dx / (2D(x,t))\}$: the
  ratio of advective to diffusive transport *across one cell*. It does not
  constrain $dt$, but it tells you whether the local physics is dominated by advection or
  diffusion.

## Euler-Maruyama (the SDE solver)

`fp1d/stochastic_solver.py` simulates the SDE in (2) directly:

$$
X_{n+1} = X_n + b(X_n, t_n) dt + \sqrt{2 D(X_n, t_n) dt} Z, \qquad Z \sim N(0, 1)
$$

for an ensemble of independent trajectories, then recovers $p(x, t)$ as a
normalized histogram of the live particles at each requested time. The
initial ensemble is sampled directly from the same initial density used by
the PDE solver (`fp1d/initial_conditions.py`), so both solvers always start
from an identical physical state.

### Why use Euler-Maruyama instead of the PDE solver

The finite-volume scheme differences $b$ and $D$ between neighboring cells,
which implicitly assumes they vary smoothly at the grid scale. If $b$ or
$D$ are rough, discontinuous (e.g. a diffusion coefficient that jumps
across a material interface), or only available as noisy/empirical
samples, that differencing has no clean meaning: the finite-volume solution
can pick up spurious oscillations or lose accuracy near the irregularity,
no matter how small $dt$ is.

Euler-Maruyama never differentiates $b$ or $D$: it only *evaluates* them
at particle positions. It therefore degrades gracefully in the presence of
non-smooth coefficients, at the cost of trading a numerically exact mesh
solution for Monte Carlo noise (statistical error $\sim 1/\sqrt{n_{trials}}$,
independent of how irregular the coefficients are). In short: **finite
volume is more accurate when $b, D$ are smooth and known everywhere;
Euler-Maruyama is more robust when they are not.**
`docs/numerical_methods_notes.md` §2 derives this from scratch, including
the Ito-calculus argument for *why* the particle histogram converges to the
same $p(x,t)$.

### Boundary conditions for Euler-Maruyama

| PDE boundary | Particle equivalent | Behavior |
|---|---|---|
| `periodic` | wrap | a particle crossing an edge re-enters on the other side |
| `neumann` | reflect | a particle crossing an edge is folded back in (no probability lost) |
| `dirichlet` | absorb | a particle crossing an edge is removed from the ensemble - matching the PDE's Dirichlet condition, under which mass is free to leave through the boundary and never returns |

See
`docs/numerical_methods_notes.md` §2.7 for more details on why the BC of (1) translate as 
the above conditions at a particle level.

## Running the GUI

```
pip install -r requirements.txt
python run_gui.py
```

### Fields

Listed in the order they appear in the window.

- **Method**: Forward Euler, Backward Euler, or Euler-Maruyama.
- **Left/right domain bound**: type a number, or `-inf`/`+inf` for an
  unbounded domain. An infinite bound is automatically truncated to a large
  finite interval for computation, and the log states clearly that this
  happened. The truncation is then *verified*: if the density turns out not
  to be negligible at the artificial boundary, the open side is enlarged and
  the problem re-solved (see `fp1d/domain_truncation.py`).
- **Spatial step dx**: cell width. The number of cells follows from it and
  the domain length; if `dx` doesn't divide the length exactly, the next
  smaller `dx` that does is used and the log says so.
- **Total time T** / **Time step dt**: simulation length and step size. If
  `dt` doesn't divide `T` exactly it is likewise shrunk to the nearest value
  that does, so a run always ends exactly at `T`.
- **Coefficient source**: `expression` to type $b$ and $D$ by hand, or
  `file` to load them from a `.npz`. Whichever you pick, the other input is
  greyed out, so it is never ambiguous which one is in effect.
- **Drift** / **Diffusivity**: Python expressions in `x` (an array of cell
  centers) and `t`, with `numpy` available as `np` (e.g. `-x*np.ones(len(x))`
  or `0.2 + 0.1*np.sin(t)`). A scalar result is broadcast over the grid.
  Only `np` and a small set of builtins are exposed; the expression cannot
  import modules or touch the filesystem. $D$ must be non-negative.
- **Coefficient file (.npz)**: an alternative to typing the above (tabulated $b$ and $D$ sampled on their own $x$ (and optionally $t$) grid,
  interpolated onto whatever the solver asks for). See
  `coefficients/README.md` for the file format. **Refresh list** re-scans
  the folder if you drop a new file in while the window is open.
- **Trials**: number of particle trajectories (Euler-Maruyama only).
- **Boundary condition**: `periodic`, `dirichlet`, or `neumann`, all in
  their homogeneous form. `periodic` additionally requires $b$ and $D$ to
  agree at the two domain edges, otherwise the flux at the seam would be
  ambiguous, and the run stops with an error saying so.
- **Initial condition**: `gaussian`, `uniform`, `bimodal`, `left-block`, or
  `custom`, always renormalized to unit mass.
- **Custom expression**: used only when the initial condition above is
  `custom`. A Python expression in `x` under the same restrictions as the
  drift field, e.g. `np.exp(-0.5*(x/0.6)**2)`, and it must be non-negative.
- **Random seed**: reproducibility for the Euler-Maruyama scheme.
- **Saved output times**: comma-separated times at which the density is
  recorded, used for the snapshot plot, the mass history, and the saved
  `.npz`. `0` and `T` are always included. This field always follows `T`:
  editing `T` resets it to the default spacing `0, T/5, ..., T` and discards
  anything you had typed, so set `T` first and customise the output times
  after.

Two read-only displays sit below the fields: a **truncation preview**, which
updates as you type and shows the domain and `dx` the run would actually use,
and an **active coefficients** box confirming which $b$ and $D$ are currently
in effect.

### Buttons

**Show diagnostics**: reports the advective CFL number $\lambda$, the
diffusion number $\mu$, the combined stability number $\lambda + 2\mu$, the
Peclet number, and the suggested Forward-Euler $dt$ - all without running
anything. A large Peclet number is the signal that the numerical diffusion
introduced by upwinding is significant relative to the true $D$ on this
grid; `docs/numerical_methods_notes.md` §1.4 quantifies it.

**Run solver**: starts the computation in a background thread (the window
stays responsive). If Forward Euler is selected and $\lambda + 2\mu > 1$, it
warns first and lets you proceed anyway.

**Stop**: requests early termination. The run still saves whatever it
computed up to that point - but what "up to that point" means depends on the
method, because the two families traverse the problem in different orders.

- *Forward/Backward Euler* march one density forward in time, so progress is
  reported as completed steps and the simulated time reached, and stopping
  truncates the **time axis**: you get $p(x,t)$ up to the moment you stopped,
  at full accuracy.
- *Euler-Maruyama* simulates trajectories in batches, each carried from $0$
  all the way to $T$ before the next batch starts. There is therefore no
  single "current time" to report, and progress is shown as **completed
  trials**. Stopping costs **samples, not time**: every trajectory that
  finished ran the whole interval, so the saved density still covers all of
  $[0, T]$ - it is simply rougher, with visibly larger fluctuations, because
  Monte Carlo error shrinks only as $1/\sqrt{N}$ in the number of
  trajectories. `simulation_parameters.json` records `trials_completed`
  alongside the requested `trials_maruyama`, so the sample count behind any
  saved estimate is always recoverable.

Each run creates a timestamped folder under `output/` containing a snapshot
plot, a mass-history plot, the raw solution (`.npz`), the run parameters
(`.json`), and an animation (`.mp4`, or `.gif` if `ffmpeg` isn't available).
The animation plays back the frames the solver actually recorded during
integration - not an artificially interpolated smoothing of a few snapshots.

## Project layout

```
Fokker-Planck Solver/
  run_gui.py                       entry point: launches the GUI
  requirements.txt                 runtime dependencies
  requirements-dev.txt             test dependencies
  fokker_planck_experiments.ipynb  the three solvers vs. exact solutions
  docs/
    numerical_methods_notes.md     full derivations behind this README
  coefficients/
    README.md                      the .npz coefficient file format
    make_examples.py               regenerates the example files below
    ou_static.npz                  b(x) = -x, D(x) = 1
    ou_breathing.npz               time-dependent b, static D
    varying_diffusion.npz          b = 0, D peaked at the center
  fp1d/                            the package
  tests/                           pytest suite
  output/                          created at runtime, one folder per run
```

### The `fp1d` package

Grouped by role rather than alphabetically, since that is roughly the order
in which a run passes through them.

**Defining the problem**

| File | What it holds                                                                                                                                                                                               |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `grid.py` | `Grid1D`, the uniform cell-centered grid (cell centers, faces, `dx`); `make_grid`, which builds one from a requested `dx`; and `finite_domain`, which substitutes finite numbers for `+/-inf` bounds        |
| `boundary_conditions.py` | `BoundaryCondition`, a one-field record naming the condition. Only the homogeneous form of each is supported                                                                                                |
| `initial_conditions.py` | The four presets and the sandboxed custom-expression evaluator, all normalized to unit mass, plus `sample_particles_from_density` so the SDE solver starts from the *same* state as the PDE solver          |
| `coefficients_io.py` | Loads $b$ and $D$ from a `.npz` file and wraps them as `(x, t)` callables, interpolating and clamping rather than extrapolating outside the saved range  |

**Solving**

| File | What it holds                                                                                                                                                                                                                                                                                                                               |
|---|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `finite_volume.py` | The Eulerian solver. `assemble_operator` builds the sparse $A$ in $\dot p = Ap$ face by face (vectorized, then summed into a `coo_matrix`); `forward_euler` and `backward_euler` share one time-stepping loop. When $b$ and $D$ don't depend on $t$, the operator (and, for the implicit scheme, its factorization) is built once and reused |
| `stochastic_solver.py` | The Lagrangian solver: Euler-Maruyama on the equivalent SDE, with the density recovered as a normalized histogram. Boundaries become particle rules (wrap, reflect, absorb). Trajectories run in batches, each taken from $0$ to $T$, which is what lets an interrupted run still describe the whole interval - from fewer samples                                                         |
| `domain_truncation.py` | Wraps either solver when a bound was `+/-inf`: solve, check whether the density at the artificial boundary is negligible, enlarge the open side(s) and re-solve if not                                                                                                                    |

**Shared machinery**

| File | What it holds |
|---|---|
| `results.py` | `SolverResult`, the single return type all three integrators produce, plus `discrete_mass` and two recorders: `SolutionRecorder` for solvers that advance in time, and `EnsembleRecorder`, which sums histogram counts over batches of trajectories so a Monte Carlo estimate is available at any point|
| `diagnostics.py` | The CFL, diffusion, combined-stability and Peclet numbers, computed pointwise over the run rather than from separately-maximized coefficients. Also the periodic-consistency check and the time-dependence probe the solvers use to decide what can be cached |

**Output and driver**

| File | What it holds |
|---|---|
| `visualization.py` | Snapshot overlay, mass-history plot, animation (mp4 with a gif fallback), and the raw `.npz` dump. Forces matplotlib's `Agg` backend, since plotting happens on a worker thread |
| `io_utils.py` | Timestamped run directories and JSON writing |
| `gui.py` | The Tkinter application. |

A run therefore flows: `gui` builds a `RunConfig` → `domain_truncation`
drives one or more attempts → each attempt builds a grid and initial
condition and calls `finite_volume` or `stochastic_solver` → the result comes
back as a `SolverResult` → `visualization` and `io_utils` write it to disk.

## Tests

```
pip install -r requirements-dev.txt
pytest tests/ -v
```

Various tests, each file covering one property:

| File | What it pins down |
|---|---|
| `test_conservation.py` | Mass is conserved *exactly* under periodic and reflecting boundaries and decays under an absorbing one |
| `test_time_stepping.py` | A run ends at `T` for any `dt`, including one that doesn't divide it. |
| `test_dirichlet_regression.py` | Forward and backward Euler agree where the boundary is doing real work, and the boundary flux converges under grid refinement with $D$ varying at the wall |
| `test_domain_truncation.py` | The adaptive-domain verdict doesn't depend on which output times were requested, and enlargement works for an initial condition far from the origin |
| `test_pde_sde_consistency.py` | The two solvers agree on absorbed mass, and on the spread produced by a spatially varying $D$ |
| `test_stationary_distribution.py` | An Ornstein-Uhlenbeck process relaxes to the right mean, and the error in the recovered stationary variance shrinks under grid refinement |
| `test_initial_conditions.py` | Presets and custom expressions normalize to unit mass, sampled particles reproduce the density, and infinite bounds are substituted correctly |
| `test_stop_event.py` | All three solvers honor an early-stop request and report it, including that an interrupted Euler-Maruyama run still spans $[0, T]$ and equals a full run with that many trials |
