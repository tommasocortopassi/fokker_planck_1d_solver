import numpy as np
from dataclasses import dataclass
from pathlib import Path
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from .api import solve
from .grid import finite_domain, make_grid, parse_bound
from .boundary_conditions import BoundaryCondition
from .diagnostics import preflight_warnings, sample_diagnostics
from .initial_conditions import build_initial_density, safe_eval
from .coefficients_io import list_coefficient_files, load_coefficient_set


# Where coefficient .npz files live: a 'coefficients' folder next to
# run_gui.py, i.e. a sibling of the fp1d package itself (this file is
# fp1d/gui.py, so its parent's parent is the project root).
COEFFICIENTS_DIR = Path(__file__).resolve().parent.parent / 'coefficients'


# Shown next to the live truncation preview whenever a domain bound is
# +/-inf, so the user knows the number displayed there is only a starting
# guess - the actual run (see domain_truncation.py) may enlarge it further
# based on the computed solution.
INFINITE_DOMAIN_PREVIEW_NOTE = (
    'Preview estimate only; the run will enlarge this further if needed.'
)


@dataclass
class RunConfig:
    """Everything the worker thread needs, read from Tkinter variables on
    the main thread only. Tkinter/Tcl calls are not thread-safe, so the
    background thread must never touch `self.vars` directly.

    `left`/`right` are numeric starting guesses for the domain bounds -
    for a side the user set to a finite number, this *is* that number;
    for a side set to +/-inf, it's a heuristic estimate that
    `domain_truncation.solve_with_adaptive_domain` will grow if needed.
    The grid itself (which depends on `dx` too, and may change size
    during that adaptive growth) is only built once solving actually
    starts, not here.
    """
    method: str
    dx: float
    left: float
    right: float
    left_is_infinite: bool
    right_is_infinite: bool
    total_time: float
    dt: float
    save_times: list
    bc: BoundaryCondition
    drift: callable
    diffusion: callable
    coeff_description: str
    trunc_msg: str
    initial_condition_name: str
    custom_expr: str
    trials: int
    seed: int


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('1D Fokker-Planck Solver')
        self.geometry('1020x980')
        self.worker = None
        self.stop_event = None
        self.progress_queue = queue.Queue()

        self.vars = {
            'method': tk.StringVar(value='Forward Euler'),
            'left': tk.StringVar(value='-5'),
            'right': tk.StringVar(value='5'),
            'dx': tk.StringVar(value='0.1'),
            'T': tk.StringVar(value='10'),
            'dt': tk.StringVar(value='0.0005'),
            'trials': tk.StringVar(value='10000'),
            'save': tk.StringVar(value='0, 2, 4, 6, 8, 10'),
            'bc_type': tk.StringVar(value='neumann'),
            'coeff_source': tk.StringVar(value='expression'),
            'drift': tk.StringVar(value='-x*np.ones(len(x))'),
            'diffusivity': tk.StringVar(value='np.ones(len(x))'),
            'coeff_file': tk.StringVar(value=''),
            'initial_condition': tk.StringVar(value='gaussian'),
            'custom_expr': tk.StringVar(value='np.exp(-0.5*(x/0.6)**2)'),
            'seed': tk.StringVar(value='12345'),
        }

        # The save-times field always follows T: any edit to T regenerates
        # the default spacing, discarding whatever was in the field. Editing
        # T means asking a different question, and output times carried over
        # from the previous one are more likely to be stale than intended -
        # several of them may not even lie in the new interval. Pick T
        # first, then the output times.
        self.vars['T'].trace_add('write', self.update_save_from_T)
        self.update_save_from_T()

        for key in ('drift', 'diffusivity', 'T', 'left', 'right', 'dx',
                    'initial_condition', 'custom_expr', 'coeff_source', 'coeff_file'):
            self.vars[key].trace_add('write', self.recompute_truncation)

        self.computed_left = float(self.vars['left'].get())
        self.computed_right = float(self.vars['right'].get())

        frame = ttk.Frame(self, padding=12)
        frame.pack(fill='both', expand=True)

        items = [
            ('Method', 'method', ['Forward Euler', 'Backward Euler', 'Euler-Maruyama']),
            ('Left domain bound', 'left', None),
            ('Right domain bound', 'right', None),
            ('Spatial step dx', 'dx', None),
            ('Total time T', 'T', None),
            ('Time step dt', 'dt', None),
            ('Coefficient source', 'coeff_source', ['expression', 'file']),
        ]
        row = 0
        for label, key, values in items:
            ttk.Label(frame, text=label).grid(row=row, column=0, sticky='w', pady=4)
            if values is None:
                widget = ttk.Entry(frame, textvariable=self.vars[key], width=42)
            else:
                widget = ttk.Combobox(frame, textvariable=self.vars[key], values=values,
                                      state='readonly', width=39)
            widget.grid(row=row, column=1, sticky='ew', pady=4)
            row += 1

        # Drift/diffusivity: either typed expressions or a loaded .npz
        # file, toggled by 'coeff_source' above. Both sets of widgets are
        # always present, but only the relevant one is enabled at a time
        # (see `_update_coefficient_widget_state`), so it's never
        # ambiguous which one is actually in effect.
        ttk.Label(frame, text='Drift').grid(row=row, column=0, sticky='w', pady=4)
        self.drift_entry = ttk.Entry(frame, textvariable=self.vars['drift'], width=42)
        self.drift_entry.grid(row=row, column=1, sticky='ew', pady=4)
        row += 1

        ttk.Label(frame, text='Diffusivity').grid(row=row, column=0, sticky='w', pady=4)
        self.diffusivity_entry = ttk.Entry(frame, textvariable=self.vars['diffusivity'], width=42)
        self.diffusivity_entry.grid(row=row, column=1, sticky='ew', pady=4)
        row += 1

        ttk.Label(frame, text='Coefficient file (.npz)').grid(row=row, column=0, sticky='w', pady=4)
        coeff_file_frame = ttk.Frame(frame)
        coeff_file_frame.grid(row=row, column=1, sticky='ew', pady=4)
        self.coeff_file_combo = ttk.Combobox(coeff_file_frame, textvariable=self.vars['coeff_file'],
                                             values=list_coefficient_files(COEFFICIENTS_DIR),
                                             state='readonly', width=30)
        self.coeff_file_combo.pack(side='left', fill='x', expand=True)
        ttk.Button(coeff_file_frame, text='Refresh list',
                   command=self.refresh_coefficient_file_list).pack(side='left', padx=(6, 0))
        row += 1

        for remaining_label, remaining_key, remaining_values in [
            ('Trials (Euler-Maruyama only)', 'trials', None),
            ('Boundary condition', 'bc_type', ['periodic', 'dirichlet', 'neumann']),
            ('Initial condition', 'initial_condition',
             ['gaussian', 'uniform', 'bimodal', 'left-block', 'custom']),
            ('Custom expression (if selected)', 'custom_expr', None),
            ('Random seed', 'seed', None),
            ('Saved output times', 'save', None),
        ]:
            ttk.Label(frame, text=remaining_label).grid(row=row, column=0, sticky='w', pady=4)
            if remaining_values is None:
                widget = ttk.Entry(frame, textvariable=self.vars[remaining_key], width=42)
            else:
                widget = ttk.Combobox(frame, textvariable=self.vars[remaining_key], values=remaining_values,
                                      state='readonly', width=39)
            widget.grid(row=row, column=1, sticky='ew', pady=4)
            row += 1

        self.truncation_label = ttk.Label(frame, text='', foreground='gray', wraplength=420)
        self.truncation_label.grid(row=row, column=0, columnspan=2, sticky='w', pady=(0, 4))
        row += 1

        ttk.Label(frame, text='Active coefficients').grid(row=row, column=0, sticky='nw', pady=4)
        self.coeff_info = tk.Text(frame, height=4, width=76)
        self.coeff_info.grid(row=row, column=1, sticky='ew', pady=4)
        row += 1

        btns = ttk.Frame(frame)
        btns.grid(row=row, column=0, columnspan=2, sticky='w', pady=8)
        ttk.Button(btns, text='Show diagnostics', command=self.show_diagnostics).pack(side='left', padx=(0, 8))
        self.run_button = ttk.Button(btns, text='Run solver', command=self.start_solver_thread)
        self.run_button.pack(side='left', padx=(0, 8))
        self.stop_button = ttk.Button(btns, text='Stop', command=self.request_stop, state='disabled')
        self.stop_button.pack(side='left')
        row += 1

        self.progress = ttk.Progressbar(frame, mode='determinate', maximum=100)
        self.progress.grid(row=row, column=0, columnspan=2, sticky='ew', pady=6)
        row += 1

        ttk.Label(frame, text='Status log').grid(row=row, column=0, sticky='nw', pady=4)
        self.log = tk.Text(frame, height=28, width=76)
        self.log.grid(row=row, column=1, sticky='nsew', pady=4)
        frame.columnconfigure(1, weight=1)
        frame.rowconfigure(row, weight=1)

        self.refresh_coefficients_text()
        for key in ('drift', 'diffusivity', 'coeff_source', 'coeff_file'):
            self.vars[key].trace_add('write', lambda *args: self.refresh_coefficients_text())
        self.vars['coeff_source'].trace_add('write', self._update_coefficient_widget_state)
        self.refresh_coefficient_file_list()
        self._update_coefficient_widget_state()
        self.recompute_truncation()
        self.after(150, self.poll_progress)


    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------

    def update_save_from_T(self, *args):
        """Reset the save-times field to the default spacing for the
        current T. Registered as a trace on T, so it runs on every edit
        and overwrites anything already in the field.
        """
        try:
            T = float(self.vars['T'].get())
        except ValueError:
            return

        values = [0, T / 5, 2 * T / 5, 3 * T / 5, 4 * T / 5, T]

        def fmt(v):
            if float(v).is_integer():
                return str(int(v))
            return f"{v:g}"

        self.vars['save'].set(', '.join(fmt(v) for v in values))

    def append_log(self, text):
        self.log.insert('end', text + '\n')
        self.log.see('end')

    def _update_coefficient_widget_state(self, *args):
        """Enable exactly the drift/diffusivity input that 'coeff_source'
        currently selects, and disable the other - so it's never
        ambiguous (to the user, or to the code) which one is actually in
        effect.
        """
        using_file = self.vars['coeff_source'].get() == 'file'
        self.drift_entry.config(state='disabled' if using_file else 'normal')
        self.diffusivity_entry.config(state='disabled' if using_file else 'normal')
        self.coeff_file_combo.config(state='readonly' if using_file else 'disabled')

    def refresh_coefficient_file_list(self):
        """Re-scan the coefficients directory for .npz files (e.g. after
        the user drops a new one in while the GUI is already open) and
        repopulate the file-choice dropdown.
        """
        files = list_coefficient_files(COEFFICIENTS_DIR)
        self.coeff_file_combo.config(values=files)
        if files and self.vars['coeff_file'].get() not in files:
            self.vars['coeff_file'].set(files[0])
        if not files:
            self.append_log(f'No .npz files found in {COEFFICIENTS_DIR}.')

    def build_coefficients(self):
        """Return `(drift, diffusion, description)` as `(x, t=0.0)`
        callables, from either the typed Python expressions or a loaded
        .npz file - whichever 'coeff_source' currently selects. This is
        the single place that turns GUI state into actual coefficient
        functions, used by `estimate_open_bounds` (the truncation
        heuristic), `refresh_coefficients_text` (the status display), and
        `parse_inputs` (the real run), so all three always agree on what
        the coefficients actually are.
        """
        if self.vars['coeff_source'].get() == 'file':
            filename = self.vars['coeff_file'].get()
            if not filename:
                raise ValueError('No coefficient file selected.')
            path = COEFFICIENTS_DIR / filename
            try:
                drift, diffusion, info = load_coefficient_set(path)
            except FileNotFoundError:
                raise ValueError(f"Coefficient file not found: {path}") from None
            t_part = (f", t in [{info['t_range'][0]:.4g}, {info['t_range'][1]:.4g}]"
                      if info['t_range'] is not None else ', static in t')
            description = (
                f"Loaded from {info['file']} "
                f"(x in [{info['x_range'][0]:.4g}, {info['x_range'][1]:.4g}]{t_part}).\n"
                f"drift is {'time-dependent' if info['drift_time_dependent'] else 'static'}; "
                f"diffusivity is {'time-dependent' if info['diffusion_time_dependent'] else 'static'}."
            )
            return drift, diffusion, description

        drift_expr = self.vars['drift'].get()
        diffusivity_expr = self.vars['diffusivity'].get()

        def evaluate(expr, what):
            def coefficient(x, t=0.0):
                val = safe_eval(expr, {'x': x, 't': t}, what)
                arr = np.asarray(val, dtype=float)
                if arr.shape == ():
                    arr = np.full_like(np.asarray(x, dtype=float), float(arr))
                return arr

            return coefficient

        drift = evaluate(drift_expr, 'drift')
        diffusion = evaluate(diffusivity_expr, 'diffusivity')

        description = f"drift = {drift_expr}\ndiffusivity = {diffusivity_expr}"
        return drift, diffusion, description

    def refresh_coefficients_text(self):
        try:
            _, _, description = self.build_coefficients()
        except Exception as exc:
            description = f'(invalid coefficients: {exc})'
        self.coeff_info.delete('1.0', 'end')
        self.coeff_info.insert('1.0', description)

    # ------------------------------------------------------------------
    # Domain-truncation preview (fast heuristic; the real, adaptive
    # truncation happens only once the solver actually runs - see
    # domain_truncation.py)
    # ------------------------------------------------------------------

    def estimate_ic_support(self, left_val, right_val, tol=1e-6):
        """Roughly locate where the initial condition is non-negligible,
        on a finite probe domain (using a fallback half-width if a side
        is +/-inf), by evaluating it on a reasonably fine auxiliary grid.
        """
        fallback = 10.0
        probe_left = left_val if np.isfinite(left_val) else -fallback
        probe_right = right_val if np.isfinite(right_val) else fallback

        requested_dx = float(self.vars['dx'].get())
        # Use a probe at least this fine regardless of the requested dx,
        # so a coarse user-chosen step doesn't hide a narrow initial
        # condition from this estimate.
        probe_dx = min(requested_dx, (probe_right - probe_left) / 200.0)
        probe_grid, _ = make_grid(probe_left, probe_right, probe_dx)

        ic_name = self.vars['initial_condition'].get()
        custom_expr = self.vars['custom_expr'].get()
        p0 = build_initial_density(ic_name, probe_grid, custom_expr)

        x = probe_grid.centers
        peak = float(np.max(np.abs(p0)))
        if peak <= 0:
            return probe_left, probe_right

        mask = np.abs(p0) >= tol * peak
        if not np.any(mask):
            return probe_left, probe_right

        return float(x[mask].min()), float(x[mask].max())

    def estimate_open_bounds(self, left_val, right_val, T=None):
        """Estimate how far the density could plausibly spread beyond the
        initial condition's support over the run, from the peak drift and
        diffusion magnitudes: `spread ~ 2*(|b|_max * T + sqrt(D_max * T))`,
        and return the resulting `(left_guess, right_guess)` bounds.

        These are plain x coordinates, ready to hand to
        `grid.finite_domain`, and they are only ever a *starting guess*
        for a truncated domain - see domain_truncation.py for how they're
        verified and grown if needed.

        Note the estimate samples b and D at t=0 only, so a coefficient
        that strengthens later in the run gets an under-estimate here;
        the adaptive loop is what actually covers that case.
        """
        if T is None:
            T = float(self.vars['T'].get())

        x_min_support, x_max_support = self.estimate_ic_support(left_val, right_val)

        x_in = left_val if np.isfinite(left_val) else x_min_support
        x_end = right_val if np.isfinite(right_val) else x_max_support

        requested_dx = float(self.vars['dx'].get())
        span = (x_end + 1.0) - (x_in - 1.0)
        n_probe = max(int(span / max(requested_dx, 1e-9)), 50)
        x_probe = np.linspace(x_in - 1.0, x_end + 1.0, n_probe)

        drift, diffusion, _ = self.build_coefficients()

        try:
            B = np.asarray(drift(x_probe, 0.0), dtype=float)
        except Exception as exc:
            raise ValueError(f'Could not evaluate drift: {exc}') from exc
        try:
            D = np.asarray(diffusion(x_probe, 0.0), dtype=float)
        except Exception as exc:
            raise ValueError(f'Could not evaluate diffusivity: {exc}') from exc

        if B.shape != x_probe.shape:
            raise ValueError('Drift must return one value per point in x.')
        if D.shape != x_probe.shape:
            raise ValueError('Diffusivity must return one value per point in x.')
        if np.any(~np.isfinite(B)):
            raise ValueError('Drift produced non-finite values.')
        if np.any(~np.isfinite(D)):
            raise ValueError('Diffusivity produced non-finite values.')
        if np.any(D < 0):
            raise ValueError('Diffusivity must be non-negative on the probed domain.')

        B_max = float(np.max(np.abs(B)))
        D_max = float(np.max(D))
        spread = 2 * (B_max * T + np.sqrt(D_max * T))
        return x_min_support - spread, x_max_support + spread

    def recompute_truncation(self, *args):
        """Live preview shown under the domain fields as the user types:
        a fast, heuristic-only estimate of the domain that will be used,
        and of how `dx` will be adjusted to tile it exactly. Runs on every
        keystroke, so it must never actually invoke a solver - that only
        happens in `run_solver_worker`, via the adaptive loop in
        domain_truncation.py.
        """
        left_text = self.vars['left'].get()
        right_text = self.vars['right'].get()

        try:
            left_val = parse_bound(left_text)
            right_val = parse_bound(right_text)
            requested_dx = float(self.vars['dx'].get())
        except ValueError:
            return

        if np.isfinite(left_val) and np.isfinite(right_val):
            self.computed_left, self.computed_right = left_val, right_val
            try:
                _, dx_msg = make_grid(left_val, right_val, requested_dx)
            except Exception as exc:
                if hasattr(self, 'truncation_label'):
                    self.truncation_label.config(text=str(exc), foreground='red')
                return
            if hasattr(self, 'truncation_label'):
                self.truncation_label.config(text=dx_msg, foreground='gray')
            return

        try:
            left_guess, right_guess = self.estimate_open_bounds(left_val, right_val)
        except ValueError as exc:
            if hasattr(self, 'truncation_label'):
                self.truncation_label.config(text=str(exc), foreground='red')
            return
        except Exception as exc:
            if hasattr(self, 'truncation_label'):
                self.truncation_label.config(text=f'Could not recompute truncation: {exc}', foreground='red')
            return

        new_left, new_right, msg = finite_domain(left_text, right_text, left_guess, right_guess)
        self.computed_left, self.computed_right = new_left, new_right
        try:
            _, dx_msg = make_grid(new_left, new_right, requested_dx)
        except Exception:
            dx_msg = ''
        full_msg = ' '.join(m for m in (msg, dx_msg, INFINITE_DOMAIN_PREVIEW_NOTE) if m)
        if hasattr(self, 'truncation_label'):
            self.truncation_label.config(text=full_msg, foreground='gray')

    # ------------------------------------------------------------------
    # Turning GUI state into a RunConfig
    # ------------------------------------------------------------------

    def parse_inputs(self) -> RunConfig:
        self.refresh_coefficients_text()

        left_text, right_text = self.vars['left'].get(), self.vars['right'].get()
        left_val, right_val = parse_bound(left_text), parse_bound(right_text)

        requested_dx = float(self.vars['dx'].get())
        if requested_dx <= 0:
            raise ValueError('dx must be positive.')

        left_is_infinite = not np.isfinite(left_val)
        right_is_infinite = not np.isfinite(right_val)

        if left_is_infinite or right_is_infinite:
            left_guess, right_guess = self.estimate_open_bounds(left_val, right_val)
            left, right, trunc_msg = finite_domain(left_text, right_text, left_guess, right_guess)
        else:
            left, right, trunc_msg = left_val, right_val, ''

        total_time = float(self.vars['T'].get())
        dt = float(self.vars['dt'].get())
        if total_time <= 0 or dt <= 0:
            raise ValueError('T and dt must be positive.')
        save_times = sorted(set(float(s.strip()) for s in self.vars['save'].get().split(',') if s.strip()))
        save_times = sorted(set([0.0, total_time] + [t for t in save_times if 0.0 <= t <= total_time]))

        bc = BoundaryCondition(self.vars['bc_type'].get())
        bc.validate()
        ic_name = self.vars['initial_condition'].get()
        custom_expr = self.vars['custom_expr'].get()

        drift, diffusion, coeff_description = self.build_coefficients()

        return RunConfig(
            method=self.vars['method'].get(),
            dx=requested_dx,
            left=left,
            right=right,
            left_is_infinite=left_is_infinite,
            right_is_infinite=right_is_infinite,
            total_time=total_time,
            dt=dt,
            save_times=save_times,
            bc=bc,
            drift=drift,
            diffusion=diffusion,
            coeff_description=coeff_description,
            trunc_msg=trunc_msg,
            initial_condition_name=ic_name,
            custom_expr=custom_expr,
            trials=int(self.vars['trials'].get()),
            seed=int(self.vars['seed'].get()),
        )

    # ------------------------------------------------------------------
    # Diagnostics (instant - never runs a solver)
    # ------------------------------------------------------------------

    def show_diagnostics(self):
        try:
            cfg = self.parse_inputs()
            grid, dx_msg = make_grid(cfg.left, cfg.right, cfg.dx)
            d = sample_diagnostics(grid, cfg.drift, cfg.diffusion, cfg.dt, cfg.total_time)
            self.log.delete('1.0', 'end')
            if cfg.trunc_msg:
                self.append_log(cfg.trunc_msg)
            if cfg.left_is_infinite or cfg.right_is_infinite:
                self.append_log(f'Note: {INFINITE_DOMAIN_PREVIEW_NOTE}')
            if dx_msg:
                self.append_log(dx_msg)
            self.append_log(f'Boundary condition: {cfg.bc.kind}')
            self.append_log(f'Coefficients: {cfg.coeff_description}')
            self.append_log(f'Grid: dx = {grid.dx:.4g}, {grid.ncells} cells, '
                             f'domain = [{grid.left:.4g}, {grid.right:.4g}]')
            self.append_log(f'Max |b(x,t)|: {d["max_abs_b"]:.4g}')
            self.append_log(f'Max D(x,t): {d["max_D"]:.4g}')
            self.append_log(f'Advective CFL number (lambda): {d["lambda"]:.4g}')
            self.append_log(f'Diffusion number (mu): {d["mu"]:.4g}')
            stability_note = 'stable' if d['stable'] else 'UNSTABLE for Forward Euler'
            self.append_log(f'Combined stability number, lambda + 2*mu: '
                             f'{d["combined_CFL"]:.4g} ({stability_note})')
            self.append_log(f'Max Peclet number: {d["Peclet"]:.4g}')
            self.append_log(f'Suggested Forward-Euler dt <= {d["dt_suggested"]:.4g}')
        except Exception as exc:
            messagebox.showerror('Input error', str(exc))

    # ------------------------------------------------------------------
    # Running the solver
    # ------------------------------------------------------------------

    def start_solver_thread(self):
        if self.worker is not None and self.worker.is_alive():
            messagebox.showinfo('Solver running', 'A computation is already in progress.')
            return
        try:
            cfg = self.parse_inputs()
            grid, _ = make_grid(cfg.left, cfg.right, cfg.dx)
            # Same checks `api.solve` runs, so the window and a script
            # warn about the same things. Conditions with no meaningful
            # "proceed anyway" - a periodic seam whose coefficients
            # disagree - raise from in here and land in the dialog below
            # as an input error, before any work starts.
            pre_warnings, _ = preflight_warnings(
                grid, cfg.drift, cfg.diffusion, cfg.dt, cfg.total_time,
                cfg.method.lower(), cfg.bc.kind)
        except Exception as exc:
            messagebox.showerror('Input error', str(exc))
            return
        if pre_warnings:
            body = '\n\n'.join(pre_warnings)
            if not messagebox.askyesno('Warning', f'{body}\n\nProceed anyway?'):
                return

        self.stop_event = threading.Event()
        self.progress['value'] = 0
        self.log.delete('1.0', 'end')
        self.append_log('Starting solver...')
        self.run_button.config(state='disabled')
        self.stop_button.config(state='normal')
        self.worker = threading.Thread(target=self.run_solver_worker, args=(cfg,), daemon=True)
        self.worker.start()

    def request_stop(self):
        if self.stop_event is not None:
            self.stop_event.set()
            self.append_log('Stop requested: finishing the current step...')
            self.stop_button.config(state='disabled')

    def run_solver_worker(self, cfg: RunConfig):
        """Run one simulation off the main thread and report back.

        The computation itself lives in `api.solve`, which this method
        and any script share; everything added here is presentation -
        forwarding progress to the queue that `poll_progress` drains,
        and formatting the summary. Nothing in this method may touch a
        Tkinter widget: `cfg` was built on the main thread precisely so
        this one never has to.
        """
        try:
            def progress_callback(step, total, current_time):
                self.progress_queue.put(('progress', step, total, current_time))

            def log(msg):
                self.progress_queue.put(('message', msg))

            if cfg.trunc_msg:
                log(cfg.trunc_msg)

            run = solve(
                method=cfg.method, drift=cfg.drift, diffusion=cfg.diffusion,
                left=cfg.left, right=cfg.right,
                left_is_infinite=cfg.left_is_infinite,
                right_is_infinite=cfg.right_is_infinite,
                dx=cfg.dx, total_time=cfg.total_time, dt=cfg.dt,
                save_times=cfg.save_times, bc=cfg.bc,
                initial_condition=cfg.initial_condition_name,
                custom_expr=cfg.custom_expr,
                trials=cfg.trials, seed=cfg.seed,
                coefficient_description=cfg.coeff_description,
                # Already asked, in `start_solver_thread`, on the main
                # thread where a dialog is legal. Prompting again from
                # here would block a worker thread the user cannot see.
                on_warning='ignore',
                progress=progress_callback, stop_event=self.stop_event,
                log=log)

            result, grid, params = run.result, run.grid, run.parameters
            files = run.save('output', log=log)

            lines = []
            if result.stopped_early:
                if result.trials_completed is not None:
                    lines.append(
                        f'Run stopped early: {result.trials_completed} of '
                        f'{cfg.trials} trials completed. The density still covers '
                        f'all of [0, {cfg.total_time:.5g}] - each finished '
                        f'trajectory ran the full interval - but is rougher, '
                        f'since Monte Carlo error grows as the sample count '
                        f'falls (roughly like 1/sqrt(N)).')
                else:
                    lines.append(f'Run stopped early by user at t = {result.final_time:.5g}.')
            lines.extend([
                f'Method used: {cfg.method}',
                f'Domain used: [{grid.left:.6g}, {grid.right:.6g}] '
                f'(dx = {grid.dx:.6g}, {grid.ncells} cells)',
                f'Boundary condition: {cfg.bc.kind}',
                f'Coefficients: {cfg.coeff_description}',
                f'Initial mass: {result.masses[0]:.10f}',
                f'Final mass: {result.masses[-1]:.10f}',
                (f'Absorbed fraction: {params["absorbed_mass_fraction"]:.4g}'
                 if cfg.bc.kind == 'dirichlet'
                 else f'Max |mass-1|: {params["max_mass_deviation_from_1"]:.3e}'),
                f'Run folder: {files["directory"].resolve()}',
                f'Snapshot plot: {files["snapshots"].resolve()}',
                f'Mass plot: {files["mass_history"].resolve()}',
                f'Solution file: {files["solution"].resolve()}',
                f'Parameters file: {files["parameters"].resolve()}',
                f'Animation: {files["animation"].resolve()}',
            ])
            self.progress_queue.put(('done', lines))
        except Exception as exc:
            self.progress_queue.put(('error', str(exc)))

    def poll_progress(self):
        while True:
            try:
                item = self.progress_queue.get_nowait()
            except queue.Empty:
                break
            kind = item[0]
            if kind == 'progress':
                done, total, current_time = item[1], item[2], item[3]
                self.progress['value'] = 100.0 * done / total
                # `current_time` is None for Euler-Maruyama. That solver
                # runs whole trajectories one batch at a time, so it has no
                # single "current time" to report - at any instant it holds
                # realizations spanning all of [0, T]. Completed trials are
                # the honest measure of how far along it is.
                if current_time is None:
                    self.append_log(f'Completed trials: {done}/{total}')
                else:
                    self.append_log(f'Progress: {done}/{total}, simulated time = {current_time:.5g}')
            elif kind == 'message':
                self.append_log(item[1])
            elif kind == 'done':
                self.run_button.config(state='normal')
                self.stop_button.config(state='disabled')
                self.progress['value'] = 100
                for line in item[1]:
                    self.append_log(line)
                messagebox.showinfo('Completed', 'Computation finished.')
            elif kind == 'error':
                self.run_button.config(state='normal')
                self.stop_button.config(state='disabled')
                messagebox.showerror('Run error', item[1])
        self.after(150, self.poll_progress)


def main():
    App().mainloop()


if __name__ == '__main__':
    main()
