"""Static plots and an animation built directly from simulated frames.

The solvers record a frame every few steps during the run (see `finite_volume._run` and
`stochastic_solver.euler_maruyama`); the animation just plays those back.
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # headless, thread-safe: solvers run in a worker
# thread while the GUI's own Tk mainloop runs on the main thread, and
# only ever needs matplotlib to save files, never to display a window.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter


def plot_snapshots(x, save_times, snapshots, filename):
    """Overlay the density p(x, t) at each requested save time."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    for t, p in zip(save_times, snapshots):
        ax.plot(x, p, label=f't = {t:g}')
    ax.set_xlabel('x')
    ax.set_ylabel('density')
    ax.set_title('Density snapshots')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, dpi=170)
    plt.close(fig)


def plot_mass_history(save_times, masses, filename):
    """Plot total probability mass, integral of p(x,t) dx, versus time.

    Mass should stay pinned at 1 for periodic/Neumann (no-flux) boundary
    conditions, since no probability can enter or leave the domain, and
    should visibly decay for a Dirichlet (absorbing) boundary. The plot is
    built to make both of those stories legible at a glance: an explicit
    reference line at the ideal starting value, axis labels that name the
    quantity rather than just "mass", and y-limits chosen so a flat,
    conserved line doesn't get exaggerated into a dramatic-looking wiggle
    by an overly tight axis range.
    """
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    save_times = np.asarray(save_times, dtype=float)
    masses = np.asarray(masses, dtype=float)
    initial_mass = float(masses[0]) if masses.size else 1.0

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(save_times, masses, marker='o', ms=4, lw=1.8, color='tab:blue',
            label='Total probability mass, $\\int p(x,t)\\,dx$')
    ax.axhline(initial_mass, color='k', ls='--', lw=1,
               label=f'Initial mass = {initial_mass:.4g}')

    ax.set_xlabel('Time $t$')
    ax.set_ylabel('Total probability mass')
    ax.set_title('Total probability mass vs. time')

    # Give the y-axis some breathing room so small (or zero) deviations
    # from the initial mass are still visibly flat, not an artifact of an
    # axis range fitted tightly to noise.
    spread = max(float(np.ptp(masses)), 1e-6 * max(initial_mass, 1.0))
    pad = 0.15 * spread + 0.02 * max(initial_mass, 1.0)
    ax.set_ylim(min(float(np.min(masses)), initial_mass) - pad,
                max(float(np.max(masses)), initial_mass) + pad)

    max_dev = float(np.max(np.abs(masses - initial_mass))) if masses.size else 0.0
    ax.text(0.02, 0.03, f'Max deviation from initial mass: {max_dev:.2e}',
            transform=ax.transAxes, fontsize=9, color='dimgray',
            va='bottom', ha='left')

    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', framealpha=0.9)
    fig.tight_layout()
    fig.savefig(filename, dpi=170)
    plt.close(fig)


def save_solution_npz(x, save_times, snapshots, masses, filename):
    """Dump the raw arrays behind the plots above, for offline analysis."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    np.savez(filename, x=np.asarray(x), times=np.asarray(save_times),
              density=np.asarray(snapshots), mass=np.asarray(masses))


def save_animation(x, frame_times, frames, filename_prefix, fps=20):
    """Animate the solver's own recorded frames; mp4 if ffmpeg is available,
    otherwise a gif. Returns the path actually written.
    """
    Path(filename_prefix).parent.mkdir(parents=True, exist_ok=True)
    n_frames = len(frame_times)
    ymax = max(1e-12, float(np.max(frames)) * 1.05)

    fig, ax = plt.subplots(figsize=(8, 5))
    line, = ax.plot([], [], lw=2)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(0.0, ymax)
    ax.set_xlabel('x')
    ax.set_ylabel('density')
    ax.grid(True, alpha=0.3)
    title = ax.set_title('')

    def update(i):
        line.set_data(x, frames[i])
        title.set_text(f't = {frame_times[i]:.5g}')
        return line, title

    anim = FuncAnimation(fig, update, init_func=lambda: update(0),
                          frames=n_frames, interval=1000 / fps, blit=True)

    mp4_file = str(filename_prefix) + '.mp4'
    try:
        anim.save(mp4_file, writer=FFMpegWriter(fps=fps, bitrate=1800))
        out = mp4_file
    except Exception:
        out = str(filename_prefix) + '.gif'
        anim.save(out, writer=PillowWriter(fps=fps))
    plt.close(fig)
    return out
