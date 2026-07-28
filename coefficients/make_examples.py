"""Generate a couple of example coefficient .npz files in this folder.

Run with `python coefficients/make_examples.py` (from the project root).
"""
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent


def make_static_ou():
    """Static Ornstein-Uhlenbeck coefficients: b(x) = -x, D(x) = 1."""
    x = np.linspace(-15, 15, 601)
    b = -x
    D = np.ones_like(x)
    np.savez(HERE / 'ou_static.npz', x=x, b=b, D=D)


def make_breathing_ou():
    """Ornstein-Uhlenbeck drift whose confining strength oscillates in
    time, paired with a constant (static) diffusivity - demonstrating
    that b and D can independently be time-dependent or not.
    """
    x = np.linspace(-15, 15, 601)
    t = np.linspace(0, 20, 41)
    b_t = np.array([-x * (1.0 + 0.5 * np.sin(2 * np.pi * tt / 10.0)) for tt in t])
    D = np.ones_like(x)
    np.savez(HERE / 'ou_breathing.npz', x=x, t=t, b=b_t, D=D)


def make_spatially_varying_diffusion():
    """Zero drift, diffusivity that varies in space (higher in the
    center, lower near the edges) - a case a typed one-line expression
    can express too, but useful as a second, simple static example.
    """
    x = np.linspace(-10, 10, 401)
    b = np.zeros_like(x)
    D = 0.2 + 0.8 * np.exp(-0.5 * (x / 3.0) ** 2)
    np.savez(HERE / 'varying_diffusion.npz', x=x, b=b, D=D)


if __name__ == '__main__':
    make_static_ou()
    make_breathing_ou()
    make_spatially_varying_diffusion()
    print(f'Wrote example .npz files to {HERE}')
