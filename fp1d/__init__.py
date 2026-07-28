"""1D Fokker-Planck solver: conservative finite-volume PDE integrator plus
an Euler-Maruyama particle (SDE) solver, sharing the same drift, diffusion,
and initial condition. Boundary conditions (periodic, Neumann, Dirichlet)
are all homogeneous. Unbounded domains are truncated adaptively - see
`domain_truncation.py`. See the top-level README.md for the derivation of
the numerical schemes.
"""
