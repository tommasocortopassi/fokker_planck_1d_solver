"""Boundary condition specification, shared by the PDE and SDE solvers.

Only the homogeneous version of each condition is supported:

kind:
    'periodic'  - the domain wraps around; p(left) and p(right) describe
                  the same physical point.
    'neumann'   - reflecting / no-flux wall: probability cannot cross the
                  boundary (zero total flux, advective + diffusive).
    'dirichlet' - the density is pinned to zero at each edge;
                  probability is free to flow out there.
"""
from dataclasses import dataclass

VALID_KINDS = ('periodic', 'dirichlet', 'neumann')


@dataclass(frozen=True)
class BoundaryCondition:
    kind: str

    def validate(self):
        if self.kind not in VALID_KINDS:
            raise ValueError(f'Boundary condition must be one of {VALID_KINDS}.')
