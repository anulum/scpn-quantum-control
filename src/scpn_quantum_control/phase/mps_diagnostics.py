# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Compressed MPS allocation and truncation diagnostics
"""Declared tensor buffers and local SVD losses, without dense state export."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray

from ..dense_budget import dense_budget_bytes


class MPSAllocationError(MemoryError):
    """Refusal to allocate the declared compressed solver buffers."""


@dataclass(frozen=True)
class MPSAllocationEstimate:
    """Component estimate; neither a reservation nor a process RSS bound.

    Attributes
    ----------
    n_sites
        Number of spin-half sites.
    bond_limit
        Requested maximum bond dimension.
    state_bytes
        Open-boundary complex128 tensors at maximal physical Schmidt ranks.
    workspace_bytes
        Declared state copies, local SVD buffers, MPO terms, inputs and records.
    budget_bytes
        The smaller of the caller ceiling and active allocation policy.

    """

    n_sites: int
    bond_limit: int
    state_bytes: int
    workspace_bytes: int
    budget_bytes: int


@dataclass(frozen=True)
class MPSObservation:
    """One compressed state observation in site order.

    Attributes
    ----------
    time
        Evolution time; dimensionless Hamiltonian-time convention.
    bond_dims
        Actual adjacent Schmidt dimensions.
    stored_bytes
        Sum of the actual tensor array byte counts, excluding Python overhead.
    norm
        State norm obtained by tensor contraction.
    local_z
        Normalized Pauli-Z expectations in MPS site order.

    """

    time: float
    bond_dims: tuple[int, ...]
    stored_bytes: int
    norm: float
    local_z: tuple[float, ...]


@dataclass(frozen=True)
class MPSRunDiagnostics:
    """Memory, contraction and truncation provenance for a real MPS run.

    Attributes
    ----------
    allocation
        Declared component estimate and admission ceiling in bytes.
    contraction_strategy
        Actual solver contraction/sweep strategy.
    observations
        Requested-time observations; no exponentially sized state arrays.
    discarded_weight
        Sum of squared local discarded singular-value norms, or unavailable.
    discarded_weight_scope
        Interpretation of that local quantity; not a global fidelity bound.
    truncation_events
        Number of local SVD reports, including zero-loss splits.
    trotter_error_estimate
        Quimb's separate time-discretization estimate, or unavailable for DMRG.
    cutoff
        Requested cutoff passed to the actual Quimb solver.
    trotter_order
        Actual Suzuki order, or not applicable for DMRG.
    time_step
        Requested TEBD timestep, or not applicable for DMRG.
    backend_version
        Version of the Quimb runtime that produced these observations.
    dense_result_exported
        False: the solver returns a compressed result, not a dense statevector.

    """

    allocation: MPSAllocationEstimate
    contraction_strategy: str
    observations: tuple[MPSObservation, ...]
    discarded_weight: float | None
    discarded_weight_scope: str
    truncation_events: int
    trotter_error_estimate: float | None
    cutoff: float
    trotter_order: int | None
    time_step: float | None
    backend_version: str
    dense_result_exported: bool = False

    @property
    def memory_budget_bytes(self) -> int:
        """The admitted component ceiling in bytes."""
        return self.allocation.budget_bytes

    def to_dict(self) -> dict[str, object]:
        """Return metadata suitable for a separate JSON companion.

        Returns
        -------
        dict[str, object]
            Diagnostic fields only; no MPS or dense amplitudes are serialized.

        """
        return asdict(self)


class DiscardedWeightLog(dict[str, float]):
    """Collect the ``error`` reports of Quimb's truncated SVD.

    The log is handed to the decomposition as its ``info`` target by the
    reporting split method of the evolution module; Quimb's gate application
    does not forward ``TEBD.split_opts["info"]`` to the decomposition.

    The pinned Quimb runtime reports the norm of discarded singular values.
    Squaring and summing those local norms does not produce a universal global
    error bound. The numerical solver and its renormalization remain unchanged.
    """

    def __init__(self) -> None:
        """Request local error reports without recording a synthetic event."""
        super().__init__({"error": 0.0})
        self.weight = 0.0
        self.events = 0

    def __setitem__(self, key: str, value: float) -> None:
        """Record a finite nonnegative local discarded norm.

        Parameters
        ----------
        key
            The Quimb diagnostic key, which must be ``error``.
        value
            Norm of discarded singular values before any renormalization.

        Raises
        ------
        ValueError
            The report is invalid or its accumulated squared norm overflows.

        """
        norm = float(value)
        weight = self.weight + norm * norm
        if key != "error" or norm < 0 or not isfinite(norm) or not isfinite(weight):
            raise ValueError("Invalid local MPS discarded singular-value norm")
        super().__setitem__(key, norm)
        self.weight = weight
        self.events += 1


def require_mps_allocation(
    n_sites: int,
    bond_limit: int,
    *,
    observations: int = 1,
    max_memory_bytes: int | None = None,
) -> MPSAllocationEstimate:
    """Admit declared compressed buffers before constructing Quimb objects.

    Parameters
    ----------
    n_sites
        At least two spin-half sites.
    bond_limit
        Positive maximum bond dimension; physical Schmidt ranks cap each cut.
    observations
        Positive number of retained time/observable records.
    max_memory_bytes
        Optional positive byte ceiling that can only lower the active policy.

    Returns
    -------
    MPSAllocationEstimate
        State storage and declared workspace component estimates in bytes.

    Raises
    ------
    ValueError
        Counts or the caller ceiling are not positive non-boolean integers.
    MPSAllocationError
        Declared compressed buffers exceed the admitted byte ceiling.

    Notes
    -----
    Counts include three MPS copies, twelve complex local SVD buffers with
    dimension at most twice the largest bond, nearest-neighbour MPO terms,
    classical model inputs and retained diagnostics. Library/Python overhead,
    BLAS internals and process RSS are not certified by this component model.

    """
    for name, value, minimum in [
        ("n_sites", n_sites, 2),
        ("bond_limit", bond_limit, 1),
        ("observations", observations, 1),
    ]:
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    budget = dense_budget_bytes()
    if max_memory_bytes is not None:
        if (
            isinstance(max_memory_bytes, bool)
            or not isinstance(max_memory_bytes, int)
            or max_memory_bytes < 1
        ):
            raise ValueError("max_memory_bytes must be a positive integer")
        budget = min(budget, max_memory_bytes)
    record_bytes = observations * (32 + 24 * n_sites)
    input_bytes = 8 * (n_sites * n_sites + n_sites)
    if record_bytes + input_bytes > budget:
        raise MPSAllocationError("MPS input and observation buffers exceed the memory cap")
    cuts = [1]
    for cut in range(1, n_sites):
        exponent = min(cut, n_sites - cut)
        cuts.append(
            bond_limit if exponent >= bond_limit.bit_length() else min(bond_limit, 1 << exponent)
        )
    cuts.append(1)
    state = 32 * sum(left * right for left, right in zip(cuts[:-1], cuts[1:], strict=True))
    largest = max(cuts)
    workspace = 3 * state + 768 * largest * largest + 256 * n_sites
    workspace += input_bytes + record_bytes
    if workspace > budget:
        raise MPSAllocationError("MPS tensor and local SVD buffers exceed the memory cap")
    return MPSAllocationEstimate(n_sites, bond_limit, state, workspace, budget)


def validate_mps_model(K: NDArray[np.float64], omega: NDArray[np.float64]) -> int:
    """Validate a finite real symmetric spin-half coupling model.

    Parameters
    ----------
    K
        Real square coupling matrix; its diagonal is unused by this XY owner.
    omega
        Real on-site frequencies with one entry per site.

    Returns
    -------
    int
        Number of sites; neither input is modified.

    Raises
    ------
    ValueError
        Dimensions, real-valued dtype, finiteness or symmetry are invalid.

    """
    if K.ndim != 2 or K.shape[0] != K.shape[1] or K.shape[0] < 2:
        raise ValueError("K must be square with at least two sites")
    if omega.shape != (K.shape[0],):
        raise ValueError("omega must have one frequency per site")
    if K.dtype.kind not in "fiu" or omega.dtype.kind not in "fiu":
        raise ValueError("K and omega must be real numeric arrays")
    if not np.all(np.isfinite(K)) or not np.all(np.isfinite(omega)):
        raise ValueError("K and omega must contain finite values")
    if not np.allclose(K, K.T, atol=1e-12, rtol=0):
        raise ValueError("K must be symmetric")
    return int(K.shape[0])
