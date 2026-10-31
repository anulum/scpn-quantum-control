# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — sparse CPU Kuramoto force and fixed-step integrators
"""Sparse CPU Kuramoto force and fixed-step integrators.

The dense networked Kuramoto primitives materialise an ``N x N`` phase-difference
matrix, which is the right correctness floor for small dense coupling matrices
but the wrong contract for ring, lattice, power-grid, connectome, and other
large sparse networks. This module accepts SciPy sparse coupling matrices,
canonicalises them into COO edge arrays, and evaluates
``F_i = sum_j K_ij sin(theta_j - theta_i)`` with ``O(E)`` memory and work.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

FloatArray: TypeAlias = NDArray[np.float64]
IndexArray: TypeAlias = NDArray[np.intp]
SparseInput: TypeAlias = "SparseKuramotoCoupling | object"
DuplicatePolicy: TypeAlias = Literal["sum", "refuse"]
"""How repeated ``(row, col)`` entries are aggregated: summed, or refused."""


@dataclass(frozen=True)
class SparseStorageReport:
    """Measured storage of a sparse coupling, by stored component.

    Attributes
    ----------
    n_oscillators:
        Number of oscillators.
    edges:
        Number of stored off-diagonal entries.
    component_bytes:
        ``(name, nbytes)`` for the row, column and weight arrays.
    stored_bytes:
        Sum of the component bytes; linear in ``edges``.
    dense_equivalent_bytes:
        Bytes a dense float64 ``N x N`` matrix would need; reported, never
        allocated.

    """

    n_oscillators: int
    edges: int
    component_bytes: tuple[tuple[str, int], ...]
    stored_bytes: int
    dense_equivalent_bytes: int

    @property
    def bytes_per_edge(self) -> float:
        """Stored bytes per edge, or zero for an edgeless coupling."""
        return 0.0 if self.edges == 0 else self.stored_bytes / self.edges


@dataclass(frozen=True)
class SparseExecutionReport:
    """Where and in which precision the sparse Kuramoto kernels run.

    Attributes
    ----------
    backend:
        Execution placement; always ``cpu``.
    library:
        Array library performing the edge gathers and reductions.
    precision:
        Floating-point type of states, weights and forces.
    gpu_status:
        ``not_implemented``: this module has no GPU kernel, whatever devices
        the host has, so it never stands in for GPU evidence.
    gpu_performance_claim:
        Always ``False``.

    """

    backend: str
    library: str
    precision: str
    gpu_status: str
    gpu_performance_claim: bool


def sparse_kuramoto_execution_report() -> SparseExecutionReport:
    """Return the fixed CPU placement and float64 precision of these kernels."""
    return SparseExecutionReport(
        backend="cpu",
        library=f"numpy {np.__version__}",
        precision="float64",
        gpu_status="not_implemented",
        gpu_performance_claim=False,
    )


@dataclass(frozen=True)
class SparseKuramotoCoupling:
    """Canonical COO edge representation of a sparse Kuramoto coupling matrix."""

    n_oscillators: int
    row: IndexArray
    col: IndexArray
    weight: FloatArray

    def __post_init__(self) -> None:
        """Validate and freeze the COO edge arrays."""
        if not isinstance(self.n_oscillators, int) or self.n_oscillators < 0:
            raise ValueError("n_oscillators must be a non-negative integer")

        row = np.ascontiguousarray(self.row, dtype=np.intp)
        col = np.ascontiguousarray(self.col, dtype=np.intp)
        weight = np.ascontiguousarray(self.weight, dtype=np.float64)
        if row.shape != col.shape or row.shape != weight.shape:
            raise ValueError("row, col, and weight must have identical one-dimensional shapes")
        if row.ndim != 1:
            raise ValueError("row, col, and weight must be one-dimensional")
        if not np.all(np.isfinite(weight)):
            raise ValueError("sparse coupling weights must be finite")
        if row.size:
            if int(np.min(row, initial=0)) < 0 or int(np.min(col, initial=0)) < 0:
                raise ValueError("sparse coupling indices must be non-negative")
            if (
                int(np.max(row, initial=0)) >= self.n_oscillators
                or int(np.max(col, initial=0)) >= self.n_oscillators
            ):
                raise ValueError("sparse coupling indices must be within n_oscillators")

        row.setflags(write=False)
        col.setflags(write=False)
        weight.setflags(write=False)
        object.__setattr__(self, "row", row)
        object.__setattr__(self, "col", col)
        object.__setattr__(self, "weight", weight)

    @property
    def nnz(self) -> int:
        """The number of stored off-diagonal coupling entries."""
        return int(self.weight.size)

    @property
    def density(self) -> float:
        """The stored-entry density relative to a dense ``N x N`` matrix."""
        if self.n_oscillators == 0:
            return 0.0
        return float(self.nnz / (self.n_oscillators * self.n_oscillators))

    def storage_report(self) -> SparseStorageReport:
        """Measure the stored row, column and weight arrays."""
        components = (
            ("row", int(self.row.nbytes)),
            ("col", int(self.col.nbytes)),
            ("weight", int(self.weight.nbytes)),
        )
        return SparseStorageReport(
            n_oscillators=self.n_oscillators,
            edges=self.nnz,
            component_bytes=components,
            stored_bytes=sum(size for _, size in components),
            dense_equivalent_bytes=self.n_oscillators * self.n_oscillators * 8,
        )

    def to_scipy_csr(self) -> Any:
        """Return the coupling as a SciPy CSR array for interoperability."""
        return sparse.csr_array(
            (self.weight, (self.row, self.col)),
            shape=(self.n_oscillators, self.n_oscillators),
            dtype=np.float64,
        )


def sparse_coupling_from_scipy(
    coupling: object, *, duplicates: DuplicatePolicy = "sum"
) -> SparseKuramotoCoupling:
    """Canonicalise a SciPy sparse matrix into sparse Kuramoto COO edges.

    Diagonal entries are discarded because ``sin(theta_i - theta_i) == 0`` and
    therefore cannot affect the Kuramoto force or any fixed-step trajectory.
    Duplicate sparse entries are summed before zero entries are dropped, or
    refused when ``duplicates="refuse"``.
    """
    if not sparse.issparse(coupling):
        raise TypeError("coupling must be a SciPy sparse matrix or SparseKuramotoCoupling")

    raw = cast(Any, coupling)
    shape = tuple(int(axis) for axis in raw.shape)
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(f"coupling must be a square sparse matrix, got shape {shape}")

    _validate_duplicate_policy(duplicates)
    coo = sparse.coo_array(raw, dtype=np.float64, copy=True)
    if duplicates == "refuse":
        order = np.lexsort((coo.col, coo.row))
        rows = np.asarray(coo.row, dtype=np.intp)[order]
        cols = np.asarray(coo.col, dtype=np.intp)[order]
        _refuse_repeated_entries(_entry_run_starts(rows, cols), rows.size)
    coo.sum_duplicates()
    row = np.asarray(coo.row, dtype=np.intp)
    col = np.asarray(coo.col, dtype=np.intp)
    weight = np.asarray(coo.data, dtype=np.float64)
    keep = (row != col) & (weight != 0.0)
    return SparseKuramotoCoupling(
        n_oscillators=shape[0],
        row=row[keep],
        col=col[keep],
        weight=weight[keep],
    )


def sparse_coupling_from_edges(
    n_oscillators: int,
    row: object,
    col: object,
    weight: object,
    *,
    duplicates: DuplicatePolicy = "sum",
) -> SparseKuramotoCoupling:
    """Canonicalise an explicit edge list under a declared duplicate policy.

    Parameters
    ----------
    n_oscillators:
        Number of oscillators.
    row, col, weight:
        Edge arrays of equal length; entry ``k`` adds ``weight[k]`` to
        ``K[row[k], col[k]]``.
    duplicates:
        ``sum`` adds repeated ``(row, col)`` entries; ``refuse`` raises.

    Returns
    -------
    SparseKuramotoCoupling
        Edges sorted by ``(row, col)``, without diagonal or zero entries.

    Raises
    ------
    ValueError
        If the arrays are malformed, out of range, or repeat an entry under
        ``refuse``.

    """
    _validate_duplicate_policy(duplicates)
    raw = SparseKuramotoCoupling(
        n_oscillators=n_oscillators,
        row=np.asarray(row),
        col=np.asarray(col),
        weight=np.asarray(weight),
    )
    order = np.lexsort((raw.col, raw.row))
    rows, cols, weights = raw.row[order], raw.col[order], raw.weight[order]
    starts = _entry_run_starts(rows, cols)
    if duplicates == "refuse":
        _refuse_repeated_entries(starts, rows.size)
    if rows.size:
        rows, cols = rows[starts], cols[starts]
        weights = np.add.reduceat(weights, starts)
    keep = (rows != cols) & (weights != 0.0)
    return SparseKuramotoCoupling(
        n_oscillators=n_oscillators, row=rows[keep], col=cols[keep], weight=weights[keep]
    )


def _validate_duplicate_policy(policy: DuplicatePolicy) -> None:
    """Refuse a duplicate policy other than ``sum`` or ``refuse``."""
    if policy not in ("sum", "refuse"):
        raise ValueError("duplicates must be 'sum' or 'refuse'")


def _entry_run_starts(rows: IndexArray, cols: IndexArray) -> IndexArray:
    """Return the first index of each run of equal ``(row, col)`` in sorted edges."""
    boundary = np.ones(rows.size, dtype=bool)
    boundary[1:] = (rows[1:] != rows[:-1]) | (cols[1:] != cols[:-1])
    return np.flatnonzero(boundary).astype(np.intp, copy=False)


def _refuse_repeated_entries(starts: IndexArray, n_entries: int) -> None:
    """Raise when sorted edges hold fewer distinct entries than stored entries."""
    if starts.size != n_entries:
        raise ValueError("sparse coupling repeats a (row, col) entry under duplicates='refuse'")


def ring_sparse_coupling(
    n_oscillators: int, coupling_strength: float = 1.0
) -> SparseKuramotoCoupling:
    """Build a bidirectional nearest-neighbour ring coupling without dense storage."""
    if not isinstance(n_oscillators, int) or n_oscillators < 1:
        raise ValueError("n_oscillators must be a positive integer")
    if not np.isfinite(coupling_strength):
        raise ValueError("coupling_strength must be finite")
    if n_oscillators == 1 or coupling_strength == 0.0:
        return SparseKuramotoCoupling(
            n_oscillators=n_oscillators,
            row=np.zeros(0, dtype=np.intp),
            col=np.zeros(0, dtype=np.intp),
            weight=np.zeros(0, dtype=np.float64),
        )

    source = np.arange(n_oscillators, dtype=np.intp)
    forward = (source + 1) % n_oscillators
    rows = np.concatenate((source, forward)).astype(np.intp, copy=False)
    cols = np.concatenate((forward, source)).astype(np.intp, copy=False)
    weights = np.full(rows.shape, float(coupling_strength), dtype=np.float64)
    return SparseKuramotoCoupling(n_oscillators=n_oscillators, row=rows, col=cols, weight=weights)


def _coerce_sparse_coupling(coupling: SparseInput) -> SparseKuramotoCoupling:
    """Return a canonical sparse coupling record."""
    if isinstance(coupling, SparseKuramotoCoupling):
        return coupling
    return sparse_coupling_from_scipy(coupling)


def _validate_phase_vector(name: str, values: object, count: int) -> FloatArray:
    """Return a finite contiguous phase/frequency vector of length ``count``."""
    vector = np.ascontiguousarray(values, dtype=np.float64)
    if vector.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _validate_step(dt: float, n_steps: int) -> tuple[float, int]:
    """Return validated fixed-step integrator settings."""
    step_size = float(dt)
    if not np.isfinite(step_size):
        raise ValueError("dt must be finite")
    if not isinstance(n_steps, int) or n_steps < 0:
        raise ValueError(f"n_steps must be a non-negative integer, got {n_steps}")
    return step_size, n_steps


def sparse_networked_kuramoto_force(theta: object, coupling: SparseInput) -> FloatArray:
    r"""Evaluate the networked Kuramoto force from sparse coupling edges.

    Returns :math:`F_i = \sum_j K_{ij}\sin(\theta_j - \theta_i)` using only the
    stored sparse entries. The result is exactly the dense
    :func:`networked_kuramoto_force` contract for the same coupling matrix.
    """
    sparse_coupling = _coerce_sparse_coupling(coupling)
    phases = _validate_phase_vector("theta", theta, sparse_coupling.n_oscillators)
    if sparse_coupling.nnz == 0:
        return np.zeros(sparse_coupling.n_oscillators, dtype=np.float64)

    edge_force = sparse_coupling.weight * np.sin(
        phases[sparse_coupling.col] - phases[sparse_coupling.row]
    )
    force = np.bincount(
        sparse_coupling.row,
        weights=edge_force,
        minlength=sparse_coupling.n_oscillators,
    )
    return np.ascontiguousarray(force, dtype=np.float64)


def _sparse_rhs(
    theta: FloatArray, omega: FloatArray, coupling: SparseKuramotoCoupling
) -> FloatArray:
    """Return ``omega + sparse_networked_kuramoto_force(theta, coupling)``."""
    return np.asarray(omega + sparse_networked_kuramoto_force(theta, coupling), dtype=np.float64)


def sparse_kuramoto_euler_trajectory(
    theta0: object,
    omega: object,
    coupling: SparseInput,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Integrate sparse networked Kuramoto dynamics with explicit Euler steps."""
    sparse_coupling = _coerce_sparse_coupling(coupling)
    phases = _validate_phase_vector("theta0", theta0, sparse_coupling.n_oscillators)
    frequencies = _validate_phase_vector("omega", omega, sparse_coupling.n_oscillators)
    step_size, step_count = _validate_step(dt, n_steps)

    trajectory = np.zeros((step_count + 1, sparse_coupling.n_oscillators), dtype=np.float64)
    trajectory[0] = phases
    current = phases
    for step in range(step_count):
        current = current + step_size * _sparse_rhs(current, frequencies, sparse_coupling)
        trajectory[step + 1] = current
    return np.ascontiguousarray(trajectory, dtype=np.float64)


def sparse_kuramoto_rk4_trajectory(
    theta0: object,
    omega: object,
    coupling: SparseInput,
    dt: float,
    n_steps: int,
) -> FloatArray:
    """Integrate sparse networked Kuramoto dynamics with fourth-order RK4 steps."""
    sparse_coupling = _coerce_sparse_coupling(coupling)
    phases = _validate_phase_vector("theta0", theta0, sparse_coupling.n_oscillators)
    frequencies = _validate_phase_vector("omega", omega, sparse_coupling.n_oscillators)
    step_size, step_count = _validate_step(dt, n_steps)
    half_step = 0.5 * step_size

    trajectory = np.zeros((step_count + 1, sparse_coupling.n_oscillators), dtype=np.float64)
    trajectory[0] = phases
    current = phases
    for step in range(step_count):
        k1 = _sparse_rhs(current, frequencies, sparse_coupling)
        k2 = _sparse_rhs(current + half_step * k1, frequencies, sparse_coupling)
        k3 = _sparse_rhs(current + half_step * k2, frequencies, sparse_coupling)
        k4 = _sparse_rhs(current + step_size * k3, frequencies, sparse_coupling)
        current = current + (step_size / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        trajectory[step + 1] = current
    return np.ascontiguousarray(trajectory, dtype=np.float64)


__all__ = [
    "DuplicatePolicy",
    "SparseExecutionReport",
    "SparseKuramotoCoupling",
    "SparseStorageReport",
    "ring_sparse_coupling",
    "sparse_coupling_from_edges",
    "sparse_coupling_from_scipy",
    "sparse_kuramoto_execution_report",
    "sparse_kuramoto_euler_trajectory",
    "sparse_kuramoto_rk4_trajectory",
    "sparse_networked_kuramoto_force",
]
