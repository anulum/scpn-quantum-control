# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Qfi
"""Quantum Fisher Information for coupling parameter estimation.

Computes the QFI matrix F_ij for the ground state |ψ_0(K)⟩ of the
XY Hamiltonian, giving the Cramér-Rao bound on estimation precision:

    Var(K̂_ij) ≥ 1 / (M × F_ij)

where M is the number of measurements.

The QFI is computed via the spectral decomposition:

    F_{ab} = 4 Σ_{m≠0} |⟨ψ_m|V_a|ψ_0⟩|² / (E_m - E_0)²

where V_a = X_iX_j + Y_iY_j is the derivative of H w.r.t. K_ij.

Reference: Braunstein & Caves, PRL 72, 3439 (1994).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian
from ..dense_budget import require_dense_allocation

QFI_CONVENTION: Final[str] = (
    "F_ab = 4 Re<d_a psi|(1 - |psi><psi|)|d_b psi> for the non-degenerate ground state psi of "
    "H(K) = -sum_{i<j} K_ij (X_i X_j + Y_i Y_j) - sum_i omega_i Z_i, one parameter K_ij per "
    "listed pair; pure states only; four times the real part (Fubini-Study metric) of the "
    "quantum geometric tensor; the Berry curvature is not included"
)
"""The convention and scope of :func:`compute_qfi`."""
QFI_PSD_TOLERANCE: Final[float] = 1.0e-10
"""Relative bound on a negative QFI eigenvalue admitted as rounding: ``-tol * max(1, max |eig|)``."""


@dataclass
class QFIResult:
    """Quantum Fisher Information computation result.

    ``metric_eigenvalues`` are the eigenvalues of ``qfi_matrix`` in ascending
    order, and ``metric_rank`` counts those above ``QFI_PSD_TOLERANCE`` times
    the largest. A rank below the number of pairs marks parameter directions
    the ground state does not resolve; they are reported, not regularised.
    """

    qfi_matrix: NDArray[np.float64]
    coupling_pairs: list[tuple[int, int]]
    precision_bounds: NDArray[np.float64]
    spectral_gap: float
    n_qubits: int
    metric_eigenvalues: NDArray[np.float64] = field(
        default_factory=lambda: np.zeros(0, dtype=np.float64)
    )
    metric_rank: int = 0
    convention: str = QFI_CONVENTION

    def precision_for(self, i: int, j: int, n_measurements: int = 10000) -> float:
        """Cramér-Rao lower bound on Var(K̂_ij) for given measurement budget."""
        idx = self.coupling_pairs.index((min(i, j), max(i, j)))
        f_ii = self.qfi_matrix[idx, idx]
        if f_ii < 1e-15:
            return float("inf")
        result: float = 1.0 / (n_measurements * f_ii)
        return result


def compute_qfi(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    pairs: list[tuple[int, int]] | None = None,
    *,
    max_dense_gib: float | None = None,
    degeneracy_tolerance: float = 1.0e-9,
) -> QFIResult:
    """Compute the Quantum Fisher Information matrix for coupling parameters.

    K: coupling matrix (symmetric, n×n)
    omega: natural frequencies (n,)
    pairs: which coupling pairs to compute QFI for (default: all nonzero K_ij)
    degeneracy_tolerance: a spectral gap at most this times ``max(1, max |E|)``
    is a degenerate ground state, for which the QFI is not defined (any state
    in the ground space could be returned) and a ``ValueError`` is raised.

    Returns QFIResult with the QFI matrix, coupling pairs, precision bounds,
    metric eigenvalues and rank, under :data:`QFI_CONVENTION`. A metric with a
    negative eigenvalue beyond :data:`QFI_PSD_TOLERANCE` is refused.
    This exact spectral implementation is dense and small-system only;
    ``max_dense_gib`` fails closed before Hamiltonian, eigensolver, or
    derivative-operator allocations exceed the active budget.
    """
    if not np.isfinite(degeneracy_tolerance) or degeneracy_tolerance <= 0.0:
        raise ValueError("degeneracy_tolerance must be finite and positive")
    n = len(omega)
    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if abs(K[i, j]) > 1e-10]

    require_dense_allocation(
        n,
        rank=2,
        object_count=max(2, 2 + len(pairs)),
        max_gib=max_dense_gib,
        label="QFI dense eigensolver and derivative workspace",
    )
    knm_to_hamiltonian(K, omega)
    H_mat = knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib)

    eigenvalues, eigenvectors = np.linalg.eigh(H_mat)
    E0 = eigenvalues[0]
    psi0 = eigenvectors[:, 0]
    gap = float(eigenvalues[1] - eigenvalues[0])
    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    if gap <= degeneracy_tolerance * scale:
        raise ValueError(
            f"the ground state is degenerate (gap {gap:.3e}); the QFI of a ground space is not "
            "defined by one state, so none is returned"
        )

    n_pairs = len(pairs)
    qfi = np.zeros((n_pairs, n_pairs))

    # Build derivative operators V_a = -(X_iX_j + Y_iY_j) for each pair
    V_ops: list[NDArray[np.complex128]] = []
    for i, j in pairs:
        xx = ["I"] * n
        xx[i] = "X"
        xx[j] = "X"
        yy = ["I"] * n
        yy[i] = "Y"
        yy[j] = "Y"
        V = SparsePauliOp(
            ["".join(reversed(xx)), "".join(reversed(yy))],
            [-1.0, -1.0],
        ).to_matrix()
        V_ops.append(V)

    # Compute matrix elements ⟨ψ_m|V_a|ψ_0⟩ for all m and a
    dim = len(eigenvalues)
    mel: NDArray[np.complex128] = np.zeros((dim, n_pairs), dtype=complex)
    for a in range(n_pairs):
        for m in range(dim):
            mel[m, a] = eigenvectors[:, m].conj() @ V_ops[a] @ psi0

    # QFI matrix: F_ab = 4 Σ_{m≠0} Re(⟨ψ_0|V_a|ψ_m⟩⟨ψ_m|V_b|ψ_0⟩) / (E_m - E_0)²
    for a in range(n_pairs):
        for b in range(a, n_pairs):
            val = 0.0
            # Every excited level lies at least the admitted gap above E0.
            for m in range(1, dim):
                denom = (eigenvalues[m] - E0) ** 2
                val += np.real(mel[m, a].conj() * mel[m, b]) / denom
            qfi[a, b] = 4.0 * val
            qfi[b, a] = qfi[a, b]

    spectrum = np.linalg.eigvalsh(qfi) if n_pairs else np.zeros(0, dtype=np.float64)
    largest = max(1.0, float(np.max(np.abs(spectrum)))) if n_pairs else 1.0
    if n_pairs and float(spectrum[0]) < -QFI_PSD_TOLERANCE * largest:
        raise ValueError(
            f"the QFI matrix has a negative eigenvalue {float(spectrum[0]):.3e}; it is not a metric"
        )
    rank = int(np.count_nonzero(spectrum > QFI_PSD_TOLERANCE * largest))

    # Precision bounds: diagonal elements give per-parameter bounds
    diag = np.diag(qfi)
    precision = np.full_like(diag, np.inf, dtype=float)
    np.divide(1.0, diag, out=precision, where=diag > 1e-15)

    return QFIResult(
        qfi_matrix=qfi,
        coupling_pairs=pairs,
        precision_bounds=precision,
        spectral_gap=gap,
        n_qubits=n,
        metric_eigenvalues=np.asarray(spectrum, dtype=np.float64),
        metric_rank=rank,
    )


def qfi_gap_tradeoff(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    max_dense_gib: float | None = None,
) -> dict[str, Any]:
    """Analyze the QFI-gap tradeoff for the coupling topology.

    Large spectral gap → robust identity but imprecise estimation.
    Small spectral gap → fragile identity but precise estimation.

    Returns dict with gap, max QFI, min precision bound, and tradeoff ratio.
    """
    result = compute_qfi(K, omega, max_dense_gib=max_dense_gib)
    max_qfi = float(np.max(np.diag(result.qfi_matrix)))
    finite_bounds = result.precision_bounds[np.isfinite(result.precision_bounds)]
    min_precision = float(np.min(finite_bounds)) if len(finite_bounds) > 0 else float("inf")

    return {
        "spectral_gap": result.spectral_gap,
        "max_qfi_diagonal": max_qfi,
        "min_precision_bound": min_precision,
        "tradeoff_ratio": result.spectral_gap * max_qfi,
        "n_pairs": len(result.coupling_pairs),
        "gap_squared_over_16": result.spectral_gap**2 / 16.0,
    }
