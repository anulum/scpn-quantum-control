# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — FIM Hamiltonian Analysis
"""Offline diagnostics for the FIM-augmented Kuramoto-XY Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectrumSummary:
    """Compact exact-spectrum summary for one FIM Hamiltonian instance."""

    n_qubits: int
    lambda_fim: float
    dimension: int
    ground_energy: float
    max_energy: float
    spectral_width: float
    spectral_gap: float | None
    mean_energy: float
    median_energy: float


def computational_magnetisations(n_qubits: int) -> np.ndarray:
    """Return total Z magnetisation for every computational basis state."""

    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    dimension = 2**n_qubits
    values = np.empty(dimension, dtype=np.int64)
    for basis_index in range(dimension):
        popcount = int(basis_index.bit_count())
        values[basis_index] = n_qubits - 2 * popcount
    return values


def fim_diagonal(n_qubits: int, lambda_fim: float) -> np.ndarray:
    """Return the diagonal entries of H_FIM = -lambda * M^2 / n."""

    magnetisations = computational_magnetisations(n_qubits).astype(np.float64)
    return -float(lambda_fim) * magnetisations**2 / float(n_qubits)


def add_fim_feedback(hamiltonian: np.ndarray, lambda_fim: float) -> np.ndarray:
    """Add the collective FIM-inspired feedback term to a dense Hamiltonian."""

    if hamiltonian.ndim != 2 or hamiltonian.shape[0] != hamiltonian.shape[1]:
        raise ValueError("hamiltonian must be a square matrix")
    dimension = hamiltonian.shape[0]
    n_qubits_float = np.log2(dimension)
    n_qubits = int(round(n_qubits_float))
    if 2**n_qubits != dimension:
        raise ValueError("hamiltonian dimension must be a power of two")
    out = np.array(hamiltonian, dtype=np.complex128, copy=True)
    diagonal = fim_diagonal(n_qubits, lambda_fim)
    out[np.diag_indices(dimension)] += diagonal
    return out


def magnetisation_sector_indices(n_qubits: int) -> dict[int, np.ndarray]:
    """Group computational basis indices by total magnetisation."""

    magnetisations = computational_magnetisations(n_qubits)
    sectors: dict[int, list[int]] = {}
    for basis_index, magnetisation in enumerate(magnetisations):
        sectors.setdefault(int(magnetisation), []).append(int(basis_index))
    return {
        magnetisation: np.asarray(indices, dtype=np.int64)
        for magnetisation, indices in sorted(sectors.items(), reverse=True)
    }


def summarise_spectrum(
    eigenvalues: np.ndarray, n_qubits: int, lambda_fim: float
) -> SpectrumSummary:
    """Summarise sorted real eigenvalues for one Hamiltonian instance."""

    values = np.sort(np.asarray(eigenvalues, dtype=np.float64))
    gap = float(values[1] - values[0]) if values.size > 1 else None
    return SpectrumSummary(
        n_qubits=n_qubits,
        lambda_fim=float(lambda_fim),
        dimension=int(values.size),
        ground_energy=float(values[0]),
        max_energy=float(values[-1]),
        spectral_width=float(values[-1] - values[0]),
        spectral_gap=gap,
        mean_energy=float(np.mean(values)),
        median_energy=float(np.median(values)),
    )


def sector_spectrum_rows(hamiltonian: np.ndarray, lambda_fim: float) -> list[dict[str, object]]:
    """Compute exact spectrum summaries inside each magnetisation sector."""

    dimension = hamiltonian.shape[0]
    n_qubits = int(round(np.log2(dimension)))
    rows: list[dict[str, object]] = []
    for magnetisation, indices in magnetisation_sector_indices(n_qubits).items():
        block = hamiltonian[np.ix_(indices, indices)]
        eigenvalues = np.linalg.eigvalsh(block)
        summary = summarise_spectrum(eigenvalues, n_qubits, lambda_fim)
        rows.append(
            {
                "n_qubits": n_qubits,
                "lambda_fim": float(lambda_fim),
                "magnetisation": int(magnetisation),
                "sector_dimension": int(indices.size),
                "fim_energy_shift": float(-float(lambda_fim) * magnetisation**2 / n_qubits),
                "ground_energy": summary.ground_energy,
                "max_energy": summary.max_energy,
                "spectral_width": summary.spectral_width,
                "spectral_gap": summary.spectral_gap,
                "mean_energy": summary.mean_energy,
                "median_energy": summary.median_energy,
            }
        )
    return rows


def adjacent_gap_ratio(eigenvalues: np.ndarray, tolerance: float = 1e-10) -> dict[str, object]:
    """Compute adjacent-gap ratio statistics after removing tiny spacings."""

    values = np.sort(np.asarray(eigenvalues, dtype=np.float64))
    spacings = np.diff(values)
    spacings = spacings[spacings > tolerance]
    if spacings.size < 2:
        return {
            "n_spacings": int(spacings.size),
            "mean_r": None,
            "median_r": None,
            "min_spacing": float(np.min(spacings)) if spacings.size else None,
            "max_spacing": float(np.max(spacings)) if spacings.size else None,
        }
    left = spacings[:-1]
    right = spacings[1:]
    ratios = np.minimum(left, right) / np.maximum(left, right)
    return {
        "n_spacings": int(spacings.size),
        "mean_r": float(np.mean(ratios)),
        "median_r": float(np.median(ratios)),
        "min_spacing": float(np.min(spacings)),
        "max_spacing": float(np.max(spacings)),
    }


def bipartite_entropy_from_statevector(
    statevector: np.ndarray,
    n_qubits: int,
    keep: list[int] | None = None,
    tolerance: float = 1e-15,
) -> float:
    """Return pure-state bipartite entropy S(keep) in bits."""

    state = np.asarray(statevector, dtype=np.complex128)
    if state.size != 2**n_qubits:
        raise ValueError("statevector length does not match n_qubits")
    if keep is None:
        keep = list(range(max(1, n_qubits // 2)))
    if not keep or len(keep) >= n_qubits:
        return 0.0
    keep_set = set(keep)
    if any(qubit < 0 or qubit >= n_qubits for qubit in keep_set):
        raise ValueError("keep contains an invalid qubit index")
    trace = [qubit for qubit in range(n_qubits) if qubit not in keep_set]
    tensor = state.reshape([2] * n_qubits)
    tensor = np.moveaxis(tensor, keep + trace, list(range(n_qubits)))
    matrix = tensor.reshape(2 ** len(keep), 2 ** len(trace))
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    probabilities = singular_values**2
    probabilities = probabilities[probabilities > tolerance]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def magnetisation_operator_diagonal(n_qubits: int) -> np.ndarray:
    """Return diagonal entries of the total magnetisation operator M."""

    return computational_magnetisations(n_qubits).astype(np.float64)


def commutator_frobenius_norm_with_diagonal(
    hamiltonian: np.ndarray,
    diagonal_operator: np.ndarray,
) -> float:
    """Return ||[H, D]||_F for a diagonal operator D."""

    if hamiltonian.shape[0] != diagonal_operator.size:
        raise ValueError("operator dimension mismatch")
    commutator = (
        hamiltonian * diagonal_operator[None, :] - diagonal_operator[:, None] * hamiltonian
    )
    return float(np.linalg.norm(commutator))


def sector_coupling_rows(hamiltonian: np.ndarray, lambda_fim: float) -> list[dict[str, object]]:
    """Measure off-sector Hamiltonian coupling for each magnetisation sector."""

    dimension = hamiltonian.shape[0]
    n_qubits = int(round(np.log2(dimension)))
    sectors = magnetisation_sector_indices(n_qubits)
    eigenvalues = np.linalg.eigvalsh(hamiltonian)
    global_ground = float(np.min(eigenvalues))
    rows: list[dict[str, object]] = []
    all_indices = np.arange(dimension, dtype=np.int64)
    for magnetisation, indices in sectors.items():
        outside = np.setdiff1d(all_indices, indices, assume_unique=True)
        off_sector = hamiltonian[np.ix_(indices, outside)]
        block = hamiltonian[np.ix_(indices, indices)]
        block_eigenvalues = np.linalg.eigvalsh(block)
        sector_ground = float(np.min(block_eigenvalues))
        rows.append(
            {
                "n_qubits": n_qubits,
                "lambda_fim": float(lambda_fim),
                "magnetisation": int(magnetisation),
                "sector_dimension": int(indices.size),
                "off_sector_frobenius_norm": float(np.linalg.norm(off_sector)),
                "sector_ground_energy": sector_ground,
                "energy_above_global_ground": float(sector_ground - global_ground),
                "fim_energy_shift": float(-float(lambda_fim) * magnetisation**2 / n_qubits),
                "ideal_unitary_sector_leakage": 0.0,
            }
        )
    return rows
