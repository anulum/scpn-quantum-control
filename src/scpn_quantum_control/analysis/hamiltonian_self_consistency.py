# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hamiltonian Self Consistency
"""Hamiltonian self-consistency loop: quantum measurement → K_nm recovery.

Closes the loop:
    K_nm (true) → build H → VQE ground state → measure correlators
    → learn_hamiltonian() → K_nm (learned) → compare

If K_learned ≈ K_true, the quantum hardware faithfully implements the
intended Hamiltonian. The reconstruction fidelity degrades with:
    1. Shot noise (finite measurement samples)
    2. Gate errors (noisy quantum state)
    3. Trotter error (approximate time evolution)

This validates the quantum simulation pipeline end-to-end.

Prior art: Hamiltonian learning for 300 trapped ions (Ising, Science
Advances 2025). Nobody has done K_nm recovery for coupled XY/Kuramoto
systems.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hamiltonian_learning import (
    HamiltonianLearningResult,
    learn_hamiltonian,
    measure_correlators,
)


@dataclass
class SelfConsistencyResult:
    """Result of the full self-consistency loop."""

    K_true: np.ndarray
    K_learned: np.ndarray
    frobenius_error: float
    elementwise_max_error: float
    relative_error: float
    correlator_error: float
    learning_result: HamiltonianLearningResult
    n_qubits: int
    shot_noise_std: float


def correlators_from_counts(
    x_counts: dict[str, int],
    y_counts: dict[str, int],
    n_qubits: int,
) -> np.ndarray:
    """Extract ⟨X_iX_j⟩ + ⟨Y_iY_j⟩ correlator matrix from hardware counts.

    This is the bridge between hardware measurement data and the
    Hamiltonian learning inverse problem.
    """
    xx = _two_point_from_counts(x_counts, n_qubits)
    yy = _two_point_from_counts(y_counts, n_qubits)
    C: np.ndarray = xx + yy
    np.fill_diagonal(C, 0.0)
    return C


def _two_point_from_counts(counts: dict[str, int], n: int) -> np.ndarray:
    """Compute ⟨Z_iZ_j⟩ from measurement counts in one basis."""
    total = sum(counts.values())
    if total == 0:
        zeros: np.ndarray = np.zeros((n, n))
        return zeros

    corr: np.ndarray = np.zeros((n, n))
    for bitstring, count in counts.items():
        bits = bitstring.replace(" ", "")
        vals = np.array([1 - 2 * int(bits[-(q + 1)]) for q in range(min(n, len(bits)))])
        corr += count * np.outer(vals, vals)
    corr /= total
    result: np.ndarray = corr
    return result


def correlator_shot_noise(
    x_counts: dict[str, int],
    y_counts: dict[str, int],
    n_qubits: int,
) -> float:
    """Estimate standard deviation of correlator due to shot noise.

    For N_shots measurements, the std of ⟨Z_iZ_j⟩ ≈ 1/√N_shots.
    The combined XX+YY std ≈ √2 / √N_shots.
    """
    n_shots_x = sum(x_counts.values())
    n_shots_y = sum(y_counts.values())
    n_shots = min(n_shots_x, n_shots_y)
    if n_shots == 0:
        return float("inf")
    return float(np.sqrt(2.0 / n_shots))


def self_consistency_from_exact(
    K_true: np.ndarray,
    omega: np.ndarray,
    maxiter: int = 100,
) -> SelfConsistencyResult:
    """Run self-consistency loop using exact (noiseless) correlators.

    This is the ideal baseline — measures how well the learning algorithm
    works without any hardware noise. Any error is from the optimization
    algorithm, not from quantum noise.
    """
    n = K_true.shape[0]

    C_measured = measure_correlators(K_true, omega)
    result = learn_hamiltonian(C_measured, omega, maxiter=maxiter)

    return _build_result(K_true, result, 0.0, n)


def self_consistency_from_counts(
    K_true: np.ndarray,
    omega: np.ndarray,
    x_counts: dict[str, int],
    y_counts: dict[str, int],
    maxiter: int = 100,
) -> SelfConsistencyResult:
    """Run self-consistency loop from hardware measurement counts.

    This is the real test — K_true is the intended Hamiltonian,
    counts are from quantum hardware. The learned K_nm should
    approximate K_true, with error from noise + optimization.
    """
    n = K_true.shape[0]

    C_measured = correlators_from_counts(x_counts, y_counts, n)
    shot_std = correlator_shot_noise(x_counts, y_counts, n)

    result = learn_hamiltonian(C_measured, omega, maxiter=maxiter)

    return _build_result(K_true, result, shot_std, n)


def self_consistency_from_noisy_sim(
    K_true: np.ndarray,
    omega: np.ndarray,
    noise_std: float = 0.05,
    n_shots: int = 8192,
    maxiter: int = 100,
    seed: int = 42,
) -> SelfConsistencyResult:
    """Run self-consistency with simulated shot noise.

    Adds Gaussian noise to exact correlators to simulate finite shots.
    Noise std = noise_std (default: 0.05, roughly 1/√400).
    """
    n = K_true.shape[0]
    rng = np.random.default_rng(seed)

    C_exact = measure_correlators(K_true, omega)
    C_noisy = C_exact + rng.normal(0, noise_std, size=C_exact.shape)
    C_noisy = (C_noisy + C_noisy.T) / 2  # keep symmetric
    np.fill_diagonal(C_noisy, 0.0)

    result = learn_hamiltonian(C_noisy, omega, maxiter=maxiter)

    shot_std = float(np.sqrt(2.0 / n_shots))
    return _build_result(K_true, result, shot_std, n)


def _build_result(
    K_true: np.ndarray,
    learning: HamiltonianLearningResult,
    shot_std: float,
    n: int,
) -> SelfConsistencyResult:
    """Compute comparison metrics."""
    K_learned = learning.K_learned
    diff = K_true - K_learned
    frob = float(np.linalg.norm(diff, "fro"))
    max_err = float(np.max(np.abs(diff)))
    norm_true = float(np.linalg.norm(K_true, "fro"))
    rel_err = frob / max(norm_true, 1e-12)

    return SelfConsistencyResult(
        K_true=K_true,
        K_learned=K_learned,
        frobenius_error=frob,
        elementwise_max_error=max_err,
        relative_error=rel_err,
        correlator_error=learning.correlator_error,
        learning_result=learning,
        n_qubits=n,
        shot_noise_std=shot_std,
    )
