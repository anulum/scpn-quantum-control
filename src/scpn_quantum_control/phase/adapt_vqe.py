# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Adaptive layered VQE
"""Adaptive layered VQE for the Kuramoto-XY ground state.

Grows a variational ansatz built from a physics-motivated operator pool
(Grimsley et al., Nature Comms 10, 3007 (2019)) until the energy converges, and
returns the variational ground-state estimate.

Why this is *layered* and not gradient-selected ADAPT
-----------------------------------------------------
The original ADAPT-VQE selection rule grows the ansatz one operator at a time by
the operator gradient ``g_k = ⟨ψ|[H, A_k]|ψ⟩``. For the Kuramoto-XY Hamiltonian
``H = -Σ K_ij(X_iX_j + Y_iY_j) - Σ ω_i Z_i`` that rule is ill-conditioned: H is
real-symmetric in the computational basis, so for *any real* state ψ every pool
gradient is identically zero — the ``i(X_iX_j+Y_iY_j)`` exchange generators give
``i ψᵀ[H,G]ψ = 0`` because ``[H,G]`` is real-antisymmetric, and the ``iY_i``
generators give zero because H has no single-spin-flip terms. Starting from the
real reference ``|0…0⟩`` (which is moreover an eigenstate, energy ``-Σω``) the
selection therefore stalls at iteration zero with no operators and the energy of
an *excited* eigenstate, while falsely reporting convergence.

The same real-state stationarity makes the *energy* gradient vanish at zero
angles, trapping small-angle optimisation in a local minimum. Both pathologies
are removed by (1) the symmetric reference ``|+⟩^{⊗n}``, (2) random non-zero
angle initialisation with restarts, and (3) growing the ansatz by full pool
layers. With these, the variational optimum reaches the exact ground state for
the systems we can diagonalise.

Operator pool (per Grimsley et al. structure)
    - i(X_iX_j + Y_iY_j) exchange generators for each coupled pair (i, j)
    - i Y_i single-qubit generators for each site i
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp
from scipy.optimize import minimize

from ..bridge.knm_hamiltonian import knm_to_dense_matrix
from ..dense_budget import require_dense_allocation

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass
class ADAPTResult:
    """Adaptive layered VQE result."""

    energy: float
    n_iterations: int  # number of ansatz layers
    n_parameters: int
    gradient_norms: list[float]  # optimiser final gradient norm per layer
    energies: list[float]  # reference energy, then best energy after each layer
    selected_operators: list[int]  # pool indices used (the full pool, per layer)
    converged: bool


def _build_operator_pool(K: FloatArray, n: int) -> list[SparsePauliOp]:
    """Build the anti-Hermitian operator pool from the K_nm coupling topology.

    Pool elements:
        - For each coupled pair (i, j): i(X_iX_j + Y_iY_j)
        - For each qubit i: i Y_i
    """
    pool: list[SparsePauliOp] = []

    for i in range(n):
        for j in range(i + 1, n):
            if abs(K[i, j]) > 1e-10:
                xx = ["I"] * n
                xx[i] = "X"
                xx[j] = "X"
                yy = ["I"] * n
                yy[i] = "Y"
                yy[j] = "Y"
                op = SparsePauliOp(
                    ["".join(reversed(xx)), "".join(reversed(yy))],
                    coeffs=[1j, 1j],
                )
                pool.append(op)

    for i in range(n):
        label = ["I"] * n
        label[i] = "Y"
        pool.append(SparsePauliOp("".join(reversed(label)), coeffs=[1j]))

    return pool


def _pool_generators_dense(K: FloatArray, n: int) -> list[ComplexArray]:
    """Dense Hermitian generators ``G_k`` derived from :func:`_build_operator_pool`.

    The pool stores anti-Hermitian ``τ_k = i G_k``; the variational gate is the
    standard ``exp(-i θ_k G_k)``, so each generator is recovered as ``τ_k / i``.
    Deriving them from the same pool keeps a single source of truth, so the
    ``selected_operators`` indices map back to ``_build_operator_pool``.
    """
    return [np.asarray((op / 1j).to_matrix(), dtype=complex) for op in _build_operator_pool(K, n)]


def _plus_reference(n: int) -> ComplexArray:
    """The symmetric reference state ``|+⟩^{⊗n}`` as a dense statevector."""
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2.0)
    state = np.ones(1, dtype=complex)
    for _ in range(n):
        state = np.kron(plus, state)
    return state


def _generator_spectra(
    generators: list[ComplexArray],
) -> list[tuple[FloatArray, ComplexArray]]:
    """Eigendecompose each Hermitian generator once for fast ``exp(-iθG)`` action."""
    return [np.linalg.eigh(generator) for generator in generators]


def _ansatz_state(
    reference: ComplexArray,
    layer_spectra: list[tuple[FloatArray, ComplexArray]],
    angles: FloatArray,
) -> ComplexArray:
    """Apply ``exp(-i θ_k G_k)`` via cached spectra: ``V (e^{-iθλ} ⊙ (V† ψ))``."""
    state = reference
    for (eigvals, eigvecs), theta in zip(layer_spectra, angles, strict=True):
        state = eigvecs @ (np.exp(-1j * theta * eigvals) * (eigvecs.conj().T @ state))
    return state


def adapt_vqe(
    K: FloatArray,
    omega: FloatArray,
    max_iterations: int = 20,
    gradient_threshold: float = 1e-3,
    maxiter_opt: int = 200,
    seed: int | None = None,
    *,
    n_restarts: int = 4,
    max_dense_gib: float | None = None,
) -> ADAPTResult:
    """Adaptive layered VQE for the Kuramoto-XY Hamiltonian.

    Grows the ansatz one full pool layer at a time, optimising all angles with
    random-restart BFGS, until the best energy stops improving by more than
    ``gradient_threshold``. Returns the variational ground-state estimate.

    Args:
        K: coupling matrix.
        omega: natural frequencies.
        max_iterations: maximum number of ansatz layers.
        gradient_threshold: energy-improvement convergence threshold between layers.
        maxiter_opt: BFGS iterations per restart.
        seed: random seed for angle initialisation (reproducible).
        n_restarts: random restarts per layer (escapes the real-state local minima).
        max_dense_gib: dense exact-statevector budget for local simulation.
    """
    n = K.shape[0]
    if n > 10:
        raise ValueError(f"n={n} too large for dense ADAPT statevector (max 10)")
    require_dense_allocation(
        n,
        rank=1,
        max_gib=max_dense_gib,
        label="ADAPT-VQE statevector",
    )

    hamiltonian = knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib)
    generators = _pool_generators_dense(K, n)
    spectra = _generator_spectra(generators)
    pool_size = len(generators)
    reference = _plus_reference(n)
    rng = np.random.default_rng(seed)

    def energy_of(state: ComplexArray) -> float:
        return float((state.conj() @ hamiltonian @ state).real)

    reference_energy = energy_of(reference)
    energies: list[float] = [reference_energy]
    grad_norms: list[float] = []
    best_energy = reference_energy
    best_angles: FloatArray = np.zeros(0)
    converged = False
    layers = 0

    for layer in range(1, max_iterations + 1):
        layer_spectra = spectra * layer
        n_params = len(layer_spectra)

        def cost(
            angles: FloatArray,
            spectra_seq: list[tuple[FloatArray, ComplexArray]] = layer_spectra,
        ) -> float:
            return energy_of(_ansatz_state(reference, spectra_seq, angles))

        layer_best_energy = np.inf
        layer_best_angles: FloatArray = np.zeros(n_params)
        layer_best_grad = np.inf
        for _restart in range(max(1, n_restarts)):
            x0 = rng.uniform(-np.pi, np.pi, n_params)
            result = minimize(cost, x0, method="BFGS", options={"maxiter": maxiter_opt})
            if float(result.fun) < layer_best_energy:
                layer_best_energy = float(result.fun)
                layer_best_angles = np.asarray(result.x, dtype=float)
                layer_best_grad = float(np.linalg.norm(np.asarray(result.jac, dtype=float)))

        energies.append(layer_best_energy)
        grad_norms.append(layer_best_grad)
        improvement = best_energy - layer_best_energy
        best_energy = layer_best_energy
        best_angles = layer_best_angles
        layers = layer

        if improvement < gradient_threshold:
            converged = True
            break

    selected_operators = (list(range(pool_size)) * layers) if pool_size else []

    return ADAPTResult(
        energy=best_energy,
        n_iterations=layers,
        n_parameters=len(best_angles),
        gradient_norms=grad_norms,
        energies=energies,
        selected_operators=selected_operators,
        converged=converged,
    )
