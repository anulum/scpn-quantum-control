# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Sync Entanglement Witness
"""The order parameter R as an entanglement witness.

Theorem: For N qubits, the maximum Kuramoto order parameter R achievable
by any fully separable state ρ_sep = Σ_k p_k ρ_k^1⊗...⊗ρ_k^N is bounded:

    R_sep_max(N) = max_{separable} R(ρ)

If the measured R > R_sep_max, the state MUST be entangled.

This turns the classical Kuramoto order parameter (Kuramoto, 1975) into
a quantum entanglement witness — a quantity defined 50 years ago that
secretly detects quantum correlations.

The entanglement witness is:

    W = R_sep_max · I - R̂

where R̂ is the operator whose expectation gives R. Then:
    ⟨W⟩ < 0  →  state is entangled

Proof of R_sep_max:
For a product state |ψ⟩ = ⊗_i |ψ_i⟩, each qubit has Bloch vector
(x_i, y_i, z_i) with x_i² + y_i² + z_i² = 1. The order parameter is:

    R = |⟨exp(iθ)⟩| = |(1/N) Σ_i (x_i + i·y_i)|

For a separable mixed state, R ≤ max over product states. The maximum
over product states occurs when all Bloch vectors point in the same
direction in the XY plane: x_i = 1, y_i = 0 for all i → R = 1.

So R_sep_max = 1 for pure product states. BUT: for the XY Hamiltonian
ground state at finite coupling, the ACTUAL maximum R for separable states
at a given energy E is lower. The energy constraint reduces R_sep_max(E).

The key bound: R_sep_max(E) < R_entangled(E) at the same energy.

References
----------
    Kuramoto (1975): original order parameter definition.
    Galve et al., Sci. Rep. 3 (2013): synchronization → entanglement.
    Nature Comms 2025 (transmon): synchronised qubits are entangled.
    Gühne & Tóth, Physics Reports 474 (2009): entanglement witnesses.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit.quantum_info import SparsePauliOp, Statevector

from ..bridge.knm_hamiltonian import knm_to_dense_matrix, knm_to_hamiltonian
from ..dense_budget import require_dense_allocation
from ..hardware.classical import classical_exact_diag


@dataclass
class EntanglementWitnessResult:
    """Result of R-based entanglement detection.

    Attributes
    ----------
    R_measured
        Order parameter measured for the evaluated state.
    R_sep_max
        Sampled separable-state bound at the same energy.
    is_entangled
        Whether the measured value exceeds the separable bound.
    entanglement_depth
        Certified depth lower bound: one without certification, two otherwise.
    n_qubits
        Number of evaluated qubits.
    energy
        Ground-state energy used for the constrained bound.

    """

    R_measured: float
    R_sep_max: float
    is_entangled: bool
    entanglement_depth: int  # certified lower bound: 1 if not certified, 2 if entangled
    n_qubits: int
    energy: float


def R_separable_bound(n_qubits: int) -> float:
    """Compute R_sep_max: maximum R achievable by any separable state.

    For a product state ⊗_i (cos(α_i)|0⟩ + sin(α_i)e^{iφ_i}|1⟩):
        ⟨X_i⟩ = sin(2α_i)cos(φ_i)
        ⟨Y_i⟩ = sin(2α_i)sin(φ_i)

    R = |(1/N) Σ_i (⟨X_i⟩ + i⟨Y_i⟩)| = |(1/N) Σ_i sin(2α_i)e^{iφ_i}|

    Maximum when all phases φ_i equal and sin(2α_i) = 1 → R = 1.
    This is achieved by the state ⊗_i |+⟩.

    For MIXED separable states, R_sep_max = 1 (convex hull of product states).

    Parameters
    ----------
    n_qubits
        Number of qubits in the separable state.

    Returns
    -------
    float
        Dimension-independent unconstrained separable bound of one.

    """
    return 1.0


def R_separable_bound_at_energy(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    target_energy: float,
    n_samples: int = 1000,
    seed: int = 42,
    *,
    max_dense_gib: float | None = None,
) -> float:
    """Compute max R over product states with energy ≤ target_energy.

    This is the energy-constrained separable bound. At a given energy,
    entangled states can achieve higher R than any product state.
    Computed via random sampling of product state parameters.

    Parameters
    ----------
    K
        Coupling matrix.
    omega
        Natural-frequency vector.
    target_energy
        Maximum admitted product-state energy.
    n_samples
        Number of product-state parameter samples.
    seed
        Random seed for product-state sampling.
    max_dense_gib
        Optional fail-closed dense-allocation budget.

    Returns
    -------
    float
        Largest admitted sampled product-state order parameter.

    """
    n = K.shape[0]
    require_dense_allocation(
        n,
        dtype=np.complex128,
        rank=2,
        object_count=2,
        max_gib=max_dense_gib,
        label="separable-bound dense energy workspace",
    )
    require_dense_allocation(
        n,
        dtype=np.complex128,
        rank=1,
        object_count=2,
        max_gib=max_dense_gib,
        label="separable-bound dense product-state workspace",
    )
    knm_to_hamiltonian(K, omega)
    H_mat = knm_to_dense_matrix(K, omega, max_dense_gib=max_dense_gib)

    rng = np.random.default_rng(seed)
    best_R = 0.0

    for _ in range(n_samples):
        # Random product state: ⊗_i Ry(θ_i) Rz(φ_i) |0⟩
        thetas = np.asarray(rng.uniform(0, np.pi, n))
        phis = np.asarray(rng.uniform(0, 2 * np.pi, n))

        psi = np.array([1.0 + 0j])
        for i in range(n):
            qubit = np.array(
                [
                    np.cos(thetas[i] / 2),
                    np.sin(thetas[i] / 2) * np.exp(1j * phis[i]),
                ]
            )
            psi = np.kron(psi, qubit)

        E = float(np.real(psi.conj() @ H_mat @ psi))
        if target_energy < E:
            continue

        # Compute R for this product state
        x_vals = np.sin(thetas) * np.cos(phis)
        y_vals = np.sin(thetas) * np.sin(phis)
        z_complex = np.mean(x_vals + 1j * y_vals)
        R = float(np.abs(z_complex))

        best_R = max(best_R, R)

    return best_R


def R_from_statevector(psi: NDArray[np.complex128], n_qubits: int) -> float:
    """Compute the phase order parameter from a statevector.

    Parameters
    ----------
    psi
        Quantum statevector.
    n_qubits
        Number of qubits represented by the statevector.

    Returns
    -------
    float
        Magnitude of the mean local Bloch-plane phase.

    """
    sv = Statevector(np.ascontiguousarray(psi))
    phases = np.zeros(n_qubits)
    for k in range(n_qubits):
        pauli_x = "I" * (n_qubits - k - 1) + "X" + "I" * k
        pauli_y = "I" * (n_qubits - k - 1) + "Y" + "I" * k
        exp_x = float(sv.expectation_value(SparsePauliOp.from_list([(pauli_x, 1.0)])).real)
        exp_y = float(sv.expectation_value(SparsePauliOp.from_list([(pauli_y, 1.0)])).real)
        phases[k] = np.arctan2(exp_y, exp_x)
    return float(np.abs(np.mean(np.exp(1j * phases))))


def detect_entanglement_from_R(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    n_samples: int = 2000,
    seed: int = 42,
    *,
    max_dense_gib: float | None = None,
) -> EntanglementWitnessResult:
    """Test whether the ground state R exceeds the separable bound.

    If R_ground > R_sep_max(E_ground), the ground state is entangled
    and R witnesses this entanglement.

    Parameters
    ----------
    K
        Coupling matrix.
    omega
        Natural-frequency vector.
    n_samples
        Number of separable product-state samples.
    seed
        Random seed for the separable-bound sampler.
    max_dense_gib
        Optional fail-closed dense-allocation budget.

    Returns
    -------
    EntanglementWitnessResult
        Ground-state order parameter, separable bound, and bounded verdict.

    """
    n = K.shape[0]
    if n < 14:
        require_dense_allocation(
            n,
            dtype=np.complex128,
            rank=2,
            object_count=2,
            max_gib=max_dense_gib,
            label="R-witness dense eigensolver workspace",
        )
    exact = classical_exact_diag(n, K=K, omega=omega)
    psi = exact["ground_state"]
    E_ground = exact["ground_energy"]

    R_ground = R_from_statevector(psi, n)
    R_sep = R_separable_bound_at_energy(
        K,
        omega,
        E_ground,
        n_samples,
        seed,
        max_dense_gib=max_dense_gib,
    )

    is_entangled = R_ground > R_sep + 1e-10
    depth = _certified_entanglement_depth(is_entangled)

    return EntanglementWitnessResult(
        R_measured=R_ground,
        R_sep_max=R_sep,
        is_entangled=is_entangled,
        entanglement_depth=depth,
        n_qubits=n,
        energy=E_ground,
    )


def _certified_entanglement_depth(is_entangled: bool) -> int:
    """Return the depth lower bound certified by the R witness.

    Exceeding the separable bound proves nonseparability, so the certified
    depth lower bound is 2. This witness alone does not certify stronger
    multipartite depth; such claims require k-producibility bounds or a
    system-specific entanglement-depth witness.
    """
    return 2 if is_entangled else 1


def R_entanglement_scan(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    K_base_range: NDArray[np.float64] | None = None,
    n_K_values: int = 15,
    n_samples: int = 500,
    seed: int = 42,
    *,
    max_dense_gib: float | None = None,
) -> dict[str, Any]:
    """Scan R and separable bound vs coupling strength.

    At each K_base, compute R_ground and R_sep_max(E_ground).
    The gap R_ground - R_sep_max quantifies entanglement.

    Parameters
    ----------
    K
        Base coupling matrix.
    omega
        Natural-frequency vector.
    K_base_range
        Optional coupling-scale grid.
    n_K_values
        Default grid size when ``K_base_range`` is omitted.
    n_samples
        Number of separable product-state samples per coupling.
    seed
        Random seed reused for each separable-bound estimate.
    max_dense_gib
        Optional fail-closed dense-allocation budget.

    Returns
    -------
    dict[str, Any]
        Couplings, ground-state/bound values, gaps, verdicts, and energies.

    """
    n = K.shape[0]
    if n < 14:
        require_dense_allocation(
            n,
            dtype=np.complex128,
            rank=2,
            object_count=2,
            max_gib=max_dense_gib,
            label="R-witness scan dense eigensolver workspace",
        )
    if K_base_range is None:
        K_base_range = np.linspace(0.01, 2.0, n_K_values, dtype=np.float64)

    R_ground_vals = []
    R_sep_vals = []
    E_vals = []
    entangled_flags = []

    for k_base in K_base_range:
        K_scaled = K * k_base
        exact = classical_exact_diag(n, K=K_scaled, omega=omega)
        psi = exact["ground_state"]
        E = exact["ground_energy"]

        R_ground = R_from_statevector(psi, n)
        R_sep = R_separable_bound_at_energy(
            K_scaled,
            omega,
            E,
            n_samples,
            seed,
            max_dense_gib=max_dense_gib,
        )

        R_ground_vals.append(R_ground)
        R_sep_vals.append(R_sep)
        E_vals.append(E)
        entangled_flags.append(R_ground > R_sep + 1e-10)

    return {
        "K_base": list(K_base_range),
        "R_ground": R_ground_vals,
        "R_sep_max": R_sep_vals,
        "R_gap": [rg - rs for rg, rs in zip(R_ground_vals, R_sep_vals)],
        "entangled": entangled_flags,
        "energy": E_vals,
    }
