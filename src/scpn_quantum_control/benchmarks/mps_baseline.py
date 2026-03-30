# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Mps Baseline
"""MPS tensor network baseline for quantum advantage comparison.

Matrix Product States (MPS) can efficiently simulate 1D quantum
systems with bounded entanglement. The bond dimension χ controls
the expressibility:

    - χ = 1: product states only (no entanglement)
    - χ = 2^(n/2): exact (full Hilbert space)
    - χ ~ poly(n): efficient classical simulation

For the Kuramoto-XY system, MPS provides the classical baseline:
if MPS at bond dimension χ matches the quantum simulation, there
is no quantum advantage at that system size.

The quantum advantage boundary is where MPS fails:
    S(n/2) > log2(χ_max)  →  MPS truncation error > ε

This module estimates:
    1. Required bond dimension for given accuracy
    2. MPS memory and time cost
    3. Comparison with quantum simulation resources
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.entanglement_spectrum import entanglement_entropy_half_chain


@dataclass
class MPSBaselineResult:
    """MPS classical baseline comparison."""

    n_qubits: int
    half_chain_entropy: float
    required_bond_dim: int  # χ needed for exact representation
    mps_memory_bytes: int  # χ² × n × sizeof(complex128)
    exact_memory_bytes: int  # 2^n × sizeof(complex128)
    compression_ratio: float  # exact / mps
    quantum_advantage_threshold: int  # n where MPS fails at χ_max
    mps_tractable: bool  # χ_required < χ_max


def required_bond_dimension(entropy: float) -> int:
    """Minimum bond dimension to represent state with given entanglement.

    χ >= 2^S where S is the half-chain entanglement entropy.
    """
    if entropy < 1e-10:
        return 1
    return max(int(np.ceil(2**entropy)), 1)


def mps_memory(n: int, chi: int) -> int:
    """Memory for MPS representation: n × χ² × 2 × sizeof(complex128)."""
    # n tensors of size (chi, 2, chi) = n × 2 × chi²
    return n * 2 * chi * chi * 16  # 16 bytes per complex128


def exact_memory(n: int) -> int:
    """Memory for full statevector: 2^n × sizeof(complex128)."""
    return int((2**n) * 16)


def quantum_advantage_n(
    chi_max: int = 1024,
    entropy_per_qubit: float = 0.5,
) -> int:
    """Estimate n where MPS at χ_max fails.

    At the BKT critical point, S ~ (c/3) log(n) with c=1.
    MPS needs χ ~ n^{1/3} at criticality — always tractable.

    For volume-law entanglement (above BKT), S ~ n/2.
    MPS needs χ ~ 2^{n/2} — exponential, quantum advantage.
    """
    # Volume law: S = entropy_per_qubit × n/2
    # χ = 2^S = 2^{entropy_per_qubit × n/2}
    # Fails when χ > χ_max: n > 2 × log2(χ_max) / entropy_per_qubit
    if entropy_per_qubit < 1e-10:
        return 1000  # always tractable
    return int(np.ceil(2.0 * np.log2(chi_max) / entropy_per_qubit))


def mps_baseline_comparison(
    K: np.ndarray,
    omega: np.ndarray,
    chi_max: int = 256,
) -> MPSBaselineResult:
    """Compare exact simulation with MPS classical baseline."""
    n = K.shape[0]
    s_half = entanglement_entropy_half_chain(K, omega)
    chi_req = required_bond_dimension(s_half)

    mem_mps = mps_memory(n, chi_req)
    mem_exact = exact_memory(n)
    compression = mem_exact / max(mem_mps, 1)

    tractable = chi_req <= chi_max
    advantage_n = quantum_advantage_n(chi_max)

    return MPSBaselineResult(
        n_qubits=n,
        half_chain_entropy=s_half,
        required_bond_dim=chi_req,
        mps_memory_bytes=mem_mps,
        exact_memory_bytes=mem_exact,
        compression_ratio=compression,
        quantum_advantage_threshold=advantage_n,
        mps_tractable=tractable,
    )
