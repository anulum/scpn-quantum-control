# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Mps Baseline
"""Entropy-based MPS storage heuristics, distinct from solver qualification.

Half-chain entropy gives a lower bound on Schmidt rank, not the exact rank or
the bond dimension needed for a chosen observable error. Exact representation
requires adequate ranks at every cut. The legacy threshold and tractability
fields remain heuristic estimates; they establish neither a simulation-error
bound nor quantum advantage. Actual bond/refinement evidence comes from the
public MPS solver diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..analysis.entanglement_spectrum import entanglement_entropy_half_chain


@dataclass
class MPSBaselineResult:
    """MPS storage estimates from half-chain entropy.

    The legacy ``required_bond_dim`` field is a lower bound, and
    ``mps_tractable`` means only that this lower bound fits ``chi_max``.
    Neither field certifies exact representation or observable convergence.
    ``quantum_advantage_threshold`` is a volume-law heuristic, not evidence
    of hardware advantage or classical simulation hardness.
    """

    n_qubits: int
    half_chain_entropy: float
    required_bond_dim: int  # entropy lower bound, not exact Schmidt rank
    mps_memory_bytes: int  # χ² × n × sizeof(complex128)
    exact_memory_bytes: int  # 2^n × sizeof(complex128)
    compression_ratio: float  # exact / mps
    quantum_advantage_threshold: int  # legacy volume-law heuristic
    mps_tractable: bool  # entropy lower bound is within χ_max
    bond_dimension_basis: str = "half-chain entropy lower bound"
    qualification_scope: str = "storage heuristic; observable convergence not established"


def required_bond_dimension(entropy: float) -> int:
    """Estimate the entropy lower bound ``ceil(2**S)``.

    Parameters
    ----------
    entropy
        Half-chain von Neumann entropy in bits.

    Returns
    -------
    int
        A lower bound on exact Schmidt rank, not the exact required bond.

    """
    if entropy < 1e-10:
        return 1
    return max(int(np.ceil(2**entropy)), 1)


def mps_memory(n: int, chi: int) -> int:
    """Estimate uniform-bond complex128 MPS tensor storage.

    Parameters
    ----------
    n
        Number of spin-half sites.
    chi
        Assumed uniform bond dimension.

    Returns
    -------
    int
        ``n * 2 * chi**2 * 16`` bytes, excluding solver workspace.

    """
    # n tensors of size (chi, 2, chi) = n × 2 × chi²
    return n * 2 * chi * chi * 16  # 16 bytes per complex128


def exact_memory(n: int) -> int:
    """Estimate dense complex128 statevector storage.

    Parameters
    ----------
    n
        Number of spin-half sites.

    Returns
    -------
    int
        ``2**n * 16`` bytes; no statevector is allocated.

    """
    return int((2**n) * 16)


def quantum_advantage_n(
    chi_max: int = 1024,
    entropy_per_qubit: float = 0.5,
) -> int:
    """Evaluate the legacy assumed-volume-law bond-cap heuristic.

    Parameters
    ----------
    chi_max
        Assumed maximum classical bond dimension.
    entropy_per_qubit
        Assumed half-chain entropy slope in bits per qubit.

    Returns
    -------
    int
        Estimated cap-crossing size; 1000 is the legacy zero-slope sentinel.
        This is not measured simulation hardness or quantum advantage.

    """
    # Volume law: S = entropy_per_qubit × n/2
    # χ = 2^S = 2^{entropy_per_qubit × n/2}
    # Fails when χ > χ_max: n > 2 × log2(χ_max) / entropy_per_qubit
    if entropy_per_qubit < 1e-10:
        return 1000  # always tractable
    return int(np.ceil(2.0 * np.log2(chi_max) / entropy_per_qubit))


def mps_baseline_comparison(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    chi_max: int = 256,
) -> MPSBaselineResult:
    """Return half-chain entropy storage heuristics for the existing model.

    Parameters
    ----------
    K
        Existing Kuramoto-XY coupling matrix, shape ``(n, n)``.
    omega
        Existing model frequencies, shape ``(n,)``.
    chi_max
        Assumed bond cap used only for the entropy lower-bound comparison.

    Returns
    -------
    MPSBaselineResult
        Entropy-based storage estimates with explicit qualification limits.
        No MPS dynamics or observable convergence is established here.

    """
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
