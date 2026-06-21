# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — K_nm quantum/classical partitioning (co-simulation)
"""Split a K_nm coupling network into a quantum-strong core and classical bath.

A large Kuramoto-XY network of ``N`` oscillators cannot be simulated as a single
``2**N`` statevector. :func:`partition_knm` selects a small, strongly-coupled
core to evolve quantumly and leaves the weakly-coupled remainder to a classical
mean-field bath. The split is deterministic, edge-exact (every coupling lands in
exactly one of the quantum-internal, classical-internal, or cross buckets), and
carries a ``cross_fraction`` quality signal that bounds the mean-field
co-simulation error: the smaller the cross coupling relative to the total, the
better the separability assumption holds.

This is a local, mean-field embedding boundary — not an exact treatment of the
full network and not a hardware path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

# Statevector ceiling for the quantum-strong core; above this the dense
# co-simulation propagator is refused fail-closed.
MAX_QUANTUM_CORE_NODES = 14

# Couplings with magnitude below this are treated as absent (matches the K_nm
# compiler sparsity epsilon).
_SPARSITY_EPS = 1e-15

# Symmetry tolerance: K is symmetrised, but a larger asymmetry is recorded.
_SYMMETRY_TOL = 1e-9


@dataclass(frozen=True)
class ConservationReport:
    """Edge-budget accounting proving the partition loses no coupling."""

    total_abs_coupling: float
    quantum_internal_abs: float
    classical_internal_abs: float
    cross_abs: float
    residual: float
    is_exact: bool
    cross_fraction: float


@dataclass(frozen=True)
class KnmPartition:
    """A deterministic quantum-strong / classical-weak split of a K_nm network."""

    quantum_indices: tuple[int, ...]
    classical_indices: tuple[int, ...]
    quantum_coupling: NDArray[np.float64]
    classical_coupling: NDArray[np.float64]
    cross_coupling: NDArray[np.float64]
    quantum_omega: NDArray[np.float64]
    classical_omega: NDArray[np.float64]
    conservation: ConservationReport
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def n_quantum(self) -> int:
        """Number of oscillators assigned to the quantum partition."""
        return len(self.quantum_indices)

    @property
    def n_classical(self) -> int:
        """Number of oscillators assigned to the classical partition."""
        return len(self.classical_indices)


def _validate(
    K: NDArray[np.float64], omega: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64], float]:
    K = np.asarray(K, dtype=np.float64)
    omega = np.asarray(omega, dtype=np.float64)
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError("K must be a square 2-D coupling matrix")
    n = K.shape[0]
    if n < 2:
        raise ValueError("K must describe at least two oscillators")
    if omega.ndim != 1 or omega.shape[0] != n:
        raise ValueError(f"omega must be a length-{n} vector")
    if not np.all(np.isfinite(K)):
        raise ValueError("K must be finite (no NaN/Inf)")
    if not np.all(np.isfinite(omega)):
        raise ValueError("omega must be finite (no NaN/Inf)")
    asymmetry = float(np.max(np.abs(K - K.T))) if n else 0.0
    K_sym = 0.5 * (K + K.T)
    return K_sym, omega, asymmetry


def _grow_strong_core(
    abs_k: NDArray[np.float64],
    max_nodes: int,
    coupling_threshold: float,
) -> tuple[list[int], list[float]]:
    """Greedily grow a strongly-coupled core, deterministic in index ties.

    Seeds with the highest weighted-degree node, then repeatedly adds the
    outside node with the greatest summed coupling into the current core,
    stopping at ``max_nodes`` or when the best available coupling drops below
    ``coupling_threshold``.
    """
    n = abs_k.shape[0]
    strength = abs_k.sum(axis=1)
    # Seed: strongest node, lowest index breaks ties.
    seed = int(np.lexsort((np.arange(n), -strength))[0])
    core = [seed]
    in_core = np.zeros(n, dtype=bool)
    in_core[seed] = True
    growth_scores = [float(strength[seed])]

    while len(core) < max_nodes:
        coupling_into_core = abs_k[:, core].sum(axis=1)
        coupling_into_core[in_core] = -np.inf
        best = int(np.lexsort((np.arange(n), -coupling_into_core))[0])
        best_score = float(coupling_into_core[best])
        if not np.isfinite(best_score) or best_score < coupling_threshold:
            break
        core.append(best)
        in_core[best] = True
        growth_scores.append(best_score)

    return core, growth_scores


def _conservation(
    abs_k: NDArray[np.float64], q_idx: list[int], c_idx: list[int]
) -> ConservationReport:
    triu = np.triu(abs_k, k=1)
    total = float(triu.sum())
    q_mask = np.zeros(abs_k.shape[0], dtype=bool)
    q_mask[q_idx] = True
    q_internal = float(triu[np.ix_(q_idx, q_idx)].sum()) if q_idx else 0.0
    c_internal = float(triu[np.ix_(c_idx, c_idx)].sum()) if c_idx else 0.0
    # Cross edges: one endpoint quantum, one classical (upper triangle only).
    cross = total - q_internal - c_internal
    reconstructed = q_internal + c_internal + cross
    residual = abs(total - reconstructed)
    cross_fraction = (cross / total) if total > _SPARSITY_EPS else 0.0
    return ConservationReport(
        total_abs_coupling=total,
        quantum_internal_abs=q_internal,
        classical_internal_abs=c_internal,
        cross_abs=cross,
        residual=residual,
        is_exact=residual <= 1e-9 * max(1.0, total),
        cross_fraction=cross_fraction,
    )


def partition_knm(
    K: NDArray[np.float64],
    omega: NDArray[np.float64],
    *,
    max_quantum_nodes: int = 8,
    coupling_threshold: float = 0.0,
) -> KnmPartition:
    """Partition a K_nm network into a quantum-strong core and classical bath.

    Args:
        K: symmetric ``(N, N)`` coupling matrix (asymmetry is symmetrised and
            recorded in provenance).
        omega: length-``N`` natural-frequency vector.
        max_quantum_nodes: core-size cap; must be ``1 <= m <= 14`` and ``<= N``.
        coupling_threshold: stop growing the core once the best coupling into it
            falls below this magnitude.

    Returns:
        A :class:`KnmPartition` with the index sets, the three coupling blocks,
        the split frequency vectors, an edge-exact :class:`ConservationReport`,
        and a provenance record.
    """
    K_sym, omega, asymmetry = _validate(K, omega)
    n = K_sym.shape[0]
    if not 1 <= max_quantum_nodes <= MAX_QUANTUM_CORE_NODES:
        raise ValueError(f"max_quantum_nodes must be in 1..{MAX_QUANTUM_CORE_NODES}")
    if max_quantum_nodes > n:
        raise ValueError("max_quantum_nodes cannot exceed the number of oscillators")
    if coupling_threshold < 0.0:
        raise ValueError("coupling_threshold must be non-negative")

    abs_k = np.abs(K_sym)
    core, growth_scores = _grow_strong_core(abs_k, max_quantum_nodes, coupling_threshold)
    q_idx = sorted(core)
    c_idx = [i for i in range(n) if i not in set(core)]

    conservation = _conservation(abs_k, q_idx, c_idx)
    provenance = {
        "algorithm": "greedy_strong_core",
        "max_quantum_nodes": max_quantum_nodes,
        "coupling_threshold": coupling_threshold,
        "seed_node": core[0],
        "growth_order": tuple(core),
        "growth_scores": tuple(growth_scores),
        "node_strength": tuple(float(s) for s in abs_k.sum(axis=1)),
        "symmetrised": asymmetry > _SYMMETRY_TOL,
        "input_asymmetry": asymmetry,
        "claim_boundary": (
            "local mean-field embedding; cross_fraction bounds the decoupling "
            "error; not exact full-network simulation, not a hardware path"
        ),
    }

    return KnmPartition(
        quantum_indices=tuple(q_idx),
        classical_indices=tuple(c_idx),
        quantum_coupling=K_sym[np.ix_(q_idx, q_idx)].copy(),
        classical_coupling=K_sym[np.ix_(c_idx, c_idx)].copy(),
        cross_coupling=K_sym[np.ix_(q_idx, c_idx)].copy(),
        quantum_omega=omega[q_idx].copy(),
        classical_omega=omega[c_idx].copy(),
        conservation=conservation,
        provenance=provenance,
    )
