# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Topology-aware fidelity kernels
"""Validated exact-statevector and classical-control kernel construction."""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray

from scpn_quantum_control.applications.quantum_kernel import (
    canonical_edge_pairs,
    encode_topology_edge_features,
)

from .schema import FloatArray, TopologyKernelConfig, TopologyKernelMatrix

_NUMERIC_CUSTODY_DECIMALS = 12


def _float_bytes(value: NDArray[np.float64]) -> bytes:
    rounded = np.round(np.asarray(value, dtype=np.float64), _NUMERIC_CUSTODY_DECIMALS)
    canonical = np.where(rounded == 0.0, 0.0, rounded)
    return np.ascontiguousarray(canonical, dtype="<f8").tobytes()


def _matrix_digest(
    values: FloatArray,
    row_ids: tuple[str, ...],
    column_ids: tuple[str, ...],
    topology_digest: str,
) -> str:
    digest = hashlib.sha256()
    digest.update(_float_bytes(values))
    digest.update("\x00".join(row_ids).encode())
    digest.update(b"\xff")
    digest.update("\x00".join(column_ids).encode())
    digest.update(topology_digest.encode())
    return digest.hexdigest()


def validate_topology(
    topology: NDArray[np.float64],
    config: TopologyKernelConfig,
) -> FloatArray:
    """Return a read-only validated coupling topology.

    Parameters
    ----------
    topology:
        Finite symmetric ``(n_qubits, n_qubits)`` matrix with zero diagonal.
        Signed, weighted off-diagonal couplings are accepted.
    config:
        Kernel configuration fixing ``n_qubits``.

    Returns
    -------
    numpy.ndarray
        Defensive read-only ``float64`` copy.

    Raises
    ------
    ValueError
        If the matrix violates shape, finiteness, symmetry, or diagonal
        requirements.

    """
    if not isinstance(config, TopologyKernelConfig):
        raise ValueError("config must be a TopologyKernelConfig")
    matrix = np.asarray(topology, dtype=np.float64)
    expected = (config.n_qubits, config.n_qubits)
    if matrix.shape != expected:
        raise ValueError(f"topology must have shape {expected}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("topology must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=1.0e-12):
        raise ValueError("topology must be symmetric")
    if not np.allclose(np.diag(matrix), 0.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("topology diagonal must be zero")
    result = np.array(matrix, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def topology_digest(topology: NDArray[np.float64], config: TopologyKernelConfig) -> str:
    """Return SHA-256 custody for an exact validated topology matrix.

    The digest binds row-major little-endian ``float64`` bytes and the qubit
    count. Numerically equal matrices with different input dtypes therefore
    receive the same digest after validation.
    """
    matrix = validate_topology(topology, config)
    digest = hashlib.sha256()
    digest.update(config.n_qubits.to_bytes(2, "big"))
    digest.update(_float_bytes(matrix))
    return digest.hexdigest()


def validate_feature_matrix(
    features: NDArray[np.float64],
    config: TopologyKernelConfig,
    *,
    name: str = "features",
) -> FloatArray:
    """Return a finite read-only edge-feature matrix within sample budgets.

    Each row must contain exactly ``config.feature_dim`` values, ordered as
    :func:`scpn_quantum_control.applications.canonical_edge_pairs`.
    """
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1:] != (config.feature_dim,):
        raise ValueError(f"{name} must have shape (n_samples, {config.feature_dim})")
    if not 1 <= matrix.shape[0] <= config.max_samples:
        raise ValueError(f"{name} sample count exceeds the configured budget")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(matrix, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def fidelity_kernel_matrix(
    row_features: NDArray[np.float64],
    column_features: NDArray[np.float64],
    topology: NDArray[np.float64],
    config: TopologyKernelConfig,
    *,
    row_ids: tuple[str, ...],
    column_ids: tuple[str, ...],
) -> TopologyKernelMatrix:
    """Evaluate a topology-aware exact-statevector fidelity kernel.

    Parameters
    ----------
    row_features, column_features:
        Edge-feature matrices in canonical upper-triangle order. Each axis is
        independently bounded by ``config.max_samples``.
    topology:
        Validated graph coupling matrix. A zero off-diagonal entry suppresses
        its aligned feature.
    config:
        Qubit, simulation, and resource policy.
    row_ids, column_ids:
        Unique identifiers matching the two feature axes.

    Returns
    -------
    TopologyKernelMatrix
        Matrix with entries ``|<phi(x_i)|phi(y_j)>|**2`` plus topology and
        content digests.

    Notes
    -----
    This routine performs dense local statevector simulation. It neither
    submits jobs nor infers performance on a quantum processor.

    """
    rows = validate_feature_matrix(row_features, config, name="row_features")
    columns = validate_feature_matrix(column_features, config, name="column_features")
    if len(row_ids) != rows.shape[0] or len(column_ids) != columns.shape[0]:
        raise ValueError("sample identifiers must match their feature axes")
    coupling = validate_topology(topology, config)
    digest = topology_digest(coupling, config)
    row_states = np.vstack(
        [
            encode_topology_edge_features(
                row,
                coupling,
                config.n_qubits,
                t=config.evolution_time,
                reps=config.trotter_reps,
                max_qubits=8,
            ).data
            for row in rows
        ]
    )
    column_states = np.vstack(
        [
            encode_topology_edge_features(
                column,
                coupling,
                config.n_qubits,
                t=config.evolution_time,
                reps=config.trotter_reps,
                max_qubits=8,
            ).data
            for column in columns
        ]
    )
    values = np.asarray(np.abs(row_states.conj() @ column_states.T) ** 2, dtype=np.float64)
    values = np.clip(values, 0.0, 1.0)
    content_digest = _matrix_digest(values, row_ids, column_ids, digest)
    return TopologyKernelMatrix(
        values=values,
        row_ids=row_ids,
        column_ids=column_ids,
        topology_digest=digest,
        content_digest=content_digest,
    )


def rbf_kernel_matrix(
    row_features: NDArray[np.float64],
    column_features: NDArray[np.float64],
    config: TopologyKernelConfig,
    *,
    gamma: float,
    row_ids: tuple[str, ...],
    column_ids: tuple[str, ...],
) -> TopologyKernelMatrix:
    """Evaluate a classical radial-basis-function comparison kernel.

    ``gamma`` must be finite and positive. The returned ``topology_digest`` is
    a stable digest of the literal control label ``classical-rbf`` and gamma;
    it must not be interpreted as graph custody.
    """
    rows = validate_feature_matrix(row_features, config, name="row_features")
    columns = validate_feature_matrix(column_features, config, name="column_features")
    if len(row_ids) != rows.shape[0] or len(column_ids) != columns.shape[0]:
        raise ValueError("sample identifiers must match their feature axes")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and positive")
    distances = np.sum((rows[:, None, :] - columns[None, :, :]) ** 2, axis=2)
    values = np.asarray(np.exp(-float(gamma) * distances), dtype=np.float64)
    digest = hashlib.sha256(f"classical-rbf:{float(gamma):.17g}".encode()).hexdigest()
    return TopologyKernelMatrix(
        values=values,
        row_ids=row_ids,
        column_ids=column_ids,
        topology_digest=digest,
        content_digest=_matrix_digest(values, row_ids, column_ids, digest),
    )


def permute_topology(
    topology: NDArray[np.float64],
    permutation: tuple[int, ...],
    config: TopologyKernelConfig,
) -> FloatArray:
    """Relabel graph nodes and return the corresponding read-only topology.

    ``permutation[new_node]`` names the original node represented at each new
    position. It must be a complete permutation of ``range(n_qubits)``.
    """
    if len(permutation) != config.n_qubits or set(permutation) != set(range(config.n_qubits)):
        raise ValueError("permutation must contain every node exactly once")
    matrix = validate_topology(topology, config)
    indices = np.asarray(permutation, dtype=np.int64)
    result = np.array(matrix[np.ix_(indices, indices)], dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def permute_edge_features(
    features: NDArray[np.float64],
    permutation: tuple[int, ...],
    config: TopologyKernelConfig,
) -> FloatArray:
    """Relabel canonical edge features consistently with graph nodes.

    For every new pair ``(i, j)``, the returned feature is read from the old
    canonical pair joining ``permutation[i]`` and ``permutation[j]``. Applying
    this together with :func:`permute_topology` should preserve fidelities.
    """
    if len(permutation) != config.n_qubits or set(permutation) != set(range(config.n_qubits)):
        raise ValueError("permutation must contain every node exactly once")
    matrix = validate_feature_matrix(features, config)
    pairs = canonical_edge_pairs(config.n_qubits)
    edge_to_index = {edge: index for index, edge in enumerate(pairs)}
    columns: list[int] = []
    for i, j in pairs:
        old_i, old_j = permutation[i], permutation[j]
        old_edge = (old_i, old_j) if old_i < old_j else (old_j, old_i)
        columns.append(edge_to_index[old_edge])
    result = np.array(matrix[:, columns], dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result
