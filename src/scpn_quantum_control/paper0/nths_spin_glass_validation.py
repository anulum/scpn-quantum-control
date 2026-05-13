# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 NTHS spin-glass validation fixture
"""Executable simulator fixture for the Paper 0 NTHS spin-glass Hamiltonian."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .spec_loader import load_macro_transition_validation_spec


@dataclass(frozen=True, slots=True)
class SpinGlassValidationConfig:
    """Numerical limits for exact finite spin-glass validation."""

    max_exact_states: int = 4096
    replica_count: int = 8
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        if self.max_exact_states < 2:
            raise ValueError("max_exact_states must be at least 2")
        if self.replica_count < 2:
            raise ValueError("replica_count must be at least 2")


@dataclass(frozen=True, slots=True)
class SpinGlassValidationResult:
    """Result of the Paper 0 NTHS spin-glass simulator fixture."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    state_count: int
    ground_state: tuple[int, ...]
    ground_state_energy: float
    ground_state_magnetisation: float
    mean_energy: float
    edwards_anderson_q: float
    ultrametric_violation: float
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


def spin_glass_energy(spins: np.ndarray, J_ij: np.ndarray, h_i: np.ndarray) -> float:
    """Evaluate ``H=-sum_{i<j}J_ij sigma_i sigma_j - sum_i h_i sigma_i``."""
    J_arr, h_arr = _validate_couplings(J_ij, h_i)
    spin_arr = _validate_spin_vector(spins, J_arr.shape[0])
    pair_energy = -0.5 * float(np.sum(J_arr * np.outer(spin_arr, spin_arr)))
    field_energy = -float(np.dot(h_arr, spin_arr))
    return pair_energy + field_energy


def magnetisation(spins: np.ndarray) -> float:
    """Return the finite-system magnetisation for binary spins."""
    spin_arr = _validate_spin_vector(spins)
    return float(np.mean(spin_arr))


def edwards_anderson_parameter(replicas: np.ndarray) -> float:
    """Return the Edwards-Anderson order parameter for matched replicas."""
    replica_arr = _validate_replicas(replicas)
    site_means = np.mean(replica_arr, axis=0)
    return float(np.mean(site_means * site_means))


def hamming_distance_matrix(replicas: np.ndarray) -> np.ndarray:
    """Return the normalised Hamming distance matrix between binary replicas."""
    replica_arr = _validate_replicas(replicas)
    count = replica_arr.shape[0]
    distances = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            distance = float(np.mean(replica_arr[i] != replica_arr[j]))
            distances[i, j] = distance
            distances[j, i] = distance
    return distances


def ultrametric_violation(distance_matrix: np.ndarray) -> float:
    """Return the maximum three-point ultrametric excess."""
    distances = np.array(distance_matrix, dtype=np.float64, copy=True)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("distance_matrix must be square")
    if not np.all(np.isfinite(distances)):
        raise ValueError("distance_matrix must contain only finite values")
    if not np.allclose(distances, distances.T, atol=1e-12, rtol=1e-12):
        raise ValueError("distance_matrix must be symmetric")
    if not np.allclose(np.diag(distances), 0.0, atol=1e-12, rtol=0.0):
        raise ValueError("distance_matrix diagonal must be zero")

    max_excess = 0.0
    count = distances.shape[0]
    for i in range(count):
        for j in range(i + 1, count):
            for k in range(j + 1, count):
                triangle = sorted(
                    (distances[i, j], distances[i, k], distances[j, k]),
                    reverse=True,
                )
                max_excess = max(max_excess, float(triangle[0] - triangle[1]))
    return max_excess


def validate_nths_spin_glass_fixture(
    J_ij: np.ndarray,
    h_i: np.ndarray,
    *,
    config: SpinGlassValidationConfig | None = None,
) -> SpinGlassValidationResult:
    """Run the source-anchored NTHS spin-glass simulator fixture."""
    cfg = config or SpinGlassValidationConfig()
    J_arr, h_arr = _validate_couplings(J_ij, h_i)
    n_spins = J_arr.shape[0]
    state_count = 2**n_spins
    if state_count > cfg.max_exact_states:
        raise ValueError(
            f"exact state count exceeds configured budget: {state_count} > {cfg.max_exact_states}"
        )

    spec = load_macro_transition_validation_spec(
        "nths.spin_glass_hamiltonian",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    states = enumerate_spin_states(n_spins)
    energies = np.array([spin_glass_energy(state, J_arr, h_arr) for state in states])
    sorted_indices = np.argsort(energies, kind="stable")
    ground_index = int(sorted_indices[0])
    ground_state = states[ground_index]
    replica_indices = sorted_indices[: min(cfg.replica_count, state_count)]
    replicas = states[replica_indices]
    distance_matrix = hamming_distance_matrix(replicas)
    q_ea = edwards_anderson_parameter(replicas)
    controls = _spin_glass_null_controls(J_arr, h_arr, ground_state, replicas)

    metadata = {
        "n_spins": n_spins,
        "exact_enumeration": True,
        "state_count": state_count,
        "replica_count": int(replicas.shape[0]),
        "paper0_spec_key": "nths.spin_glass_hamiltonian",
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "energy_min": float(energies[ground_index]),
        "energy_max": float(np.max(energies)),
    }
    return SpinGlassValidationResult(
        spec_key="nths.spin_glass_hamiltonian",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_equation_ids=tuple(str(item) for item in spec["source_equation_ids"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        state_count=state_count,
        ground_state=tuple(int(item) for item in ground_state),
        ground_state_energy=float(energies[ground_index]),
        ground_state_magnetisation=magnetisation(ground_state),
        mean_energy=float(np.mean(energies)),
        edwards_anderson_q=q_ea,
        ultrametric_violation=ultrametric_violation(distance_matrix),
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def enumerate_spin_states(n_spins: int) -> np.ndarray:
    """Enumerate all binary spin states in lexicographic bit order."""
    if n_spins < 1:
        raise ValueError("n_spins must be positive")
    state_count = 2**n_spins
    bit_patterns = np.arange(state_count, dtype=np.uint64)[:, None]
    bit_indices = np.arange(n_spins, dtype=np.uint64)[None, :]
    bits = (bit_patterns >> bit_indices) & np.uint64(1)
    return np.where(bits == 1, 1, -1).astype(np.int8, copy=False)


def _spin_glass_null_controls(
    J_ij: np.ndarray,
    h_i: np.ndarray,
    ground_state: np.ndarray,
    replicas: np.ndarray,
) -> dict[str, float]:
    shuffled = _deterministic_shuffled_couplings(J_ij)
    zero_field = np.zeros_like(h_i)
    ferromagnetic = np.abs(J_ij)
    np.fill_diagonal(ferromagnetic, 0.0)
    aligned = np.ones(J_ij.shape[0], dtype=np.int8)
    return {
        "shuffled_coupling_energy_delta": abs(
            spin_glass_energy(ground_state, J_ij, h_i)
            - spin_glass_energy(ground_state, shuffled, h_i)
        ),
        "zero_field_energy_delta": abs(
            spin_glass_energy(ground_state, J_ij, h_i)
            - spin_glass_energy(ground_state, J_ij, zero_field)
        ),
        "ferromagnetic_aligned_magnetisation_abs": abs(magnetisation(aligned)),
        "ferromagnetic_aligned_energy": spin_glass_energy(aligned, ferromagnetic, zero_field),
        "matched_disorder_q_EA": edwards_anderson_parameter(replicas),
    }


def _deterministic_shuffled_couplings(J_ij: np.ndarray) -> np.ndarray:
    if J_ij.shape[0] < 3:
        return cast(np.ndarray, J_ij.copy())
    permutation = np.roll(np.arange(J_ij.shape[0]), 2)
    shuffled = J_ij[np.ix_(permutation, permutation)].copy()
    np.fill_diagonal(shuffled, 0.0)
    return cast(np.ndarray, shuffled)


def _validate_couplings(J_ij: np.ndarray, h_i: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    J_arr = np.array(J_ij, dtype=np.float64, copy=True)
    h_arr = np.array(h_i, dtype=np.float64, copy=True)
    if J_arr.ndim != 2 or J_arr.shape[0] != J_arr.shape[1]:
        raise ValueError(f"J_ij must be a square matrix, got shape {J_arr.shape}")
    n_spins = J_arr.shape[0]
    if h_arr.shape != (n_spins,):
        raise ValueError(f"h_i must have shape ({n_spins},), got {h_arr.shape}")
    if not np.all(np.isfinite(J_arr)):
        raise ValueError("J_ij must contain only finite values")
    if not np.all(np.isfinite(h_arr)):
        raise ValueError("h_i must contain only finite values")
    if not np.allclose(J_arr, J_arr.T, atol=1e-12, rtol=1e-12):
        raise ValueError("J_ij must be symmetric")
    np.fill_diagonal(J_arr, 0.0)
    return J_arr, h_arr


def _validate_spin_vector(spins: np.ndarray, expected_size: int | None = None) -> np.ndarray:
    spin_arr = np.array(spins, dtype=np.int8, copy=True)
    if spin_arr.ndim != 1:
        raise ValueError("spins must be a one-dimensional vector")
    if expected_size is not None and spin_arr.shape != (expected_size,):
        raise ValueError(f"spins must have shape ({expected_size},), got {spin_arr.shape}")
    if not np.all((spin_arr == -1) | (spin_arr == 1)):
        raise ValueError("spins must contain only -1 or +1 values")
    return spin_arr


def _validate_replicas(replicas: np.ndarray) -> np.ndarray:
    replica_arr = np.array(replicas, dtype=np.int8, copy=True)
    if replica_arr.ndim != 2:
        raise ValueError("replicas must be a two-dimensional array")
    if replica_arr.shape[0] < 2:
        raise ValueError("at least two replicas are required")
    if replica_arr.shape[1] < 1:
        raise ValueError("replicas must contain at least one spin")
    if not np.all((replica_arr == -1) | (replica_arr == 1)):
        raise ValueError("replicas must contain only -1 or +1 values")
    return replica_arr


__all__ = [
    "SpinGlassValidationConfig",
    "SpinGlassValidationResult",
    "edwards_anderson_parameter",
    "enumerate_spin_states",
    "hamming_distance_matrix",
    "magnetisation",
    "spin_glass_energy",
    "ultrametric_violation",
    "validate_nths_spin_glass_fixture",
]
