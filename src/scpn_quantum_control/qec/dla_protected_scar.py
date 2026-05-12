# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA-Protected Scar Memory
"""DLA-protected scar-memory prototypes with finite-time revivals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

from .dla_protected_subspace import (
    DLAProtectedSubspaceSpec,
    DLAProtectionCertificate,
    certify_dla_protected_subspace,
    evaluate_dla_protected_memory,
)

FloatArray: TypeAlias = NDArray[np.float64]
ComplexArray: TypeAlias = NDArray[np.complex128]


@dataclass(frozen=True)
class DLAProtectedScarSpec:
    """Configuration for a protected scar-memory revival prototype."""

    memory_spec: DLAProtectedSubspaceSpec = field(
        default_factory=lambda: DLAProtectedSubspaceSpec(
            n_logical=4,
            code_distance=3,
            target_parity=0,
        )
    )
    revival_period: float = 1.0
    n_time_steps: int = 16
    min_revival_fidelity: float = 0.99
    min_protected_weight: float = 0.99
    min_scar_support: float = 0.99
    max_parity_leakage: float = 1e-12

    def __post_init__(self) -> None:
        if self.revival_period <= 0.0 or not np.isfinite(self.revival_period):
            raise ValueError("revival_period must be finite and positive")
        if int(self.n_time_steps) != self.n_time_steps or self.n_time_steps < 2:
            raise ValueError("n_time_steps must be an integer >= 2")
        for value, name in (
            (self.min_revival_fidelity, "min_revival_fidelity"),
            (self.min_protected_weight, "min_protected_weight"),
            (self.min_scar_support, "min_scar_support"),
            (self.max_parity_leakage, "max_parity_leakage"),
        ):
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

    @property
    def energy_spacing(self) -> float:
        """Energy spacing that produces a full revival at ``revival_period``."""
        return float(2.0 * np.pi / self.revival_period)


@dataclass(frozen=True)
class DLAProtectedScarPrototype:
    """Circuit, state, and diagonal Hamiltonian for a protected scar memory."""

    spec: DLAProtectedScarSpec
    certificate: DLAProtectionCertificate
    scar_logical_words: tuple[tuple[int, ...], ...]
    scar_basis_indices: tuple[int, ...]
    initial_state: ComplexArray
    hamiltonian_diagonal: FloatArray
    preparation_circuit: QuantumCircuit

    @property
    def n_scar_states(self) -> int:
        """Number of protected basis states carrying the scar packet."""
        return len(self.scar_basis_indices)

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable prototype summary."""
        return {
            "n_logical": self.spec.memory_spec.n_logical,
            "code_distance": self.spec.memory_spec.code_distance,
            "target_parity": self.spec.memory_spec.target_parity,
            "revival_period": self.spec.revival_period,
            "energy_spacing": self.spec.energy_spacing,
            "n_scar_states": self.n_scar_states,
            "scar_basis_indices": list(self.scar_basis_indices),
            "preparation_depth": self.preparation_circuit.depth(),
            "certificate": self.certificate.to_dict(),
        }


@dataclass(frozen=True)
class DLAProtectedScarSimulationResult:
    """Revival and leakage metrics for a protected scar-memory trajectory."""

    prototype: DLAProtectedScarPrototype
    times: FloatArray
    survival_probability: FloatArray
    protected_weight: FloatArray
    code_weight: FloatArray
    target_parity_weight: FloatArray
    opposite_parity_weight: FloatArray
    total_weight: FloatArray
    scar_support: FloatArray
    failure_reasons: tuple[str, ...] = field(default_factory=tuple)
    backend: str = "python:numpy_phase_evolution"

    @property
    def final_revival_fidelity(self) -> float:
        """Survival probability at the final sampled time."""
        return float(self.survival_probability[-1])

    @property
    def min_protected_weight(self) -> float:
        """Smallest probability weight in the protected memory sector."""
        return float(np.min(self.protected_weight))

    @property
    def max_parity_leakage(self) -> float:
        """Largest probability weight in the opposite DLA parity sector."""
        return float(np.max(self.opposite_parity_weight))

    @property
    def min_scar_support(self) -> float:
        """Smallest probability weight carried by the scar basis states."""
        return float(np.min(self.scar_support))

    @property
    def midcycle_survival(self) -> float:
        """Survival probability at the sample nearest half a revival period."""
        half_time = 0.5 * self.prototype.spec.revival_period
        index = int(np.argmin(np.abs(self.times - half_time)))
        return float(self.survival_probability[index])

    @property
    def passes(self) -> bool:
        """True iff the revival and leakage criteria pass."""
        return not self.failure_reasons

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable simulation report."""
        return {
            "final_revival_fidelity": self.final_revival_fidelity,
            "midcycle_survival": self.midcycle_survival,
            "min_protected_weight": self.min_protected_weight,
            "max_parity_leakage": self.max_parity_leakage,
            "min_scar_support": self.min_scar_support,
            "passes": self.passes,
            "failure_reasons": list(self.failure_reasons),
            "backend": self.backend,
            "prototype": self.prototype.to_dict(),
        }


def build_dla_protected_scar_prototype(
    spec: DLAProtectedScarSpec | None = None,
    *,
    scar_logical_words: Sequence[Sequence[int]] | None = None,
) -> DLAProtectedScarPrototype:
    """Build a protected two-word scar memory with an exact revival period."""
    resolved_spec = spec or DLAProtectedScarSpec()
    certificate = certify_dla_protected_subspace(resolved_spec.memory_spec)
    resolved_words = _resolve_scar_words(certificate, scar_logical_words)
    basis_indices = tuple(
        _basis_index_from_logical_word(word, resolved_spec.memory_spec.code_distance)
        for word in resolved_words
    )
    initial_state = _equal_superposition_state(
        basis_indices,
        resolved_spec.memory_spec.hilbert_dim,
    )
    hamiltonian_diagonal = _scar_hamiltonian_diagonal(
        basis_indices,
        resolved_spec.memory_spec.hilbert_dim,
        resolved_spec.energy_spacing,
    )
    circuit = _scar_preparation_circuit(resolved_spec.memory_spec, resolved_words, initial_state)
    return DLAProtectedScarPrototype(
        spec=resolved_spec,
        certificate=certificate,
        scar_logical_words=resolved_words,
        scar_basis_indices=basis_indices,
        initial_state=initial_state,
        hamiltonian_diagonal=hamiltonian_diagonal,
        preparation_circuit=circuit,
    )


def simulate_dla_protected_scar_memory(
    prototype: DLAProtectedScarPrototype | None = None,
    *,
    spec: DLAProtectedScarSpec | None = None,
) -> DLAProtectedScarSimulationResult:
    """Simulate protected scar-memory revival and leakage metrics."""
    resolved = prototype or build_dla_protected_scar_prototype(spec)
    times = np.linspace(
        0.0,
        resolved.spec.revival_period,
        resolved.spec.n_time_steps + 1,
        dtype=np.float64,
    )
    states = _evolve_diagonal(resolved.initial_state, resolved.hamiltonian_diagonal, times)
    probabilities = np.ascontiguousarray(np.abs(states) ** 2, dtype=np.float64)
    survival = np.asarray(
        np.abs(states @ np.conjugate(resolved.initial_state)) ** 2,
        dtype=np.float64,
    )
    protected, code, target, opposite, total, backend = _trajectory_memory_metrics(
        probabilities,
        resolved.spec.memory_spec,
    )
    scar_support = np.asarray(
        np.sum(probabilities[:, list(resolved.scar_basis_indices)], axis=1),
        dtype=np.float64,
    )
    failures = _scar_failure_reasons(
        resolved,
        final_revival_fidelity=float(survival[-1]),
        min_protected_weight=float(np.min(protected)),
        max_parity_leakage=float(np.max(opposite)),
        min_scar_support=float(np.min(scar_support)),
    )
    return DLAProtectedScarSimulationResult(
        prototype=resolved,
        times=times,
        survival_probability=survival,
        protected_weight=protected,
        code_weight=code,
        target_parity_weight=target,
        opposite_parity_weight=opposite,
        total_weight=total,
        scar_support=scar_support,
        failure_reasons=failures,
        backend=backend,
    )


def evaluate_dla_protected_scar_counts(
    counts_by_time: Sequence[dict[str, int]],
    *,
    prototype: DLAProtectedScarPrototype,
) -> DLAProtectedScarSimulationResult:
    """Evaluate measured count snapshots against the scar-memory criteria."""
    spec = prototype.spec.memory_spec
    probabilities = np.vstack(
        [
            _probabilities_from_counts(counts, spec.n_physical, spec.hilbert_dim)
            for counts in counts_by_time
        ]
    )
    times = np.linspace(
        0.0,
        prototype.spec.revival_period,
        len(counts_by_time),
        dtype=np.float64,
    )
    protected, code, target, opposite, total, backend = _trajectory_memory_metrics(
        probabilities,
        spec,
    )
    scar_support = np.asarray(
        np.sum(probabilities[:, list(prototype.scar_basis_indices)], axis=1),
        dtype=np.float64,
    )
    failures = _scar_failure_reasons(
        prototype,
        final_revival_fidelity=float(scar_support[-1]),
        min_protected_weight=float(np.min(protected)),
        max_parity_leakage=float(np.max(opposite)),
        min_scar_support=float(np.min(scar_support)),
    )
    return DLAProtectedScarSimulationResult(
        prototype=prototype,
        times=times,
        survival_probability=scar_support,
        protected_weight=protected,
        code_weight=code,
        target_parity_weight=target,
        opposite_parity_weight=opposite,
        total_weight=total,
        scar_support=scar_support,
        failure_reasons=failures,
        backend=f"{backend}:counts",
    )


def _resolve_scar_words(
    certificate: DLAProtectionCertificate,
    requested: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, ...], ...]:
    if requested is not None:
        words = tuple(tuple(int(bit) for bit in word) for word in requested)
    else:
        words = _default_sync_pair(certificate)
    if len(words) < 2:
        raise ValueError("at least two scar logical words are required for a revival")
    protected = set(certificate.protected_logical_words)
    for word in words:
        if word not in protected:
            raise ValueError("scar logical words must lie in the protected DLA sector")
        if _logical_agreement(word) < certificate.spec.sync_agreement_threshold:
            raise ValueError("scar logical words must satisfy the synchronisation threshold")
    return words


def _default_sync_pair(
    certificate: DLAProtectionCertificate,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    zeros = tuple(0 for _ in range(certificate.spec.n_logical))
    ones = tuple(1 for _ in range(certificate.spec.n_logical))
    candidates = tuple(word for word in (zeros, ones) if word in certificate.sync_logical_words)
    if len(candidates) >= 2:
        return candidates[0], candidates[1]
    if len(certificate.sync_logical_words) >= 2:
        return certificate.sync_logical_words[0], certificate.sync_logical_words[1]
    raise ValueError("the protected sync sector must contain at least two scar words")


def _equal_superposition_state(indices: Sequence[int], dim: int) -> ComplexArray:
    state = np.zeros(dim, dtype=np.complex128)
    amplitude = 1.0 / np.sqrt(len(indices))
    for index in indices:
        state[int(index)] = amplitude
    return state


def _scar_hamiltonian_diagonal(
    indices: Sequence[int],
    dim: int,
    energy_spacing: float,
) -> FloatArray:
    diagonal = np.zeros(dim, dtype=np.float64)
    for level, index in enumerate(indices):
        diagonal[int(index)] = level * energy_spacing
    return diagonal


def _evolve_diagonal(
    initial_state: ComplexArray,
    hamiltonian_diagonal: FloatArray,
    times: FloatArray,
) -> ComplexArray:
    phases = np.exp(-1j * np.outer(times, hamiltonian_diagonal))
    return np.asarray(phases * initial_state[np.newaxis, :], dtype=np.complex128)


def _trajectory_memory_metrics(
    probabilities: FloatArray,
    spec: DLAProtectedSubspaceSpec,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, str]:
    try:
        import scpn_quantum_engine as engine

        if hasattr(engine, "dla_protected_trajectory_metrics"):
            protected, code, target, opposite, total = engine.dla_protected_trajectory_metrics(
                np.ascontiguousarray(probabilities, dtype=np.float64),
                spec.n_logical,
                spec.code_distance,
                spec.target_parity,
            )
            return (
                np.asarray(protected, dtype=np.float64),
                np.asarray(code, dtype=np.float64),
                np.asarray(target, dtype=np.float64),
                np.asarray(opposite, dtype=np.float64),
                np.asarray(total, dtype=np.float64),
                "rust:dla_protected_trajectory_metrics",
            )
    except (ImportError, AttributeError, ValueError):
        pass

    protected_values = []
    code_values = []
    target_values = []
    opposite_values = []
    total_values = []
    for row in probabilities:
        result = evaluate_dla_protected_memory(row, spec=spec)
        protected_values.append(result.protected_weight)
        code_values.append(result.code_weight)
        target_values.append(result.target_parity_weight)
        opposite_values.append(result.opposite_parity_weight)
        total_values.append(result.total_weight)
    return (
        np.asarray(protected_values, dtype=np.float64),
        np.asarray(code_values, dtype=np.float64),
        np.asarray(target_values, dtype=np.float64),
        np.asarray(opposite_values, dtype=np.float64),
        np.asarray(total_values, dtype=np.float64),
        "python:evaluate_dla_protected_memory",
    )


def _scar_failure_reasons(
    prototype: DLAProtectedScarPrototype,
    *,
    final_revival_fidelity: float,
    min_protected_weight: float,
    max_parity_leakage: float,
    min_scar_support: float,
) -> tuple[str, ...]:
    spec = prototype.spec
    reasons: list[str] = []
    if not prototype.certificate.is_provable:
        reasons.append("protection_certificate_failed")
    if final_revival_fidelity < spec.min_revival_fidelity:
        reasons.append("revival_fidelity_below_threshold")
    if min_protected_weight < spec.min_protected_weight:
        reasons.append("protected_weight_below_threshold")
    if max_parity_leakage > spec.max_parity_leakage:
        reasons.append("parity_leakage_above_threshold")
    if min_scar_support < spec.min_scar_support:
        reasons.append("scar_support_below_threshold")
    return tuple(reasons)


def _scar_preparation_circuit(
    spec: DLAProtectedSubspaceSpec,
    words: tuple[tuple[int, ...], ...],
    initial_state: ComplexArray,
) -> QuantumCircuit:
    circuit = QuantumCircuit(spec.n_physical, name="dla_protected_scar_memory")
    zeros = tuple(0 for _ in range(spec.n_logical))
    ones = tuple(1 for _ in range(spec.n_logical))
    if len(words) == 2 and set(words) == {zeros, ones}:
        circuit.h(0)
        for qubit in range(1, spec.n_physical):
            circuit.cx(0, qubit)
        return circuit
    circuit.initialize(initial_state, range(spec.n_physical))
    return circuit


def _probabilities_from_counts(counts: dict[str, int], n_qubits: int, dim: int) -> FloatArray:
    total = int(sum(counts.values()))
    if total <= 0:
        raise ValueError("counts must contain positive shot total")
    probabilities = np.zeros(dim, dtype=np.float64)
    for bitstring, shots in counts.items():
        clean = bitstring.replace(" ", "")
        if len(clean) != n_qubits or any(bit not in "01" for bit in clean):
            raise ValueError(f"bitstrings must have length {n_qubits} and contain only 0/1 bits")
        if shots < 0:
            raise ValueError("counts must be non-negative")
        probabilities[int(clean[::-1], 2)] += shots / total
    return probabilities


def _basis_index_from_logical_word(word: Sequence[int], code_distance: int) -> int:
    index = 0
    block_mask = (1 << code_distance) - 1
    for logical, bit in enumerate(word):
        if int(bit) == 1:
            index |= block_mask << (logical * code_distance)
    return index


def _logical_agreement(word: Sequence[int]) -> float:
    if len(word) < 2:
        return 1.0
    same = 0
    total = 0
    for i, left in enumerate(word):
        for right in word[i + 1 :]:
            same += int(left == right)
            total += 1
    return same / total
