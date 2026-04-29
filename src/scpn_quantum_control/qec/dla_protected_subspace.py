# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — DLA-Protected Logical Synchronisation
"""DLA-protected logical synchronisation subspaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from qiskit import QuantumCircuit

from ..analysis.dla_parity_theorem import parity_sector_dimensions, predicted_dla_dimension

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class DLAProtectedSubspaceSpec:
    """Specification for a fixed-parity logical repetition-code sector."""

    n_logical: int
    code_distance: int = 3
    target_parity: int = 0
    min_protected_weight: float = 0.9
    min_sync_weight: float = 0.75
    max_parity_leakage: float = 0.05
    max_code_leakage: float = 0.1
    sync_agreement_threshold: float = 1.0

    def __post_init__(self) -> None:
        _require_positive_int(self.n_logical, "n_logical")
        _require_positive_int(self.code_distance, "code_distance")
        if self.code_distance % 2 == 0:
            raise ValueError("code_distance must be odd so block parity tracks the logical bit")
        if self.target_parity not in (0, 1):
            raise ValueError("target_parity must be 0 or 1")
        for value, name in (
            (self.min_protected_weight, "min_protected_weight"),
            (self.min_sync_weight, "min_sync_weight"),
            (self.max_parity_leakage, "max_parity_leakage"),
            (self.max_code_leakage, "max_code_leakage"),
            (self.sync_agreement_threshold, "sync_agreement_threshold"),
        ):
            _require_unit_interval(value, name)

    @property
    def n_physical(self) -> int:
        """Number of physical qubits in the logical memory register."""
        return self.n_logical * self.code_distance

    @property
    def hilbert_dim(self) -> int:
        """Dense Hilbert-space dimension of the physical register."""
        if self.n_physical > 24:
            raise ValueError("dense protected-subspace masks are limited to 24 physical qubits")
        return 1 << self.n_physical

    @property
    def protected_logical_dim(self) -> int:
        """Number of repetition-code logical words in the target parity sector."""
        return 1 << max(self.n_logical - 1, 0)


@dataclass(frozen=True)
class DLAProtectionCertificate:
    """Analytic certificate for the protected logical sector."""

    spec: DLAProtectedSubspaceSpec
    physical_dla_dimension: int
    even_sector_dim: int
    odd_sector_dim: int
    protected_logical_dim: int
    protected_basis_indices: tuple[int, ...]
    sync_basis_indices: tuple[int, ...]
    protected_logical_words: tuple[tuple[int, ...], ...]
    sync_logical_words: tuple[tuple[int, ...], ...]
    proof_obligations: dict[str, bool]

    @property
    def is_provable(self) -> bool:
        """True when all finite-dimensional certificate obligations hold."""
        return all(self.proof_obligations.values())

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable certificate summary."""
        return {
            "n_logical": self.spec.n_logical,
            "code_distance": self.spec.code_distance,
            "target_parity": self.spec.target_parity,
            "n_physical": self.spec.n_physical,
            "physical_dla_dimension": self.physical_dla_dimension,
            "even_sector_dim": self.even_sector_dim,
            "odd_sector_dim": self.odd_sector_dim,
            "protected_logical_dim": self.protected_logical_dim,
            "n_protected_basis_indices": len(self.protected_basis_indices),
            "n_sync_basis_indices": len(self.sync_basis_indices),
            "proof_obligations": self.proof_obligations,
            "is_provable": self.is_provable,
        }


@dataclass(frozen=True)
class DLAProtectedMemoryPrototype:
    """Circuit-level prototype for one protected logical memory word."""

    spec: DLAProtectedSubspaceSpec
    logical_word: tuple[int, ...]
    basis_index: int
    circuit: QuantumCircuit
    certificate: DLAProtectionCertificate

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable prototype summary."""
        return {
            "logical_word": list(self.logical_word),
            "basis_index": self.basis_index,
            "n_qubits": self.circuit.num_qubits,
            "depth": self.circuit.depth(),
            "certificate": self.certificate.to_dict(),
        }


@dataclass(frozen=True)
class DLAProtectedWitnessResult:
    """Witness result for a DLA-protected logical synchronisation memory."""

    spec: DLAProtectedSubspaceSpec
    protected_weight: float
    code_weight: float
    target_parity_weight: float
    opposite_parity_weight: float
    sync_weight: float
    logical_sync_order: float
    total_weight: float
    failure_reasons: tuple[str, ...] = field(default_factory=tuple)

    @property
    def protected_leakage(self) -> float:
        """Probability mass outside the fixed-parity repetition-code sector."""
        return max(0.0, self.total_weight - self.protected_weight)

    @property
    def code_leakage(self) -> float:
        """Probability mass outside repetition-code memory blocks."""
        return max(0.0, self.total_weight - self.code_weight)

    @property
    def parity_leakage(self) -> float:
        """Probability mass in the opposite physical parity sector."""
        return self.opposite_parity_weight

    @property
    def passes(self) -> bool:
        """True iff all configured failure criteria pass."""
        return not self.failure_reasons

    def to_dict(self) -> dict[str, Any]:
        """Return a serialisable witness report."""
        return {
            "protected_weight": self.protected_weight,
            "code_weight": self.code_weight,
            "target_parity_weight": self.target_parity_weight,
            "opposite_parity_weight": self.opposite_parity_weight,
            "sync_weight": self.sync_weight,
            "logical_sync_order": self.logical_sync_order,
            "total_weight": self.total_weight,
            "protected_leakage": self.protected_leakage,
            "code_leakage": self.code_leakage,
            "parity_leakage": self.parity_leakage,
            "passes": self.passes,
            "failure_reasons": list(self.failure_reasons),
        }


class DLAProtectedLogicalSyncWitness:
    """Evaluate fixed-parity logical memory and synchronisation criteria."""

    def __init__(self, spec: DLAProtectedSubspaceSpec | None = None) -> None:
        self.spec = spec or DLAProtectedSubspaceSpec(n_logical=2)

    def __call__(
        self,
        probabilities: FloatArray | None = None,
        *,
        counts: Mapping[str, int] | None = None,
    ) -> DLAProtectedWitnessResult:
        """Evaluate a probability vector or measurement counts."""
        return evaluate_dla_protected_memory(
            probabilities=probabilities,
            counts=counts,
            spec=self.spec,
        )


def certify_dla_protected_subspace(
    spec: DLAProtectedSubspaceSpec,
) -> DLAProtectionCertificate:
    """Build an analytic DLA parity certificate for a logical memory sector."""
    even_dim, odd_dim = parity_sector_dimensions(spec.n_physical)
    protected_words = protected_logical_words(spec)
    sync_words = tuple(
        word
        for word in protected_words
        if _logical_agreement(word) >= spec.sync_agreement_threshold
    )
    protected_indices = tuple(
        _basis_index_from_logical_word(word, spec.code_distance) for word in protected_words
    )
    sync_indices = tuple(
        _basis_index_from_logical_word(word, spec.code_distance) for word in sync_words
    )
    obligations = {
        "odd_repetition_distance": spec.code_distance % 2 == 1,
        "fixed_global_parity": all(
            sum(word) % 2 == spec.target_parity for word in protected_words
        ),
        "sector_dimension_matches": len(protected_indices) == spec.protected_logical_dim,
        "sync_subspace_inside_protected": set(sync_indices).issubset(protected_indices),
        "xy_dla_has_two_parity_sectors": predicted_dla_dimension(spec.n_physical)
        == 2 * ((1 << (spec.n_physical - 1)) ** 2 - 1),
    }
    return DLAProtectionCertificate(
        spec=spec,
        physical_dla_dimension=predicted_dla_dimension(spec.n_physical),
        even_sector_dim=even_dim,
        odd_sector_dim=odd_dim,
        protected_logical_dim=spec.protected_logical_dim,
        protected_basis_indices=protected_indices,
        sync_basis_indices=sync_indices,
        protected_logical_words=protected_words,
        sync_logical_words=sync_words,
        proof_obligations=obligations,
    )


def build_dla_protected_memory_prototype(
    spec: DLAProtectedSubspaceSpec,
    logical_word: Sequence[int] | None = None,
) -> DLAProtectedMemoryPrototype:
    """Build a circuit preparing one protected logical repetition-code word."""
    resolved = tuple(int(bit) for bit in (logical_word or _default_sync_word(spec)))
    _validate_logical_word(resolved, spec)
    circuit = QuantumCircuit(spec.n_physical, name="dla_protected_sync_memory")
    for logical, bit in enumerate(resolved):
        if bit == 0:
            continue
        start = logical * spec.code_distance
        for qubit in range(start, start + spec.code_distance):
            circuit.x(qubit)
    return DLAProtectedMemoryPrototype(
        spec=spec,
        logical_word=resolved,
        basis_index=_basis_index_from_logical_word(resolved, spec.code_distance),
        circuit=circuit,
        certificate=certify_dla_protected_subspace(spec),
    )


def protected_memory_mask(spec: DLAProtectedSubspaceSpec) -> BoolArray:
    """Return a dense mask for the fixed-parity repetition-code sector."""
    try:
        import scpn_quantum_engine as _engine

        if hasattr(_engine, "dla_protected_memory_mask"):
            return np.asarray(
                _engine.dla_protected_memory_mask(
                    spec.n_logical,
                    spec.code_distance,
                    spec.target_parity,
                ),
                dtype=np.bool_,
            )
    except (ImportError, AttributeError, ValueError):
        pass
    return _protected_memory_mask_numpy(spec)


def sync_memory_mask(spec: DLAProtectedSubspaceSpec) -> BoolArray:
    """Return a dense mask for synchronised words inside the protected sector."""
    mask = np.zeros(spec.hilbert_dim, dtype=np.bool_)
    for word in certify_dla_protected_subspace(spec).sync_logical_words:
        mask[_basis_index_from_logical_word(word, spec.code_distance)] = True
    return mask


def protected_logical_words(spec: DLAProtectedSubspaceSpec) -> tuple[tuple[int, ...], ...]:
    """Enumerate logical words whose parity equals the target DLA sector."""
    words: list[tuple[int, ...]] = []
    for word_index in range(1 << spec.n_logical):
        word = tuple((word_index >> bit) & 1 for bit in range(spec.n_logical))
        if sum(word) % 2 == spec.target_parity:
            words.append(word)
    return tuple(words)


def evaluate_dla_protected_memory(
    probabilities: FloatArray | None = None,
    *,
    counts: Mapping[str, int] | None = None,
    spec: DLAProtectedSubspaceSpec,
) -> DLAProtectedWitnessResult:
    """Evaluate memory, parity, and synchronisation weights."""
    probs = _resolve_probabilities(probabilities=probabilities, counts=counts, spec=spec)
    protected_weight, code_weight, target_weight, opposite_weight, total_weight = _memory_metrics(
        probs,
        spec,
    )
    sync_weight = float(np.sum(probs[sync_memory_mask(spec)]))
    logical_sync_order = _weighted_logical_sync_order(probs, spec)
    failures = _failure_reasons(
        spec,
        protected_weight=protected_weight,
        sync_weight=sync_weight,
        parity_leakage=opposite_weight,
        code_leakage=max(0.0, total_weight - code_weight),
    )
    return DLAProtectedWitnessResult(
        spec=spec,
        protected_weight=protected_weight,
        code_weight=code_weight,
        target_parity_weight=target_weight,
        opposite_parity_weight=opposite_weight,
        sync_weight=sync_weight,
        logical_sync_order=logical_sync_order,
        total_weight=total_weight,
        failure_reasons=failures,
    )


def _memory_metrics(
    probabilities: FloatArray, spec: DLAProtectedSubspaceSpec
) -> tuple[float, float, float, float, float]:
    try:
        import scpn_quantum_engine as _engine

        if hasattr(_engine, "dla_protected_memory_metrics"):
            protected, code, target, opposite, total = _engine.dla_protected_memory_metrics(
                np.ascontiguousarray(probabilities, dtype=np.float64),
                spec.n_logical,
                spec.code_distance,
                spec.target_parity,
            )
            return (
                float(protected),
                float(code),
                float(target),
                float(opposite),
                float(total),
            )
    except (ImportError, AttributeError, ValueError):
        pass
    return _memory_metrics_numpy(probabilities, spec)


def _protected_memory_mask_numpy(spec: DLAProtectedSubspaceSpec) -> BoolArray:
    mask = np.zeros(spec.hilbert_dim, dtype=np.bool_)
    for word in protected_logical_words(spec):
        mask[_basis_index_from_logical_word(word, spec.code_distance)] = True
    return mask


def _memory_metrics_numpy(
    probabilities: FloatArray, spec: DLAProtectedSubspaceSpec
) -> tuple[float, float, float, float, float]:
    probs = _validate_probabilities(probabilities, spec)
    protected = float(np.sum(probs[protected_memory_mask(spec)]))
    code = 0.0
    for word in _all_logical_words(spec.n_logical):
        code += float(probs[_basis_index_from_logical_word(word, spec.code_distance)])
    parity_mask = np.fromiter(
        ((index.bit_count() % 2) == spec.target_parity for index in range(spec.hilbert_dim)),
        dtype=np.bool_,
        count=spec.hilbert_dim,
    )
    target = float(np.sum(probs[parity_mask]))
    opposite = float(np.sum(probs[~parity_mask]))
    return protected, code, target, opposite, float(np.sum(probs))


def _resolve_probabilities(
    *,
    probabilities: FloatArray | None,
    counts: Mapping[str, int] | None,
    spec: DLAProtectedSubspaceSpec,
) -> FloatArray:
    if probabilities is not None and counts is not None:
        raise ValueError("provide either probabilities or counts, not both")
    if counts is not None:
        return _probabilities_from_counts(counts, spec)
    if probabilities is None:
        raise ValueError("probabilities or counts must be provided")
    return _validate_probabilities(probabilities, spec)


def _probabilities_from_counts(
    counts: Mapping[str, int], spec: DLAProtectedSubspaceSpec
) -> FloatArray:
    probs = np.zeros(spec.hilbert_dim, dtype=np.float64)
    total = int(sum(counts.values()))
    if total <= 0:
        raise ValueError("counts must contain positive shot total")
    for bitstring, shots in counts.items():
        clean = bitstring.replace(" ", "")
        if len(clean) != spec.n_physical or any(bit not in "01" for bit in clean):
            raise ValueError(
                f"bitstrings must have length {spec.n_physical} and contain only 0/1 bits"
            )
        if shots < 0:
            raise ValueError("counts must be non-negative")
        probs[int(clean[::-1], 2)] += shots / total
    return probs


def _validate_probabilities(
    probabilities: FloatArray, spec: DLAProtectedSubspaceSpec
) -> FloatArray:
    probs = np.asarray(probabilities, dtype=np.float64)
    if probs.shape != (spec.hilbert_dim,):
        raise ValueError(f"probabilities must have shape ({spec.hilbert_dim},), got {probs.shape}")
    if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
        raise ValueError("probabilities must contain finite non-negative values")
    return np.ascontiguousarray(probs, dtype=np.float64)


def _failure_reasons(
    spec: DLAProtectedSubspaceSpec,
    *,
    protected_weight: float,
    sync_weight: float,
    parity_leakage: float,
    code_leakage: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if protected_weight < spec.min_protected_weight:
        reasons.append("protected_weight_below_threshold")
    if sync_weight < spec.min_sync_weight:
        reasons.append("sync_weight_below_threshold")
    if parity_leakage > spec.max_parity_leakage:
        reasons.append("parity_leakage_above_threshold")
    if code_leakage > spec.max_code_leakage:
        reasons.append("code_leakage_above_threshold")
    return tuple(reasons)


def _weighted_logical_sync_order(
    probabilities: FloatArray, spec: DLAProtectedSubspaceSpec
) -> float:
    numerator = 0.0
    denominator = 0.0
    for word in protected_logical_words(spec):
        weight = float(probabilities[_basis_index_from_logical_word(word, spec.code_distance)])
        numerator += weight * _logical_agreement(word)
        denominator += weight
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


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


def _all_logical_words(n_logical: int) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple((word_index >> bit) & 1 for bit in range(n_logical))
        for word_index in range(1 << n_logical)
    )


def _basis_index_from_logical_word(word: Sequence[int], code_distance: int) -> int:
    index = 0
    block_mask = (1 << code_distance) - 1
    for logical, bit in enumerate(word):
        if int(bit) == 1:
            index |= block_mask << (logical * code_distance)
    return index


def _default_sync_word(spec: DLAProtectedSubspaceSpec) -> tuple[int, ...]:
    zeros = tuple(0 for _ in range(spec.n_logical))
    if sum(zeros) % 2 == spec.target_parity:
        return zeros
    ones = tuple(1 for _ in range(spec.n_logical))
    if sum(ones) % 2 == spec.target_parity:
        return ones
    for word in protected_logical_words(spec):
        return word
    raise ValueError("no protected logical word exists")


def _validate_logical_word(word: tuple[int, ...], spec: DLAProtectedSubspaceSpec) -> None:
    if len(word) != spec.n_logical:
        raise ValueError(f"logical_word must contain {spec.n_logical} bits")
    if any(bit not in (0, 1) for bit in word):
        raise ValueError("logical_word must contain only 0/1 bits")
    if sum(word) % 2 != spec.target_parity:
        raise ValueError("logical_word does not match the target DLA parity sector")


def _require_positive_int(value: int, name: str) -> None:
    if int(value) != value or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _require_unit_interval(value: float, name: str) -> None:
    if not np.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
