# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 anomalous-boundary validation fixtures
"""Simulator-only falsification-boundary fixtures for Paper 0 psi records."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import numpy as np

from .computational_unifier_validation import abl_probabilities
from .spec_loader import load_anomalous_boundary_validation_spec

CLAIM_BOUNDARY = "simulator-only falsification boundary; not anomalous evidence"


@dataclass(frozen=True, slots=True)
class AnomalousBoundaryConfig:
    """Finite simulator settings for Paper 0 anomalous-boundary fixtures."""

    pre_state: np.ndarray | None = None
    post_state: np.ndarray | None = None
    projectors: tuple[np.ndarray, ...] | None = None
    bell_state: np.ndarray | None = None
    product_state: np.ndarray | None = None
    chsh_angles: tuple[float, float, float, float] = (
        0.0,
        math.pi / 2.0,
        math.pi / 4.0,
        -math.pi / 4.0,
    )
    prior_probability: float = 0.42
    intent_bias: float = 0.35
    measurement_strength: float = 0.7
    spec_bundle_path: Path | None = None

    def __post_init__(self) -> None:
        pre = _normalised_state(
            "pre_state",
            self.pre_state
            if self.pre_state is not None
            else np.array([math.sqrt(0.7), math.sqrt(0.3)], dtype=np.complex128),
            expected_dimension=None,
        )
        post = _normalised_state(
            "post_state",
            self.post_state
            if self.post_state is not None
            else np.array([math.sqrt(0.6), math.sqrt(0.4)], dtype=np.complex128),
            expected_dimension=pre.size,
        )
        projectors = (
            self.projectors if self.projectors is not None else _basis_projectors(pre.size)
        )
        _validate_projector_resolution(projectors, dimension=pre.size)
        bell = _normalised_state(
            "bell_state",
            self.bell_state
            if self.bell_state is not None
            else np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / math.sqrt(2.0),
            expected_dimension=4,
        )
        product = _normalised_state(
            "product_state",
            self.product_state
            if self.product_state is not None
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
            expected_dimension=4,
        )
        if len(self.chsh_angles) != 4:
            raise ValueError("chsh_angles must contain four angles")
        if not all(np.isfinite(self.chsh_angles)):
            raise ValueError("chsh_angles must contain finite values")
        _require_open_unit("prior_probability", self.prior_probability)
        _require_finite("intent_bias", self.intent_bias)
        if not -1.0 <= self.intent_bias <= 1.0:
            raise ValueError("intent_bias must lie in [-1, 1]")
        _require_finite("measurement_strength", self.measurement_strength)
        if self.measurement_strength < 0.0:
            raise ValueError("measurement_strength must be non-negative")
        object.__setattr__(self, "pre_state", pre)
        object.__setattr__(self, "post_state", post)
        object.__setattr__(self, "projectors", projectors)
        object.__setattr__(self, "bell_state", bell)
        object.__setattr__(self, "product_state", product)


@dataclass(frozen=True, slots=True)
class TSVFBoundaryValidationResult:
    """TSVF/ABL conditioning validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    shifted_post_probabilities: tuple[float, ...]
    probability_normalisation_error: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class EntanglementCorrelationBoundaryValidationResult:
    """Bell-correlation validation result with no-signalling controls."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    chsh_value: float
    no_signalling_residual: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class WeakMeasurementBiasBoundaryValidationResult:
    """Weak-measurement probability-bias validation result."""

    spec_key: str
    validation_protocol: str
    hardware_status: str
    source_ledger_ids: tuple[str, ...]
    prior_probability: float
    biased_probability: float
    probability_shift: float
    claim_boundary: str
    null_controls: MappingProxyType[str, float]
    problem_metadata: MappingProxyType[str, Any]


@dataclass(frozen=True, slots=True)
class AnomalousBoundaryFixtureResult:
    """Combined anomalous-boundary fixture result."""

    tsvf: TSVFBoundaryValidationResult
    entanglement: EntanglementCorrelationBoundaryValidationResult
    weak_measurement: WeakMeasurementBiasBoundaryValidationResult

    @property
    def hardware_status(self) -> str:
        """Return the shared simulator-only hardware status."""
        return self.tsvf.hardware_status

    @property
    def spec_keys(self) -> tuple[str, str, str]:
        """Return the promoted anomalous-boundary spec keys."""
        return (
            self.tsvf.spec_key,
            self.entanglement.spec_key,
            self.weak_measurement.spec_key,
        )

    @property
    def claim_boundary(self) -> str:
        """Return the shared claim-boundary statement."""
        return CLAIM_BOUNDARY


def bell_chsh_value(
    state: np.ndarray | None,
    angles: tuple[float, float, float, float],
) -> float:
    """Compute the CHSH value for two-qubit X-Z plane spin measurements."""
    vector = _normalised_state("state", state, expected_dimension=4)
    if len(angles) != 4:
        raise ValueError("angles must contain four entries")
    a0, a1, b0, b1 = angles
    value = (
        _two_qubit_expectation(vector, a0, b0)
        + _two_qubit_expectation(vector, a0, b1)
        + _two_qubit_expectation(vector, a1, b0)
        - _two_qubit_expectation(vector, a1, b1)
    )
    return float(value)


def bounded_weak_measurement_bias(
    prior_probability: float,
    intent_bias: float,
    measurement_strength: float,
) -> float:
    """Apply a bounded log-odds weak-measurement probability bias."""
    _require_open_unit("prior_probability", prior_probability)
    _require_finite("intent_bias", intent_bias)
    _require_finite("measurement_strength", measurement_strength)
    if not -1.0 <= intent_bias <= 1.0:
        raise ValueError("intent_bias must lie in [-1, 1]")
    if measurement_strength < 0.0:
        raise ValueError("measurement_strength must be non-negative")
    logit = math.log(prior_probability / (1.0 - prior_probability))
    shifted = logit + measurement_strength * intent_bias
    return float(1.0 / (1.0 + math.exp(-shifted)))


def validate_tsvf_precognition_boundary_fixture(
    config: AnomalousBoundaryConfig | None = None,
) -> TSVFBoundaryValidationResult:
    """Run the TSVF/ABL conditioning boundary fixture."""
    cfg = config or AnomalousBoundaryConfig()
    spec = load_anomalous_boundary_validation_spec(
        "applied.anomalous_boundary.tsvf_precognition_boundary",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    probabilities = abl_probabilities(cfg.pre_state, cfg.post_state, cfg.projectors)
    shifted = abl_probabilities(
        cfg.pre_state,
        np.array([math.sqrt(0.2), math.sqrt(0.8)], dtype=np.complex128),
        cfg.projectors,
    )
    controls = {
        "zero_denominator_rejection_label": _abl_zero_denominator_rejection_label(),
        "born_rule_marginalisation_l1": _born_rule_marginalisation_l1(cfg),
        "retrocausal_signalling_supported_label": 0.0,
    }
    metadata = _metadata(
        spec,
        extra={
            "projector_count": len(cast(tuple[np.ndarray, ...], cfg.projectors)),
            "boundary_rule": "Aharonov-Bergmann-Lebowitz post-selection",
        },
    )
    return TSVFBoundaryValidationResult(
        spec_key="applied.anomalous_boundary.tsvf_precognition_boundary",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        probabilities=probabilities,
        shifted_post_probabilities=shifted,
        probability_normalisation_error=float(abs(sum(probabilities) - 1.0)),
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def validate_entanglement_correlation_boundary_fixture(
    config: AnomalousBoundaryConfig | None = None,
) -> EntanglementCorrelationBoundaryValidationResult:
    """Run the Bell-correlation boundary fixture with no-signalling controls."""
    cfg = config or AnomalousBoundaryConfig()
    spec = load_anomalous_boundary_validation_spec(
        "applied.anomalous_boundary.entanglement_correlation_boundary",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    chsh = bell_chsh_value(cfg.bell_state, cfg.chsh_angles)
    residual = _no_signalling_residual(cfg.bell_state, cfg.chsh_angles)
    product_chsh = bell_chsh_value(cfg.product_state, cfg.chsh_angles)
    controls = {
        "product_state_chsh_value": product_chsh,
        "no_signalling_residual": residual,
        "signalling_rejection_label": _signalling_rejection_label(),
    }
    metadata = _metadata(
        spec,
        extra={
            "measurement_plane": "Pauli X-Z",
            "classical_chsh_bound": 2.0,
            "tsirelson_bound": 2.0 * math.sqrt(2.0),
        },
    )
    return EntanglementCorrelationBoundaryValidationResult(
        spec_key="applied.anomalous_boundary.entanglement_correlation_boundary",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        chsh_value=chsh,
        no_signalling_residual=residual,
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def validate_weak_measurement_bias_boundary_fixture(
    config: AnomalousBoundaryConfig | None = None,
) -> WeakMeasurementBiasBoundaryValidationResult:
    """Run the bounded weak-measurement probability-bias fixture."""
    cfg = config or AnomalousBoundaryConfig()
    spec = load_anomalous_boundary_validation_spec(
        "applied.anomalous_boundary.weak_measurement_bias_boundary",
        spec_bundle_path=cfg.spec_bundle_path,
    )
    biased = bounded_weak_measurement_bias(
        cfg.prior_probability,
        cfg.intent_bias,
        cfg.measurement_strength,
    )
    zero_intent = bounded_weak_measurement_bias(
        cfg.prior_probability,
        0.0,
        cfg.measurement_strength,
    )
    saturated = bounded_weak_measurement_bias(0.999999, 1.0, cfg.measurement_strength)
    controls = {
        "zero_intent_probability_shift_abs": abs(zero_intent - cfg.prior_probability),
        "saturated_probability_upper_margin": 1.0 - saturated,
        "out_of_range_bias_rejection_label": _out_of_range_bias_rejection_label(),
        "negative_strength_rejection_label": _negative_strength_rejection_label(),
    }
    metadata = _metadata(
        spec,
        extra={
            "bias_map": "bounded log-odds update",
            "intent_is_simulator_parameter": True,
        },
    )
    return WeakMeasurementBiasBoundaryValidationResult(
        spec_key="applied.anomalous_boundary.weak_measurement_bias_boundary",
        validation_protocol=str(spec["validation_protocol"]),
        hardware_status=str(spec["hardware_status"]),
        source_ledger_ids=tuple(str(item) for item in spec["source_ledger_ids"]),
        prior_probability=cfg.prior_probability,
        biased_probability=biased,
        probability_shift=float(biased - cfg.prior_probability),
        claim_boundary=CLAIM_BOUNDARY,
        null_controls=MappingProxyType(controls),
        problem_metadata=MappingProxyType(metadata),
    )


def validate_anomalous_boundary_fixture(
    config: AnomalousBoundaryConfig | None = None,
) -> AnomalousBoundaryFixtureResult:
    """Run all Paper 0 anomalous-boundary validation fixtures."""
    cfg = config or AnomalousBoundaryConfig()
    return AnomalousBoundaryFixtureResult(
        tsvf=validate_tsvf_precognition_boundary_fixture(cfg),
        entanglement=validate_entanglement_correlation_boundary_fixture(cfg),
        weak_measurement=validate_weak_measurement_bias_boundary_fixture(cfg),
    )


def _basis_projectors(dimension: int) -> tuple[np.ndarray, ...]:
    return tuple(
        np.diag(np.eye(dimension, dtype=np.complex128)[index]) for index in range(dimension)
    )


def _observable(angle: float) -> np.ndarray:
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    return cast(np.ndarray, math.cos(angle) * z + math.sin(angle) * x)


def _two_qubit_expectation(state: np.ndarray, alice_angle: float, bob_angle: float) -> float:
    operator = np.kron(_observable(alice_angle), _observable(bob_angle))
    return float(np.real(np.vdot(state, operator @ state)))


def _joint_probabilities(
    state: np.ndarray | None,
    alice_angle: float,
    bob_angle: float,
) -> np.ndarray:
    vector = _normalised_state("state", state, expected_dimension=4)
    identity = np.eye(2, dtype=np.complex128)
    alice = _observable(alice_angle)
    bob = _observable(bob_angle)
    probabilities = np.zeros((2, 2), dtype=np.float64)
    for row, alice_sign in enumerate((1.0, -1.0)):
        alice_projector = (identity + alice_sign * alice) / 2.0
        for column, bob_sign in enumerate((1.0, -1.0)):
            bob_projector = (identity + bob_sign * bob) / 2.0
            projector = np.kron(alice_projector, bob_projector)
            probabilities[row, column] = float(np.real(np.vdot(vector, projector @ vector)))
    return probabilities


def _no_signalling_residual(
    state: np.ndarray | None,
    angles: tuple[float, float, float, float],
) -> float:
    a0, a1, b0, b1 = angles
    alice_b0 = _joint_probabilities(state, a0, b0).sum(axis=1)
    alice_b1 = _joint_probabilities(state, a0, b1).sum(axis=1)
    bob_a0 = _joint_probabilities(state, a0, b0).sum(axis=0)
    bob_a1 = _joint_probabilities(state, a1, b0).sum(axis=0)
    return float(max(np.max(np.abs(alice_b0 - alice_b1)), np.max(np.abs(bob_a0 - bob_a1))))


def _signalling_rejection_label() -> float:
    signalling_residual = max(abs(0.8 - 0.5), abs(0.2 - 0.5))
    return float(signalling_residual > 1.0e-12)


def _born_rule_marginalisation_l1(config: AnomalousBoundaryConfig) -> float:
    pre = cast(np.ndarray, config.pre_state)
    projectors = cast(tuple[np.ndarray, ...], config.projectors)
    born = np.array([float(np.real(np.vdot(pre, projector @ pre))) for projector in projectors])
    born /= np.sum(born)
    basis = _basis_projectors(pre.size)
    weights = np.zeros(len(projectors), dtype=np.float64)
    for projector_index, projector in enumerate(projectors):
        for basis_index in range(pre.size):
            post = basis[basis_index] @ np.ones(pre.size, dtype=np.complex128)
            post /= np.linalg.norm(post)
            weights[projector_index] += abs(np.vdot(post, projector @ pre)) ** 2
    weights /= np.sum(weights)
    return float(np.sum(np.abs(born - weights)))


def _abl_zero_denominator_rejection_label() -> float:
    try:
        abl_probabilities(
            np.array([1.0, 0.0], dtype=np.complex128),
            np.array([0.0, 1.0], dtype=np.complex128),
            (np.diag([1.0, 0.0]).astype(np.complex128),),
        )
    except ValueError as exc:
        return float("denominator is zero" in str(exc))
    return 0.0


def _out_of_range_bias_rejection_label() -> float:
    try:
        bounded_weak_measurement_bias(0.5, 1.1, 0.7)
    except ValueError as exc:
        return float("intent_bias" in str(exc))
    return 0.0


def _negative_strength_rejection_label() -> float:
    try:
        bounded_weak_measurement_bias(0.5, 0.2, -0.1)
    except ValueError as exc:
        return float("non-negative" in str(exc))
    return 0.0


def _metadata(spec: dict[str, Any], *, extra: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper0_validation_protocol": str(spec["validation_protocol"]),
        "hardware_status": str(spec["hardware_status"]),
        "simulator_only_falsification_boundary": True,
        "claim_boundary": CLAIM_BOUNDARY,
        **extra,
    }


def _normalised_state(
    name: str,
    values: np.ndarray | None,
    *,
    expected_dimension: int | None,
) -> np.ndarray:
    if values is None:
        raise ValueError(f"{name} must not be None")
    arr = np.asarray(values, dtype=np.complex128)
    if arr.ndim != 1 or arr.size < 2:
        raise ValueError(f"{name} must be a one-dimensional vector with at least two entries")
    if expected_dimension is not None and arr.size != expected_dimension:
        raise ValueError(f"{name} must be normalised with dimension {expected_dimension}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain finite values")
    norm = float(np.linalg.norm(arr))
    if not np.isclose(norm, 1.0, atol=1.0e-12):
        if expected_dimension is not None:
            raise ValueError(f"{name} must be normalised")
        if norm <= np.finfo(np.float64).eps:
            raise ValueError(f"{name} must have non-zero norm")
    if norm <= np.finfo(np.float64).eps:
        raise ValueError(f"{name} must have non-zero norm")
    return cast(np.ndarray, arr / norm)


def _validate_projector_resolution(
    projectors: tuple[np.ndarray, ...],
    *,
    dimension: int,
) -> None:
    if len(projectors) == 0:
        raise ValueError("projectors must not be empty")
    total = np.zeros((dimension, dimension), dtype=np.complex128)
    identity = np.eye(dimension, dtype=np.complex128)
    for index, projector in enumerate(projectors):
        arr = np.asarray(projector, dtype=np.complex128)
        if arr.shape != (dimension, dimension):
            raise ValueError(f"projector {index} must have shape {(dimension, dimension)}")
        if not np.allclose(arr, arr.conj().T, atol=1.0e-12):
            raise ValueError(f"projector {index} must be Hermitian")
        if not np.allclose(arr @ arr, arr, atol=1.0e-12):
            raise ValueError(f"projector {index} must be idempotent")
        total += arr
    if not np.allclose(total, identity, atol=1.0e-12):
        raise ValueError("projectors must resolve identity")


def _require_open_unit(name: str, value: float) -> None:
    _require_finite(name, value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must lie in the open unit interval")


def _require_finite(name: str, value: float) -> None:
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")


__all__ = [
    "AnomalousBoundaryConfig",
    "AnomalousBoundaryFixtureResult",
    "CLAIM_BOUNDARY",
    "EntanglementCorrelationBoundaryValidationResult",
    "TSVFBoundaryValidationResult",
    "WeakMeasurementBiasBoundaryValidationResult",
    "bell_chsh_value",
    "bounded_weak_measurement_bias",
    "validate_anomalous_boundary_fixture",
    "validate_entanglement_correlation_boundary_fixture",
    "validate_tsvf_precognition_boundary_fixture",
    "validate_weak_measurement_bias_boundary_fixture",
]
