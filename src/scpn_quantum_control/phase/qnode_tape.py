# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNode Tape
"""QNode-style differentiable tape records for supported phase objectives."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import FiniteShotSampleProvenance, Parameter, ParameterShiftRule
from ..differentiable_result_contracts import (
    ParameterShiftSampleRecord,
    StochasticGradientResult,
)
from .gradient_backend import QuantumGradientPlan, plan_quantum_gradient_backend
from .gradient_tape import TapeGradientRecord, gradient_tape

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]
PhaseQNodeTapeKind = Literal["deterministic", "finite_shot", "provider_boundary"]

EVIDENCE_CLASS = "phase_qnode_differentiable_tape"
CLAIM_BOUNDARY = (
    "supported phase QNode tape evidence only; not hardware execution unless "
    "hardware_execution is true; not arbitrary QNode autodiff or provider submission"
)


@dataclass(frozen=True)
class PhaseQNodeTapeRecord:
    """One QNode-style differentiable execution or fail-closed boundary record."""

    qnode_name: str
    objective_name: str
    observable: str
    kind: PhaseQNodeTapeKind
    backend: str
    plan: QuantumGradientPlan
    value: float
    gradient: FloatArray
    standard_error: FloatArray | None
    confidence_radius: FloatArray | None
    shot_count: int | None
    seed: int | None
    provider: str | None
    requested_job_id: str | None
    failure_reason: str
    alternatives: tuple[str, ...]
    sample_records: tuple[ParameterShiftSampleRecord, ...] = ()
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY
    hardware_execution: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "qnode_name", _as_non_empty_string("qnode_name", self.qnode_name))
        object.__setattr__(
            self,
            "objective_name",
            _as_non_empty_string("objective_name", self.objective_name),
        )
        object.__setattr__(self, "observable", _as_non_empty_string("observable", self.observable))
        object.__setattr__(self, "backend", _as_non_empty_string("backend", self.backend))
        object.__setattr__(self, "value", _as_finite_scalar("value", self.value))
        gradient = _as_finite_vector("gradient", self.gradient, allow_empty=True)
        standard_error = _as_optional_finite_vector("standard_error", self.standard_error)
        confidence_radius = _as_optional_finite_vector(
            "confidence_radius",
            self.confidence_radius,
        )
        if standard_error is not None and standard_error.shape != gradient.shape:
            raise ValueError("standard_error must match gradient shape")
        if confidence_radius is not None and confidence_radius.shape != gradient.shape:
            raise ValueError("confidence_radius must match gradient shape")
        if self.shot_count is not None and self.shot_count <= 0:
            raise ValueError("shot_count must be positive or None")
        if self.seed is not None and self.seed < 0:
            raise ValueError("seed must be non-negative or None")
        object.__setattr__(self, "gradient", gradient)
        object.__setattr__(self, "standard_error", standard_error)
        object.__setattr__(self, "confidence_radius", confidence_radius)
        object.__setattr__(self, "provider", _as_optional_string("provider", self.provider))
        object.__setattr__(
            self,
            "requested_job_id",
            _as_optional_string("requested_job_id", self.requested_job_id),
        )
        object.__setattr__(self, "failure_reason", str(self.failure_reason))
        object.__setattr__(self, "alternatives", tuple(str(item) for item in self.alternatives))
        sample_records = tuple(self.sample_records)
        for sample in sample_records:
            if not isinstance(sample, ParameterShiftSampleRecord):
                raise ValueError("sample_records must contain ParameterShiftSampleRecord values")
            if sample.parameter_index >= gradient.size:
                raise ValueError("sample_records parameter_index must fit the gradient shape")
        if self.kind == "finite_shot" and self.plan.supported and not self.failure_reason:
            if not sample_records:
                raise ValueError("finite-shot QNode tape records must include sample_records")
        elif sample_records:
            raise ValueError("sample_records are only valid for finite-shot QNode tape records")
        object.__setattr__(self, "sample_records", sample_records)
        object.__setattr__(
            self,
            "evidence_class",
            _as_non_empty_string("evidence_class", self.evidence_class),
        )
        object.__setattr__(
            self,
            "claim_boundary",
            _as_non_empty_string("claim_boundary", self.claim_boundary),
        )

    @property
    def supported(self) -> bool:
        """Return true when the record executed under a supported backend plan."""
        return self.plan.supported and not self.failure_reason

    @property
    def fail_closed(self) -> bool:
        """Return true when the record intentionally refused execution."""
        return not self.supported

    @property
    def parameter_shift_evaluations(self) -> int:
        """Return planned objective evaluations for the recorded derivative."""
        return self.plan.evaluations

    @property
    def sample_record_count(self) -> int:
        """Return shifted plus/minus sample records attached to the tape record."""
        return len(self.sample_records)

    @property
    def total_shots(self) -> int | None:
        """Return total shot budget when the record has a finite-shot plan."""
        if self.plan.shots is None:
            return None if self.supported else 0
        return self.plan.evaluations * self.plan.shots

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready QNode tape record."""
        return {
            "qnode_name": self.qnode_name,
            "objective_name": self.objective_name,
            "observable": self.observable,
            "kind": self.kind,
            "backend": self.backend,
            "plan": _plan_to_dict(self.plan),
            "supported": self.supported,
            "fail_closed": self.fail_closed,
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "standard_error": None
            if self.standard_error is None
            else self.standard_error.tolist(),
            "confidence_radius": (
                None if self.confidence_radius is None else self.confidence_radius.tolist()
            ),
            "shot_count": self.shot_count,
            "seed": self.seed,
            "provider": self.provider,
            "requested_job_id": self.requested_job_id,
            "failure_reason": self.failure_reason,
            "alternatives": list(self.alternatives),
            "sample_record_count": self.sample_record_count,
            "sample_records": [sample.to_dict() for sample in self.sample_records],
            "parameter_shift_evaluations": self.parameter_shift_evaluations,
            "total_shots": self.total_shots,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
        }


@dataclass(frozen=True)
class PhaseQNodeTapeReadinessSuiteResult:
    """Default QNode tape readiness evidence across supported and blocked routes."""

    records: tuple[PhaseQNodeTapeRecord, ...]
    evidence_class: str = EVIDENCE_CLASS
    claim_boundary: str = CLAIM_BOUNDARY
    hardware_execution: bool = False

    @property
    def record_count(self) -> int:
        """Return number of suite records."""
        return len(self.records)

    @property
    def supported_count(self) -> int:
        """Return number of supported records."""
        return sum(1 for record in self.records if record.supported)

    @property
    def fail_closed_count(self) -> int:
        """Return number of fail-closed records."""
        return sum(1 for record in self.records if record.fail_closed)

    @property
    def total_parameter_shift_evaluations(self) -> int:
        """Return total planned parameter-shift evaluations."""
        return sum(record.parameter_shift_evaluations for record in self.records)

    @property
    def total_shots(self) -> int:
        """Return total finite-shot budget across records."""
        return sum(int(record.total_shots or 0) for record in self.records)

    @property
    def passed(self) -> bool:
        """Return true when default supported routes pass and hardware is blocked."""
        return (
            self.record_count > 0
            and all(
                record.supported for record in self.records if record.kind != "provider_boundary"
            )
            and all(
                record.fail_closed for record in self.records if record.kind == "provider_boundary"
            )
            and not self.hardware_execution
        )

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready readiness evidence."""
        return {
            "passed": self.passed,
            "record_count": self.record_count,
            "supported_count": self.supported_count,
            "fail_closed_count": self.fail_closed_count,
            "total_parameter_shift_evaluations": self.total_parameter_shift_evaluations,
            "total_shots": self.total_shots,
            "records": [record.to_dict() for record in self.records],
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
        }


class PhaseQNodeTape:
    """Context-managed QNode-style tape for supported phase objectives."""

    def __init__(
        self,
        *,
        qnode_name: str,
        observable: str,
        backend: str = "statevector",
        shots: int | None = None,
        seed: int | None = None,
        confidence_level: float = 0.95,
        allow_hardware: bool = False,
    ) -> None:
        """Create a QNode tape with shared execution metadata."""
        self.qnode_name = _as_non_empty_string("qnode_name", qnode_name)
        self.observable = _as_non_empty_string("observable", observable)
        self.backend = _as_non_empty_string("backend", backend)
        self.shots = _as_optional_positive_int("shots", shots)
        self.seed = _as_optional_non_negative_int("seed", seed)
        self.confidence_level = _as_confidence_level(confidence_level)
        self.allow_hardware = bool(allow_hardware)
        self._records: list[PhaseQNodeTapeRecord] = []
        self._active = False

    def __enter__(self) -> PhaseQNodeTape:
        """Activate the tape context."""
        if self._active:
            raise RuntimeError("phase QNode tape is already active")
        self._active = True
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Deactivate the tape context."""
        self._active = False

    @property
    def records(self) -> tuple[PhaseQNodeTapeRecord, ...]:
        """Return immutable view of QNode records."""
        return tuple(self._records)

    def clear(self) -> None:
        """Clear recorded QNode evidence."""
        self._records.clear()

    def record_parameter_shift(
        self,
        objective_name: str,
        objective: ScalarObjective,
        params: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = None,
        rule: ParameterShiftRule | None = None,
    ) -> PhaseQNodeTapeRecord:
        """Record a deterministic QNode parameter-shift derivative."""
        self._require_active()
        clean_name = _as_non_empty_string("objective_name", objective_name)
        with gradient_tape(
            backend=self.backend,
            shots=self.shots,
            seed=self.seed,
            confidence_level=self.confidence_level,
            allow_hardware=self.allow_hardware,
        ) as tape:
            inner = tape.record_parameter_shift(
                clean_name,
                objective,
                params,
                parameters=parameters,
                rule=rule,
            )
        return self._append_from_tape_record(clean_name, "deterministic", inner)

    def record_finite_shot_parameter_shift(
        self,
        objective_name: str,
        *,
        plus_values: ArrayLike,
        minus_values: ArrayLike,
        plus_variances: ArrayLike,
        minus_variances: ArrayLike,
        sample_provenance: Mapping[str, object] | FiniteShotSampleProvenance | None = None,
        value: float = 0.0,
        parameters: Sequence[Parameter] | None = None,
        rule: ParameterShiftRule | None = None,
        confidence_z: float = 1.959963984540054,
    ) -> PhaseQNodeTapeRecord:
        """Record a finite-shot QNode parameter-shift replay with uncertainty.

        Parameters
        ----------
        objective_name:
            Non-empty objective label stored in the QNode tape.
        plus_values, minus_values:
            Materialised plus/minus shifted objective estimates.
        plus_variances, minus_variances:
            Per-estimate finite-shot variances matching the shifted values.
        sample_provenance:
            Source metadata for the materialised finite-shot tensors. The
            mapping or record must include ``sample_seed``, ``shot_batch_id``,
            and ``source_class``.
        value:
            Scalar objective value associated with the gradient record.
        parameters:
            Optional parameter metadata and trainable mask.
        rule:
            Optional parameter-shift rule.
        confidence_z:
            Normal-approximation multiplier used for confidence radii.

        Returns
        -------
        PhaseQNodeTapeRecord
            QNode tape record containing finite-shot uncertainty evidence.
        """
        self._require_active()
        clean_name = _as_non_empty_string("objective_name", objective_name)
        with gradient_tape(
            backend=self.backend,
            shots=self.shots,
            seed=self.seed,
            confidence_level=self.confidence_level,
            allow_hardware=self.allow_hardware,
        ) as tape:
            inner = tape.record_finite_shot_parameter_shift(
                clean_name,
                plus_values=plus_values,
                minus_values=minus_values,
                plus_variances=plus_variances,
                minus_variances=minus_variances,
                sample_provenance=sample_provenance,
                value=value,
                parameters=parameters,
                rule=rule,
                confidence_z=confidence_z,
            )
        return self._append_from_tape_record(clean_name, "finite_shot", inner)

    def record_provider_boundary(
        self,
        objective_name: str,
        *,
        provider: str,
        requested_job_id: str | None = None,
        n_params: int = 1,
        shift_terms: int = 1,
    ) -> PhaseQNodeTapeRecord:
        """Record provider-gradient readiness without submitting a provider job."""
        self._require_active()
        clean_name = _as_non_empty_string("objective_name", objective_name)
        clean_provider = _as_non_empty_string("provider", provider)
        params_count = _as_positive_int("n_params", n_params)
        terms_count = _as_positive_int("shift_terms", shift_terms)
        plan = plan_quantum_gradient_backend(
            self.backend,
            n_params=params_count,
            shift_terms=terms_count,
            method="stochastic_parameter_shift" if self.shots is not None else "parameter_shift",
            shots=self.shots,
            seed=self.seed,
            finite_shot=self.shots is not None,
            confidence_level=self.confidence_level,
            allow_hardware=self.allow_hardware,
        )
        failure_reason = "; ".join(plan.reasons)
        if plan.supported:
            failure_reason = "provider execution not submitted by QNode tape readiness record"
        record = PhaseQNodeTapeRecord(
            qnode_name=self.qnode_name,
            objective_name=clean_name,
            observable=self.observable,
            kind="provider_boundary",
            backend=plan.backend,
            plan=plan,
            value=0.0,
            gradient=np.empty(0, dtype=np.float64),
            standard_error=None,
            confidence_radius=None,
            shot_count=plan.shots,
            seed=plan.seed,
            provider=clean_provider,
            requested_job_id=requested_job_id,
            failure_reason=failure_reason,
            alternatives=plan.alternatives,
            hardware_execution=False,
        )
        self._records.append(record)
        return record

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready tape metadata."""
        return {
            "qnode_name": self.qnode_name,
            "observable": self.observable,
            "backend": self.backend,
            "record_count": len(self._records),
            "supported_count": sum(1 for record in self._records if record.supported),
            "fail_closed_count": sum(1 for record in self._records if record.fail_closed),
            "records": [record.to_dict() for record in self._records],
            "evidence_class": EVIDENCE_CLASS,
            "claim_boundary": CLAIM_BOUNDARY,
        }

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError(
                "phase QNode tape records can only be created inside an active context"
            )

    def _append_from_tape_record(
        self,
        objective_name: str,
        kind: PhaseQNodeTapeKind,
        inner: TapeGradientRecord,
    ) -> PhaseQNodeTapeRecord:
        standard_error = inner.standard_error
        confidence_radius = inner.confidence_radius
        sample_records = (
            inner.result.records if isinstance(inner.result, StochasticGradientResult) else ()
        )
        record = PhaseQNodeTapeRecord(
            qnode_name=self.qnode_name,
            objective_name=objective_name,
            observable=self.observable,
            kind=kind,
            backend=inner.plan.backend,
            plan=inner.plan,
            value=inner.value,
            gradient=inner.gradient,
            standard_error=standard_error,
            confidence_radius=confidence_radius,
            shot_count=inner.plan.shots,
            seed=inner.plan.seed,
            provider=None,
            requested_job_id=None,
            failure_reason="" if inner.plan.supported else "; ".join(inner.plan.reasons),
            alternatives=inner.plan.alternatives,
            sample_records=sample_records,
            hardware_execution=False,
        )
        self._records.append(record)
        return record


def phase_qnode_tape(
    *,
    qnode_name: str,
    observable: str,
    backend: str = "statevector",
    shots: int | None = None,
    seed: int | None = None,
    confidence_level: float = 0.95,
    allow_hardware: bool = False,
) -> PhaseQNodeTape:
    """Return a context-managed phase QNode tape."""
    return PhaseQNodeTape(
        qnode_name=qnode_name,
        observable=observable,
        backend=backend,
        shots=shots,
        seed=seed,
        confidence_level=confidence_level,
        allow_hardware=allow_hardware,
    )


def run_phase_qnode_tape_readiness_suite() -> PhaseQNodeTapeReadinessSuiteResult:
    """Run default QNode tape readiness evidence for supported and blocked routes."""
    records: list[PhaseQNodeTapeRecord] = []

    def energy(params: FloatArray) -> float:
        return float(np.cos(params[0]) + 0.25 * np.sin(params[1]))

    with phase_qnode_tape(
        qnode_name="kuramoto_xy_vqe",
        observable="energy",
        backend="statevector",
    ) as tape:
        records.append(
            tape.record_parameter_shift(
                "deterministic_energy",
                energy,
                np.array([0.2, -0.4], dtype=np.float64),
            )
        )

    with phase_qnode_tape(
        qnode_name="bounded_phase_qnn",
        observable="mean_squared_error",
        backend="finite_shot_simulator",
        shots=1024,
        seed=31,
    ) as tape:
        records.append(
            tape.record_finite_shot_parameter_shift(
                "finite_shot_loss",
                plus_values=np.array([1.2, -0.3], dtype=np.float64),
                minus_values=np.array([0.8, -0.7], dtype=np.float64),
                plus_variances=np.array([0.04, 0.09], dtype=np.float64),
                minus_variances=np.array([0.05, 0.07], dtype=np.float64),
                sample_provenance={
                    "sample_seed": "phase-qnode-tape-readiness-seed",
                    "shot_batch_id": "phase-qnode-tape-readiness-batch",
                    "source_class": "synthetic_fixture",
                },
                value=0.375,
            )
        )

    with phase_qnode_tape(
        qnode_name="hardware_vqe_candidate",
        observable="energy",
        backend="hardware",
        shots=4096,
        seed=7,
    ) as tape:
        records.append(
            tape.record_provider_boundary(
                "hardware_gradient",
                provider="hardware_qpu",
                requested_job_id="blocked-before-submit",
            )
        )

    return PhaseQNodeTapeReadinessSuiteResult(records=tuple(records))


def _plan_to_dict(plan: QuantumGradientPlan) -> dict[str, object]:
    return {
        "backend": plan.backend,
        "family": plan.family,
        "method": plan.method,
        "supported": plan.supported,
        "n_params": plan.n_params,
        "shift_terms": plan.shift_terms,
        "evaluations": plan.evaluations,
        "shots": plan.shots,
        "seed": plan.seed,
        "finite_shot": plan.finite_shot,
        "confidence_level": plan.confidence_level,
        "requires_hardware_approval": plan.requires_hardware_approval,
        "reasons": list(plan.reasons),
        "alternatives": list(plan.alternatives),
    }


def _as_non_empty_string(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _as_optional_string(name: str, value: str | None) -> str | None:
    if value is None:
        return None
    return _as_non_empty_string(name, value)


def _as_finite_scalar(name: str, value: float) -> float:
    scalar = float(value)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_positive_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _as_optional_positive_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    return _as_positive_int(name, value)


def _as_optional_non_negative_int(name: str, value: int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _as_confidence_level(value: float) -> float:
    confidence = float(value)
    if not np.isfinite(confidence) or confidence <= 0.0 or confidence >= 1.0:
        raise ValueError("confidence_level must be between zero and one")
    return confidence


def _as_finite_vector(name: str, values: ArrayLike, *, allow_empty: bool = False) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not allow_empty and vector.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return cast(FloatArray, vector.astype(np.float64, copy=True))


def _as_optional_finite_vector(name: str, values: ArrayLike | None) -> FloatArray | None:
    if values is None:
        return None
    return _as_finite_vector(name, values, allow_empty=True)


__all__ = [
    "PhaseQNodeTape",
    "PhaseQNodeTapeKind",
    "PhaseQNodeTapeReadinessSuiteResult",
    "PhaseQNodeTapeRecord",
    "phase_qnode_tape",
    "run_phase_qnode_tape_readiness_suite",
]
