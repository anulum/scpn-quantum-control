# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Gradient Tape
"""Context-managed quantum-gradient tape for phase objectives."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..differentiable import (
    FiniteShotSampleProvenance,
    GradientResult,
    Parameter,
    ParameterShiftRule,
    StochasticGradientResult,
    value_and_parameter_shift_grad,
)
from .gradient_backend import QuantumGradientPlan, plan_quantum_gradient_backend
from .param_shift import parameter_shift_gradient_with_uncertainty

FloatArray: TypeAlias = NDArray[np.float64]
ScalarObjective = Callable[[FloatArray], float]
TapeKind = Literal["deterministic", "stochastic"]
TapeContractStatus = Literal["supported", "fail_closed"]


GRADIENT_TAPE_CONTRACT_CLAIM_BOUNDARY = (
    "DP-003 gradient-tape contract audit only; supported records are local phase "
    "parameter-shift or materialised finite-shot replay, while arbitrary Python "
    "mutation semantics, provider execution, hardware gradients, and benchmark "
    "promotion remain outside this tape contract"
)
"""Claim boundary for the executable gradient-tape contract audit."""


@dataclass(frozen=True)
class TapeGradientRecord:
    """One recorded quantum-gradient evaluation."""

    name: str
    kind: TapeKind
    plan: QuantumGradientPlan
    result: GradientResult | StochasticGradientResult
    parameter_fingerprint: str
    replay_fingerprint: str
    contract_notes: tuple[str, ...]

    @property
    def gradient(self) -> FloatArray:
        """Return the recorded gradient vector."""
        return self.result.gradient

    @property
    def value(self) -> float:
        """Return the recorded objective value."""
        return self.result.value

    @property
    def evaluations(self) -> int:
        """Return planned quantum objective evaluations, excluding tape bookkeeping."""
        return self.plan.evaluations

    @property
    def method(self) -> str:
        """Return the replay method recorded by the gradient result."""
        return self.result.method

    @property
    def shift_terms(self) -> int:
        """Return the number of parameter-shift terms used per parameter."""
        return self.plan.shift_terms

    @property
    def standard_error(self) -> FloatArray | None:
        """Return finite-shot standard errors when the record is stochastic."""
        if isinstance(self.result, StochasticGradientResult):
            return self.result.standard_error
        return None

    @property
    def confidence_radius(self) -> FloatArray | None:
        """Return finite-shot confidence radii when the record is stochastic."""
        if isinstance(self.result, StochasticGradientResult):
            return self.result.confidence_radius
        return None

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready replay provenance for audit logs and notebooks."""
        standard_error = self.standard_error
        confidence_radius = self.confidence_radius
        return {
            "name": self.name,
            "kind": self.kind,
            "backend": self.plan.backend,
            "plan_method": self.plan.method,
            "method": self.method,
            "n_params": self.plan.n_params,
            "shift_terms": self.plan.shift_terms,
            "evaluations": self.plan.evaluations,
            "shots": self.plan.shots,
            "seed": self.plan.seed,
            "finite_shot": self.plan.finite_shot,
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "standard_error": (None if standard_error is None else standard_error.tolist()),
            "confidence_radius": (
                None if confidence_radius is None else confidence_radius.tolist()
            ),
            "parameter_fingerprint": self.parameter_fingerprint,
            "replay_fingerprint": self.replay_fingerprint,
            "contract_notes": list(self.contract_notes),
        }


@dataclass(frozen=True)
class GradientTapeContractCheck:
    """One DP-003 gradient-tape contract check.

    Parameters
    ----------
    name
        Stable check identifier.
    status
        ``"supported"`` when the behaviour executes, or ``"fail_closed"``
        when the unsupported route is intentionally rejected.
    evidence
        Human-readable evidence collected by the executable audit.
    blocked_reason
        Rejection reason for fail-closed checks, or ``None`` for supported
        checks.
    """

    name: str
    status: TapeContractStatus
    evidence: tuple[str, ...]
    blocked_reason: str | None = None

    @property
    def supported(self) -> bool:
        """Return true when the audited behaviour is supported."""
        return self.status == "supported"

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready contract-check metadata."""
        return {
            "name": self.name,
            "status": self.status,
            "supported": self.supported,
            "evidence": list(self.evidence),
            "blocked_reason": self.blocked_reason,
        }


@dataclass(frozen=True)
class GradientTapeContractAuditResult:
    """Executable DP-003 contract audit for the phase gradient tape.

    Parameters
    ----------
    checks
        Ordered supported and fail-closed contract checks.
    passed
        Whether every expected contract check produced evidence.
    claim_boundary
        Boundary text limiting the audit to local tape replay semantics.
    """

    checks: tuple[GradientTapeContractCheck, ...]
    passed: bool
    claim_boundary: str = GRADIENT_TAPE_CONTRACT_CLAIM_BOUNDARY

    @property
    def supported_checks(self) -> tuple[GradientTapeContractCheck, ...]:
        """Return contract checks that executed as supported."""
        return tuple(check for check in self.checks if check.supported)

    @property
    def fail_closed_checks(self) -> tuple[GradientTapeContractCheck, ...]:
        """Return contract checks that intentionally failed closed."""
        return tuple(check for check in self.checks if not check.supported)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready audit metadata."""
        return {
            "checks": [check.to_dict() for check in self.checks],
            "passed": self.passed,
            "claim_boundary": self.claim_boundary,
        }


def _as_parameter_vector(name: str, values: ArrayLike) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _shift_term_count(rule: ParameterShiftRule | None) -> int:
    if rule is None:
        return 1
    return len(rule.terms)


def _as_finite_shot_array(name: str, values: ArrayLike) -> FloatArray:
    records = np.asarray(values, dtype=float)
    if records.ndim not in (1, 2):
        raise ValueError(f"{name} must be a one- or two-dimensional array")
    if records.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(records)):
        raise ValueError(f"{name} must contain only finite values")
    return records.astype(np.float64, copy=True)


def _require_matching_finite_shot_shape(
    name: str,
    values: ArrayLike,
    expected_shape: tuple[int, ...],
) -> FloatArray:
    records = _as_finite_shot_array(name, values)
    if records.shape != expected_shape:
        raise ValueError(f"{name} must match plus_values shape {expected_shape}")
    return records


def _array_fingerprint(values: FloatArray) -> str:
    digest = hashlib.sha256()
    digest.update(str(values.shape).encode("utf-8"))
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _record_fingerprint(
    *,
    name: str,
    kind: TapeKind,
    plan: QuantumGradientPlan,
    value: float,
    gradient: FloatArray,
) -> str:
    digest = hashlib.sha256()
    digest.update(name.encode("utf-8"))
    digest.update(kind.encode("utf-8"))
    digest.update(plan.backend.encode("utf-8"))
    digest.update(plan.method.encode("utf-8"))
    digest.update(str(plan.evaluations).encode("utf-8"))
    digest.update(repr(float(value)).encode("utf-8"))
    digest.update(_array_fingerprint(gradient).encode("utf-8"))
    return digest.hexdigest()


def _guard_scalar_objective(objective: ScalarObjective) -> ScalarObjective:
    def guarded(params: FloatArray) -> float:
        before = params.copy()
        value = float(objective(params))
        if not np.array_equal(params, before):
            raise ValueError(
                "gradient tape objectives must not mutate parameter arrays during replay"
            )
        if not np.isfinite(value):
            raise ValueError("gradient tape objective replay must return a finite scalar")
        return value

    return guarded


def _assert_replay_stable(objective: ScalarObjective, params: FloatArray) -> None:
    first = objective(params.copy())
    second = objective(params.copy())
    if not np.isclose(first, second, rtol=0.0, atol=0.0):
        raise ValueError("gradient tape objective replay must be stable for identical parameters")


class QuantumGradientTape:
    """Context manager that records supported phase-gradient evaluations."""

    def __init__(
        self,
        *,
        backend: str = "statevector",
        shots: int | None = None,
        seed: int | None = None,
        confidence_level: float = 0.95,
        allow_hardware: bool = False,
    ) -> None:
        """Create a tape with backend policy shared by all records."""
        self.backend = backend
        self.shots = shots
        self.seed = seed
        self.confidence_level = confidence_level
        self.allow_hardware = allow_hardware
        self._records: list[TapeGradientRecord] = []
        self._active = False

    def __enter__(self) -> QuantumGradientTape:
        """Activate the tape context."""
        if self._active:
            raise RuntimeError("gradient tape is already active")
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
    def records(self) -> tuple[TapeGradientRecord, ...]:
        """Return immutable view of recorded gradient evaluations."""
        return tuple(self._records)

    def clear(self) -> None:
        """Clear recorded evaluations while preserving backend policy."""
        self._records.clear()

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError(
                "gradient tape records can only be created inside an active context"
            )

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("record name must be a non-empty string")
        return name.strip()

    @staticmethod
    def _ensure_supported(plan: QuantumGradientPlan) -> None:
        if plan.fail_closed:
            joined = "; ".join(plan.reasons)
            raise ValueError(f"backend gradient plan is unsupported: {joined}")

    def record_parameter_shift(
        self,
        name: str,
        objective: ScalarObjective,
        params: ArrayLike,
        *,
        parameters: Sequence[Parameter] | None = None,
        rule: ParameterShiftRule | None = None,
    ) -> TapeGradientRecord:
        """Record deterministic parameter-shift value and gradient."""
        self._require_active()
        record_name = self._validate_name(name)
        values = _as_parameter_vector("params", params)
        shift_terms = _shift_term_count(rule)
        plan = plan_quantum_gradient_backend(
            self.backend,
            n_params=values.size,
            shift_terms=shift_terms,
            method="parameter_shift",
            seed=self.seed,
            allow_hardware=self.allow_hardware,
        )
        self._ensure_supported(plan)
        guarded_objective = _guard_scalar_objective(objective)
        _assert_replay_stable(guarded_objective, values)
        result = value_and_parameter_shift_grad(
            guarded_objective,
            values,
            parameters=parameters,
            rule=rule,
        )
        gradient = result.gradient.astype(np.float64, copy=True)
        record = TapeGradientRecord(
            name=record_name,
            kind="deterministic",
            plan=plan,
            result=result,
            parameter_fingerprint=_array_fingerprint(values),
            replay_fingerprint=_record_fingerprint(
                name=record_name,
                kind="deterministic",
                plan=plan,
                value=result.value,
                gradient=gradient,
            ),
            contract_notes=(
                "input parameters copied before replay",
                "objective mutation guard passed",
                "identical-parameter replay stability passed",
            ),
        )
        self._records.append(record)
        return record

    def record_finite_shot_parameter_shift(
        self,
        name: str,
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
    ) -> TapeGradientRecord:
        """Record finite-shot parameter-shift gradient with uncertainty.

        Parameters
        ----------
        name:
            Non-empty record label.
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
        TapeGradientRecord
            Tape record containing the stochastic gradient result.
        """
        self._require_active()
        record_name = self._validate_name(name)
        plus = _as_finite_shot_array("plus_values", plus_values)
        shift_terms = _shift_term_count(rule)
        if plus.ndim == 1:
            if shift_terms != 1:
                raise ValueError(
                    "multi-term finite-shot tape records require two-dimensional "
                    "plus_values with shape (shift_terms, n_params)"
                )
            n_params = plus.size
        else:
            if plus.shape[0] != shift_terms:
                raise ValueError(
                    "plus_values first axis must match the parameter-shift "
                    f"rule term count {shift_terms}"
                )
            n_params = plus.shape[1]
        minus = _require_matching_finite_shot_shape(
            "minus_values",
            minus_values,
            plus.shape,
        )
        plus_var = _require_matching_finite_shot_shape(
            "plus_variances",
            plus_variances,
            plus.shape,
        )
        minus_var = _require_matching_finite_shot_shape(
            "minus_variances",
            minus_variances,
            plus.shape,
        )
        plan = plan_quantum_gradient_backend(
            self.backend,
            n_params=n_params,
            shift_terms=shift_terms,
            method="stochastic_parameter_shift",
            shots=self.shots,
            seed=self.seed,
            finite_shot=True,
            confidence_level=self.confidence_level,
            allow_hardware=self.allow_hardware,
        )
        self._ensure_supported(plan)
        if plan.shots is None:
            raise ValueError("finite-shot tape records require an explicit shot plan")
        result = parameter_shift_gradient_with_uncertainty(
            plus,
            minus,
            plus_var,
            minus_var,
            shots=plan.shots,
            sample_provenance=sample_provenance,
            backend=plan.backend,
            value=value,
            parameters=parameters,
            rule=rule,
            confidence_level=self.confidence_level,
            confidence_z=confidence_z,
        )
        record = TapeGradientRecord(
            name=record_name,
            kind="stochastic",
            plan=plan,
            result=result,
            parameter_fingerprint=_array_fingerprint(plus),
            replay_fingerprint=_record_fingerprint(
                name=record_name,
                kind="stochastic",
                plan=plan,
                value=result.value,
                gradient=result.gradient.astype(np.float64, copy=True),
            ),
            contract_notes=(
                "materialised finite-shot records copied before replay",
                "sample provenance required",
                "backend planner accepted finite-shot replay",
            ),
        )
        self._records.append(record)
        return record


def run_gradient_tape_contract_audit() -> GradientTapeContractAuditResult:
    """Run executable DP-003 checks for the phase gradient tape contract.

    Returns
    -------
    GradientTapeContractAuditResult
        Ordered evidence for nested tape isolation, persistent reuse,
        mutation/alias protection, and replay-stability fail-closed behaviour.
    """
    checks = [
        _audit_independent_nested_tapes(),
        _audit_same_tape_reentry_fails_closed(),
        _audit_persistent_reuse(),
        _audit_parameter_alias_snapshot(),
        _audit_objective_mutation_fails_closed(),
        _audit_unstable_replay_fails_closed(),
    ]
    passed = (
        len(checks) == 6
        and len([check for check in checks if check.supported]) == 3
        and len([check for check in checks if not check.supported]) == 3
        and all(check.evidence for check in checks)
    )
    return GradientTapeContractAuditResult(checks=tuple(checks), passed=passed)


def _audit_independent_nested_tapes() -> GradientTapeContractCheck:
    """Audit independent nested tape contexts and record isolation."""
    params = np.array([0.2], dtype=float)

    def objective(values: FloatArray) -> float:
        return float(np.cos(values[0]))

    with gradient_tape(backend="statevector") as outer:
        outer_record = outer.record_parameter_shift("outer", objective, params)
        with gradient_tape(backend="statevector") as inner:
            inner_record = inner.record_parameter_shift("inner", objective, params)
    if len(outer.records) != 1 or len(inner.records) != 1:  # pragma: no cover
        raise RuntimeError("nested tape audit expected isolated record buffers")
    return GradientTapeContractCheck(
        name="independent_nested_tapes",
        status="supported",
        evidence=(
            f"outer={outer_record.replay_fingerprint}",
            f"inner={inner_record.replay_fingerprint}",
        ),
    )


def _audit_same_tape_reentry_fails_closed() -> GradientTapeContractCheck:
    """Audit fail-closed re-entry for the same active tape."""
    tape = QuantumGradientTape(backend="statevector")
    blocked_reason = ""
    with tape:
        try:
            tape.__enter__()
        except RuntimeError as exc:
            blocked_reason = str(exc)
    if not blocked_reason:  # pragma: no cover
        raise RuntimeError("same tape re-entry did not fail closed")
    return GradientTapeContractCheck(
        name="same_tape_reentry",
        status="fail_closed",
        evidence=("active tape rejected second enter",),
        blocked_reason=blocked_reason,
    )


def _audit_persistent_reuse() -> GradientTapeContractCheck:
    """Audit record persistence after exit and explicit clear/reuse."""
    tape = QuantumGradientTape(backend="statevector")
    params = np.array([0.4], dtype=float)

    def objective(values: FloatArray) -> float:
        return float(np.sin(values[0]))

    with tape:
        first = tape.record_parameter_shift("first", objective, params)
    persisted = len(tape.records) == 1 and tape.records[0] is first
    tape.clear()
    with tape:
        second = tape.record_parameter_shift("second", objective, params)
    if (
        not persisted or len(tape.records) != 1 or tape.records[0] is not second
    ):  # pragma: no cover
        raise RuntimeError("persistent tape reuse audit failed")
    return GradientTapeContractCheck(
        name="persistent_reuse",
        status="supported",
        evidence=(
            f"first={first.replay_fingerprint}",
            f"second={second.replay_fingerprint}",
        ),
    )


def _audit_parameter_alias_snapshot() -> GradientTapeContractCheck:
    """Audit that external parameter mutation cannot alter a stored record."""
    params = np.array([0.5], dtype=float)

    def objective(values: FloatArray) -> float:
        return float(np.cos(values[0]))

    with gradient_tape(backend="statevector") as tape:
        record = tape.record_parameter_shift("alias_snapshot", objective, params)
    before = record.to_dict()
    params[0] = 10.0
    after = record.to_dict()
    if before != after:  # pragma: no cover
        raise RuntimeError("external parameter alias mutation changed tape record")
    return GradientTapeContractCheck(
        name="parameter_alias_snapshot",
        status="supported",
        evidence=(record.parameter_fingerprint, record.replay_fingerprint),
    )


def _audit_objective_mutation_fails_closed() -> GradientTapeContractCheck:
    """Audit fail-closed behaviour for objectives mutating replay inputs."""
    params = np.array([0.6], dtype=float)

    def mutating_objective(values: FloatArray) -> float:
        values[0] = values[0] + 1.0
        return float(values[0])

    blocked_reason = ""
    with gradient_tape(backend="statevector") as tape:
        try:
            tape.record_parameter_shift("mutating", mutating_objective, params)
        except ValueError as exc:
            blocked_reason = str(exc)
    if not blocked_reason:  # pragma: no cover
        raise RuntimeError("mutating objective did not fail closed")
    return GradientTapeContractCheck(
        name="objective_mutation",
        status="fail_closed",
        evidence=("mutation guard rejected replay input mutation",),
        blocked_reason=blocked_reason,
    )


def _audit_unstable_replay_fails_closed() -> GradientTapeContractCheck:
    """Audit fail-closed behaviour for control-flow/stateful replay drift."""
    params = np.array([0.7], dtype=float)
    calls = 0

    def stateful_objective(values: FloatArray) -> float:
        nonlocal calls
        calls += 1
        return float(values[0] + calls)

    blocked_reason = ""
    with gradient_tape(backend="statevector") as tape:
        try:
            tape.record_parameter_shift("stateful", stateful_objective, params)
        except ValueError as exc:
            blocked_reason = str(exc)
    if not blocked_reason:  # pragma: no cover
        raise RuntimeError("unstable objective replay did not fail closed")
    return GradientTapeContractCheck(
        name="control_flow_replay_stability",
        status="fail_closed",
        evidence=(f"calls_before_block={calls}",),
        blocked_reason=blocked_reason,
    )


def gradient_tape(
    *,
    backend: str = "statevector",
    shots: int | None = None,
    seed: int | None = None,
    confidence_level: float = 0.95,
    allow_hardware: bool = False,
) -> QuantumGradientTape:
    """Return a context-managed quantum-gradient tape."""
    return QuantumGradientTape(
        backend=backend,
        shots=shots,
        seed=seed,
        confidence_level=confidence_level,
        allow_hardware=allow_hardware,
    )


__all__ = [
    "GRADIENT_TAPE_CONTRACT_CLAIM_BOUNDARY",
    "GradientTapeContractAuditResult",
    "GradientTapeContractCheck",
    "QuantumGradientTape",
    "TapeGradientRecord",
    "TapeContractStatus",
    "TapeKind",
    "gradient_tape",
    "run_gradient_tape_contract_audit",
]
