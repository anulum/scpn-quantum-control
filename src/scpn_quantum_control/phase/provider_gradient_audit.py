# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Provider Gradient Readiness Audit
"""Executable readiness audit for provider-safe quantum gradients."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..differentiable import ParameterShiftRule
from .gradient_backend import QuantumGradientPlan, plan_quantum_gradient_backend
from .param_shift import multi_frequency_parameter_shift_rule
from .provider_gradient import (
    ProviderExpectationSample,
    ProviderGradientExecutionResult,
    execute_provider_parameter_shift_gradient,
)

FloatArray: TypeAlias = NDArray[np.float64]
ProviderReadinessOutcome = Literal["supported", "plan_blocked", "execution_blocked"]
ProviderReadinessSampler = Callable[[FloatArray, int | None], ProviderExpectationSample]


@dataclass(frozen=True)
class ProviderGradientReadinessScenario:
    """One executable provider-gradient readiness scenario."""

    name: str
    backend: str
    values: FloatArray
    shots: int | None
    rule: ParameterShiftRule | None
    expected_gradient: FloatArray | None
    expected_outcome: ProviderReadinessOutcome
    description: str

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("scenario name must be non-empty")
        if not self.backend.strip():
            raise ValueError("scenario backend must be non-empty")
        values = np.asarray(self.values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("scenario values must be a non-empty finite vector")
        if self.shots is not None and (
            isinstance(self.shots, bool) or not isinstance(self.shots, int) or self.shots <= 0
        ):
            raise ValueError("scenario shots must be positive or None")
        if self.expected_gradient is not None:
            expected_gradient = np.asarray(self.expected_gradient, dtype=np.float64)
            if expected_gradient.shape != values.shape or not np.all(
                np.isfinite(expected_gradient)
            ):
                raise ValueError("expected_gradient must match values and be finite")
            object.__setattr__(self, "expected_gradient", expected_gradient)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "name", self.name.strip())
        object.__setattr__(self, "backend", self.backend.strip())
        object.__setattr__(self, "description", self.description.strip())

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready scenario metadata."""
        return {
            "name": self.name,
            "backend": self.backend,
            "values": self.values.tolist(),
            "shots": self.shots,
            "rule_terms": 1 if self.rule is None else len(self.rule.terms),
            "expected_gradient": None
            if self.expected_gradient is None
            else self.expected_gradient.tolist(),
            "expected_outcome": self.expected_outcome,
            "description": self.description,
        }


@dataclass(frozen=True)
class ProviderGradientReadinessRecord:
    """Result of one provider-gradient readiness scenario."""

    scenario: ProviderGradientReadinessScenario
    plan: QuantumGradientPlan
    result: ProviderGradientExecutionResult | None
    supported: bool
    passed: bool
    failure_reason: str | None
    max_abs_error: float | None
    claim_boundary: str

    @property
    def blocked(self) -> bool:
        """Return true when the scenario was intentionally blocked."""
        return not self.supported

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready readiness record metadata."""
        return {
            "scenario": self.scenario.to_dict(),
            "plan": _plan_to_dict(self.plan),
            "result": None if self.result is None else self.result.to_dict(),
            "supported": self.supported,
            "passed": self.passed,
            "failure_reason": self.failure_reason,
            "max_abs_error": self.max_abs_error,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ProviderGradientReadinessAuditResult:
    """Executable support matrix for provider-gradient readiness."""

    records: tuple[ProviderGradientReadinessRecord, ...]
    passed: bool
    claim_boundary: str

    @property
    def supported_records(self) -> tuple[ProviderGradientReadinessRecord, ...]:
        """Return scenarios that executed and matched their gradient references."""
        return tuple(record for record in self.records if record.supported)

    @property
    def blocked_records(self) -> tuple[ProviderGradientReadinessRecord, ...]:
        """Return scenarios that fail closed by plan or execution guard."""
        return tuple(record for record in self.records if record.blocked)

    @property
    def failing_records(self) -> tuple[ProviderGradientReadinessRecord, ...]:
        """Return scenarios whose observed outcome did not match the expectation."""
        return tuple(record for record in self.records if not record.passed)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready provider-gradient audit metadata."""
        return {
            "records": [record.to_dict() for record in self.records],
            "passed": self.passed,
            "claim_boundary": self.claim_boundary,
        }


def default_provider_gradient_readiness_scenarios() -> tuple[
    ProviderGradientReadinessScenario, ...
]:
    """Return built-in provider-gradient support and fail-closed scenarios."""
    single_values = np.array([0.2, -0.4], dtype=np.float64)
    single_gradient = np.array(
        [-math.sin(single_values[0]), 0.25 * math.cos(single_values[1])],
        dtype=np.float64,
    )
    multi_values = np.array([0.4], dtype=np.float64)
    multi_rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    multi_gradient = np.array(
        [math.cos(multi_values[0]) - 0.2 * math.sin(2.0 * multi_values[0])],
        dtype=np.float64,
    )
    return (
        ProviderGradientReadinessScenario(
            name="statevector_parameter_shift",
            backend="statevector",
            values=single_values,
            shots=None,
            rule=None,
            expected_gradient=single_gradient,
            expected_outcome="supported",
            description="deterministic local provider callback with standard parameter-shift",
        ),
        ProviderGradientReadinessScenario(
            name="finite_shot_parameter_shift",
            backend="qasm_simulator",
            values=single_values,
            shots=400,
            rule=None,
            expected_gradient=single_gradient,
            expected_outcome="supported",
            description="finite-shot local provider callback with variance metadata",
        ),
        ProviderGradientReadinessScenario(
            name="multi_frequency_finite_shot",
            backend="qasm_simulator",
            values=multi_values,
            shots=300,
            rule=multi_rule,
            expected_gradient=multi_gradient,
            expected_outcome="supported",
            description="multi-frequency parameter-shift callback with per-term variance metadata",
        ),
        ProviderGradientReadinessScenario(
            name="hardware_without_policy",
            backend="ibm_quantum",
            values=single_values,
            shots=1024,
            rule=None,
            expected_gradient=None,
            expected_outcome="plan_blocked",
            description="hardware alias must fail closed without explicit gradient policy approval",
        ),
        ProviderGradientReadinessScenario(
            name="unknown_backend",
            backend="new_vendor_backend",
            values=single_values,
            shots=None,
            rule=None,
            expected_gradient=None,
            expected_outcome="plan_blocked",
            description="unknown provider family must fail closed with simulator alternatives",
        ),
        ProviderGradientReadinessScenario(
            name="finite_shot_missing_variance",
            backend="qasm_simulator",
            values=single_values,
            shots=400,
            rule=None,
            expected_gradient=None,
            expected_outcome="execution_blocked",
            description="finite-shot callbacks must return variance for every shifted sample",
        ),
    )


def run_provider_gradient_readiness_audit(
    scenarios: tuple[ProviderGradientReadinessScenario, ...] | None = None,
    *,
    tolerance: float = 1e-10,
) -> ProviderGradientReadinessAuditResult:
    """Run executable provider-gradient readiness checks.

    The audit intentionally mixes successful local callback routes with blocked
    hardware, unknown-backend, and malformed-sample routes. A passing audit means
    supported paths produce the expected gradients and unsupported paths refuse
    execution with explicit reasons; it is not a hardware execution claim.
    """
    if tolerance <= 0.0 or not math.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    selected = default_provider_gradient_readiness_scenarios() if scenarios is None else scenarios
    if not selected:
        raise ValueError("at least one provider-gradient readiness scenario is required")
    records = tuple(
        _run_provider_gradient_readiness_scenario(scenario, tolerance) for scenario in selected
    )
    passed = all(record.passed for record in records)
    return ProviderGradientReadinessAuditResult(
        records=records,
        passed=passed,
        claim_boundary=(
            "provider-gradient readiness audit only; supported records are local "
            "callback executions, blocked records are fail-closed governance evidence, "
            "and no live hardware-gradient claim is implied"
        ),
    )


def _run_provider_gradient_readiness_scenario(
    scenario: ProviderGradientReadinessScenario,
    tolerance: float,
) -> ProviderGradientReadinessRecord:
    rule_terms = 1 if scenario.rule is None else len(scenario.rule.terms)
    plan = plan_quantum_gradient_backend(
        scenario.backend,
        n_params=scenario.values.size,
        shift_terms=rule_terms,
        shots=scenario.shots,
        finite_shot=scenario.shots is not None,
    )
    try:
        result = execute_provider_parameter_shift_gradient(
            _sampler_for_scenario(scenario),
            scenario.values,
            backend=scenario.backend,
            shots=scenario.shots,
            rule=scenario.rule,
        )
    except ValueError as exc:
        failure_reason = str(exc)
        expected_block = scenario.expected_outcome in {"plan_blocked", "execution_blocked"}
        plan_blocked = plan.fail_closed
        execution_blocked = not plan.fail_closed
        passed = expected_block and (
            (scenario.expected_outcome == "plan_blocked" and plan_blocked)
            or (scenario.expected_outcome == "execution_blocked" and execution_blocked)
        )
        return ProviderGradientReadinessRecord(
            scenario=scenario,
            plan=plan,
            result=None,
            supported=False,
            passed=passed,
            failure_reason=failure_reason,
            max_abs_error=None,
            claim_boundary="blocked provider-gradient scenario; no gradient result emitted",
        )

    max_abs_error = _max_abs_error(result.gradient, scenario.expected_gradient)
    supported = scenario.expected_outcome == "supported" and max_abs_error <= tolerance
    return ProviderGradientReadinessRecord(
        scenario=scenario,
        plan=result.plan,
        result=result,
        supported=supported,
        passed=supported,
        failure_reason=None
        if supported
        else "gradient route executed but did not match expectation",
        max_abs_error=max_abs_error,
        claim_boundary=result.claim_boundary,
    )


def _sampler_for_scenario(
    scenario: ProviderGradientReadinessScenario,
) -> ProviderReadinessSampler:
    if scenario.name == "multi_frequency_finite_shot":
        return _multi_frequency_sampler
    if scenario.name == "finite_shot_missing_variance":
        return _missing_variance_sampler
    return _single_frequency_sampler


def _single_frequency_objective(values: FloatArray) -> float:
    return float(math.cos(values[0]) + 0.25 * math.sin(values[1]))


def _single_frequency_sampler(values: FloatArray, shots: int | None) -> ProviderExpectationSample:
    variance = None if shots is None else 0.04
    return ProviderExpectationSample(
        value=_single_frequency_objective(values),
        variance=variance,
        shots=shots,
        metadata={"audit_objective": "single_frequency"},
    )


def _multi_frequency_objective(values: FloatArray) -> float:
    return float(math.sin(values[0]) + 0.1 * math.cos(2.0 * values[0]))


def _multi_frequency_sampler(values: FloatArray, shots: int | None) -> ProviderExpectationSample:
    return ProviderExpectationSample(
        value=_multi_frequency_objective(values),
        variance=0.05 if shots is not None else None,
        shots=shots,
        metadata={"audit_objective": "multi_frequency"},
    )


def _missing_variance_sampler(values: FloatArray, shots: int | None) -> ProviderExpectationSample:
    return ProviderExpectationSample(
        value=_single_frequency_objective(values),
        shots=shots,
        metadata={"audit_objective": "missing_variance"},
    )


def _max_abs_error(actual: FloatArray, expected: FloatArray | None) -> float:
    if expected is None:
        return math.inf
    return float(np.max(np.abs(actual - expected)))


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
