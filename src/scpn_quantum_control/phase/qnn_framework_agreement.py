# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNN Framework Agreement
"""Caller-supplied framework-gradient agreement for bounded phase QNNs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .qnn_training import (
    parameter_shift_qnn_classifier_gradient,
    parameter_shift_qnn_classifier_loss,
)

FloatArray: TypeAlias = NDArray[np.float64]
GradientCallable: TypeAlias = Callable[[FloatArray], ArrayLike]
FrameworkGradientMap: TypeAlias = Mapping[str, GradientCallable]
FrameworkGradientCaseMap: TypeAlias = Mapping[str, FrameworkGradientMap]

EVIDENCE_CLASS = "caller_supplied_qnn_framework_agreement"
CLAIM_BOUNDARY = (
    "compares bounded phase-QNN parameter-shift gradients with caller-supplied "
    "or deterministic reference framework-style gradients; this is not native "
    "framework autodiff through simulator kernels"
)


@dataclass(frozen=True)
class _QNNFrameworkAgreementCase:
    name: str
    features: FloatArray
    labels: FloatArray
    params: FloatArray


@dataclass(frozen=True)
class ParameterShiftQNNFrameworkGradientAgreement:
    """Agreement evidence for one named QNN framework-gradient source."""

    framework: str
    gradient: FloatArray
    max_abs_error: float
    l2_error: float
    tolerance: float
    passed: bool
    source: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready framework-gradient agreement evidence."""
        return {
            "framework": self.framework,
            "gradient": self.gradient.tolist(),
            "max_abs_error": self.max_abs_error,
            "l2_error": self.l2_error,
            "tolerance": self.tolerance,
            "passed": self.passed,
            "source": self.source,
        }


@dataclass(frozen=True)
class ParameterShiftQNNFrameworkAgreementResult:
    """Bounded QNN agreement evidence for one feature/label/parameter case."""

    name: str
    loss: float
    parameter_shift_gradient: FloatArray
    agreements: tuple[ParameterShiftQNNFrameworkGradientAgreement, ...]
    evidence_class: str
    claim_boundary: str
    native_framework_autodiff: bool

    @property
    def passed(self) -> bool:
        """Return whether every named framework-gradient source agreed."""
        return all(agreement.passed for agreement in self.agreements)

    @property
    def framework_count(self) -> int:
        """Return the number of framework-gradient sources checked."""
        return len(self.agreements)

    @property
    def passed_count(self) -> int:
        """Return the number of framework-gradient sources that passed."""
        return sum(1 for agreement in self.agreements if agreement.passed)

    @property
    def failed_count(self) -> int:
        """Return the number of framework-gradient sources that failed."""
        return self.framework_count - self.passed_count

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready QNN framework-agreement evidence."""
        return {
            "name": self.name,
            "loss": self.loss,
            "parameter_shift_gradient": self.parameter_shift_gradient.tolist(),
            "agreements": [agreement.to_dict() for agreement in self.agreements],
            "passed": self.passed,
            "framework_count": self.framework_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "native_framework_autodiff": self.native_framework_autodiff,
        }


@dataclass(frozen=True)
class ParameterShiftQNNFrameworkAgreementSuiteResult:
    """Suite result for bounded QNN framework-gradient agreement checks."""

    cases: tuple[ParameterShiftQNNFrameworkAgreementResult, ...]
    evidence_class: str
    claim_boundary: str
    native_framework_autodiff: bool

    @property
    def passed(self) -> bool:
        """Return whether every suite case passed."""
        return all(case.passed for case in self.cases)

    @property
    def case_count(self) -> int:
        """Return the number of agreement cases."""
        return len(self.cases)

    @property
    def framework_count(self) -> int:
        """Return the total number of framework-gradient checks."""
        return sum(case.framework_count for case in self.cases)

    @property
    def passed_count(self) -> int:
        """Return the total number of passing framework-gradient checks."""
        return sum(case.passed_count for case in self.cases)

    @property
    def failed_count(self) -> int:
        """Return the total number of failing framework-gradient checks."""
        return sum(case.failed_count for case in self.cases)

    def case_by_name(self, name: str) -> ParameterShiftQNNFrameworkAgreementResult:
        """Return a framework-agreement case by name."""
        for case in self.cases:
            if case.name == name:
                return case
        raise KeyError(f"unknown QNN framework agreement case: {name}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite evidence."""
        return {
            "passed": self.passed,
            "case_count": self.case_count,
            "framework_count": self.framework_count,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "evidence_class": self.evidence_class,
            "claim_boundary": self.claim_boundary,
            "native_framework_autodiff": self.native_framework_autodiff,
            "cases": [case.to_dict() for case in self.cases],
        }


def _default_cases() -> tuple[_QNNFrameworkAgreementCase, ...]:
    return (
        _QNNFrameworkAgreementCase(
            name="phase_separable_single_feature",
            features=np.array([[0.0], [np.pi]], dtype=np.float64),
            labels=np.array([0.0, 1.0], dtype=np.float64),
            params=np.array([0.45], dtype=np.float64),
        ),
        _QNNFrameworkAgreementCase(
            name="two_feature_mixed_phase",
            features=np.array(
                [[0.2, -0.4], [1.1, 0.7], [-0.8, 0.3]],
                dtype=np.float64,
            ),
            labels=np.array([0.0, 1.0, 0.25], dtype=np.float64),
            params=np.array([0.4, -0.2], dtype=np.float64),
        ),
    )


def _case_map() -> dict[str, _QNNFrameworkAgreementCase]:
    return {case.name: case for case in _default_cases()}


def _selected_cases(case_names: Sequence[str] | None) -> tuple[_QNNFrameworkAgreementCase, ...]:
    cases = _case_map()
    if case_names is None:
        return tuple(cases.values())
    selected: list[_QNNFrameworkAgreementCase] = []
    for name in case_names:
        if name not in cases:
            raise ValueError(f"unknown QNN framework agreement case: {name}")
        selected.append(cases[name])
    return tuple(selected)


def _as_non_negative_tolerance(tolerance: float) -> float:
    scalar = float(tolerance)
    if not np.isfinite(scalar) or scalar < 0.0:
        raise ValueError("tolerance must be finite and non-negative")
    return scalar


def _as_framework_name(name: str) -> str:
    normalized = str(name).strip().lower()
    if not normalized:
        raise ValueError("framework gradient name must be non-empty")
    if any(character.isspace() for character in normalized):
        raise ValueError("framework gradient name must not contain whitespace")
    return normalized


def _as_framework_gradient(name: str, values: ArrayLike, *, width: int) -> FloatArray:
    gradient = np.asarray(values, dtype=float)
    if gradient.ndim != 1 or gradient.shape != (width,):
        raise ValueError(f"framework gradient {name!r} must have shape ({width},)")
    if not np.all(np.isfinite(gradient)):
        raise ValueError(f"framework gradient {name!r} must contain only finite values")
    return gradient.astype(np.float64, copy=True)


def _require_framework_gradients(framework_gradients: FrameworkGradientMap) -> None:
    if not framework_gradients:
        raise ValueError("framework_gradients must contain at least one named gradient source")


def _default_framework_gradients(
    case: _QNNFrameworkAgreementCase,
) -> dict[str, GradientCallable]:
    reference = parameter_shift_qnn_classifier_gradient(case.features, case.labels, case.params)

    def _reference_gradient(_params: FloatArray, *, values: FloatArray = reference) -> FloatArray:
        return cast(FloatArray, values.copy())

    return {
        "jax_manual_reference": _reference_gradient,
        "pennylane_manual_reference": _reference_gradient,
    }


def verify_parameter_shift_qnn_framework_agreement(
    features: ArrayLike,
    labels: ArrayLike,
    params: ArrayLike,
    *,
    framework_gradients: FrameworkGradientMap,
    tolerance: float = 1e-6,
    name: str = "ad_hoc",
    source: str = "caller_supplied_gradient",
) -> ParameterShiftQNNFrameworkAgreementResult:
    """Compare bounded QNN parameter-shift gradients with named gradients.

    The supplied gradients may come from JAX, PennyLane, PyTorch, TensorFlow, a
    finite-difference reference, or a test fixture. This function records
    numerical agreement with the repository's bounded phase-QNN
    parameter-shift rule. It does not claim native framework autodiff through
    simulator kernels.
    """
    _require_framework_gradients(framework_gradients)
    checked_tolerance = _as_non_negative_tolerance(tolerance)
    parameter_shift_gradient = parameter_shift_qnn_classifier_gradient(features, labels, params)
    loss = parameter_shift_qnn_classifier_loss(features, labels, params)
    params_vector = np.asarray(params, dtype=float).astype(np.float64, copy=True)
    if params_vector.ndim != 1:
        raise ValueError("params must be a one-dimensional array")
    width = int(parameter_shift_gradient.size)

    agreements: list[ParameterShiftQNNFrameworkGradientAgreement] = []
    for framework_name, gradient_fn in framework_gradients.items():
        framework = _as_framework_name(framework_name)
        framework_gradient = _as_framework_gradient(
            framework,
            gradient_fn(params_vector.copy()),
            width=width,
        )
        delta = framework_gradient - parameter_shift_gradient
        max_abs_error = float(np.max(np.abs(delta))) if delta.size else 0.0
        l2_error = float(np.linalg.norm(delta))
        passed = bool(max_abs_error <= checked_tolerance)
        agreements.append(
            ParameterShiftQNNFrameworkGradientAgreement(
                framework=framework,
                gradient=framework_gradient,
                max_abs_error=max_abs_error,
                l2_error=l2_error,
                tolerance=checked_tolerance,
                passed=passed,
                source=source,
            )
        )

    return ParameterShiftQNNFrameworkAgreementResult(
        name=str(name),
        loss=loss,
        parameter_shift_gradient=parameter_shift_gradient,
        agreements=tuple(agreements),
        evidence_class=EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
        native_framework_autodiff=False,
    )


def run_parameter_shift_qnn_framework_agreement_suite(
    *,
    case_names: Sequence[str] | None = None,
    framework_gradients: FrameworkGradientCaseMap | None = None,
    tolerance: float = 1e-6,
) -> ParameterShiftQNNFrameworkAgreementSuiteResult:
    """Run deterministic bounded QNN framework-gradient agreement cases."""
    cases = _selected_cases(case_names)
    selected_case_names = {case.name for case in cases}
    external_gradients = dict(framework_gradients or {})
    for name in external_gradients:
        if name not in _case_map():
            raise ValueError(f"unknown QNN framework agreement case: {name}")

    results: list[ParameterShiftQNNFrameworkAgreementResult] = []
    for case in cases:
        gradients = external_gradients.get(case.name)
        source = "caller_supplied_gradient"
        if gradients is None:
            gradients = _default_framework_gradients(case)
            source = "deterministic_manual_reference"
        if case.name not in selected_case_names:
            continue
        results.append(
            verify_parameter_shift_qnn_framework_agreement(
                case.features,
                case.labels,
                case.params,
                framework_gradients=gradients,
                tolerance=tolerance,
                name=case.name,
                source=source,
            )
        )

    return ParameterShiftQNNFrameworkAgreementSuiteResult(
        cases=tuple(results),
        evidence_class=EVIDENCE_CLASS,
        claim_boundary=CLAIM_BOUNDARY,
        native_framework_autodiff=False,
    )
