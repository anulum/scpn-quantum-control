# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Objective Audit
"""Reviewer-facing correctness evidence for composed phase objectives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .objectives import (
    ComposedObjectiveTrainingCertificate,
    ComposedObjectiveTrainingResult,
    ComposedPhaseObjective,
    build_phase_control_objective,
    train_composed_phase_objective,
    validate_composed_objective_training,
)

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ComposedObjectiveGradientAgreement:
    """Finite-difference agreement record for one composed objective."""

    objective_name: str
    params: FloatArray
    exact_gradient: FloatArray
    finite_difference_gradient: FloatArray
    max_abs_error: float
    max_relative_error: float
    absolute_tolerance: float
    relative_tolerance: float
    finite_difference_step: float
    finite_difference_evaluations: int
    passed: bool
    parameter_shift_compatible: bool
    term_names: tuple[str, ...]
    claim_boundary: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready gradient-agreement evidence."""
        return {
            "objective_name": self.objective_name,
            "params": self.params.tolist(),
            "exact_gradient": self.exact_gradient.tolist(),
            "finite_difference_gradient": self.finite_difference_gradient.tolist(),
            "max_abs_error": self.max_abs_error,
            "max_relative_error": self.max_relative_error,
            "absolute_tolerance": self.absolute_tolerance,
            "relative_tolerance": self.relative_tolerance,
            "finite_difference_step": self.finite_difference_step,
            "finite_difference_evaluations": self.finite_difference_evaluations,
            "passed": self.passed,
            "parameter_shift_compatible": self.parameter_shift_compatible,
            "term_names": list(self.term_names),
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True)
class ComposedObjectiveAuditSuiteResult:
    """Built-in objective correctness and convergence audit suite."""

    pure_gradient: ComposedObjectiveGradientAgreement
    hybrid_gradient: ComposedObjectiveGradientAgreement
    pure_training: ComposedObjectiveTrainingResult
    hybrid_training: ComposedObjectiveTrainingResult
    pure_certificate: ComposedObjectiveTrainingCertificate
    hybrid_certificate: ComposedObjectiveTrainingCertificate
    pure_parameter_shift_gate_passed: bool
    hybrid_parameter_shift_gate_failed: bool
    hybrid_parameter_shift_error: str
    unsupported_scenarios: tuple[str, ...]
    passed: bool
    claim_boundary: str

    @property
    def gradient_records(self) -> tuple[ComposedObjectiveGradientAgreement, ...]:
        """Return gradient agreement records in audit order."""
        return (self.pure_gradient, self.hybrid_gradient)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready audit-suite evidence."""
        return {
            "pure_gradient": self.pure_gradient.to_dict(),
            "hybrid_gradient": self.hybrid_gradient.to_dict(),
            "pure_training": self.pure_training.to_dict(),
            "hybrid_training": self.hybrid_training.to_dict(),
            "pure_certificate": self.pure_certificate.to_dict(),
            "hybrid_certificate": self.hybrid_certificate.to_dict(),
            "pure_parameter_shift_gate_passed": self.pure_parameter_shift_gate_passed,
            "hybrid_parameter_shift_gate_failed": self.hybrid_parameter_shift_gate_failed,
            "hybrid_parameter_shift_error": self.hybrid_parameter_shift_error,
            "unsupported_scenarios": list(self.unsupported_scenarios),
            "passed": self.passed,
            "claim_boundary": self.claim_boundary,
        }


def _as_vector(name: str, values: ArrayLike) -> FloatArray:
    vector = np.asarray(values, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _positive_float(name: str, value: float) -> float:
    scalar = float(value)
    if scalar <= 0.0 or not np.isfinite(scalar):
        raise ValueError(f"{name} must be a positive finite scalar")
    return scalar


def _finite_difference_gradient(
    objective: ComposedPhaseObjective,
    params: FloatArray,
    *,
    step: float,
) -> FloatArray:
    gradient = np.zeros_like(params)
    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (objective(plus) - objective(minus)) / (2.0 * step)
    return gradient


def verify_composed_objective_gradient(
    objective: ComposedPhaseObjective,
    params: ArrayLike,
    *,
    finite_difference_step: float = 1e-6,
    absolute_tolerance: float = 1e-5,
    relative_tolerance: float = 1e-5,
) -> ComposedObjectiveGradientAgreement:
    """Verify exact term-wise gradients against central finite differences."""
    vector = _as_vector("params", params)
    step = _positive_float("finite_difference_step", finite_difference_step)
    abs_tol = _positive_float("absolute_tolerance", absolute_tolerance)
    rel_tol = _positive_float("relative_tolerance", relative_tolerance)
    exact = objective.evaluate(vector).gradient
    finite_difference = _finite_difference_gradient(objective, vector, step=step)
    abs_error = np.abs(exact - finite_difference)
    denominator = np.maximum(np.abs(finite_difference), abs_tol)
    relative_error = abs_error / denominator
    max_abs_error = float(np.max(abs_error))
    max_relative_error = float(np.max(relative_error))
    return ComposedObjectiveGradientAgreement(
        objective_name=objective.name,
        params=vector,
        exact_gradient=exact,
        finite_difference_gradient=finite_difference,
        max_abs_error=max_abs_error,
        max_relative_error=max_relative_error,
        absolute_tolerance=abs_tol,
        relative_tolerance=rel_tol,
        finite_difference_step=step,
        finite_difference_evaluations=2 * vector.size,
        passed=max_abs_error <= abs_tol or max_relative_error <= rel_tol,
        parameter_shift_compatible=objective.parameter_shift_compatible,
        term_names=objective.term_names,
        claim_boundary=(
            "central finite-difference diagnostic for small smooth composed "
            "phase objectives; not a scalable hardware-gradient method"
        ),
    )


def _certificate_passed(certificate: ComposedObjectiveTrainingCertificate) -> bool:
    gates = (
        certificate.within_gradient_tolerance,
        certificate.within_target_value_tolerance,
        certificate.min_decrease_satisfied,
    )
    return certificate.monotone_accepted_values and all(gate is not False for gate in gates)


def run_composed_objective_audit_suite() -> ComposedObjectiveAuditSuiteResult:
    """Run the built-in composed-objective correctness and convergence audit."""
    initial = np.array([0.8, -0.7], dtype=np.float64)
    pure_objective = build_phase_control_objective(
        2,
        energy_weight=1.0,
        fidelity_target=np.zeros(2, dtype=np.float64),
        fidelity_weight=0.2,
        regularization_center=np.zeros(2, dtype=np.float64),
        regularization_weight=0.05,
        symmetry_pairs=((0, 1),),
        symmetry_weight=0.1,
    )
    hybrid_objective = build_phase_control_objective(
        2,
        energy_weight=1.0,
        fidelity_target=np.zeros(2, dtype=np.float64),
        fidelity_weight=0.2,
        regularization_center=np.zeros(2, dtype=np.float64),
        regularization_weight=0.05,
        symmetry_pairs=((0, 1),),
        symmetry_weight=0.1,
        safety_bounds=(-1.0, 1.0),
        safety_weight=0.2,
    )

    pure_gradient = verify_composed_objective_gradient(pure_objective, initial)
    hybrid_gradient = verify_composed_objective_gradient(hybrid_objective, initial)
    pure_gate_passed = True
    pure_objective.require_parameter_shift_compatible()
    hybrid_gate_failed = False
    hybrid_error = ""
    try:
        hybrid_objective.require_parameter_shift_compatible()
    except ValueError as exc:
        hybrid_gate_failed = True
        hybrid_error = str(exc)

    pure_training = train_composed_phase_objective(
        pure_objective,
        initial,
        learning_rate=0.35,
        max_steps=40,
    )
    hybrid_training = train_composed_phase_objective(
        hybrid_objective,
        initial,
        learning_rate=0.35,
        max_steps=40,
    )
    pure_certificate = validate_composed_objective_training(
        pure_training,
        min_decrease=0.1,
    )
    hybrid_certificate = validate_composed_objective_training(
        hybrid_training,
        min_decrease=0.1,
    )
    passed = (
        pure_gradient.passed
        and hybrid_gradient.passed
        and pure_gate_passed
        and hybrid_gate_failed
        and _certificate_passed(pure_certificate)
        and _certificate_passed(hybrid_certificate)
    )
    return ComposedObjectiveAuditSuiteResult(
        pure_gradient=pure_gradient,
        hybrid_gradient=hybrid_gradient,
        pure_training=pure_training,
        hybrid_training=hybrid_training,
        pure_certificate=pure_certificate,
        hybrid_certificate=hybrid_certificate,
        pure_parameter_shift_gate_passed=pure_gate_passed,
        hybrid_parameter_shift_gate_failed=hybrid_gate_failed,
        hybrid_parameter_shift_error=hybrid_error,
        unsupported_scenarios=(
            "smooth box-safety penalties are analytic-only and fail pure parameter-shift gates",
            "finite-difference agreement is a local diagnostic, not a hardware-gradient route",
            "training certificates show local descent evidence, not global optimality",
        ),
        passed=passed,
        claim_boundary=(
            "local correctness and convergence evidence for composed phase "
            "objectives; no hardware execution, throughput, or global "
            "optimality claim is implied"
        ),
    )
