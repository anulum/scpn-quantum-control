# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Phase Gradient Backend Planner
"""Tests for phase/gradient_backend.py quantum-gradient planning."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

import scpn_quantum_control.phase.gradient_backend as gradient_backend_module
from scpn_quantum_control.phase import (
    explain_quantum_gradient_method,
    multi_frequency_parameter_shift_rule,
    plan_parameter_shift_shots,
    plan_quantum_gradient_backend,
)
from scpn_quantum_control.phase.gradient_backend import (
    QuantumGradientBackendCapability,
    QuantumGradientMethodExplanation,
    QuantumGradientPlan,
    QuantumGradientRejectedMethod,
    quantum_gradient_backend_capability,
)
from scpn_quantum_control.phase.param_shift import parameter_shift_gradient_with_uncertainty

SAMPLE_PROVENANCE = {
    "sample_seed": "phase-gradient-backend-test-seed",
    "shot_batch_id": "phase-gradient-backend-test-batch",
    "source_class": "caller_supplied",
}


def test_statevector_backend_auto_selects_deterministic_parameter_shift() -> None:
    plan = plan_quantum_gradient_backend("statevector", n_params=3)

    assert plan.supported
    assert not plan.fail_closed
    assert plan.method == "parameter_shift"
    assert plan.evaluations == 6
    assert plan.shots is None
    assert not plan.finite_shot


def test_method_explanation_describes_statevector_decision_and_rejections() -> None:
    explanation = explain_quantum_gradient_method("statevector", n_params=3)
    payload = explanation.to_dict()

    assert isinstance(explanation, QuantumGradientMethodExplanation)
    assert explanation.supported
    assert explanation.selected_method == "parameter_shift"
    assert explanation.selected_plan.evaluations == 6
    assert explanation.shot_policy.to_dict() == {
        "finite_shot": False,
        "requested_shots": None,
        "planned_shots": None,
        "defaulted": False,
        "confidence_level": None,
        "seed": None,
        "reasons": ["deterministic route does not consume finite-shot samples"],
    }
    rejections = {row.method: row for row in explanation.rejected_methods}
    assert rejections["stochastic_parameter_shift"].supported_if_requested is False
    assert (
        "backend has no finite-shot estimator support"
        in rejections["stochastic_parameter_shift"].reasons
    )
    assert rejections["finite_difference"].to_dict()["reasons"] == [
        "finite_difference is diagnostic-only and not a promoted quantum gradient"
    ]
    assert "finite_difference_diagnostic" in explanation.fallback_path
    assert payload["selected_method"] == "parameter_shift"
    selected_plan = cast(dict[str, object], payload["selected_plan"])
    rejected_methods = cast(list[dict[str, object]], payload["rejected_methods"])
    assert selected_plan["evaluations"] == 6
    assert rejected_methods[0]["method"] == "stochastic_parameter_shift"


def test_backend_plan_accounts_for_multi_frequency_shift_terms() -> None:
    plan = plan_quantum_gradient_backend("statevector", n_params=3, shift_terms=4)

    assert plan.supported
    assert plan.shift_terms == 4
    assert plan.evaluations == 24


def test_finite_shot_backend_auto_selects_stochastic_parameter_shift() -> None:
    plan = plan_quantum_gradient_backend(
        "qasm_simulator",
        n_params=4,
        shots=2048,
        seed=0,
    )

    assert plan.supported
    assert plan.method == "stochastic_parameter_shift"
    assert plan.evaluations == 8
    assert plan.shots == 2048
    assert plan.seed == 0
    assert plan.confidence_level == 0.95


def test_method_explanation_records_finite_shot_default_policy() -> None:
    explanation = explain_quantum_gradient_method(
        "qasm_simulator",
        n_params=2,
        seed=9,
    )

    assert explanation.selected_method == "stochastic_parameter_shift"
    assert explanation.shot_policy.finite_shot
    assert explanation.shot_policy.planned_shots == 4096
    assert explanation.shot_policy.defaulted
    assert explanation.shot_policy.seed == 9
    assert "finite-shot route uses backend default shots" in explanation.shot_policy.reasons
    rejections = {row.method: row for row in explanation.rejected_methods}
    assert rejections["parameter_shift"].supported_if_requested is False
    assert (
        "finite-shot execution needs stochastic_parameter_shift"
        in rejections["parameter_shift"].reasons
    )
    assert "spsa" in explanation.fallback_path
    assert "increase_shots_or_use_statevector" in explanation.fallback_path


def test_method_explanation_respects_explicit_spsa_request() -> None:
    explanation = explain_quantum_gradient_method(
        "qasm_simulator",
        n_params=5,
        method="spsa",
        shots=512,
        confidence_level=0.9,
    )

    assert explanation.requested_method == "spsa"
    assert explanation.selected_method == "spsa"
    assert explanation.selected_plan.evaluations == 2
    assert explanation.shot_policy.planned_shots == 512
    assert explanation.shot_policy.confidence_level == 0.9
    rejections = {row.method: row for row in explanation.rejected_methods}
    assert rejections["stochastic_parameter_shift"].supported_if_requested
    assert rejections["stochastic_parameter_shift"].reasons == (
        "caller explicitly requested spsa",
    )


def test_hardware_backend_fails_closed_without_policy_approval() -> None:
    plan = plan_quantum_gradient_backend("ibm_quantum", n_params=2, shots=1024)

    assert plan.fail_closed
    assert plan.method == "unsupported"
    assert plan.requires_hardware_approval
    assert "hardware gradient execution requires explicit hardware policy approval" in plan.reasons


def test_method_explanation_preserves_hardware_fail_closed_fallbacks() -> None:
    explanation = explain_quantum_gradient_method("ibm_quantum", n_params=2, shots=1024)

    assert not explanation.supported
    assert explanation.selected_method == "unsupported"
    assert explanation.shot_policy.reasons == (
        "unsupported plan does not allocate executable shots",
    )
    assert explanation.fallback_path == ("statevector_simulator", "finite_shot_simulator")
    assert all(not row.supported_if_requested for row in explanation.rejected_methods)
    assert {"hardware gradient execution requires explicit hardware policy approval"} == {
        reason for row in explanation.rejected_methods for reason in row.reasons
    }


def test_unknown_backend_is_unsupported_with_safe_alternatives() -> None:
    capability = quantum_gradient_backend_capability("new_vendor_backend")
    plan = plan_quantum_gradient_backend("new_vendor_backend", n_params=2)

    assert capability.family == "unknown"
    assert plan.fail_closed
    assert "statevector_simulator" in plan.alternatives
    assert "finite_shot_simulator" in plan.alternatives


def test_method_explanation_preserves_unknown_backend_boundaries() -> None:
    explanation = explain_quantum_gradient_method(
        "new_vendor_backend",
        n_params=2,
        finite_shot=True,
        seed=3,
        confidence_level=0.8,
    )

    assert not explanation.supported
    assert explanation.selected_plan.backend == "new_vendor_backend"
    assert explanation.shot_policy.finite_shot
    assert explanation.shot_policy.seed == 3
    assert explanation.shot_policy.confidence_level == 0.8
    assert explanation.fallback_path == ("statevector_simulator", "finite_shot_simulator")
    assert {"unknown backend has no registered gradient capability"} == {
        reason for row in explanation.rejected_methods for reason in row.reasons
    }


def test_planner_rejects_invalid_method_and_shape_controls() -> None:
    with pytest.raises(ValueError, match="method must be one of"):
        plan_quantum_gradient_backend("statevector", n_params=2, method="adjoint")
    with pytest.raises(ValueError, match="n_params"):
        plan_quantum_gradient_backend("statevector", n_params=0)
    with pytest.raises(ValueError, match="shift_terms"):
        plan_quantum_gradient_backend("statevector", n_params=2, shift_terms=0)
    with pytest.raises(ValueError, match="shots"):
        plan_quantum_gradient_backend("qasm_simulator", n_params=2, shots=0)
    with pytest.raises(ValueError, match="backend"):
        plan_quantum_gradient_backend(" ", n_params=2)
    with pytest.raises(ValueError, match="seed"):
        plan_quantum_gradient_backend("qasm_simulator", n_params=2, seed=-1)
    with pytest.raises(ValueError, match="confidence_level"):
        plan_quantum_gradient_backend("qasm_simulator", n_params=2, confidence_level=1.0)


def test_planner_records_explicitly_unsupported_and_unavailable_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def no_parameter_shift_capability(_backend: str) -> QuantumGradientBackendCapability:
        return QuantumGradientBackendCapability(
            backend="custom_backend",
            family="custom",
            supports_parameter_shift=False,
            supports_finite_shot=False,
            supports_adjoint=False,
            supports_spsa=False,
            hardware=False,
            default_shots=None,
            notes=("custom backend without parameter-shift",),
        )

    hardware_missing_shots = plan_quantum_gradient_backend(
        "hardware",
        n_params=2,
        method="stochastic_parameter_shift",
        allow_hardware=True,
    )
    disabled = plan_quantum_gradient_backend(
        "statevector",
        n_params=2,
        method="unsupported",
    )
    monkeypatch.setattr(
        gradient_backend_module,
        "quantum_gradient_backend_capability",
        no_parameter_shift_capability,
    )
    unsupported_parameter_shift = plan_quantum_gradient_backend(
        "custom_backend",
        n_params=2,
        method="parameter_shift",
    )

    assert unsupported_parameter_shift.fail_closed
    assert "backend does not support parameter-shift" in unsupported_parameter_shift.reasons
    assert hardware_missing_shots.fail_closed
    assert "shots are required for finite-shot gradient planning" in hardware_missing_shots.reasons
    assert disabled.fail_closed
    assert "method explicitly disabled" in disabled.reasons


def test_internal_explanation_helpers_cover_future_supported_rejection_edges() -> None:
    selected_parameter_shift = QuantumGradientPlan(
        backend="future_backend",
        family="future",
        method="parameter_shift",
        supported=True,
        n_params=1,
        shift_terms=1,
        evaluations=2,
        shots=None,
        seed=None,
        finite_shot=False,
        confidence_level=None,
        requires_hardware_approval=False,
        reasons=("future backend",),
        alternatives=(),
    )
    selected_experimental = QuantumGradientPlan(
        backend="future_backend",
        family="future",
        method="experimental",
        supported=True,
        n_params=1,
        shift_terms=1,
        evaluations=1,
        shots=None,
        seed=None,
        finite_shot=False,
        confidence_level=None,
        requires_hardware_approval=False,
        reasons=("future backend",),
        alternatives=(),
    )
    selected_spsa = QuantumGradientPlan(
        backend="future_backend",
        family="future",
        method="spsa",
        supported=True,
        n_params=1,
        shift_terms=1,
        evaluations=2,
        shots=128,
        seed=None,
        finite_shot=True,
        confidence_level=0.95,
        requires_hardware_approval=False,
        reasons=("future backend",),
        alternatives=(),
    )
    supported_candidate = QuantumGradientPlan(
        backend="future_backend",
        family="future",
        method="spsa",
        supported=True,
        n_params=1,
        shift_terms=1,
        evaluations=2,
        shots=128,
        seed=None,
        finite_shot=True,
        confidence_level=0.95,
        requires_hardware_approval=False,
        reasons=("future fallback",),
        alternatives=(),
    )

    assert gradient_backend_module._candidate_rejection_reasons(
        "spsa",
        candidate_plan=supported_candidate,
        selected_plan=selected_parameter_shift,
        requested_method="auto",
    ) == ("deterministic local parameter-shift route has lower estimator noise",)
    assert gradient_backend_module._candidate_rejection_reasons(
        "spsa",
        candidate_plan=supported_candidate,
        selected_plan=selected_experimental,
        requested_method="auto",
    ) == ("experimental selected by backend planner",)
    assert gradient_backend_module._candidate_rejection_reasons(
        "stochastic_parameter_shift",
        candidate_plan=supported_candidate,
        selected_plan=selected_spsa,
        requested_method="auto",
    ) == ("caller selected SPSA diagnostic fallback",)
    assert gradient_backend_module._fallback_path(
        selected_experimental,
        (
            QuantumGradientRejectedMethod(
                method="spsa",
                reasons=("future fallback",),
                supported_if_requested=True,
            ),
        ),
    ) == ("spsa",)


def test_parameter_shift_uncertainty_propagates_finite_shot_noise() -> None:
    result = parameter_shift_gradient_with_uncertainty(
        plus_values=np.array([1.2, -0.3], dtype=float),
        minus_values=np.array([0.8, -0.7], dtype=float),
        plus_variances=np.array([0.04, 0.09], dtype=float),
        minus_variances=np.array([0.04, 0.09], dtype=float),
        shots=400,
        sample_provenance=SAMPLE_PROVENANCE,
        value=0.5,
    )

    np.testing.assert_allclose(result.gradient, np.array([0.2, 0.2], dtype=float))
    assert result.method == "parameter_shift_shot_noise"
    assert result.shots.shape == (2, 2)
    assert np.all(result.standard_error > 0.0)
    assert np.all(result.confidence_radius >= result.standard_error)
    assert result.records[0].sample_seed == SAMPLE_PROVENANCE["sample_seed"]
    assert result.records[0].shot_batch_id == SAMPLE_PROVENANCE["shot_batch_id"]
    assert result.records[0].source_class == SAMPLE_PROVENANCE["source_class"]


def test_phase_uncertainty_wrapper_accepts_multi_frequency_records() -> None:
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    result = parameter_shift_gradient_with_uncertainty(
        plus_values=np.array([[1.2], [0.8]], dtype=float),
        minus_values=np.array([[0.4], [0.1]], dtype=float),
        plus_variances=np.array([[0.04], [0.09]], dtype=float),
        minus_variances=np.array([[0.05], [0.10]], dtype=float),
        shots=np.array([[400], [500]], dtype=float),
        sample_provenance=SAMPLE_PROVENANCE,
        rule=rule,
        backend="qasm_simulator",
    )

    expected_gradient = sum(
        coefficient * (plus - minus)
        for (plus, minus), (_shift, coefficient) in zip(
            [(1.2, 0.4), (0.8, 0.1)],
            rule.terms,
            strict=True,
        )
    )

    assert result.method == "multi_frequency_parameter_shift_shot_noise"
    assert result.shots.shape == (len(rule.terms), 2, 1)
    np.testing.assert_allclose(result.gradient, np.array([expected_gradient]))
    assert result.standard_error[0] > 0.0


def test_parameter_shift_uncertainty_fails_closed_for_hardware_backend() -> None:
    with pytest.raises(ValueError, match="hardware gradient execution requires"):
        parameter_shift_gradient_with_uncertainty(
            plus_values=np.array([1.0], dtype=float),
            minus_values=np.array([0.5], dtype=float),
            plus_variances=np.array([0.1], dtype=float),
            minus_variances=np.array([0.1], dtype=float),
            shots=128,
            backend="hardware",
        )


def test_parameter_shift_shot_allocation_bounds_variance_target() -> None:
    allocation = plan_parameter_shift_shots(
        plus_variances=np.array([0.04, 0.09], dtype=float),
        minus_variances=np.array([0.04, 0.09], dtype=float),
        target_standard_error=0.02,
        min_shots=10,
        max_shots_per_evaluation=1000,
    )

    assert allocation.shots.shape == (2, 2)
    assert np.all(allocation.shots >= 10)
    assert np.all(allocation.shots <= 1000)
    assert np.all(allocation.predicted_standard_error <= 0.02)


def test_phase_shot_allocation_accepts_multi_frequency_rule() -> None:
    rule = multi_frequency_parameter_shift_rule([1.0, 2.0])
    allocation = plan_parameter_shift_shots(
        plus_variances=np.array([[0.04], [0.09]], dtype=float),
        minus_variances=np.array([[0.04], [0.09]], dtype=float),
        target_standard_error=0.03,
        rule=rule,
        min_shots=10,
    )

    assert allocation.method == "multi_frequency_parameter_shift_target_se"
    assert allocation.shots.shape == (len(rule.terms), 2, 1)
    assert allocation.predicted_standard_error[0] <= 0.03
