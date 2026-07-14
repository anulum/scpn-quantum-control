# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — differentiable API tests
# scpn-quantum-control -- tests for unified differentiable API facade
"""Tests for scpn_quantum_control.differentiable_api."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
from _differentiable_api_test_helpers import _require_torch_backend
from numpy.typing import NDArray

import scpn_quantum_control as scpn
from scpn_quantum_control.analysis.finite_size_scaling import DEFAULT_FSS_SYSTEM_SIZES
from scpn_quantum_control.differentiable_api import (
    DifferentiabilityDiagnosticReport,
    DifferentiableDashboardCapabilityRow,
    DifferentiableDashboardCapabilityState,
    DifferentiableDashboardStatus,
    UnifiedDifferentiableAPIResult,
    UnifiedDifferentiableOperation,
    differentiable_api,
    differentiable_benchmark_report,
    differentiable_compile_report,
    differentiable_dashboard_status,
    differentiable_frontend_report,
    differentiable_gradient,
    differentiable_hessian,
    differentiable_jacobian,
    differentiable_qfi_fss_report,
    differentiable_support_report,
    differentiable_transform_algebra_report,
    differentiable_value,
    explain_differentiability,
)
from scpn_quantum_control.differentiable_transform_algebra import (
    REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES,
    TransformAlgebraAudit,
    TransformAlgebraCase,
    assert_transform_algebra_audit_passes,
    run_transform_algebra_audit,
)

FloatArray = NDArray[np.float64]


def _scalar_objective(values: FloatArray) -> float:
    return float(values[0] ** 2 + 3.0 * values[1])


def _vector_objective(values: FloatArray) -> FloatArray:
    return np.array([values[0] + values[1], values[0] * values[1]], dtype=float)


def _periodic_objective(values: FloatArray) -> float:
    return float(np.cos(values[0]) + np.sin(values[1]))


def test_unified_differentiable_value_and_gradient_share_schema() -> None:
    values = np.array([2.0, -1.0], dtype=float)

    value = differentiable_value(_scalar_objective, values, method="finite_difference")
    gradient = differentiable_gradient(_scalar_objective, values, method="finite_difference")

    assert isinstance(value, UnifiedDifferentiableAPIResult)
    assert value.operation == "value"
    assert value.supported
    assert value.gradient is None
    assert value.value == pytest.approx(1.0)
    assert value.to_dict()["fail_closed"] is False

    assert gradient.operation == "gradient"
    assert gradient.supported
    assert gradient.gradient is not None
    np.testing.assert_allclose(gradient.gradient, np.array([4.0, 3.0]), atol=1e-5)
    assert gradient.to_dict()["gradient"] == pytest.approx([4.0, 3.0], abs=1e-5)
    assert "no hardware execution" in gradient.claim_boundary


def test_unified_differentiable_jacobian_and_hessian_routes_are_explicit() -> None:
    values = np.array([2.0, -1.0], dtype=float)

    jacobian = differentiable_jacobian(_vector_objective, values)
    hessian = differentiable_hessian(_scalar_objective, values)

    assert jacobian.operation == "jacobian"
    assert jacobian.jacobian is not None
    np.testing.assert_allclose(
        jacobian.jacobian,
        np.array([[1.0, 1.0], [-1.0, 2.0]], dtype=float),
        atol=1e-5,
    )
    assert jacobian.payload["objective_value"] == pytest.approx([1.0, -2.0])

    assert hessian.operation == "hessian"
    assert hessian.hessian is not None
    np.testing.assert_allclose(
        hessian.hessian,
        np.array([[2.0, 0.0], [0.0, 0.0]], dtype=float),
        atol=1e-4,
    )
    assert hessian.value == pytest.approx(1.0)


def test_unified_dispatcher_routes_numeric_operations_and_defaults() -> None:
    values = np.array([0.3, -0.2], dtype=float)

    value_default = differentiable_api(
        "value",
        objective=_periodic_objective,
        values=values,
    )
    value_explicit = differentiable_api(
        "value",
        objective=_periodic_objective,
        values=values,
        method="finite_difference",
        step=1.0e-6,
    )
    gradient_default = differentiable_api(
        "gradient",
        objective=_periodic_objective,
        values=values,
    )
    jacobian_default = differentiable_api(
        "jacobian",
        objective=_vector_objective,
        values=values,
    )
    jacobian_explicit = differentiable_api(
        "jacobian",
        objective=_vector_objective,
        values=values,
        method="finite_difference",
        step=5.0e-7,
    )
    hessian_default = differentiable_api(
        "hessian",
        objective=_scalar_objective,
        values=values,
    )
    hessian_explicit = differentiable_api(
        "hessian",
        objective=_scalar_objective,
        values=values,
        method="finite_difference",
        step=5.0e-5,
    )

    expected_value = _periodic_objective(values)
    assert value_default.method == "parameter_shift"
    assert value_default.value == pytest.approx(expected_value)
    assert value_explicit.method == "finite_difference_central"
    assert value_explicit.value == pytest.approx(expected_value)
    assert gradient_default.gradient is not None
    np.testing.assert_allclose(
        gradient_default.gradient,
        np.array([-np.sin(values[0]), np.cos(values[1])], dtype=float),
        atol=1.0e-12,
    )
    assert jacobian_default.jacobian is not None
    assert jacobian_explicit.jacobian is not None
    np.testing.assert_allclose(
        jacobian_default.jacobian,
        jacobian_explicit.jacobian,
        atol=1.0e-8,
    )
    assert hessian_default.hessian is not None
    assert hessian_explicit.hessian is not None
    np.testing.assert_allclose(
        hessian_default.hessian,
        hessian_explicit.hessian,
        atol=1.0e-4,
    )


def test_unified_differentiable_support_report_fails_closed_for_unsupported_route() -> None:
    report = differentiable_support_report(
        gate="unregistered_gate",
        observable="pauli_expectation",
    )

    assert report.operation == "support_report"
    assert report.fail_closed
    assert report.payload["supported"] is False
    assert "no registered parameter-shift generator" in report.payload["blocked_reasons"][0]
    payload = cast(dict[str, object], report.to_dict()["payload"])
    assert payload["requires_hardware_policy"] is False


def test_explain_differentiability_reports_reasons_and_matrices() -> None:
    report = explain_differentiability(
        gate="unregistered_gate",
        observable="pauli_expectation",
        backend="hardware",
        shots=1024,
    )

    assert isinstance(report, DifferentiabilityDiagnosticReport)
    assert report.fail_closed
    assert "no registered parameter-shift generator" in report.blocked_reasons[0]
    assert "statevector_simulator" in report.suggested_alternatives
    assert "finite_shot_simulator" in report.suggested_alternatives
    assert report.support_payload["requires_hardware_policy"] is True
    assert "do not execute objectives" in report.claim_boundary

    dependency_rows = {str(row["framework"]): row for row in report.dependency_matrix}
    assert dependency_rows["jax"]["optional_dependency"] == "jax"
    assert dependency_rows["pytorch"]["supported"] is True
    assert dependency_rows["provider_hardware_gradient"]["supported"] is False

    device_rows = {str(row["backend"]): row for row in report.device_matrix}
    assert device_rows["hardware_qpu"]["hardware"] is True
    assert device_rows["statevector_simulator"]["supports_parameter_shift"] is True

    backend_rows = {str(row["backend"]): row for row in report.backend_matrix}
    assert backend_rows["hardware_qpu"]["fail_closed"] is True
    assert backend_rows["finite_shot_simulator"]["method"] == "stochastic_parameter_shift"
    report_payload = report.to_dict()
    dependency_payload = cast(list[dict[str, object]], report_payload["dependency_matrix"])
    assert dependency_payload[0]["framework"] == "jax"


def test_unified_differentiable_compile_report_filters_registered_primitives() -> None:
    report = differentiable_compile_report(
        primitive_identities=("scpn.program_ad.array:getitem@1",)
    )
    default_dispatch = differentiable_api("compile_report")
    explicit_dispatch = differentiable_api(
        "compile_report",
        primitive_identities=("scpn.program_ad.array:getitem@1",),
        method="jvp_vjp_adjoint",
    )

    assert report.operation == "compile_report"
    assert report.supported
    assert report.payload["primitive_count"] == 1
    assert report.payload["primitive_identities"] == ["scpn.program_ad.array:getitem@1"]
    assert "scpn_diff.primitive" in str(report.payload["mlir"])
    assert int(default_dispatch.payload["primitive_count"]) > 1
    assert explicit_dispatch.to_dict() == report.to_dict()

    with pytest.raises(ValueError, match="unknown primitive identities"):
        differentiable_compile_report(primitive_identities=("missing:primitive@1",))
    with pytest.raises(ValueError, match="must be non-empty"):
        differentiable_compile_report(primitive_identities=())


def test_unified_differentiable_transform_algebra_report_is_bounded() -> None:
    report = differentiable_transform_algebra_report()
    dispatched = differentiable_api("transform_algebra_report")

    assert report.operation == "transform_algebra_report"
    assert report.supported is True
    assert report.method == "differentiable_transform_algebra"
    assert report.payload["missing_categories"] == []
    categories = cast(list[str], report.payload["categories"])
    blocked_count = cast(int, report.payload["blocked_count"])
    support_rows = cast(list[dict[str, object]], report.payload["support_matrix"])
    support_row_ids = {str(row["row_id"]) for row in support_rows}
    assert set(categories) == set(REQUIRED_TRANSFORM_ALGEBRA_CATEGORIES)
    assert "native_grad_vmap" in support_row_ids
    assert "registered_custom_rules" in support_row_ids
    assert "program_ad_jvp_vjp" in support_row_ids
    assert "quantum_gradient_native_nesting" in support_row_ids
    assert "unsupported_structured_container" in support_row_ids
    assert blocked_count >= 4
    assert "finite differences remain diagnostic" in report.claim_boundary
    assert dispatched.to_dict() == report.to_dict()


def test_unified_differentiable_qfi_fss_report_is_bounded() -> None:
    report = differentiable_qfi_fss_report(
        system_sizes=[2, 3],
        k_range=np.linspace(0.5, 3.0, 6),
    )
    dispatched = differentiable_api(
        "qfi_fss_report",
        system_sizes=[2, 3],
        k_range=np.linspace(0.5, 3.0, 6),
    )

    assert report.operation == "qfi_fss_report"
    assert report.supported is True
    assert report.method == "qfi_finite_size_scaling"
    assert report.payload["system_sizes"] == [2, 3]
    assert report.payload["bkt_fit"]["model"] == "bkt_log_correction"
    assert report.payload["power_fit"]["model"] == "power_law_nu_1"
    assert "no hardware" in report.claim_boundary
    assert dispatched.to_dict() == report.to_dict()
    assert scpn.differentiable_qfi_fss_report is differentiable_qfi_fss_report


def test_unified_differentiable_qfi_fss_report_uses_default_fss_sizes() -> None:
    report = differentiable_qfi_fss_report(k_range=np.linspace(0.5, 3.0, 6))

    assert report.payload["system_sizes"] == list(DEFAULT_FSS_SYSTEM_SIZES)


def test_unified_differentiable_benchmark_report_is_non_performance_evidence() -> None:
    _require_torch_backend()

    report = differentiable_api("benchmark_report")

    assert report.operation == "benchmark_report"
    assert report.supported
    assert report.payload["program_ad_case_count"] > 0
    assert report.payload["quantum_gradient_case_count"] > 0
    assert report.payload["support_audit_passed"] is True
    assert "not isolated performance" in report.claim_boundary
    assert scpn.differentiable_benchmark_report is differentiable_benchmark_report


def test_unified_differentiable_dispatcher_and_root_exports() -> None:
    values = np.array([2.0, -1.0], dtype=float)
    calls = {"count": 0}

    def frontend_objective(inputs: FloatArray) -> object:
        calls["count"] += 1
        return np.sin(inputs[0]) + inputs[1]

    gradient = differentiable_api(
        "gradient",
        objective=_scalar_objective,
        values=values,
        method="finite_difference",
    )
    support = differentiable_api(
        "support_report",
        gate="ry",
        observable="pauli_expectation",
        n_params=2,
    )
    diagnostic = differentiable_api(
        "diagnostic_report",
        gate="unregistered_gate",
        observable="pauli_expectation",
    )
    dashboard = differentiable_api("dashboard_status")
    frontend_direct = differentiable_frontend_report(frontend_objective)
    frontend_dispatched = differentiable_api("frontend_report", objective=frontend_objective)

    assert gradient.gradient is not None
    np.testing.assert_allclose(gradient.gradient, np.array([4.0, 3.0]), atol=1e-5)
    assert calls == {"count": 0}
    assert support.supported
    assert diagnostic.fail_closed
    assert diagnostic.payload["blocked_reasons"]
    assert dashboard.supported
    assert dashboard.payload["status_api_ready"] is True
    assert dashboard.payload["rows"]
    assert frontend_direct.operation == "frontend_report"
    assert frontend_direct.supported is True
    assert frontend_direct.payload["frontend_ready"] is True
    assert frontend_direct.payload["bytecode_basic_block_count"] > 0
    assert frontend_direct.payload["source_region_count"] > 0
    assert frontend_direct.payload["source_bytecode_line_map_count"] > 0
    assert frontend_direct.payload["symbol_scope_entry_count"] > 0
    assert frontend_direct.payload["unsupported_semantic_diagnostic_count"] == 0
    assert frontend_direct.payload["unsupported_semantic_diagnostics"] == []
    assert int(frontend_direct.payload["source_start_line"]) > 0
    assert int(frontend_direct.payload["source_end_line"]) >= int(
        frontend_direct.payload["source_start_line"]
    )
    first_line_map = frontend_direct.payload["source_bytecode_line_map"][0]
    assert int(first_line_map["line_number"]) > 0
    assert first_line_map["absolute_line_number"] is None or int(
        first_line_map["absolute_line_number"]
    ) >= int(frontend_direct.payload["source_start_line"])
    assert first_line_map["region_ids"]
    assert len(str(frontend_direct.payload["frontend_digest"])) == 64
    assert frontend_dispatched.supported is True
    assert frontend_dispatched.payload["bytecode_instruction_count"] > 0
    assert scpn.DifferentiableDashboardStatus is DifferentiableDashboardStatus
    assert scpn.DifferentiableDashboardCapabilityRow is DifferentiableDashboardCapabilityRow
    assert scpn.DifferentiableDashboardCapabilityState is DifferentiableDashboardCapabilityState
    assert scpn.differentiable_dashboard_status is differentiable_dashboard_status
    assert scpn.differentiable_frontend_report is differentiable_frontend_report
    assert scpn.differentiable_transform_algebra_report is differentiable_transform_algebra_report
    assert scpn.explain_differentiability is explain_differentiability
    assert scpn.differentiable_api is differentiable_api
    assert scpn.differentiable_gradient is differentiable_gradient
    assert scpn.differentiable_value is differentiable_value
    assert scpn.TransformAlgebraAudit is TransformAlgebraAudit
    assert scpn.TransformAlgebraCase is TransformAlgebraCase
    assert scpn.run_transform_algebra_audit is run_transform_algebra_audit
    assert scpn.assert_transform_algebra_audit_passes is assert_transform_algebra_audit_passes

    with pytest.raises(ValueError, match="objective is required"):
        differentiable_api("gradient", values=values)
    with pytest.raises(ValueError, match="values are required"):
        differentiable_api("gradient", objective=_scalar_objective)
    with pytest.raises(ValueError, match="unsupported unified differentiable operation"):
        differentiable_api(cast(UnifiedDifferentiableOperation, "unsupported"))
