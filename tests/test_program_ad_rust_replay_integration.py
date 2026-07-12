# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD rust replay integration tests
"""Integration tests for scalar, branch, stencil, gather, and structural Rust replay."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from _program_ad_rust_bridge_test_fixtures import (
    _ARRAY_ELEMENTWISE_BROADCAST_SUM_PROGRAM_AD_IR,
    _STRUCTURAL_ARRAY_PROGRAM_AD_IR,
    _STRUCTURAL_ASSEMBLY_PROGRAM_AD_IR,
)

from scpn_quantum_control.differentiable import (
    Parameter,
    program_adjoint_replay_gradient,
    whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_rust_bridge import (
    value_and_grad_program_ad_effect_ir_with_rust,
)


def test_rust_program_ad_value_and_gradient_replays_executed_branch_trace() -> None:
    """Rust Program AD replay should preserve executed scalar branch semantics."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array([0.4, -0.2], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        x, y = trace_values
        return (x * x if x > y else y * y) + 2.0 * y + np.sin(x)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(Parameter("x"), Parameter("y")),
    )
    assert result.program_ir is not None
    assert result.program_ir.control_regions
    assert result.program_ir.phi_nodes
    assert not result.program_ir.alias_edges
    assert "runtime_branch" in {region.kind for region in result.program_ir.control_regions}, (
        result.program_ir.control_regions
    )
    assert '"kind":"runtime_branch"' in result.program_ir.serialization
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)
    assert rust_result.supported_effect_count == len(result.program_ir.effects)
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_replays_scalar_primitive_family_trace() -> None:
    """Rust Program AD replay should preserve emitted scalar primitive-family traces."""
    pytest.importorskip("scpn_quantum_engine")
    values = np.array([0.4, -0.2, 0.25, 0.1], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        x, y, z, w = trace_values
        return (
            np.sqrt(x + 2.0)
            + np.tanh(y)
            + np.log1p(z)
            + np.expm1(w)
            + np.reciprocal(x + 3.0)
            + np.arcsin(0.2 * y)
            + np.arccos(0.1 * z)
            + abs(w + 1.0)
        )

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=(
            Parameter("x"),
            Parameter("y"),
            Parameter("z"),
            Parameter("w"),
        ),
    )
    assert result.program_ir is not None
    assert not result.program_ir.alias_edges
    assert not result.program_ir.control_regions
    operations = {effect.operation for effect in result.program_ir.effects}
    assert {
        "sqrt",
        "tanh",
        "log1p",
        "expm1",
        "reciprocal",
        "arcsin",
        "arccos",
        "abs",
    } <= operations

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)
    assert rust_result.supported_effect_count == len(result.program_ir.effects)
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_replays_static_stencil_trace() -> None:
    """Rust Program AD replay should preserve compact static ``np.gradient`` adjoints."""
    pytest.importorskip("scpn_quantum_engine")
    values = np.array([1.0, -2.0, 0.5, 3.0, -1.5], dtype=np.float64)
    weights = np.array([0.5, -1.0, 0.25, 2.0, -0.75], dtype=np.float64)
    coordinates = np.array([0.0, 0.5, 1.5, 3.0, 5.0], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        gradient = np.gradient(trace_values, coordinates, edge_order=2)
        return np.sum(gradient * weights)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )

    assert result.program_ir is not None
    assert any(
        effect.operation.startswith("stencil:gradient:shape:5:axis:0:edge:2")
        for effect in result.program_ir.effects
        if effect.operation is not None
    )
    np.testing.assert_allclose(
        program_adjoint_replay_gradient(result),
        result.gradient,
        atol=1.0e-12,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)
    assert rust_result.supported_effect_count == len(result.program_ir.effects)
    assert "static_stencil_primitives" in rust_result.claim_boundary


def test_rust_program_ad_value_and_gradient_replays_static_take_trace() -> None:
    """Rust Program AD replay should preserve Python-emitted static gather adjoints."""
    pytest.importorskip("scpn_quantum_engine")
    values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    weights = np.array([1.5, -2.0, 0.25], dtype=np.float64)

    def objective(trace_values: Any) -> object:
        gathered = np.take(trace_values, [2, 0, 2])
        return np.sum(gathered * weights)

    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"x{index}") for index in range(values.size)),
    )
    assert result.program_ir is not None
    assert any(
        edge.kind == "alias_analysis"
        and edge.source == "assignment_binding"
        and edge.target.startswith("source:")
        for edge in result.program_ir.alias_edges
    )
    assert any(
        edge.kind == "expression_rebinding_alias"
        and edge.source.startswith("expr:")
        and "np.take" in edge.source
        and edge.target.startswith("name:")
        for edge in result.program_ir.alias_edges
    )
    assert any(
        edge.kind == "view_alias" and edge.target.startswith("view:take:")
        for edge in result.program_ir.alias_edges
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(result.program_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(result.value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, result.gradient, atol=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, [-2.0, 0.0, 1.75, 0.0], atol=1.0e-12)
    assert rust_result.supported_effect_count == len(result.program_ir.effects)
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_replays_array_elementwise_broadcast_sum() -> None:
    """Rust Program AD replay should handle shaped elementwise array adjoints."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array([0.2, -0.3, 0.5, 1.25], dtype=np.float64)
    x = values[:3]
    bias = float(values[3])
    expected_value = float(np.sum(np.sin(x) * (x + bias)))
    expected_gradient = np.concatenate(
        (
            np.cos(x) * (x + bias) + np.sin(x),
            np.array([np.sum(np.sin(x))], dtype=np.float64),
        )
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _ARRAY_ELEMENTWISE_BROADCAST_SUM_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(expected_value, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == ("%0[0]", "%0[1]", "%0[2]", "%1")
    assert rust_result.supported_effect_count == 6
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_replays_structural_array_ops() -> None:
    """Rust Program AD replay should handle structural array adjoints."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array([2.0, 5.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float64)
    expected_gradient = np.array(
        [
            15.0,
            20.0,
            2.0 / 6.0,
            5.0 / 6.0,
            2.0 / 6.0,
            5.0 / 6.0,
            2.0 / 6.0,
            5.0 / 6.0,
        ],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STRUCTURAL_ARRAY_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(130.0, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%1[0]",
        "%1[1]",
        "%1[2]",
        "%1[3]",
        "%1[4]",
        "%1[5]",
    )
    assert rust_result.supported_effect_count == 8
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_replays_structural_assembly_ops() -> None:
    """Rust Program AD replay should handle static concatenate/stack adjoints."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [2.0, 5.0, 7.0, 11.0, 1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [11.0, 32.0, 23.0, 44.0, 2.0, 5.0, 7.0, 11.0, 2.0, 7.0, 5.0, 11.0],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STRUCTURAL_ASSEMBLY_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(827.0, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%1[0]",
        "%1[1]",
        "%2[0]",
        "%2[1]",
        "%2[2]",
        "%2[3]",
        "%3[0]",
        "%3[1]",
        "%3[2]",
        "%3[3]",
    )
    assert rust_result.supported_effect_count == 11
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )
