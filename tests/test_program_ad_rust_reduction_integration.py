# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Program AD Rust Reduction Integration Tests
"""Integration tests for bounded reduction and source-map Rust replay."""

from __future__ import annotations

import numpy as np
import pytest
from _program_ad_rust_bridge_test_fixtures import (
    _ARRAY_ELEMENTWISE_VECTOR_OBJECTIVE_PROGRAM_AD_IR,
    _ORDER_STATISTIC_TIE_PROGRAM_AD_IR,
    _PRODUCT_SINGLE_ZERO_PROGRAM_AD_IR,
    _STATIC_AXIS_REDUCTION_PROGRAM_AD_IR,
    _STATIC_ORDER_STATISTIC_REDUCTION_PROGRAM_AD_IR,
    _STATIC_PRODUCT_REDUCTION_PROGRAM_AD_IR,
    _STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR,
    _STATIC_TRAPEZOID_FULL_GRID_PROGRAM_AD_IR,
    _STATIC_TRAPEZOID_REDUCTION_PROGRAM_AD_IR,
    _STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR,
    _STD_ZERO_VARIANCE_PROGRAM_AD_IR,
)

from scpn_quantum_control.program_ad_rust_bridge import (
    value_and_grad_program_ad_effect_ir_with_rust,
)


def test_rust_program_ad_value_and_gradient_replays_static_axis_reductions() -> None:
    """Rust Program AD replay should handle static-axis sum/mean adjoints."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 7.0, 11.0],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [
            10.0 + 7.0 / 3.0,
            20.0 + 7.0 / 3.0,
            30.0 + 7.0 / 3.0,
            10.0 + 11.0 / 3.0,
            20.0 + 11.0 / 3.0,
            30.0 + 11.0 / 3.0,
            5.0,
            7.0,
            9.0,
            2.0,
            5.0,
        ],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_AXIS_REDUCTION_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(529.0, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%0[2]",
        "%0[3]",
        "%0[4]",
        "%0[5]",
        "%1[0]",
        "%1[1]",
        "%1[2]",
        "%2[0]",
        "%2[1]",
    )
    assert rust_result.supported_effect_count == 10
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_rejects_reduction_without_axis_metadata() -> None:
    """Shaped reduction replay should require static-axis metadata."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    missing_axis = _STATIC_AXIS_REDUCTION_PROGRAM_AD_IR.replace(
        '"operation": "sum:axis:0"',
        '"operation": "sum"',
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        missing_axis,
        np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 10.0, 20.0, 30.0, 7.0, 11.0],
            dtype=np.float64,
        ),
    )

    assert rust_result.supported is False
    assert any("requires static axis metadata" in reason for reason in rust_result.blocked_reasons)


def test_rust_program_ad_value_and_gradient_replays_static_source_map_indexing() -> None:
    """Rust Program AD replay should scatter static source-map adjoints."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [20.0, 60.0, 40.0, 50.0, 3.0, 1.0, 3.0, -1.5, 4.0, 2.0],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(400.0, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%0[2]",
        "%0[3]",
        "%1[0]",
        "%1[1]",
        "%1[2]",
        "%1[3]",
        "%1[4]",
        "%1[5]",
    )
    assert rust_result.supported_effect_count == 5
    assert (
        rust_result.claim_boundary
        == "bounded_rust_program_ad_ir_elementwise_structural_array_static_source_map_static_reductions_static_signal_primitives_static_interpolation_primitives_static_stencil_primitives_static_cumulative_primitives_value_and_gradient_static_linalg_primitives_dynamic_boundary_fail_closed_audit_executed_branch_view_assignment_and_expression_alias_metadata_only_no_llvm_jit"
    )


def test_rust_program_ad_value_and_gradient_rejects_source_map_without_metadata() -> None:
    """Static source-map replay should require explicit map metadata."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    missing_map = _STATIC_SOURCE_MAP_INDEXING_PROGRAM_AD_IR.replace(
        '"operation": "index_map:s2,s0,s2,c-1.5,s3,s1"',
        '"operation": "index_map"',
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        missing_map,
        np.array(
            [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
            dtype=np.float64,
        ),
    )

    assert rust_result.supported is False
    assert any(
        "requires static source-map metadata" in reason for reason in rust_result.blocked_reasons
    )


def test_rust_program_ad_value_and_gradient_replays_static_product_reductions() -> None:
    """Rust Program AD replay should handle static product reductions."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.5, -1.0, 2.0, 3.0, -0.25, 0.1],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [92.0, 40.0, 42.0, 11.0, 6.4, 13.0, 4.0, 10.0, 18.0, 6.0, 120.0, 720.0],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_PRODUCT_REDUCTION_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(88.0, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%0[2]",
        "%0[3]",
        "%0[4]",
        "%0[5]",
        "%1[0]",
        "%1[1]",
        "%1[2]",
        "%2[0]",
        "%2[1]",
        "%3",
    )
    assert rust_result.supported_effect_count == 14


def test_rust_program_ad_value_and_gradient_replays_single_zero_product() -> None:
    """Rust Program AD product replay should differentiate one zero input."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _PRODUCT_SINGLE_ZERO_PROGRAM_AD_IR,
        np.array([0.0, 2.0, 3.0, 4.0], dtype=np.float64),
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(0.0, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, [24.0, 0.0, 0.0, 0.0], atol=1.0e-12)


def test_rust_program_ad_value_and_gradient_rejects_multi_zero_product() -> None:
    """Rust Program AD product replay should fail closed on multi-zero products."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _PRODUCT_SINGLE_ZERO_PROGRAM_AD_IR,
        np.array([0.0, 2.0, 0.0, 4.0], dtype=np.float64),
    )

    assert rust_result.supported is False
    assert any(
        "prod gradient supports at most one zero input" in reason
        for reason in rust_result.blocked_reasons
    )


def test_rust_program_ad_value_and_gradient_replays_static_variance_std_reductions() -> None:
    """Rust Program AD replay should handle population variance/std reductions."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 0.5, -1.25, 2.0, 1.5, -0.75, 0.25],
        dtype=np.float64,
    )
    sqrt_14 = np.sqrt(14.0)
    sqrt_6 = np.sqrt(6.0)
    expected_gradient = np.array(
        [
            -0.5 - 2.0 / sqrt_14 - 2.0 / 9.0,
            1.875 - 0.5 / sqrt_14 - 5.0 / 36.0,
            -3.0 + 2.5 / sqrt_14 + 1.0 / 36.0,
            0.5 + 0.75 / sqrt_6 - 1.0 / 18.0,
            -1.875 + 1.0 / 9.0,
            3.0 - 0.75 / sqrt_6 + 5.0 / 18.0,
            1.0,
            2.25,
            2.25,
            sqrt_14 / 3.0,
            2.0 * sqrt_6 / 3.0,
            35.0 / 9.0,
        ],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(
        455.0 / 144.0 + 0.5 * (sqrt_14 - sqrt_6),
        abs=1.0e-12,
    )
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%0[2]",
        "%0[3]",
        "%0[4]",
        "%0[5]",
        "%1[0]",
        "%1[1]",
        "%1[2]",
        "%2[0]",
        "%2[1]",
        "%3",
    )
    assert rust_result.supported_effect_count == 14


def test_rust_program_ad_value_and_gradient_replays_corrected_variance_std_reductions() -> None:
    """Rust Program AD replay should support static ddof/correction metadata."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 0.5, -1.0, 2.0, 1.25, -0.75, 0.6],
        dtype=np.float64,
    )
    row_std = np.sqrt(7.0 / 3.0)
    corrected_ir = (
        _STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR.replace(
            '"operation": "var:axis:0"',
            '"operation": "var:axis:0:ddof:1"',
        )
        .replace(
            '"operation": "std:axis:-1"',
            '"operation": "std:axis:-1:correction:1"',
        )
        .replace('"operation": "var"', '"operation": "var:ddof:2"')
    )
    expected_gradient = np.array(
        [
            -1.8 - 5.0 / (6.0 * row_std),
            2.5 - 5.0 / (24.0 * row_std),
            -5.9 + 25.0 / (24.0 * row_std),
            1.175,
            -2.6,
            6.625,
            2.0,
            4.5,
            4.5,
            row_std,
            2.0,
            35.0 / 6.0,
        ],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(corrected_ir, values)

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(7.5 + 1.25 * row_std, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.supported_effect_count == 14


def test_rust_program_ad_value_and_gradient_rejects_degenerate_moment_correction() -> None:
    """Rust Program AD replay should reject non-positive variance denominators."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    invalid_ir = _STATIC_VARIANCE_STD_REDUCTION_PROGRAM_AD_IR.replace(
        '"operation": "std:axis:-1"',
        '"operation": "std:axis:-1:ddof:3"',
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        invalid_ir,
        np.array(
            [1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 0.5, -1.25, 2.0, 1.5, -0.75, 0.25],
            dtype=np.float64,
        ),
    )

    assert rust_result.supported is False
    assert any(
        "correction must be less than reduction group size" in reason
        for reason in rust_result.blocked_reasons
    )


def test_rust_program_ad_value_and_gradient_rejects_zero_variance_std() -> None:
    """Rust Program AD std replay should fail closed at zero variance."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STD_ZERO_VARIANCE_PROGRAM_AD_IR,
        np.array([2.0, 2.0, 2.0], dtype=np.float64),
    )

    assert rust_result.supported is False
    assert any(
        "std gradient requires positive variance" in reason
        for reason in rust_result.blocked_reasons
    )


def test_rust_program_ad_value_and_gradient_replays_order_statistic_reductions() -> None:
    """Rust Program AD replay should route strict order-statistic adjoints."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [
            3.0,
            -2.0,
            0.5,
            1.0,
            -1.5,
            2.0,
            0.7,
            -1.3,
            0.25,
            1.1,
            -0.4,
            0.8,
            -0.6,
            0.9,
            1.2,
            -0.4,
            0.75,
            -1.1,
            0.5,
        ],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [
            2.0625,
            0.825,
            1.175,
            0.4375,
            -2.725,
            0.625,
            3.0,
            -1.5,
            2.0,
            -2.0,
            -1.5,
            3.0,
            -2.0,
            0.75,
            -0.75,
            -0.25,
            2.5,
            -1.625,
            1.625,
        ],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_ORDER_STATISTIC_REDUCTION_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(10.9, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%0[2]",
        "%0[3]",
        "%0[4]",
        "%0[5]",
        "%1[0]",
        "%1[1]",
        "%1[2]",
        "%2[0]",
        "%2[1]",
        "%3",
        "%4",
        "%5",
        "%6[0]",
        "%6[1]",
        "%7[0]",
        "%7[1]",
        "%7[2]",
    )
    assert rust_result.supported_effect_count == 32


def test_rust_program_ad_value_and_gradient_rejects_order_statistic_ties() -> None:
    """Rust Program AD order-statistic replay should fail closed on ties."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _ORDER_STATISTIC_TIE_PROGRAM_AD_IR,
        np.array([2.0, 2.0, 1.0], dtype=np.float64),
    )

    assert rust_result.supported is False
    assert any("strictly ordered values" in reason for reason in rust_result.blocked_reasons)


def test_rust_program_ad_value_and_gradient_replays_static_trapezoid_reductions() -> None:
    """Rust Program AD replay should route static-grid trapezoid adjoints."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 4.0, 0.5, -1.5, 3.0, 2.0, -1.5, 0.5],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [0.3125, 1.125, 0.875, -0.0625, -0.625, -0.5, 2.625, 0.4375, 1.75],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_TRAPEZOID_REDUCTION_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(5.46875, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.parameter_targets == (
        "%0[0]",
        "%0[1]",
        "%0[2]",
        "%0[3]",
        "%0[4]",
        "%0[5]",
        "%1[0]",
        "%1[1]",
        "%2",
    )
    assert rust_result.supported_effect_count == 10


def test_rust_program_ad_value_and_gradient_replays_full_grid_trapezoid_reductions() -> None:
    """Rust Program AD replay should support full-shape static trapezoid grids."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    values = np.array(
        [1.0, 2.0, 4.0, 0.5, -1.5, 3.0, 2.0, -1.5, 0.5],
        dtype=np.float64,
    )
    expected_gradient = np.array(
        [0.3125, 1.125, 0.875, -0.25, -1.0, -0.6875, 2.625, 0.5, 1.75],
        dtype=np.float64,
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _STATIC_TRAPEZOID_FULL_GRID_PROGRAM_AD_IR,
        values,
    )

    assert rust_result.supported is True, rust_result.blocked_reasons
    assert rust_result.value == pytest.approx(5.375, abs=1.0e-12)
    np.testing.assert_allclose(rust_result.gradient, expected_gradient, atol=1.0e-12)
    assert rust_result.supported_effect_count == 10


def test_rust_program_ad_value_and_gradient_rejects_invalid_trapezoid_metadata() -> None:
    """Rust Program AD trapezoid replay should validate static grid metadata."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    invalid_ir = _STATIC_TRAPEZOID_REDUCTION_PROGRAM_AD_IR.replace(
        '"operation": "trapezoid:axis:1:x:0,0.25,1.0"',
        '"operation": "trapezoid:axis:1:x:0,1"',
    )

    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        invalid_ir,
        np.array([1.0, 2.0, 4.0, 0.5, -1.5, 3.0, 2.0, -1.5, 0.5], dtype=np.float64),
    )

    assert rust_result.supported is False
    assert any(
        "x metadata length must match integration axis size" in reason
        for reason in rust_result.blocked_reasons
    )


def test_rust_program_ad_value_and_gradient_rejects_vector_objective() -> None:
    """Rust Program AD replay should fail closed on non-scalar objectives."""
    engine = pytest.importorskip("scpn_quantum_engine")
    assert callable(getattr(engine, "program_ad_effect_ir_interpret_value_and_gradient", None))
    rust_result = value_and_grad_program_ad_effect_ir_with_rust(
        _ARRAY_ELEMENTWISE_VECTOR_OBJECTIVE_PROGRAM_AD_IR,
        np.array([0.2, -0.3], dtype=np.float64),
    )

    assert rust_result.supported is False
    assert rust_result.value is None
    assert rust_result.gradient.size == 0
    assert any("requires a scalar objective" in reason for reason in rust_result.blocked_reasons)
