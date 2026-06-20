# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD linalg registry contract tests
"""Tests for Program AD linear-algebra primitive registry contracts."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    PrimitiveIdentity,
    primitive_complete_contract_for,
    primitive_contract_for,
    primitive_static_argument_rule_for,
    whole_program_value_and_grad,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_linalg_primitives_are_registry_policy_gated() -> None:
    """Supported Program AD linalg primitives should expose registry contracts."""

    for name in (
        "det",
        "inv",
        "solve",
        "trace",
        "diag",
        "diagflat",
        "matrix_power",
        "multi_dot",
        "eig",
    ):
        identity = PrimitiveIdentity("scpn.program_ad.linalg", name, "1")
        contract = primitive_contract_for(identity)
        assert contract.identity == identity
        assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
        assert contract.effect == "pure"
        assert contract.lowering_metadata["program_ad"] == "operator_intercepted_trace"
        assert (
            contract.lowering_metadata["mlir"]
            == "available: scpn_diff linalg dialect interchange; executable lowering blocked"
        )
        assert contract.lowering_metadata["mlir_op"] == f"scpn_diff.linalg.{name}"
        assert contract.batching_rule is not None
        assert contract.dtype_rule is not None
        assert contract.shape_rule is not None
        with pytest.raises(ValueError, match="incomplete primitive contract"):
            primitive_complete_contract_for(identity)

    expected_default_factories = {
        "det": "singular_matrix_rank_drop",
        "inv": "singular_matrix_inverse",
    }
    for name, boundary in expected_default_factories.items():
        metadata = primitive_contract_for(f"scpn.program_ad.linalg:{name}").lowering_metadata
        assert metadata["static_derivative_factory"] == "not_required"
        assert metadata["static_signature"] == "none"
        assert metadata["nondifferentiable_boundary"] == boundary
        assert metadata["nondifferentiable_boundary_policy"] == "fail_closed"

    values = np.array([2.0, -0.5, 0.75, 1.5], dtype=np.float64)

    def determinant_objective(flat_values: Any) -> object:
        return np.linalg.det(np.reshape(flat_values, (2, 2)))

    result = whole_program_value_and_grad(determinant_objective, values)
    _assert_allclose(
        result.gradient,
        np.array([1.5, -0.75, 0.5, 2.0], dtype=np.float64),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_program_ad_linalg_nondifferentiable_boundary_metadata_is_explicit() -> None:
    """Linalg contracts should expose singular-boundary fail-closed semantics."""

    expected_boundaries = {
        "det": "singular_matrix_rank_drop",
        "inv": "singular_matrix_inverse",
        "solve": "singular_or_incompatible_linear_system",
        "matrix_power": "negative_power_singular_matrix",
        "multi_dot": "static_shape_alignment",
        "eig": "real_simple_diagonalizable_eigensystem",
    }
    for name, boundary in expected_boundaries.items():
        metadata = primitive_contract_for(f"scpn.program_ad.linalg:{name}").lowering_metadata
        assert metadata["nondifferentiable_boundary"] == boundary
        assert metadata["nondifferentiable_boundary_policy"] == "fail_closed"


def test_program_ad_linalg_primitive_shape_dtype_rules_are_specialized() -> None:
    """Supported linalg primitive contracts should expose concrete shape and dtype rules."""

    matrix = np.array([[2.0, -0.5], [0.75, 1.5]], dtype=np.float64)
    rhs_vector = np.array([1.0, -0.25], dtype=np.float64)
    rhs_matrix = np.array([[1.0, -0.25], [0.5, 1.5]], dtype=np.float64)
    right = np.array([[0.25, 1.0, -0.5], [1.5, -0.25, 0.75]], dtype=np.float64)

    contracts = {
        name: primitive_contract_for(PrimitiveIdentity("scpn.program_ad.linalg", name, "1"))
        for name in ("det", "inv", "solve", "matrix_power", "multi_dot")
    }
    assert contracts["det"].shape_rule is not None
    assert contracts["inv"].shape_rule is not None
    assert contracts["solve"].shape_rule is not None
    assert contracts["matrix_power"].shape_rule is not None
    assert contracts["multi_dot"].shape_rule is not None
    assert contracts["det"].dtype_rule is not None
    assert contracts["inv"].dtype_rule is not None

    assert cast(Any, contracts["det"].shape_rule)((matrix,)) == ()
    assert cast(Any, contracts["inv"].shape_rule)((matrix,)) == (2, 2)
    assert cast(Any, contracts["solve"].shape_rule)((matrix, rhs_vector)) == (2,)
    assert cast(Any, contracts["solve"].shape_rule)((matrix, rhs_matrix)) == (2, 2)
    assert cast(Any, contracts["matrix_power"].shape_rule)((matrix, 3)) == (2, 2)
    assert cast(Any, contracts["multi_dot"].shape_rule)(((rhs_vector, matrix, right),)) == (3,)
    assert cast(Any, contracts["det"].dtype_rule)((matrix,)) == "float64"
    assert cast(Any, contracts["inv"].dtype_rule)((matrix,)) == "float64"

    with pytest.raises(ValueError, match="requires a square matrix"):
        cast(Any, contracts["det"].shape_rule)((np.reshape(np.arange(6.0), (2, 3)),))
    with pytest.raises(ValueError, match="vector length must match matrix"):
        cast(Any, contracts["solve"].shape_rule)((matrix, np.arange(3.0)))
    with pytest.raises(ValueError, match="static integer power"):
        cast(Any, contracts["matrix_power"].shape_rule)((matrix, 1.5))
    with pytest.raises(ValueError, match="dimensions must align"):
        cast(Any, contracts["multi_dot"].shape_rule)(
            ((rhs_vector, np.ones((3, 2), dtype=np.float64)),)
        )


def test_program_ad_linalg_static_argument_rules_are_specialized() -> None:
    """Supported linalg primitive contracts should expose static-argument policies."""

    matrix = np.array([[2.0, -0.5], [0.75, 1.5]], dtype=np.float64)
    vector = np.array([1.0, -0.25], dtype=np.float64)
    right = np.array([[0.25, 1.0, -0.5], [1.5, -0.25, 0.75]], dtype=np.float64)

    contracts = {
        name: primitive_contract_for(PrimitiveIdentity("scpn.program_ad.linalg", name, "1"))
        for name in ("matrix_power", "multi_dot")
    }
    det_static = primitive_static_argument_rule_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "det", "1")
    )
    inv_static = primitive_static_argument_rule_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "inv", "1")
    )
    matrix_power_static = primitive_static_argument_rule_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "matrix_power", "1")
    )
    multi_dot_static = primitive_static_argument_rule_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "multi_dot", "1")
    )

    assert det_static((matrix,)) == ()
    assert inv_static((matrix,)) == ()
    assert matrix_power_static((matrix, np.int64(-2))) == (-2,)
    assert multi_dot_static(((vector, matrix, right),)) == ((2,), (2, 2), (2, 3))
    assert (
        primitive_contract_for(
            PrimitiveIdentity("scpn.program_ad.linalg", "matrix_power", "1")
        ).static_argument_rule
        is matrix_power_static
    )
    assert callable(matrix_power_static)
    assert (
        contracts["matrix_power"].lowering_metadata["static_derivative_factory"]
        == "program_ad_linalg_matrix_power_derivative_rule"
    )
    assert contracts["matrix_power"].lowering_metadata["static_signature"] == "power:i64"
    assert (
        contracts["multi_dot"].lowering_metadata["static_derivative_factory"]
        == "program_ad_linalg_multi_dot_derivative_rule"
    )
    assert (
        contracts["multi_dot"].lowering_metadata["static_signature"]
        == "operand_shapes:ranked_tensor_shape_sequence"
    )

    with pytest.raises(ValueError, match="integer power"):
        matrix_power_static((matrix, 1.5))
    with pytest.raises(ValueError, match="static operand sequence"):
        multi_dot_static((np.reshape(np.arange(4.0), (2, 2)),))
