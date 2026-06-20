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
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    PrimitiveIdentity,
    PrimitiveTransformRule,
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


def test_program_ad_linalg_primitive_batching_rules_vectorize_outputs() -> None:
    """Registered linalg batching rules should vectorize pure NumPy primitive calls."""

    det_contract = primitive_contract_for(PrimitiveIdentity("scpn.program_ad.linalg", "det", "1"))
    solve_contract = primitive_contract_for(
        PrimitiveIdentity("scpn.program_ad.linalg", "solve", "1")
    )
    assert det_contract.batching_rule is not None
    assert solve_contract.batching_rule is not None

    matrices = np.array(
        [
            [[2.0, -0.5], [0.75, 1.5]],
            [[1.25, 0.25], [-0.5, 2.0]],
        ],
        dtype=np.float64,
    )
    rhs = np.array([[1.0, -0.25], [0.5, 1.5]], dtype=np.float64)

    det_values = cast(Any, det_contract.batching_rule)(np.linalg.det, (matrices,), (0,), 0)
    solve_values = cast(Any, solve_contract.batching_rule)(
        np.linalg.solve, (matrices, rhs), (0, 0), 0
    )

    _assert_allclose(det_values, np.linalg.det(matrices), rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        solve_values,
        np.stack([np.linalg.solve(matrices[index], rhs[index]) for index in range(2)]),
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def _transform_rule_from_contract(contract: Any) -> PrimitiveTransformRule:
    """Return a transform rule snapshot for restoring registry mutations."""

    return PrimitiveTransformRule(
        identity=contract.identity,
        derivative_rule=contract.derivative_rule,
        batching_rule=contract.batching_rule,
        lowering_rule=contract.lowering_rule,
        lowering_metadata=contract.lowering_metadata,
        shape_rule=contract.shape_rule,
        dtype_rule=contract.dtype_rule,
        static_argument_rule=contract.static_argument_rule,
        nondifferentiable_policy=contract.nondifferentiable_policy,
        effect=contract.effect,
    )


def test_program_ad_linalg_primitives_validate_registry_rules_at_dispatch() -> None:
    """Supported linalg primitives must execute through registry validation rules."""

    originals = {
        name: primitive_contract_for(f"scpn.program_ad.linalg:{name}")
        for name in (
            "det",
            "diag",
            "diagflat",
            "eig",
            "eigh",
            "eigvals",
            "eigvalsh",
            "inv",
            "matrix_power",
            "multi_dot",
            "pinv",
            "solve",
            "svd",
            "trace",
        )
    }
    calls: dict[str, set[str]] = {name: set() for name in originals}

    for name, original in originals.items():
        assert original.shape_rule is not None
        assert original.dtype_rule is not None
        assert original.static_argument_rule is not None

        def shape_rule(args: tuple[object, ...], *, primitive_name: str = name) -> tuple[int, ...]:
            calls[primitive_name].add("shape")
            return cast(tuple[int, ...], cast(Any, originals[primitive_name].shape_rule)(args))

        def dtype_rule(args: tuple[object, ...], *, primitive_name: str = name) -> str:
            calls[primitive_name].add("dtype")
            return str(cast(Any, originals[primitive_name].dtype_rule)(args))

        def static_argument_rule(
            args: tuple[object, ...], *, primitive_name: str = name
        ) -> tuple[object, ...]:
            calls[primitive_name].add("static")
            return cast(
                tuple[object, ...], cast(Any, originals[primitive_name].static_argument_rule)(args)
            )

        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            PrimitiveTransformRule(
                identity=original.identity,
                derivative_rule=original.derivative_rule,
                batching_rule=original.batching_rule,
                lowering_rule=original.lowering_rule,
                lowering_metadata=original.lowering_metadata,
                shape_rule=shape_rule,
                dtype_rule=dtype_rule,
                static_argument_rule=static_argument_rule,
                nondifferentiable_policy=original.nondifferentiable_policy,
                effect=original.effect,
            ),
            overwrite=True,
        )

    def objective(values: Any) -> object:
        raw_matrix = np.reshape(values, (2, 2))
        matrix = 0.5 * (raw_matrix + np.transpose(raw_matrix))
        rhs = values[:2]
        eig_values, _eig_vectors = np.linalg.eig(matrix)
        eigh_values, _eigh_vectors = np.linalg.eigh(matrix)
        return (
            np.linalg.det(matrix)
            + np.sum(np.linalg.inv(matrix))
            + np.sum(np.linalg.solve(matrix, rhs))
            + np.trace(matrix)
            + np.sum(np.diag(matrix))
            + np.sum(np.diagflat(rhs))
            + np.sum(np.linalg.matrix_power(matrix, 2))
            + np.linalg.multi_dot((rhs, matrix, values[2:]))
            + np.sum(eig_values)
            + np.sum(eigh_values)
            + np.sum(np.linalg.eigvals(matrix))
            + np.sum(np.linalg.eigvalsh(matrix))
            + np.sum(np.linalg.svd(matrix, compute_uv=False))
            + np.sum(np.linalg.pinv(matrix, rcond=1.0e-12))
        )

    values = np.array([2.0, 0.25, 0.25, 1.5], dtype=np.float64)
    try:
        result = whole_program_value_and_grad(objective, values)
    finally:
        for original in originals.values():
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
                _transform_rule_from_contract(original), overwrite=True
            )

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    assert calls == {
        "det": {"shape", "dtype", "static"},
        "diag": {"shape", "dtype", "static"},
        "diagflat": {"shape", "dtype", "static"},
        "eig": {"shape", "dtype", "static"},
        "eigh": {"shape", "dtype", "static"},
        "eigvals": {"shape", "dtype", "static"},
        "eigvalsh": {"shape", "dtype", "static"},
        "inv": {"shape", "dtype", "static"},
        "matrix_power": {"shape", "dtype", "static"},
        "multi_dot": {"shape", "dtype", "static"},
        "pinv": {"shape", "dtype", "static"},
        "solve": {"shape", "dtype", "static"},
        "svd": {"shape", "dtype", "static"},
        "trace": {"shape", "dtype", "static"},
    }
