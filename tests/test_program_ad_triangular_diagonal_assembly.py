# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD triangular diagonal assembly tests
# scpn-quantum-control -- Program AD triangular and diagonal assembly tests
"""Tests for Program AD triangular, diagonal, and diagflat primitive semantics."""

from __future__ import annotations

from typing import Any, Callable, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    CustomDerivativeRule,
    Parameter,
    PrimitiveIdentity,
    primitive_complete_contract_for,
    primitive_contract_for,
    program_ad_assembly_diagonal_derivative_rule,
    program_ad_assembly_tril_derivative_rule,
    program_ad_assembly_triu_derivative_rule,
    program_ad_linalg_diag_derivative_rule,
    program_ad_linalg_diagflat_derivative_rule,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_assembly_primitives import (
    program_ad_assembly_diagonal_derivative_rule as extracted_diagonal_derivative_rule,
)
from scpn_quantum_control.program_ad_assembly_primitives import (
    program_ad_assembly_tril_derivative_rule as extracted_tril_derivative_rule,
)
from scpn_quantum_control.program_ad_assembly_primitives import (
    program_ad_assembly_triu_derivative_rule as extracted_triu_derivative_rule,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
TriangularFactory = Callable[..., CustomDerivativeRule]
TriangularNumpyFn = Callable[..., FloatArray]


def test_program_ad_triangular_diagonal_facade_uses_extracted_factories() -> None:
    """The compatibility facade should expose the extracted assembly factories."""

    assert program_ad_assembly_diagonal_derivative_rule is extracted_diagonal_derivative_rule
    assert program_ad_assembly_tril_derivative_rule is extracted_tril_derivative_rule
    assert program_ad_assembly_triu_derivative_rule is extracted_triu_derivative_rule


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_triangular_masks_preserve_zeroed_adjoint() -> None:
    """Program AD tril/triu should pass adjoints only through unmasked entries."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values[:6], (2, 3))
        cube = np.reshape(values, (2, 2, 3))
        lower_matrix = np.tril(matrix)
        upper_matrix = np.triu(matrix, k=1)
        lower_cube = np.tril(cube, k=-1)
        upper_cube = np.triu(cube)
        return (
            np.sum(lower_matrix * np.array([[1.0, -2.0, 3.0], [0.5, 1.5, -1.0]]))
            + np.sum(upper_matrix * np.array([[-0.25, 0.75, -1.25], [2.0, -0.5, 1.0]]))
            + np.sum(
                lower_cube
                * np.array(
                    [[[0.1, -0.2, 0.3], [2.0, -0.4, 0.5]], [[0.6, -0.7, 0.8], [-1.5, 0.9, -1.0]]]
                )
            )
            + np.sum(
                upper_cube
                * np.array(
                    [[[0.2, -0.4, 0.6], [1.0, -0.8, 0.5]], [[-0.3, 0.7, -0.9], [0.4, 1.1, -1.2]]]
                )
            )
        )

    values = np.linspace(-1.1, 1.1, 12, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_gradient = np.array(
        [1.2, 0.35, -0.65, 2.5, 0.7, 1.5, -0.3, 0.7, -0.9, -1.5, 1.1, -1.2],
        dtype=np.float64,
    )
    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_triangular_masks_reject_invalid_contracts() -> None:
    """Program AD tril/triu should fail closed on invalid rank and k contracts."""

    with pytest.raises(ValueError, match="rank >= 2"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.tril(values)),
            np.arange(1.0, 7.0, dtype=np.float64),
        )

    with pytest.raises(ValueError, match="static integer k"):
        whole_program_value_and_grad(
            lambda values: np.sum(cast(Any, np.triu)(np.reshape(values, (2, 3)), 0.5)),
            np.arange(1.0, 7.0, dtype=np.float64),
        )


def test_program_ad_diagonal_preserves_offset_axis_adjoint() -> None:
    """Program AD np.diagonal should gather exact offset/axis adjoints."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values[:9], (3, 3))
        tensor = np.reshape(values, (2, 2, 3))
        main = np.diagonal(matrix)
        upper = np.diagonal(matrix, offset=1)
        batched = np.diagonal(tensor, offset=1, axis1=1, axis2=2)
        return (
            np.sum(main * np.array([1.0, -2.0, 3.0]))
            + np.sum(upper * np.array([0.5, -1.5]))
            + np.sum(batched * np.array([[0.25, -0.75], [1.25, -2.0]]))
        )

    values = np.linspace(-1.0, 1.0, 12, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_gradient = np.array(
        [1.0, 0.75, 0.0, 0.0, -2.0, -2.25, 0.0, 1.25, 3.0, 0.0, 0.0, -2.0],
        dtype=np.float64,
    )
    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_diagonal_rejects_invalid_contracts() -> None:
    """Program AD np.diagonal should fail closed on invalid static contracts."""

    with pytest.raises(ValueError, match="rank >= 2"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.diagonal(values)),
            np.arange(1.0, 7.0, dtype=np.float64),
        )

    with pytest.raises(ValueError, match="static integer offset"):
        whole_program_value_and_grad(
            lambda values: np.sum(cast(Any, np.diagonal)(np.reshape(values, (2, 3)), offset=0.5)),
            np.arange(1.0, 7.0, dtype=np.float64),
        )


def test_program_ad_diagflat_flattens_source_into_offset_diagonal_adjoint() -> None:
    """Program AD np.diagflat should flatten sources before diagonal scatter adjoints."""

    lower_weights = np.zeros((7, 7), dtype=np.float64)
    lower_weights[1, 0] = 0.5
    lower_weights[2, 1] = -1.0
    lower_weights[3, 2] = 1.5
    lower_weights[4, 3] = 2.0
    lower_weights[5, 4] = -0.25
    lower_weights[6, 5] = 0.75
    upper_weights = np.zeros((5, 5), dtype=np.float64)
    upper_weights[0, 2] = -2.0
    upper_weights[1, 3] = 0.25
    upper_weights[2, 4] = 1.25

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        lower = np.diagflat(matrix, k=-1)
        upper = np.diagflat(values[:3], k=2)
        return np.sum(lower * lower_weights) + np.sum(upper * upper_weights)

    values = np.arange(1.0, 7.0, dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_gradient = np.array([-1.5, -0.75, 2.75, 2.0, -0.25, 0.75], dtype=np.float64)
    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_diagflat_rejects_invalid_static_contracts() -> None:
    """Program AD np.diagflat should fail closed on invalid static offset contracts."""

    with pytest.raises(ValueError, match="integer offset"):
        whole_program_value_and_grad(
            lambda values: np.sum(cast(Any, np.diagflat)(values, k=0.5)),
            np.arange(1.0, 7.0, dtype=np.float64),
        )


def test_program_ad_linalg_diagonal_family_uses_primitive_adjoint_replay_ir() -> None:
    """Program AD diagonal-family linalg calls should emit compact primitive replay IR."""

    vector_diag_weights = np.zeros((4, 4), dtype=np.float64)
    vector_diag_weights[1, 0] = 0.25
    vector_diag_weights[2, 1] = -0.75
    vector_diag_weights[3, 2] = 1.5
    diagflat_weights = np.zeros((5, 5), dtype=np.float64)
    diagflat_weights[0, 1] = -1.0
    diagflat_weights[1, 2] = 0.5
    diagflat_weights[2, 3] = 2.0
    diagflat_weights[3, 4] = -0.25

    def objective(values: Any) -> object:
        matrix = np.reshape(values[:6], (2, 3))
        vector = values[6:9]
        source = np.reshape(values[9:], (2, 2))
        return (
            1.75 * np.trace(matrix, offset=1)
            + 0.5 * np.sum(np.diag(matrix, k=-1))
            + np.sum(np.diag(vector, k=-1) * vector_diag_weights)
            + np.sum(np.diagflat(source, k=1) * diagflat_weights)
        )

    values = np.array(
        [2.0, -0.5, 1.0, 0.75, 1.5, -2.0, -1.25, 0.5, 2.25, 1.0, -0.75, 1.5, 0.25],
        dtype=np.float64,
    )
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"d{index}") for index in range(values.size)),
    )

    expected = np.zeros_like(values)
    expected[1] = 1.75
    expected[5] = 1.75
    expected[3] = 0.5
    expected[6:9] = np.array([0.25, -0.75, 1.5], dtype=np.float64)
    expected[9:] = np.array([-1.0, 0.5, 2.0, -0.25], dtype=np.float64)

    assert [node.op for node in result.ir_nodes if node.op.startswith("linalg:trace:")] == [
        "linalg:trace:2x3:offset:1"
    ]
    assert [node.op for node in result.ir_nodes if node.op.startswith("linalg:diag:")] == [
        "linalg:diag:2x3:offset:-1:extract:0",
        "linalg:diag:3:offset:-1:construct:0",
        "linalg:diag:3:offset:-1:construct:1",
        "linalg:diag:3:offset:-1:construct:2",
    ]
    assert [node.op for node in result.ir_nodes if node.op.startswith("linalg:diagflat:")] == [
        "linalg:diagflat:2x2:offset:1:construct:0",
        "linalg:diagflat:2x2:offset:1:construct:1",
        "linalg:diagflat:2x2:offset:1:construct:2",
        "linalg:diagflat:2x2:offset:1:construct:3",
    ]
    assert result.adjoint_result is not None
    assert result.adjoint_result.supported is True
    assert result.adjoint_result.unsupported_ops == ()
    _assert_allclose(result.gradient, expected, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(program_adjoint_gradient(result), expected, rtol=1.0e-12, atol=1.0e-12)


def _assert_triangular_contract_and_rule(
    name: str,
    k: int,
    factory: TriangularFactory,
    numpy_fn: TriangularNumpyFn,
) -> None:
    """Assert triangular registry metadata and direct derivative rules."""

    matrix = np.array([[1.0, -2.0, 0.5], [3.0, -1.5, 2.0]], dtype=np.float64)
    tangent = np.array([[0.25, -0.5, 0.75], [-1.0, 1.25, -1.5]], dtype=np.float64)
    cotangent = np.array([[0.2, -0.4, 0.6], [-0.8, 1.0, -1.2]], dtype=np.float64)

    contract = primitive_contract_for(f"scpn.program_ad.assembly:{name}")
    assert contract.identity == PrimitiveIdentity("scpn.program_ad.assembly", name, "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["mlir_op"] == f"scpn_diff.assembly.{name}"
    assert (
        contract.lowering_metadata["static_derivative_factory"]
        == f"program_ad_assembly_{name}_derivative_rule"
    )
    assert contract.lowering_metadata["static_signature"] == "source_shape:rank_ge_2;k"
    assert contract.shape_rule is not None
    assert contract.shape_rule((matrix, k)) == matrix.shape
    assert contract.dtype_rule is not None
    assert contract.dtype_rule((matrix, k)) == "float64"
    assert contract.static_argument_rule is not None
    assert contract.static_argument_rule((matrix, k)) == (matrix.shape, k)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(contract.identity)

    rule = factory(matrix.shape, k=k)
    assert rule.name == f"program_ad_assembly_{name}_k{k}_direct_rule"
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    jvp_rule = rule.jvp_rule
    vjp_rule = rule.vjp_rule
    _assert_allclose(rule.value_fn(matrix.reshape(-1)), numpy_fn(matrix, k=k).reshape(-1))
    _assert_allclose(
        jvp_rule(matrix.reshape(-1), tangent.reshape(-1)),
        numpy_fn(tangent, k=k).reshape(-1),
    )
    _assert_allclose(
        vjp_rule(matrix.reshape(-1), cotangent.reshape(-1)),
        numpy_fn(cotangent, k=k).reshape(-1),
    )


def test_program_ad_assembly_triangular_contracts_and_direct_rules() -> None:
    """np.tril and np.triu should expose exact static triangular mask contracts."""

    _assert_triangular_contract_and_rule(
        "tril",
        -1,
        cast(TriangularFactory, program_ad_assembly_tril_derivative_rule),
        cast(TriangularNumpyFn, np.tril),
    )
    _assert_triangular_contract_and_rule(
        "triu",
        1,
        cast(TriangularFactory, program_ad_assembly_triu_derivative_rule),
        cast(TriangularNumpyFn, np.triu),
    )


def test_program_ad_assembly_diagonal_contract_and_direct_rule() -> None:
    """np.diagonal should expose exact static gather and scatter-add adjoints."""

    matrix = np.arange(12, dtype=np.float64).reshape(3, 4) - 2.0
    tangent = np.linspace(-0.5, 0.6, num=12, dtype=np.float64).reshape(3, 4)
    cotangent = np.array([0.25, -0.75, 1.5], dtype=np.float64)
    source_indices: IntArray = np.arange(matrix.size, dtype=np.int64).reshape(matrix.shape)
    selected_indices = np.diagonal(source_indices, offset=1, axis1=0, axis2=1).reshape(-1)
    expected_adjoint = np.zeros(matrix.size, dtype=np.float64)
    np.add.at(expected_adjoint, selected_indices, cotangent)

    contract = primitive_contract_for("scpn.program_ad.assembly:diagonal")
    assert contract.identity == PrimitiveIdentity("scpn.program_ad.assembly", "diagonal", "1")
    assert contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert contract.effect == "pure"
    assert contract.lowering_metadata["mlir_op"] == "scpn_diff.assembly.diagonal"
    assert (
        contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_assembly_diagonal_derivative_rule"
    )
    assert (
        contract.lowering_metadata["static_signature"]
        == "source_shape:rank_ge_2;offset_axis_pair;output_shape"
    )
    assert contract.shape_rule is not None
    assert contract.shape_rule((matrix, 1, 0, 1)) == (3,)
    assert contract.dtype_rule is not None
    assert contract.dtype_rule((matrix, 1, 0, 1)) == "float64"
    assert contract.static_argument_rule is not None
    assert contract.static_argument_rule((matrix, 1, 0, 1)) == (matrix.shape, 1, 0, 1, (3,))
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(contract.identity)

    rule = program_ad_assembly_diagonal_derivative_rule(matrix.shape, offset=1, axis1=0, axis2=1)
    assert rule.name == "program_ad_assembly_diagonal_offset1_axis0_1_direct_rule"
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    jvp_rule = rule.jvp_rule
    vjp_rule = rule.vjp_rule
    _assert_allclose(
        rule.value_fn(matrix.reshape(-1)),
        np.diagonal(matrix, offset=1, axis1=0, axis2=1).reshape(-1),
    )
    _assert_allclose(
        jvp_rule(matrix.reshape(-1), tangent.reshape(-1)),
        np.diagonal(tangent, offset=1, axis1=0, axis2=1).reshape(-1),
    )
    _assert_allclose(vjp_rule(matrix.reshape(-1), cotangent), expected_adjoint)


def test_program_ad_linalg_diag_and_diagflat_direct_rules() -> None:
    """np.diag and np.diagflat should expose exact static linalg direct rules."""

    vector = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    vector_tangent = np.array([0.25, -0.5, 0.75], dtype=np.float64)
    vector_diag_rule = program_ad_linalg_diag_derivative_rule((3,), k=-1)
    assert vector_diag_rule.name == "program_ad_linalg_diag_3_offset_-1_direct_rule"
    assert vector_diag_rule.jvp_rule is not None
    assert vector_diag_rule.vjp_rule is not None
    vector_diag_jvp = vector_diag_rule.jvp_rule
    vector_diag_vjp = vector_diag_rule.vjp_rule
    _assert_allclose(
        vector_diag_rule.value_fn(vector),
        np.diag(vector, k=-1).reshape(-1),
    )
    _assert_allclose(
        vector_diag_jvp(vector, vector_tangent),
        np.diag(vector_tangent, k=-1).reshape(-1),
    )

    vector_diag_cotangent = np.arange(16, dtype=np.float64).reshape(4, 4)
    _assert_allclose(
        vector_diag_vjp(vector, vector_diag_cotangent.reshape(-1)),
        np.diag(vector_diag_cotangent, k=-1),
    )

    matrix = np.array([[1.0, -2.0, 0.5], [3.0, -1.5, 2.0]], dtype=np.float64)
    matrix_tangent = np.array([[0.25, -0.5, 0.75], [-1.0, 1.25, -1.5]], dtype=np.float64)
    matrix_diag_rule = program_ad_linalg_diag_derivative_rule((2, 3), k=1)
    assert matrix_diag_rule.name == "program_ad_linalg_diag_2x3_offset_1_direct_rule"
    assert matrix_diag_rule.jvp_rule is not None
    assert matrix_diag_rule.vjp_rule is not None
    matrix_diag_jvp = matrix_diag_rule.jvp_rule
    matrix_diag_vjp = matrix_diag_rule.vjp_rule
    _assert_allclose(
        matrix_diag_rule.value_fn(matrix.reshape(-1)),
        np.diag(matrix, k=1),
    )
    _assert_allclose(
        matrix_diag_jvp(matrix.reshape(-1), matrix_tangent.reshape(-1)),
        np.diag(matrix_tangent, k=1),
    )
    expected_matrix_diag_vjp = np.zeros_like(matrix)
    expected_matrix_diag_vjp[0, 1] = 0.2
    expected_matrix_diag_vjp[1, 2] = -0.4
    _assert_allclose(
        matrix_diag_vjp(matrix.reshape(-1), np.array([0.2, -0.4], dtype=np.float64)),
        expected_matrix_diag_vjp.reshape(-1),
    )
    with pytest.raises(ValueError, match="rank-1 or rank-2"):
        program_ad_linalg_diag_derivative_rule((2, 2, 2))

    rectangular = np.array([[1.0, -2.0, 0.5], [3.0, -1.5, 2.0]], dtype=np.float64)
    rectangular_tangent = np.array([[0.25, -0.5, 0.75], [-1.0, 1.25, -1.5]], dtype=np.float64)
    diagflat_rule = program_ad_linalg_diagflat_derivative_rule((2, 3), k=-1)
    assert diagflat_rule.name == "program_ad_linalg_diagflat_2x3_offset_-1_direct_rule"
    assert diagflat_rule.jvp_rule is not None
    assert diagflat_rule.vjp_rule is not None
    diagflat_jvp = diagflat_rule.jvp_rule
    diagflat_vjp = diagflat_rule.vjp_rule
    rectangular_flat_diag = np.diagflat(rectangular, k=-1)
    _assert_allclose(
        diagflat_rule.value_fn(rectangular.reshape(-1)),
        rectangular_flat_diag.reshape(-1),
    )
    _assert_allclose(
        diagflat_jvp(rectangular.reshape(-1), rectangular_tangent.reshape(-1)),
        np.diagflat(rectangular_tangent, k=-1).reshape(-1),
    )
    diagflat_cotangent = np.arange(rectangular_flat_diag.size, dtype=np.float64).reshape(
        rectangular_flat_diag.shape
    )
    _assert_allclose(
        diagflat_vjp(rectangular.reshape(-1), diagflat_cotangent.reshape(-1)),
        np.diag(diagflat_cotangent, k=-1).reshape(rectangular.shape).reshape(-1),
    )
    with pytest.raises(ValueError, match="integer offset"):
        program_ad_linalg_diagflat_derivative_rule((2, 3), k=cast(Any, 1.5))


def test_program_ad_assembly_triangular_and_diagonal_batching_rules_map_outer_axes() -> None:
    """Triangular and diagonal batching should map only non-structural outer axes."""

    triangular_contract = primitive_contract_for("scpn.program_ad.assembly:tril")
    diagonal_contract = primitive_contract_for("scpn.program_ad.assembly:diagonal")
    assert triangular_contract.batching_rule is not None
    assert diagonal_contract.batching_rule is not None
    triangular_batching_rule = triangular_contract.batching_rule
    diagonal_batching_rule = diagonal_contract.batching_rule

    def tril_fn(source: FloatArray, k: int) -> FloatArray:
        return cast(FloatArray, np.tril(source, k=k))

    def diagonal_fn(source: FloatArray, offset: int, axis1: int, axis2: int) -> FloatArray:
        return cast(FloatArray, np.diagonal(source, offset=offset, axis1=axis1, axis2=axis2))

    triangular_batch = np.array(
        [
            [[1.0, -2.0, 0.5], [3.0, -1.5, 2.0]],
            [[-0.25, 0.75, -1.25], [1.5, -2.5, 3.5]],
        ],
        dtype=np.float64,
    )
    expected_triangular = np.stack(
        [np.tril(triangular_batch[index], k=-1) for index in range(2)],
        axis=0,
    )
    _assert_allclose(
        triangular_batching_rule(tril_fn, (triangular_batch, -1), (0, None), 0),
        expected_triangular,
    )
    _assert_allclose(
        triangular_batching_rule(tril_fn, (triangular_batch, -1), (0, None), 1),
        np.moveaxis(expected_triangular, 0, 1),
    )
    with pytest.raises(ValueError, match="matrix axes"):
        triangular_batching_rule(tril_fn, (triangular_batch, -1), (1, None), 0)

    diagonal_batch = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
    expected_diagonal = np.stack(
        [np.diagonal(diagonal_batch[index], offset=1, axis1=0, axis2=1) for index in range(2)],
        axis=0,
    )
    _assert_allclose(
        diagonal_batching_rule(
            diagonal_fn,
            (diagonal_batch, 1, 1, 2),
            (0, None, None, None),
            0,
        ),
        expected_diagonal,
    )
    with pytest.raises(ValueError, match="diagonal axes"):
        diagonal_batching_rule(
            diagonal_fn,
            (diagonal_batch, 1, 1, 2),
            (1, None, None, None),
            0,
        )
