# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD split assembly tests
"""Tests for Program AD split and partition assembly primitive semantics."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    Parameter,
    PrimitiveIdentity,
    primitive_complete_contract_for,
    primitive_contract_for,
    program_ad_assembly_split_derivative_rule,
    program_adjoint_gradient,
    whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_assembly_primitives import (
    program_ad_assembly_split_derivative_rule as extracted_split_derivative_rule,
)

FloatArray = NDArray[np.float64]


def test_program_ad_assembly_split_facade_uses_extracted_factory() -> None:
    """The compatibility facade should expose the extracted split factory."""

    assert program_ad_assembly_split_derivative_rule is extracted_split_derivative_rule


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD result payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def test_program_ad_split_family_preserves_gather_adjoint() -> None:
    """Program AD split-family calls should replay exact gather adjoints."""

    def objective(values: Any) -> object:
        matrix = np.reshape(values, (2, 3))
        cube = np.reshape(values, (1, 2, 3))
        first, second, third = np.split(values, [2, 4])
        top, bottom = np.vsplit(matrix, 2)
        left, middle, right = np.hsplit(matrix, [1, 2])
        depth0, depth1 = np.dsplit(cube, [1])
        uneven0, uneven1, uneven2, uneven3 = np.array_split(values, 4)
        return (
            np.sum(first * np.array([1.0, -2.0]))
            + np.sum(second * np.array([0.5, 3.0]))
            + np.sum(third * np.array([-1.5, 2.5]))
            + np.sum(top * np.array([[0.25, -0.5, 0.75]]))
            + np.sum(bottom * np.array([[-1.0, 1.5, -2.0]]))
            + np.sum(left * np.array([[2.0], [-0.25]]))
            + np.sum(middle * np.array([[-1.25], [0.5]]))
            + np.sum(right * np.array([[1.75], [-0.75]]))
            + np.sum(depth0 * np.array([[[0.4], [-0.6]]]))
            + np.sum(depth1 * np.array([[[0.2, -0.3], [0.8, -0.9]]]))
            + np.sum(uneven0 * np.array([0.05, -0.1]))
            + np.sum(uneven1 * np.array([0.15, -0.2]))
            + np.sum(uneven2 * np.array([0.25]))
            + np.sum(uneven3 * np.array([-0.3]))
        )

    values = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.0], dtype=np.float64)
    result = whole_program_value_and_grad(
        objective,
        values,
        parameters=tuple(Parameter(f"theta_{index}") for index in range(values.size)),
    )

    expected_gradient = np.array([3.7, -3.65, 2.85, 0.95, 1.55, -1.45], dtype=np.float64)
    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(result.gradient, expected_gradient, rtol=1.0e-12, atol=1.0e-12)
    _assert_allclose(
        program_adjoint_gradient(result), expected_gradient, rtol=1.0e-12, atol=1.0e-12
    )


def test_program_ad_split_family_rejects_invalid_static_sections() -> None:
    """Program AD split-family calls should fail closed on invalid split contracts."""

    with pytest.raises(ValueError, match="static split"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.split(values, 4)[0]),
            np.arange(1.0, 7.0, dtype=np.float64),
        )

    with pytest.raises(ValueError, match="static split"):
        whole_program_value_and_grad(
            lambda values: np.sum(np.vsplit(values, 2)[0]),
            np.arange(1.0, 7.0, dtype=np.float64),
        )


def test_program_ad_assembly_split_family_contract_and_direct_rule() -> None:
    """Split-family calls should expose fail-closed assembly primitive direct rules."""

    matrix = np.array([[1.0, -2.0, 0.5], [3.0, -1.5, 2.0]], dtype=np.float64)
    tangent = np.array([[0.25, -0.5, 0.75], [-1.0, 1.25, -1.5]], dtype=np.float64)
    cotangent = np.array([0.2, -0.4, 0.6, -0.8, 1.0, -1.2], dtype=np.float64)
    source_indices = np.arange(matrix.size, dtype=np.int64).reshape(matrix.shape)

    split_contract = primitive_contract_for("scpn.program_ad.assembly:split")
    assert split_contract.identity == PrimitiveIdentity("scpn.program_ad.assembly", "split", "1")
    assert split_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert split_contract.effect == "pure"
    assert split_contract.lowering_metadata["mlir_op"] == "scpn_diff.assembly.split"
    assert (
        split_contract.lowering_metadata["static_derivative_factory"]
        == "program_ad_assembly_split_derivative_rule"
    )
    assert (
        split_contract.lowering_metadata["static_signature"]
        == "source_shape:ranked_tensor_shape;indices_or_sections;axis;part_shapes"
    )
    assert split_contract.shape_rule is not None
    assert split_contract.shape_rule((matrix, [1, 2], 1)) == (6,)
    assert split_contract.dtype_rule is not None
    assert split_contract.dtype_rule((matrix, [1, 2], 1)) == "float64"
    assert split_contract.static_argument_rule is not None
    assert split_contract.static_argument_rule((matrix, [1, 2], 1)) == (
        (2, 3),
        (1, 2),
        1,
        ((2, 1), (2, 1), (2, 1)),
    )
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(split_contract.identity)

    rule = program_ad_assembly_split_derivative_rule(matrix.shape, [1, 2], axis=1)
    assert rule.name == "program_ad_assembly_split_axis1_3_parts_direct_rule"
    assert rule.jvp_rule is not None
    assert rule.vjp_rule is not None
    jvp_rule = rule.jvp_rule
    vjp_rule = rule.vjp_rule
    expected_parts = np.split(matrix, [1, 2], axis=1)
    expected_tangent_parts = np.split(tangent, [1, 2], axis=1)
    expected_indices = np.concatenate(
        [part.reshape(-1) for part in np.split(source_indices, [1, 2], axis=1)]
    )
    expected_adjoint = np.zeros(matrix.size, dtype=np.float64)
    np.add.at(expected_adjoint, expected_indices, cotangent)
    _assert_allclose(
        rule.value_fn(matrix.reshape(-1)),
        np.concatenate([part.reshape(-1) for part in expected_parts]),
    )
    _assert_allclose(
        jvp_rule(matrix.reshape(-1), tangent.reshape(-1)),
        np.concatenate([part.reshape(-1) for part in expected_tangent_parts]),
    )
    _assert_allclose(vjp_rule(matrix.reshape(-1), cotangent), expected_adjoint)

    array_split_contract = primitive_contract_for("scpn.program_ad.assembly:array_split")
    assert array_split_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.assembly", "array_split", "1"
    )
    assert array_split_contract.shape_rule is not None
    assert array_split_contract.shape_rule((matrix.reshape(-1), 4, 0)) == (6,)
    assert array_split_contract.static_argument_rule is not None
    assert array_split_contract.static_argument_rule((matrix.reshape(-1), 4, 0)) == (
        (6,),
        4,
        0,
        ((2,), (2,), (1,), (1,)),
    )
    array_rule = program_ad_assembly_split_derivative_rule((6,), 4, split_name="array_split")
    assert array_rule.name == "program_ad_assembly_array_split_axis0_4_parts_direct_rule"
    assert array_rule.jvp_rule is not None
    assert array_rule.vjp_rule is not None
    array_jvp_rule = array_rule.jvp_rule
    array_vjp_rule = array_rule.vjp_rule
    _assert_allclose(array_rule.value_fn(matrix.reshape(-1)), matrix.reshape(-1))
    _assert_allclose(array_jvp_rule(matrix.reshape(-1), tangent.reshape(-1)), tangent.reshape(-1))
    _assert_allclose(array_vjp_rule(matrix.reshape(-1), cotangent), cotangent)

    hsplit_contract = primitive_contract_for("scpn.program_ad.assembly:hsplit")
    vsplit_contract = primitive_contract_for("scpn.program_ad.assembly:vsplit")
    dsplit_contract = primitive_contract_for("scpn.program_ad.assembly:dsplit")
    for split_name, variant_contract in (
        ("hsplit", hsplit_contract),
        ("vsplit", vsplit_contract),
        ("dsplit", dsplit_contract),
    ):
        assert variant_contract.identity == PrimitiveIdentity(
            "scpn.program_ad.assembly", split_name, "1"
        )
        assert variant_contract.lowering_metadata["mlir_op"] == (
            f"scpn_diff.assembly.{split_name}"
        )
        assert (
            variant_contract.lowering_metadata["static_derivative_factory"]
            == "program_ad_assembly_split_derivative_rule"
        )
        assert variant_contract.lowering_metadata["static_signature"] == (
            "source_shape:ranked_tensor_shape;indices_or_sections;axis;part_shapes"
        )
    assert hsplit_contract.shape_rule is not None
    assert hsplit_contract.shape_rule((matrix, [1, 2], 1)) == (6,)
    assert hsplit_contract.dtype_rule is not None
    assert hsplit_contract.dtype_rule((matrix, [1, 2], 1)) == "float64"
    assert hsplit_contract.static_argument_rule is not None
    assert hsplit_contract.static_argument_rule((matrix, [1, 2], 1)) == (
        (2, 3),
        (1, 2),
        1,
        ((2, 1), (2, 1), (2, 1)),
    )
    assert vsplit_contract.shape_rule is not None
    assert vsplit_contract.shape_rule((matrix, 2, 0)) == (6,)
    assert vsplit_contract.dtype_rule is not None
    assert vsplit_contract.dtype_rule((matrix, 2, 0)) == "float64"
    assert vsplit_contract.static_argument_rule is not None
    assert vsplit_contract.static_argument_rule((matrix, 2, 0)) == (
        (2, 3),
        2,
        0,
        ((1, 3), (1, 3)),
    )
    cube = matrix.reshape(1, 2, 3)
    assert dsplit_contract.shape_rule is not None
    assert dsplit_contract.shape_rule((cube, [1], 2)) == (6,)
    assert dsplit_contract.dtype_rule is not None
    assert dsplit_contract.dtype_rule((cube, [1], 2)) == "float64"
    assert dsplit_contract.static_argument_rule is not None
    assert dsplit_contract.static_argument_rule((cube, [1], 2)) == (
        (1, 2, 3),
        (1,),
        2,
        ((1, 2, 1), (1, 2, 2)),
    )
    for variant_contract in (hsplit_contract, vsplit_contract, dsplit_contract):
        with pytest.raises(ValueError, match="incomplete primitive contract"):
            primitive_complete_contract_for(variant_contract.identity)


def test_program_ad_assembly_split_variant_boundaries_are_explicit() -> None:
    """Split-family contracts should expose fail-closed static partition boundaries."""

    expected_boundaries = {
        "hsplit": "static_hsplit_sections_gather_scatter",
        "vsplit": "static_vsplit_sections_gather_scatter",
        "dsplit": "static_dsplit_sections_gather_scatter",
    }
    for split_name, boundary in expected_boundaries.items():
        metadata = primitive_contract_for(
            PrimitiveIdentity("scpn.program_ad.assembly", split_name, "1")
        ).lowering_metadata
        assert metadata["nondifferentiable_boundary"] == boundary
        assert metadata["nondifferentiable_boundary_policy"] == "fail_closed"


def test_program_ad_assembly_split_batching_rule_maps_partition_outputs() -> None:
    """Split batching should map source batches and preserve partition structure."""

    contract = primitive_contract_for("scpn.program_ad.assembly:split")
    assert contract.batching_rule is not None
    batching_rule = contract.batching_rule

    def split_fn(
        source: FloatArray, indices_or_sections: object, axis: int
    ) -> tuple[FloatArray, ...]:
        return tuple(
            cast(
                tuple[FloatArray, ...],
                cast(Any, np.split)(source, indices_or_sections, axis=axis),
            )
        )

    batch = np.array(
        [
            [[1.0, -2.0, 0.5], [3.0, -1.5, 2.0]],
            [[-0.25, 0.75, -1.25], [1.5, -2.5, 3.5]],
        ],
        dtype=np.float64,
    )
    expected = tuple(
        np.stack(
            [parts[part_index] for parts in (np.split(item, [1, 2], axis=1) for item in batch)],
            axis=0,
        )
        for part_index in range(3)
    )

    result = batching_rule(split_fn, (batch, [1, 2], 2), (0, None, None), 0)
    assert isinstance(result, tuple)
    assert len(result) == 3
    for observed, expected_part in zip(result, expected, strict=True):
        _assert_allclose(observed, expected_part)

    moved = batching_rule(split_fn, (batch, [1, 2], 2), (0, None, None), 1)
    assert isinstance(moved, tuple)
    for observed, expected_part in zip(moved, expected, strict=True):
        _assert_allclose(observed, np.moveaxis(expected_part, 0, 1))

    with pytest.raises(ValueError, match="keeps split metadata static"):
        batching_rule(split_fn, (batch, [1, 2], 2), (0, 0, None), 0)
    with pytest.raises(ValueError, match="cannot map the split axis"):
        batching_rule(split_fn, (batch, [1, 2], 0), (0, None, None), 0)
