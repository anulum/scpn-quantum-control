# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — program AD selection registry tests
# scpn-quantum-control -- Program AD selection primitive registry tests
"""Tests for Program AD selection primitive registry contracts."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest

from scpn_quantum_control.differentiable import (
    program_adjoint_gradient,
    whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_registry import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    PrimitiveContract,
    PrimitiveDTypeRule,
    PrimitiveIdentity,
    PrimitiveShapeRule,
    PrimitiveStaticArgumentRule,
    PrimitiveTransformRule,
    primitive_complete_contract_for,
    primitive_contract_for,
)


def _assert_allclose(
    actual: object, expected: object, *, rtol: float = 1.0e-7, atol: float = 0.0
) -> None:
    """Assert NumPy closeness across dynamically typed Program AD payloads."""

    cast(Any, np.testing.assert_allclose)(actual, expected, rtol=rtol, atol=atol)


def _transform_rule_from_contract(contract: PrimitiveContract) -> PrimitiveTransformRule:
    """Return a registry transform snapshot from a primitive contract."""

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


def test_program_ad_selection_primitives_are_registry_policy_gated() -> None:
    """Selection primitives should expose first-class primitive registry contracts."""

    vector = np.array([-1.0, 0.25, 2.0], dtype=np.float64)
    condition = np.array([True, False, True])
    upper = np.array([0.5, 0.75, 2.0], dtype=np.float64)

    where_contract = primitive_contract_for("scpn.program_ad.selection:where")
    assert where_contract.identity == PrimitiveIdentity("scpn.program_ad.selection", "where", "1")
    assert where_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert where_contract.effect == "pure"
    assert where_contract.lowering_metadata["mlir_op"] == "scpn_diff.selection.where"
    assert where_contract.shape_rule is not None
    assert where_contract.shape_rule((condition, vector, 1.0)) == (3,)
    assert where_contract.dtype_rule is not None
    assert where_contract.dtype_rule((condition, vector, 1.0)) == "float64"
    assert where_contract.static_argument_rule is not None
    assert where_contract.static_argument_rule((condition, vector, 1.0)) == (
        "static_condition",
        (True, False, True),
        (3,),
    )

    clip_contract = primitive_contract_for("scpn.program_ad.selection:clip")
    assert clip_contract.identity == PrimitiveIdentity("scpn.program_ad.selection", "clip", "1")
    assert clip_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert clip_contract.effect == "pure"
    assert clip_contract.lowering_metadata["mlir_op"] == "scpn_diff.selection.clip"
    assert clip_contract.shape_rule is not None
    assert clip_contract.shape_rule((vector, -0.5, upper)) == (3,)
    assert clip_contract.dtype_rule is not None
    assert clip_contract.dtype_rule((vector, -0.5, upper)) == "float64"
    assert clip_contract.static_argument_rule is not None
    assert clip_contract.static_argument_rule((vector, -0.5, upper)) == ()

    sort_contract = primitive_contract_for("scpn.program_ad.selection:sort")
    assert sort_contract.identity == PrimitiveIdentity("scpn.program_ad.selection", "sort", "1")
    assert sort_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert sort_contract.effect == "pure"
    assert sort_contract.lowering_metadata["mlir_op"] == "scpn_diff.selection.sort"
    assert sort_contract.shape_rule is not None
    assert sort_contract.shape_rule((vector, None, "stable")) == (3,)
    matrix = np.reshape(np.arange(6.0, dtype=np.float64), (2, 3))
    assert sort_contract.shape_rule((matrix, 1, "quicksort")) == (2, 3)
    assert sort_contract.dtype_rule is not None
    assert sort_contract.dtype_rule((vector, None, "stable")) == "float64"
    assert sort_contract.static_argument_rule is not None
    assert sort_contract.static_argument_rule((matrix, 1, "stable")) == (
        "axis",
        1,
        "kind",
        "stable",
    )

    select_contract = primitive_contract_for("scpn.program_ad.selection:select")
    assert select_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.selection", "select", "1"
    )
    assert select_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert select_contract.effect == "pure"
    assert select_contract.lowering_metadata["mlir_op"] == "scpn_diff.selection.select"
    assert select_contract.shape_rule is not None
    assert select_contract.shape_rule(((condition,), (vector,), 0.0)) == (3,)
    assert select_contract.dtype_rule is not None
    assert select_contract.dtype_rule(((condition,), (vector,), 0.0)) == "float64"
    assert select_contract.static_argument_rule is not None
    assert select_contract.static_argument_rule(((condition,), (vector,), 0.0))[:2] == (
        "branch_count",
        1,
    )

    piecewise_contract = primitive_contract_for("scpn.program_ad.selection:piecewise")
    assert piecewise_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.selection", "piecewise", "1"
    )
    assert piecewise_contract.lowering_metadata["mlir_op"] == "scpn_diff.selection.piecewise"
    assert piecewise_contract.shape_rule is not None
    assert piecewise_contract.shape_rule((vector, (condition,), (lambda item: item, 0.0))) == (3,)
    assert piecewise_contract.dtype_rule is not None
    assert piecewise_contract.dtype_rule((vector, (condition,), (lambda item: item, 0.0))) == (
        "float64"
    )
    assert piecewise_contract.static_argument_rule is not None
    assert piecewise_contract.static_argument_rule(
        (vector, (condition,), (lambda item: item, 0.0))
    )[-2:] == ("has_default", True)

    choose_contract = primitive_contract_for("scpn.program_ad.selection:choose")
    selector = np.array([0, 1, 0], dtype=np.int64)
    assert choose_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.selection", "choose", "1"
    )
    assert choose_contract.lowering_metadata["mlir_op"] == "scpn_diff.selection.choose"
    assert choose_contract.shape_rule is not None
    assert choose_contract.shape_rule((selector, (vector, upper), "raise")) == (3,)
    assert choose_contract.dtype_rule is not None
    assert choose_contract.dtype_rule((selector, (vector, upper), "raise")) == "float64"
    assert choose_contract.static_argument_rule is not None
    assert choose_contract.static_argument_rule((selector, (vector, upper), "raise"))[-2:] == (
        "mode",
        "raise",
    )

    compress_contract = primitive_contract_for("scpn.program_ad.selection:compress")
    assert compress_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.selection", "compress", "1"
    )
    assert compress_contract.lowering_metadata["mlir_op"] == "scpn_diff.selection.compress"
    assert compress_contract.shape_rule is not None
    assert compress_contract.shape_rule((condition, vector, None)) == (2,)
    assert compress_contract.dtype_rule is not None
    assert compress_contract.dtype_rule((condition, vector, None)) == "float64"
    assert compress_contract.static_argument_rule is not None
    assert compress_contract.static_argument_rule((condition, vector, None))[-2:] == (
        "axis",
        None,
    )

    extract_contract = primitive_contract_for("scpn.program_ad.selection:extract")
    assert extract_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.selection", "extract", "1"
    )
    assert extract_contract.lowering_metadata["mlir_op"] == "scpn_diff.selection.extract"
    assert extract_contract.shape_rule is not None
    assert extract_contract.shape_rule((condition, vector)) == (2,)
    assert extract_contract.dtype_rule is not None
    assert extract_contract.dtype_rule((condition, vector)) == "float64"
    assert extract_contract.static_argument_rule is not None
    assert extract_contract.static_argument_rule((condition, vector))[-2:] == (
        "indices",
        (0, 2),
    )

    argmax_contract = primitive_contract_for("scpn.program_ad.selection:argmax")
    assert argmax_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.selection", "argmax", "1"
    )
    assert argmax_contract.nondifferentiable_policy == "program_ad_trace_exact_fail_closed"
    assert argmax_contract.effect == "pure"
    assert argmax_contract.lowering_metadata["program_ad"] == (
        "unsupported_index_selection_fail_closed"
    )
    assert argmax_contract.lowering_metadata["mlir_op"] == "scpn_diff.selection.argmax"
    assert argmax_contract.shape_rule is not None
    assert argmax_contract.shape_rule((matrix, 1)) == (2,)
    assert argmax_contract.shape_rule((matrix, None)) == ()
    assert argmax_contract.dtype_rule is not None
    assert argmax_contract.dtype_rule((matrix, 1)) == "int64"
    assert argmax_contract.static_argument_rule is not None
    assert argmax_contract.static_argument_rule((matrix, 1)) == ("axis", 1)

    argmin_contract = primitive_contract_for("scpn.program_ad.selection:argmin")
    assert argmin_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.selection", "argmin", "1"
    )
    assert argmin_contract.lowering_metadata["program_ad"] == (
        "unsupported_index_selection_fail_closed"
    )
    assert argmin_contract.shape_rule is not None
    assert argmin_contract.shape_rule((matrix, 0)) == (3,)

    argsort_contract = primitive_contract_for("scpn.program_ad.selection:argsort")
    assert argsort_contract.identity == PrimitiveIdentity(
        "scpn.program_ad.selection", "argsort", "1"
    )
    assert argsort_contract.lowering_metadata["program_ad"] == (
        "unsupported_index_selection_fail_closed"
    )
    assert argsort_contract.shape_rule is not None
    assert argsort_contract.shape_rule((matrix, 1, "stable")) == (2, 3)
    assert argsort_contract.shape_rule((matrix, None, "stable")) == (6,)
    assert argsort_contract.dtype_rule is not None
    assert argsort_contract.dtype_rule((matrix, 1, "stable")) == "int64"
    assert argsort_contract.static_argument_rule is not None
    assert argsort_contract.static_argument_rule((matrix, 1, "stable")) == (
        "axis",
        1,
        "kind",
        "stable",
    )

    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(where_contract.identity)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(clip_contract.identity)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(sort_contract.identity)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(argmax_contract.identity)
    with pytest.raises(ValueError, match="incomplete primitive contract"):
        primitive_complete_contract_for(select_contract.identity)


def test_program_ad_selection_boundary_metadata_is_explicit() -> None:
    """Selection contracts should expose fail-closed branch and clipping boundaries."""

    expected_boundaries = {
        "where": "predicate_branch_boundary",
        "clip": "clipping_boundary_and_bound_order",
        "sort": "strict_total_order_required",
        "select": "static_condition_sequence_branch_fold",
        "piecewise": "static_condition_sequence_callable_fold",
        "choose": "static_integer_selector_gather",
        "compress": "static_boolean_mask_gather",
        "extract": "static_boolean_mask_flat_gather",
        "argmax": "integer_index_selection_nondifferentiable",
        "argmin": "integer_index_selection_nondifferentiable",
        "argsort": "integer_index_permutation_nondifferentiable",
    }
    for name, boundary in expected_boundaries.items():
        metadata = primitive_contract_for(
            PrimitiveIdentity("scpn.program_ad.selection", name, "1")
        ).lowering_metadata
        assert metadata["nondifferentiable_boundary"] == boundary
        assert metadata["nondifferentiable_boundary_policy"] == "fail_closed"


def test_program_ad_selection_primitives_validate_registry_rules_at_dispatch() -> None:
    """Selection trace execution should validate registry shape, dtype, and static rules."""

    originals = {
        name: primitive_contract_for(f"scpn.program_ad.selection:{name}")
        for name in ("where", "clip", "sort")
    }
    calls: dict[str, set[str]] = {name: set() for name in originals}

    for name, original in originals.items():
        assert original.shape_rule is not None
        assert original.dtype_rule is not None
        assert original.static_argument_rule is not None
        contract_shape_rule = original.shape_rule
        contract_dtype_rule = original.dtype_rule
        contract_static_argument_rule = original.static_argument_rule

        def shape_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            bound_rule: PrimitiveShapeRule = contract_shape_rule,
        ) -> tuple[int, ...]:
            calls[primitive_name].add("shape")
            return bound_rule(args)

        def dtype_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            bound_rule: PrimitiveDTypeRule = contract_dtype_rule,
        ) -> str:
            calls[primitive_name].add("dtype")
            return bound_rule(args)

        def static_argument_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            bound_rule: PrimitiveStaticArgumentRule = contract_static_argument_rule,
        ) -> tuple[object, ...]:
            calls[primitive_name].add("static")
            return bound_rule(args)

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

    try:
        result = whole_program_value_and_grad(
            lambda values: np.sum(
                np.where(values > np.array([-0.5, 0.0, 1.0]), values**2, -values)
                + np.clip(
                    values + np.array([0.1, -0.2, 0.3]),
                    -0.75,
                    np.array([0.5, 0.75, 2.0]),
                )
                + np.sort(values, kind="stable")
            ),
            np.array([-1.0, 0.4, 1.2], dtype=np.float64),
        )
    finally:
        for original in originals.values():
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
                _transform_rule_from_contract(original), overwrite=True
            )

    _assert_allclose(result.gradient, np.array([0.0, 2.8, 4.4]))
    assert calls == {
        "where": {"shape", "dtype", "static"},
        "clip": {"shape", "dtype", "static"},
        "sort": {"shape", "dtype", "static"},
    }


def test_program_ad_selection_fold_primitives_validate_registry_rules_at_dispatch() -> None:
    """Selection fold primitives should execute through registry validation rules."""

    originals = {
        name: primitive_contract_for(f"scpn.program_ad.selection:{name}")
        for name in ("select", "piecewise", "choose", "compress", "extract")
    }
    calls: dict[str, set[str]] = {name: set() for name in originals}

    for name, original in originals.items():
        assert original.shape_rule is not None
        assert original.dtype_rule is not None
        assert original.static_argument_rule is not None
        contract_shape_rule = original.shape_rule
        contract_dtype_rule = original.dtype_rule
        contract_static_argument_rule = original.static_argument_rule

        def shape_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            bound_rule: PrimitiveShapeRule = contract_shape_rule,
        ) -> tuple[int, ...]:
            calls[primitive_name].add("shape")
            return bound_rule(args)

        def dtype_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            bound_rule: PrimitiveDTypeRule = contract_dtype_rule,
        ) -> str:
            calls[primitive_name].add("dtype")
            return bound_rule(args)

        def static_argument_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            bound_rule: PrimitiveStaticArgumentRule = contract_static_argument_rule,
        ) -> tuple[object, ...]:
            calls[primitive_name].add("static")
            return bound_rule(args)

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

    selector = np.array([0, 1, 0, 1], dtype=np.int64)
    mask = np.array([True, False, True, False], dtype=np.bool_)

    def objective(values: Any) -> object:
        selected = np.select(
            [values < -0.5, values > 1.0],
            [values * values, 2.0 * values],
            default=-values,
        )
        piecewise = np.piecewise(
            values,
            [values < -0.5, values > 1.0],
            [lambda item: item * item, lambda item: 2.0 * item, lambda item: -item],
        )
        chosen = np.choose(selector, [values, 2.0 * values])
        compressed = np.compress(mask, values)
        extracted = np.extract(mask, values)
        return (
            np.sum(selected)
            + np.sum(piecewise)
            + np.sum(chosen)
            + np.sum(compressed)
            + np.sum(extracted)
        )

    values = np.array([-1.0, -0.25, 0.75, 1.5], dtype=np.float64)
    try:
        result = whole_program_value_and_grad(objective, values)
    finally:
        for original in originals.values():
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
                _transform_rule_from_contract(original), overwrite=True
            )

    assert result.value == pytest.approx(float(cast(Any, objective(values))))
    _assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)
    assert calls == {
        "select": {"shape", "dtype", "static"},
        "piecewise": {"shape", "dtype", "static"},
        "choose": {"shape", "dtype", "static"},
        "compress": {"shape", "dtype", "static"},
        "extract": {"shape", "dtype", "static"},
    }


def test_program_ad_index_selection_primitives_validate_registry_rules_at_dispatch() -> None:
    """Index-valued selection boundaries should still validate registry metadata."""

    originals = {
        name: primitive_contract_for(f"scpn.program_ad.selection:{name}")
        for name in ("argmax", "argmin", "argsort")
    }
    calls: dict[str, set[str]] = {name: set() for name in originals}

    for name, original in originals.items():
        assert original.shape_rule is not None
        assert original.dtype_rule is not None
        assert original.static_argument_rule is not None
        contract_shape_rule = original.shape_rule
        contract_dtype_rule = original.dtype_rule
        contract_static_argument_rule = original.static_argument_rule

        def shape_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            bound_rule: PrimitiveShapeRule = contract_shape_rule,
        ) -> tuple[int, ...]:
            calls[primitive_name].add("shape")
            return bound_rule(args)

        def dtype_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            bound_rule: PrimitiveDTypeRule = contract_dtype_rule,
        ) -> str:
            calls[primitive_name].add("dtype")
            return bound_rule(args)

        def static_argument_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            bound_rule: PrimitiveStaticArgumentRule = contract_static_argument_rule,
        ) -> tuple[object, ...]:
            calls[primitive_name].add("static")
            return bound_rule(args)

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

    try:
        values = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        with pytest.raises(
            ValueError, match="registered nondifferentiable integer selection primitives"
        ):
            whole_program_value_and_grad(lambda vector: np.argmax(vector), values)
        with pytest.raises(
            ValueError, match="registered nondifferentiable integer selection primitives"
        ):
            whole_program_value_and_grad(
                lambda vector: np.reshape(vector, (2, 2)).argmin(axis=1)[0], values
            )
        with pytest.raises(
            ValueError, match="registered nondifferentiable integer selection primitives"
        ):
            whole_program_value_and_grad(
                lambda vector: np.argsort(vector, kind="stable")[0], values
            )
    finally:
        for original in originals.values():
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
                _transform_rule_from_contract(original), overwrite=True
            )

    assert calls == {
        "argmax": {"shape", "dtype", "static"},
        "argmin": {"shape", "dtype", "static"},
        "argsort": {"shape", "dtype", "static"},
    }
