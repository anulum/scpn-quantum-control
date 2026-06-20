# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# © Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Program AD runtime registry dispatch tests
"""Tests for Program AD runtime registry-dispatch contract validation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import (
    program_adjoint_gradient,
    whole_program_value_and_grad,
)
from scpn_quantum_control.program_ad_registry import (
    DEFAULT_CUSTOM_DERIVATIVE_REGISTRY,
    PrimitiveContract,
    PrimitiveTransformRule,
    primitive_contract_for,
)

Objective = Callable[[Any], object]
TransformFactory = Callable[[PrimitiveContract], PrimitiveTransformRule]


def _transform_rule_from_contract(contract: PrimitiveContract) -> PrimitiveTransformRule:
    """Return a mutable registry transform that exactly mirrors a contract."""

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


def _without_batching_rule(contract: PrimitiveContract) -> PrimitiveTransformRule:
    """Return a transform with the batching facet removed."""

    return PrimitiveTransformRule(
        identity=contract.identity,
        derivative_rule=contract.derivative_rule,
        batching_rule=None,
        lowering_rule=contract.lowering_rule,
        lowering_metadata=contract.lowering_metadata,
        shape_rule=contract.shape_rule,
        dtype_rule=contract.dtype_rule,
        static_argument_rule=contract.static_argument_rule,
        nondifferentiable_policy=contract.nondifferentiable_policy,
        effect=contract.effect,
    )


def _without_boundary_metadata(contract: PrimitiveContract) -> PrimitiveTransformRule:
    """Return a transform with incomplete fail-closed boundary metadata."""

    return PrimitiveTransformRule(
        identity=contract.identity,
        derivative_rule=contract.derivative_rule,
        batching_rule=contract.batching_rule,
        lowering_rule=contract.lowering_rule,
        lowering_metadata={
            "mlir_op": "scpn_diff.shape.reshape",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=contract.shape_rule,
        dtype_rule=contract.dtype_rule,
        static_argument_rule=contract.static_argument_rule,
        nondifferentiable_policy=contract.nondifferentiable_policy,
        effect=contract.effect,
    )


def _without_mlir_op(contract: PrimitiveContract) -> PrimitiveTransformRule:
    """Return a transform with boundary metadata but no MLIR operation key."""

    return PrimitiveTransformRule(
        identity=contract.identity,
        derivative_rule=contract.derivative_rule,
        batching_rule=contract.batching_rule,
        lowering_rule=contract.lowering_rule,
        lowering_metadata={
            "nondifferentiable_boundary": "rank2_broadcast_contract",
            "nondifferentiable_boundary_policy": "fail_closed",
        },
        shape_rule=contract.shape_rule,
        dtype_rule=contract.dtype_rule,
        static_argument_rule=contract.static_argument_rule,
        nondifferentiable_policy=contract.nondifferentiable_policy,
        effect=contract.effect,
    )


def _without_dtype_rule(contract: PrimitiveContract) -> PrimitiveTransformRule:
    """Return a transform with the dtype facet removed."""

    return PrimitiveTransformRule(
        identity=contract.identity,
        derivative_rule=contract.derivative_rule,
        batching_rule=contract.batching_rule,
        lowering_rule=contract.lowering_rule,
        lowering_metadata=contract.lowering_metadata,
        shape_rule=contract.shape_rule,
        dtype_rule=None,
        static_argument_rule=contract.static_argument_rule,
        nondifferentiable_policy=contract.nondifferentiable_policy,
        effect=contract.effect,
    )


def _without_static_argument_rule(contract: PrimitiveContract) -> PrimitiveTransformRule:
    """Return a transform with the static-argument facet removed."""

    return PrimitiveTransformRule(
        identity=contract.identity,
        derivative_rule=contract.derivative_rule,
        batching_rule=contract.batching_rule,
        lowering_rule=contract.lowering_rule,
        lowering_metadata=contract.lowering_metadata,
        shape_rule=contract.shape_rule,
        dtype_rule=contract.dtype_rule,
        static_argument_rule=None,
        nondifferentiable_policy=contract.nondifferentiable_policy,
        effect=contract.effect,
    )


def test_program_ad_runtime_dispatch_requires_complete_registry_contracts() -> None:
    """Program AD execution should fail closed when any runtime registry facet is missing."""

    scenarios: tuple[
        tuple[str, Objective, NDArray[np.float64], TransformFactory, str],
        ...,
    ] = (
        (
            "scpn.program_ad.reduction:sum",
            lambda values: np.sum(values),
            np.array([1.0, 2.0, 3.0], dtype=np.float64),
            _without_batching_rule,
            "batching_rule",
        ),
        (
            "scpn.program_ad.shape:reshape",
            lambda values: np.sum(np.reshape(values, (2, 2))),
            np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64),
            _without_boundary_metadata,
            "nondifferentiable_boundary",
        ),
        (
            "scpn.program_ad.product:matmul",
            lambda values: np.sum(
                np.matmul(np.reshape(values, (2, 2)), np.reshape(values, (2, 2)))
            ),
            np.array([1.0, 0.25, -0.5, 2.0], dtype=np.float64),
            _without_mlir_op,
            "mlir_op",
        ),
        (
            "scpn.program_ad.linalg:trace",
            lambda values: np.trace(np.reshape(values, (2, 2))),
            np.array([1.0, 0.25, -0.5, 2.0], dtype=np.float64),
            _without_dtype_rule,
            "dtype_rule",
        ),
        (
            "scpn.program_ad.selection:clip",
            lambda values: np.sum(np.clip(values, -0.5, 0.5)),
            np.array([-1.0, 0.25, 2.0], dtype=np.float64),
            _without_static_argument_rule,
            "static_argument_rule",
        ),
    )

    for identity, objective, values, transform_factory, missing in scenarios:
        original = primitive_contract_for(identity)
        DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
            transform_factory(original), overwrite=True
        )
        try:
            with pytest.raises(
                ValueError,
                match=rf"incomplete program AD .* runtime contract.*{missing}",
            ):
                whole_program_value_and_grad(objective, values, trace=False)
        finally:
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
                _transform_rule_from_contract(original), overwrite=True
            )


def test_program_ad_assembly_primitives_validate_registry_rules_at_dispatch() -> None:
    """Supported assembly primitives must execute through registry validation rules."""

    originals = {
        name: primitive_contract_for(f"scpn.program_ad.assembly:{name}")
        for name in (
            "append",
            "array_split",
            "block",
            "broadcast_arrays",
            "broadcast_to",
            "concatenate",
            "column_stack",
            "diagonal",
            "dstack",
            "dsplit",
            "full_like",
            "hsplit",
            "hstack",
            "ones_like",
            "split",
            "stack",
            "tril",
            "triu",
            "vstack",
            "vsplit",
            "zeros_like",
        )
    }
    calls: dict[str, set[str]] = {name: set() for name in originals}

    for name, original in originals.items():
        assert original.shape_rule is not None
        assert original.dtype_rule is not None
        assert original.static_argument_rule is not None
        contract_shape_rule = cast(Any, original.shape_rule)
        contract_dtype_rule = cast(Any, original.dtype_rule)
        contract_static_argument_rule = cast(Any, original.static_argument_rule)

        def shape_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            contract_rule: Any = contract_shape_rule,
        ) -> tuple[int, ...]:
            calls[primitive_name].add("shape")
            return cast(tuple[int, ...], contract_rule(args))

        def dtype_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            contract_rule: Any = contract_dtype_rule,
        ) -> str:
            calls[primitive_name].add("dtype")
            return cast(str, contract_rule(args))

        def static_argument_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            contract_rule: Any = contract_static_argument_rule,
        ) -> tuple[object, ...]:
            calls[primitive_name].add("static")
            return cast(tuple[object, ...], contract_rule(args))

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

    values = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.0], dtype=np.float64)

    def objective(source: Any) -> object:
        matrix = np.reshape(source, (2, 3))
        left = matrix[:, :1]
        right = matrix[:, 1:]
        top = matrix[:1, :]
        bottom = matrix[1:, :]
        cube = matrix.reshape((1, 2, 3))
        broadcast_left, broadcast_right = np.broadcast_arrays(left, top)
        split_parts = np.split(matrix, [1, 2], axis=1)
        array_split_parts = np.array_split(source, 4, axis=0)
        hsplit_parts = np.hsplit(matrix, [1, 2])
        vsplit_parts = np.vsplit(matrix, 2)
        dsplit_parts = np.dsplit(cube, [1])
        return (
            np.sum(np.concatenate((left, right), axis=1))
            + np.sum(np.stack((top, bottom), axis=0))
            + np.sum(np.hstack((left, right)))
            + np.sum(np.vstack((top, bottom)))
            + np.sum(np.column_stack((source[:3], source[3:])))
            + np.sum(np.dstack((top, bottom)))
            + np.sum(np.append(left, right[:, :1], axis=1))
            + np.sum(np.block([[left, right]]))
            + np.sum(np.broadcast_to(source[:1], (2, 1)))
            + np.sum(broadcast_left)
            + np.sum(broadcast_right)
            + np.sum(np.zeros_like(source))
            + np.sum(np.ones_like(source))
            + np.sum(np.full_like(source, 2.0))
            + np.sum(np.tril(matrix, k=-1))
            + np.sum(np.triu(matrix, k=1))
            + np.sum(np.diagonal(matrix, offset=0, axis1=0, axis2=1))
            + sum(np.sum(part) for part in split_parts)
            + sum(np.sum(part) for part in array_split_parts)
            + sum(np.sum(part) for part in hsplit_parts)
            + sum(np.sum(part) for part in vsplit_parts)
            + sum(np.sum(part) for part in dsplit_parts)
        )

    try:
        result = whole_program_value_and_grad(objective, values)
    finally:
        for original in originals.values():
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
                _transform_rule_from_contract(original), overwrite=True
            )

    assert result.value == pytest.approx(float(cast(Any, objective)(values)))
    assert calls == {
        "append": {"shape", "dtype", "static"},
        "array_split": {"shape", "dtype", "static"},
        "block": {"shape", "dtype", "static"},
        "broadcast_arrays": {"shape", "dtype", "static"},
        "broadcast_to": {"shape", "dtype", "static"},
        "concatenate": {"shape", "dtype", "static"},
        "column_stack": {"shape", "dtype", "static"},
        "diagonal": {"shape", "dtype", "static"},
        "dstack": {"shape", "dtype", "static"},
        "dsplit": {"shape", "dtype", "static"},
        "full_like": {"shape", "dtype", "static"},
        "hsplit": {"shape", "dtype", "static"},
        "hstack": {"shape", "dtype", "static"},
        "ones_like": {"shape", "dtype", "static"},
        "split": {"shape", "dtype", "static"},
        "stack": {"shape", "dtype", "static"},
        "tril": {"shape", "dtype", "static"},
        "triu": {"shape", "dtype", "static"},
        "vstack": {"shape", "dtype", "static"},
        "vsplit": {"shape", "dtype", "static"},
        "zeros_like": {"shape", "dtype", "static"},
    }


def test_program_ad_reduction_and_cumulative_primitives_validate_registry_rules_at_dispatch() -> (
    None
):
    """Supported reduction and cumulative primitives must use registry validation."""

    originals = {
        name: primitive_contract_for(f"scpn.program_ad.reduction:{name}")
        for name in (
            "max",
            "mean",
            "median",
            "min",
            "percentile",
            "prod",
            "quantile",
            "std",
            "sum",
            "trapezoid",
            "var",
        )
    }
    originals.update(
        {
            name: primitive_contract_for(f"scpn.program_ad.cumulative:{name}")
            for name in ("cumprod", "cumsum", "diff")
        }
    )
    calls: dict[str, set[str]] = {name: set() for name in originals}

    for name, original in originals.items():
        assert original.shape_rule is not None
        assert original.dtype_rule is not None
        assert original.static_argument_rule is not None
        contract_shape_rule = cast(Any, original.shape_rule)
        contract_dtype_rule = cast(Any, original.dtype_rule)
        contract_static_argument_rule = cast(Any, original.static_argument_rule)

        def shape_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            contract_rule: Any = contract_shape_rule,
        ) -> tuple[int, ...]:
            calls[primitive_name].add("shape")
            return cast(tuple[int, ...], contract_rule(args))

        def dtype_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            contract_rule: Any = contract_dtype_rule,
        ) -> str:
            calls[primitive_name].add("dtype")
            return cast(str, contract_rule(args))

        def static_argument_rule(
            args: tuple[object, ...],
            *,
            primitive_name: str = name,
            contract_rule: Any = contract_static_argument_rule,
        ) -> tuple[object, ...]:
            calls[primitive_name].add("static")
            return cast(tuple[object, ...], contract_rule(args))

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

    values = np.array([1.0, -2.0, 0.5, 3.0, -1.5, 2.0], dtype=np.float64)

    def objective(source: Any) -> object:
        matrix = np.reshape(source, (2, 3))
        shifted = matrix + 3.0
        grid = np.array([0.0, 0.5, 2.0], dtype=np.float64)
        return (
            np.sum(matrix)
            + np.sum(np.prod(shifted, axis=1))
            + np.sum(np.mean(matrix, axis=0))
            + np.sum(matrix.max(axis=0))
            - np.sum(matrix.min(axis=1))
            + np.sum(matrix.var(axis=1, ddof=1))
            + np.sum(matrix.std(axis=0, ddof=1))
            + np.median(source)
            + np.sum(np.quantile(matrix, 0.25, axis=1))
            + np.sum(np.percentile(matrix, 75.0, axis=0))
            + np.sum(np.trapezoid(matrix, x=grid, axis=1))
            + np.sum(np.cumsum(source))
            + np.sum(np.cumprod(source + 3.0))
            + np.sum(np.diff(source, n=2))
        )

    try:
        result = whole_program_value_and_grad(objective, values)
    finally:
        for original in originals.values():
            DEFAULT_CUSTOM_DERIVATIVE_REGISTRY.register_transform(
                _transform_rule_from_contract(original), overwrite=True
            )

    assert result.value == pytest.approx(float(cast(Any, objective)(values)))
    np.testing.assert_allclose(program_adjoint_gradient(result), result.gradient, atol=1.0e-12)
    assert calls == {
        "cumprod": {"shape", "dtype", "static"},
        "cumsum": {"shape", "dtype", "static"},
        "diff": {"shape", "dtype", "static"},
        "max": {"shape", "dtype", "static"},
        "mean": {"shape", "dtype", "static"},
        "median": {"shape", "dtype", "static"},
        "min": {"shape", "dtype", "static"},
        "percentile": {"shape", "dtype", "static"},
        "prod": {"shape", "dtype", "static"},
        "quantile": {"shape", "dtype", "static"},
        "std": {"shape", "dtype", "static"},
        "sum": {"shape", "dtype", "static"},
        "trapezoid": {"shape", "dtype", "static"},
        "var": {"shape", "dtype", "static"},
    }
