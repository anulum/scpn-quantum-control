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
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.differentiable import whole_program_value_and_grad
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
