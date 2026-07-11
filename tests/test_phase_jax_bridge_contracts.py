# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Bridge Result Contract Tests
"""Contract and compatibility tests for the JAX result-record leaf."""

from __future__ import annotations

import ast
import inspect
from dataclasses import is_dataclass
from typing import Any, cast

import numpy as np

import scpn_quantum_control.phase as phase
import scpn_quantum_control.phase.jax_bridge as jax_bridge
import scpn_quantum_control.phase.jax_bridge_contracts as contracts

CONTRACT_NAMES: tuple[str, ...] = (
    "PhaseJAXCloudValidationRunSpec",
    "PhaseJAXCustomVJPQNNGradientResult",
    "PhaseJAXGradientAgreementResult",
    "PhaseJAXJITCompatibilityResult",
    "PhaseJAXMaturityAuditResult",
    "PhaseJAXNativeQNNGradientResult",
    "PhaseJAXNestedTransformAlgebraResult",
    "PhaseJAXNestedTransformRoute",
    "PhaseJAXParameterShiftResult",
    "PhaseJAXPhaseQNodeAOTExportResult",
    "PhaseJAXPhaseQNodeLoweringMatrixResult",
    "PhaseJAXPhaseQNodeLoweringRoute",
    "PhaseJAXPhaseQNodeNativeTransformResult",
    "PhaseJAXPhaseQNodePyTreeTransformResult",
    "PhaseJAXPhaseQNodeShardingTransformResult",
    "PhaseJAXPhaseQNodeStatevectorResult",
    "PhaseJAXPyTreeCompatibilityResult",
    "PhaseJAXShardingCompatibilityResult",
    "PhaseJAXVMAPCompatibilityResult",
)


def test_jax_contract_exports_preserve_public_identity() -> None:
    """Preserve every contract class across leaf, bridge, and phase imports."""
    assert tuple(contracts.__all__) == CONTRACT_NAMES
    for name in CONTRACT_NAMES:
        contract = getattr(contracts, name)
        assert getattr(jax_bridge, name) is contract
        assert getattr(phase, name) is contract
        assert is_dataclass(contract)
        assert cast(Any, contract).__dataclass_params__.frozen is True


def test_jax_contract_module_has_no_relative_import_back_edge() -> None:
    """Keep the contract leaf independent from phase execution modules."""
    tree = ast.parse(inspect.getsource(contracts))
    relative_imports = [
        node for node in tree.body if isinstance(node, ast.ImportFrom) and node.level > 0
    ]
    assert relative_imports == []


def test_nested_transform_contract_serializes_routes_and_copies_arrays() -> None:
    """Serialize nested-transform routes without exposing mutable source arrays."""
    source = np.array([[0.1, 0.2]], dtype=np.float64)
    route = contracts.PhaseJAXNestedTransformRoute(
        name="jit_under_vmap_grad",
        status="passed",
        reason="bounded local parity",
        requires=("jax.jit", "jax.vmap", "jax.grad"),
    )
    result = contracts.PhaseJAXNestedTransformAlgebraResult(
        jit_under_vmap_gradients=source,
        jit_vmap_gradients=source,
        pytree_gradient_vector=source[0],
        parameter_shift_batch_gradients=source,
        parameter_shift_pytree_gradient=source[0],
        max_abs_error=0.0,
        l2_error=0.0,
        tolerance=1e-6,
        routes=(route,),
    )

    payload = result.to_dict()
    source[0, 0] = 99.0

    assert result.passed
    assert result.bounded_transform_algebra_ready
    assert result.route_status("jit_under_vmap_grad") == "passed"
    routes = cast(dict[str, dict[str, object]], payload["routes"])
    assert routes["jit_under_vmap_grad"]["requires"] == ["jax.jit", "jax.vmap", "jax.grad"]
    np.testing.assert_allclose(
        cast(Any, payload["jit_under_vmap_gradients"]),
        [[0.1, 0.2]],
    )


def test_maturity_contract_normalizes_nested_result_evidence() -> None:
    """Normalize nested result objects through the maturity evidence envelope."""
    route = contracts.PhaseJAXPhaseQNodeLoweringRoute(
        name="bounded_qnn_native_gradient",
        status="passed",
        reason="native local evidence",
        host_callback=False,
    )
    matrix = contracts.PhaseJAXPhaseQNodeLoweringMatrixResult(routes=(route,))
    maturity = contracts.PhaseJAXMaturityAuditResult(
        bounded_model_ready=True,
        ready_for_provider_exceedance=False,
        evidence={"lowering_matrix": matrix},
        required_capabilities={"provider_execution": "blocked"},
        open_gaps=("provider_execution",),
    )

    payload = maturity.to_dict()

    evidence = cast(dict[str, dict[str, object]], payload["evidence"])
    assert evidence["lowering_matrix"]["bounded_no_host_callback_routes_ready"] is True
    assert payload["ready_for_provider_exceedance"] is False
    assert payload["open_gaps"] == ["provider_execution"]


def test_json_ready_pytree_converts_numpy_leaves_recursively() -> None:
    """Convert NumPy arrays and scalars in nested PyTrees to plain values."""
    tree = {
        "weights": np.array([0.1, 0.2], dtype=np.float64),
        "metadata": (np.float64(0.3), {"count": np.int64(2)}),
    }

    assert contracts._json_ready_pytree(tree) == {
        "weights": [0.1, 0.2],
        "metadata": [0.3, {"count": 2}],
    }
