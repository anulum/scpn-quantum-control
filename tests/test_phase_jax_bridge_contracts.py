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
import json
from dataclasses import MISSING, fields, is_dataclass
from typing import Any, cast, get_type_hints

import numpy as np
import pytest
from _phase_jax_bridge_test_helpers import _FakeJAX

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


def test_route_lookups_refuse_unknown_names() -> None:
    """Route lookups find the named route, not the first, and refuse names they do not hold."""
    nested = contracts.PhaseJAXNestedTransformAlgebraResult(
        jit_under_vmap_gradients=np.zeros((1, 1)),
        jit_vmap_gradients=np.zeros((1, 1)),
        pytree_gradient_vector=np.zeros(1),
        parameter_shift_batch_gradients=np.zeros((1, 1)),
        parameter_shift_pytree_gradient=np.zeros(1),
        max_abs_error=0.0,
        l2_error=0.0,
        tolerance=1e-6,
        routes=(
            contracts.PhaseJAXNestedTransformRoute("jit_route", "passed", "local", ("jax.jit",)),
            contracts.PhaseJAXNestedTransformRoute("qnode_route", "blocked", "open", ("jax",)),
        ),
    )
    matrix = contracts.PhaseJAXPhaseQNodeLoweringMatrixResult(
        routes=(
            contracts.PhaseJAXPhaseQNodeLoweringRoute("native", "passed", "local", False),
            contracts.PhaseJAXPhaseQNodeLoweringRoute("provider", "blocked", "open", True),
        )
    )

    assert nested.route_status("qnode_route") == "blocked"
    assert matrix.route_status("provider") == "blocked"
    with pytest.raises(KeyError, match="unknown JAX nested-transform route: missing"):
        nested.route_status("missing")
    with pytest.raises(KeyError, match="unknown JAX Phase-QNode lowering route: missing"):
        matrix.route_status("missing")


def test_maturity_contract_keeps_plain_evidence_values() -> None:
    """Evidence without its own to_dict is serialised as given."""
    maturity = contracts.PhaseJAXMaturityAuditResult(
        bounded_model_ready=False,
        ready_for_provider_exceedance=False,
        evidence={"operator_note": "awaiting isolated host"},
        required_capabilities={},
        open_gaps=(),
    )

    payload = maturity.to_dict()

    assert payload["evidence"] == {"operator_note": "awaiting isolated host"}


def test_jax_results_serialise_list_pytrees_and_native_gradients(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A list inside a parameter PyTree stays a list, and native gradients serialise."""
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_FakeJAX(), np))
    features = np.array([[0.0, 0.2], [np.pi, np.pi + 0.2]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)

    pytree = jax_bridge.run_jax_pytree_compatibility_audit(
        features=features,
        labels=labels,
        params_pytree={"layers": [{"weight": np.array([0.25])}, np.array([0.45])]},
    )
    native = jax_bridge.jax_native_qnn_value_and_grad(features, labels, np.array([0.25, 0.45]))

    gradients = cast(dict[str, object], pytree.to_dict()["native_gradient_pytree"])
    layers = cast(list[object], gradients["layers"])
    assert isinstance(layers, list)
    assert isinstance(layers[0], dict) and set(layers[0]) == {"weight"}
    native_payload = native.to_dict()
    assert native_payload["gradient"] == native.gradient.tolist()
    assert native_payload["method"] == "jax_native_bounded_phase_qnn_value_and_grad"


def _probe_value(hint: object) -> object:
    """Return a small deterministic value for one evidence field annotation."""
    text = str(hint)
    if hint is bool:
        return True
    if hint is int:
        return 2
    if hint is float:
        return 0.25
    if hint is str:
        return "probe"
    if "complex" in text:
        return np.array([0.6 + 0.8j, 0.0 - 0.5j])
    if "ndarray" in text.lower():
        return np.array([0.5, -0.25])
    if text == "tuple[str, ...]":
        return ("jit", "grad")
    if text == "tuple[tuple[int, ...], ...]":
        return ((2,),)
    if hint is object:
        return {"weights": np.array([0.5, -0.25])}
    raise AssertionError(f"no probe value for {text}")


@pytest.mark.parametrize(
    "result_type",
    [
        contracts.PhaseJAXPhaseQNodeStatevectorResult,
        contracts.PhaseJAXPhaseQNodeNativeTransformResult,
        contracts.PhaseJAXPhaseQNodePyTreeTransformResult,
        contracts.PhaseJAXPhaseQNodeShardingTransformResult,
    ],
)
def test_registered_phase_qnode_results_serialise_without_jax(result_type: type[Any]) -> None:
    """Registered Phase-QNode JAX evidence serialises to plain JSON without importing JAX.

    Real JAX lowering produces these results in the integration tests; their
    JSON contract is checked here directly, so it holds on hosts without JAX.
    """
    hints = get_type_hints(result_type)
    values = {
        field.name: _probe_value(hints[field.name])
        for field in fields(result_type)
        if field.default is MISSING
    }
    result = result_type(**values)

    payload = result.to_dict()

    assert json.loads(json.dumps(payload)) == payload
    defaults = {
        field.name: field.default for field in fields(result_type) if field.default is not MISSING
    }
    assert payload["method"] == defaults["method"]
    assert payload["claim_boundary"] == defaults["claim_boundary"]
    for name, value in values.items():
        if isinstance(value, np.ndarray) and np.isrealobj(value) and name in payload:
            assert payload[name] == value.tolist()
    if "state" in values:
        state = cast(np.ndarray, values["state"])
        assert payload["state_real"] == state.real.tolist()
        assert payload["state_imag"] == state.imag.tolist()
