# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX Compatibility Integration Tests
"""Integration tests for JAX compatibility and nested-transform routes."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np
import pytest
from _phase_jax_bridge_test_helpers import (
    _FakeJAX,
)

import scpn_quantum_control.phase.jax_bridge as jax_bridge
from scpn_quantum_control.phase import (
    PhaseJAXJITCompatibilityResult,
    PhaseJAXNestedTransformAlgebraResult,
    PhaseJAXPyTreeCompatibilityResult,
    PhaseJAXShardingCompatibilityResult,
    PhaseJAXVMAPCompatibilityResult,
    parameter_shift_qnn_classifier_gradient,
    run_jax_jit_compatibility_audit,
    run_jax_nested_transform_algebra_audit,
    run_jax_pytree_compatibility_audit,
    run_jax_sharding_compatibility_audit,
    run_jax_vmap_compatibility_audit,
)


def test_phase_jax_jit_compatibility_audit_separates_native_and_callback_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))

    result = run_jax_jit_compatibility_audit(
        features=np.array([[0.0], [np.pi]], dtype=float),
        labels=np.array([0.0, 1.0], dtype=float),
        params=np.array([0.45], dtype=float),
        tolerance=1e-12,
    )

    assert isinstance(result, PhaseJAXJITCompatibilityResult)
    assert result.passed
    assert result.native_qnn_jitted
    assert result.custom_vjp_qnn_jitted
    assert result.custom_vjp_registered
    assert result.parameter_shift_jitted
    assert result.parameter_shift_host_callback
    assert not result.native_qnn_host_callback
    assert not result.custom_vjp_qnn_host_callback
    assert result.max_abs_error <= 1e-12
    assert result.claim_boundary == "bounded_jax_jit_compatibility"
    assert "parameter_shift_host_callback" in result.unsupported_native_routes
    assert fake_jax.jit_calls == 3
    assert result.to_dict()["passed"] is True


def test_phase_jax_jit_compatibility_audit_fails_closed_without_jit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoJITJAX(_FakeJAX):
        jit: Any = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_NoJITJAX(), np))

    with pytest.raises(RuntimeError, match="JAX JIT"):
        run_jax_jit_compatibility_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params=np.array([0.45], dtype=float),
        )


def test_phase_jax_vmap_compatibility_audit_batches_native_and_custom_vjp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params_batch = np.array([[0.25], [0.45], [0.65]], dtype=float)

    result = run_jax_vmap_compatibility_audit(
        features=features,
        labels=labels,
        params_batch=params_batch,
        tolerance=1e-10,
    )

    expected_gradients = np.vstack(
        [parameter_shift_qnn_classifier_gradient(features, labels, row) for row in params_batch]
    )
    assert isinstance(result, PhaseJAXVMAPCompatibilityResult)
    assert result.passed
    assert result.batch_size == 3
    assert result.native_qnn_vmapped
    assert result.custom_vjp_qnn_vmapped
    assert result.custom_vjp_registered
    assert not result.native_qnn_host_callback
    assert not result.custom_vjp_qnn_host_callback
    assert result.max_abs_error <= 1e-10
    assert result.claim_boundary == "bounded_jax_vmap_compatibility"
    assert "parameter_shift_host_loop_reference" in result.unsupported_native_routes
    assert fake_jax.vmap_calls == 2
    np.testing.assert_allclose(result.native_gradients, expected_gradients, atol=1e-10)
    np.testing.assert_allclose(result.custom_vjp_gradients, expected_gradients, atol=1e-10)
    assert result.to_dict()["passed"] is True


def test_phase_jax_vmap_compatibility_audit_fails_closed_without_vmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoVMAPJAX(_FakeJAX):
        vmap: Any = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_NoVMAPJAX(), np))

    with pytest.raises(RuntimeError, match="JAX vmap"):
        run_jax_vmap_compatibility_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params_batch=np.array([[0.25], [0.45]], dtype=float),
        )


def test_phase_jax_sharding_compatibility_audit_batches_native_and_custom_vjp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    fake_jax.local_device_count_value = 2
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    features = np.array([[0.0], [np.pi]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params_batch = np.array([[0.25], [0.45]], dtype=float)

    result = run_jax_sharding_compatibility_audit(
        features=features,
        labels=labels,
        params_batch=params_batch,
        tolerance=1e-10,
    )

    expected_gradients = np.vstack(
        [parameter_shift_qnn_classifier_gradient(features, labels, row) for row in params_batch]
    )
    assert isinstance(result, PhaseJAXShardingCompatibilityResult)
    assert result.passed
    assert result.batch_size == 2
    assert result.local_device_count == 2
    assert result.sharding_mode == "pmap_multi_device"
    assert result.native_qnn_pmapped
    assert result.custom_vjp_qnn_pmapped
    assert result.custom_vjp_registered
    assert not result.native_qnn_host_callback
    assert not result.custom_vjp_qnn_host_callback
    assert result.max_abs_error <= 1e-10
    assert result.claim_boundary == "bounded_jax_pmap_sharding_compatibility"
    assert "parameter_shift_host_loop_reference" in result.unsupported_native_routes
    assert fake_jax.pmap_calls == 2
    np.testing.assert_allclose(result.native_gradients, expected_gradients, atol=1e-10)
    np.testing.assert_allclose(result.custom_vjp_gradients, expected_gradients, atol=1e-10)
    assert result.to_dict()["passed"] is True


def test_phase_jax_sharding_compatibility_audit_fails_closed_without_pmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoPMAPJAX(_FakeJAX):
        pmap: Any = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_NoPMAPJAX(), np))

    with pytest.raises(RuntimeError, match="JAX pmap"):
        run_jax_sharding_compatibility_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params_batch=np.array([[0.25], [0.45]], dtype=float),
        )


def test_phase_jax_pytree_compatibility_audit_round_trips_gradient_structure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    features = np.array([[0.0, 0.2, 0.4], [np.pi, np.pi + 0.2, np.pi + 0.4]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params_tree = {
        "encoder": np.array([0.25, 0.45], dtype=float),
        "readout": {"phase": np.array([0.65], dtype=float)},
    }

    result = run_jax_pytree_compatibility_audit(
        features=features,
        labels=labels,
        params_pytree=params_tree,
        tolerance=1e-10,
    )

    expected = parameter_shift_qnn_classifier_gradient(
        features,
        labels,
        np.array([0.25, 0.45, 0.65], dtype=float),
    )
    assert isinstance(result, PhaseJAXPyTreeCompatibilityResult)
    assert result.passed
    assert result.leaf_count == 2
    assert result.parameter_count == 3
    assert result.native_qnn_pytree
    assert result.custom_vjp_qnn_pytree
    assert result.custom_vjp_registered
    assert not result.native_qnn_host_callback
    assert not result.custom_vjp_qnn_host_callback
    assert result.max_abs_error <= 1e-10
    assert result.claim_boundary == "bounded_jax_pytree_compatibility"
    np.testing.assert_allclose(result.parameter_vector, np.array([0.25, 0.45, 0.65]))
    np.testing.assert_allclose(result.native_gradient_vector, expected, atol=1e-10)
    np.testing.assert_allclose(result.custom_vjp_gradient_vector, expected, atol=1e-10)
    np.testing.assert_allclose(
        cast(Any, result.native_gradient_pytree)["encoder"],
        expected[:2],
        atol=1e-10,
    )
    np.testing.assert_allclose(
        cast(Any, result.custom_vjp_gradient_pytree)["readout"]["phase"],
        expected[2:],
        atol=1e-10,
    )
    assert result.to_dict()["passed"] is True


def test_phase_jax_pytree_compatibility_audit_fails_closed_without_tree_util(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _NoTreeUtilJAX(_FakeJAX):
        tree_util: ClassVar[Any] = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_NoTreeUtilJAX(), np))

    with pytest.raises(RuntimeError, match="PyTree"):
        run_jax_pytree_compatibility_audit(
            features=np.array([[0.0], [np.pi]], dtype=float),
            labels=np.array([0.0, 1.0], dtype=float),
            params_pytree={"phase": np.array([0.45], dtype=float)},
        )


def test_phase_jax_nested_transform_algebra_audit_verifies_bounded_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_jax = _FakeJAX()
    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (fake_jax, np))
    features = np.array([[0.0, 0.2], [np.pi, np.pi + 0.2]], dtype=float)
    labels = np.array([0.0, 1.0], dtype=float)
    params_batch = np.array([[0.25, 0.45], [0.35, 0.55]], dtype=float)
    params_pytree = {"encoder": np.array([0.25], dtype=float), "readout": (np.array([0.45]),)}

    result = run_jax_nested_transform_algebra_audit(
        features=features,
        labels=labels,
        params_batch=params_batch,
        params_pytree=params_pytree,
        tolerance=1e-8,
    )

    expected = np.vstack(
        [parameter_shift_qnn_classifier_gradient(features, labels, row) for row in params_batch]
    )
    assert isinstance(result, PhaseJAXNestedTransformAlgebraResult)
    assert result.passed
    assert result.bounded_transform_algebra_ready
    assert not result.ready_for_provider_exceedance
    assert result.route_status("jit_value_and_grad_under_vmap") == "passed"
    assert result.route_status("jit_vmap_value_and_grad") == "passed"
    assert result.route_status("jit_value_and_grad_pytree") == "passed"
    assert result.route_status("arbitrary_phase_qnode_jacfwd_jacrev") == "blocked"
    assert "hardware_provider_callback_transform_safety" in result.open_gaps
    assert result.claim_boundary == "bounded_jax_nested_transform_algebra"
    assert fake_jax.jit_calls >= 3
    assert fake_jax.vmap_calls >= 2
    np.testing.assert_allclose(result.jit_under_vmap_gradients, expected, atol=1e-8)
    np.testing.assert_allclose(result.jit_vmap_gradients, expected, atol=1e-8)
    np.testing.assert_allclose(
        result.pytree_gradient_vector,
        parameter_shift_qnn_classifier_gradient(features, labels, np.array([0.25, 0.45])),
        atol=1e-8,
    )
    payload = result.to_dict()
    assert cast(Any, payload["routes"])["arbitrary_phase_qnode_hessian"]["status"] == "blocked"
