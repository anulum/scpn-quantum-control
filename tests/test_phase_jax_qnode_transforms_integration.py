# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — JAX QNode Transform Integration Tests
"""Integration tests for registered Phase-QNode JAX transforms."""

from __future__ import annotations

from typing import Any, ClassVar, cast

import numpy as np
import pytest
from _phase_jax_bridge_test_helpers import (
    _FakeJAX,
)

import scpn_quantum_control.phase.jax_bridge as jax_bridge
from scpn_quantum_control.phase import (
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseJAXPhaseQNodeNativeTransformResult,
    PhaseJAXPhaseQNodePyTreeTransformResult,
    PhaseJAXPhaseQNodeStatevectorResult,
    PhaseQNodeCircuit,
    SparsePauliHamiltonian,
    execute_phase_qnode_circuit,
    is_phase_jax_available,
    jax_phase_qnode_native_transform_audit,
    jax_phase_qnode_pytree_transform_audit,
    jax_phase_qnode_value_and_grad,
    parameter_shift_phase_qnode_gradient,
)


def test_phase_jax_registered_qnode_native_transform_audit_uses_no_callback() -> None:
    """Registered Phase-QNode transforms should lower through native JAX APIs."""
    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("ry", (0,), 0),
            ("rx", (1,), 1),
            ("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    params = np.array([0.37, -0.21], dtype=float)

    result = jax_phase_qnode_native_transform_audit(
        circuit,
        params,
        tangent=np.array([0.25, -0.15], dtype=float),
        batch_offsets=np.array([[0.0, 0.0], [0.03, -0.01]], dtype=float),
        tolerance=5e-5,
    )
    reference = parameter_shift_phase_qnode_gradient(circuit, params)

    assert isinstance(result, PhaseJAXPhaseQNodeNativeTransformResult)
    assert result.passed
    assert result.native_framework_autodiff
    assert not result.host_callback
    assert result.jit_value_and_grad
    assert result.vmap_value_and_grad
    assert set(result.transform_names) == {
        "grad",
        "value_and_grad",
        "jacfwd",
        "jacrev",
        "hessian",
        "jvp",
        "vjp",
        "vmap",
        "jit",
    }
    np.testing.assert_allclose(result.value, reference.value, atol=5e-5)
    np.testing.assert_allclose(result.gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.value_and_grad_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.jacfwd_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.jacrev_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.vjp_cotangent_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.hessian, result.hessian.T, atol=5e-5)
    assert result.max_abs_hessian_symmetry_error <= 5e-5
    assert result.batch_params.shape == (2, 2)
    assert result.vmap_gradients.shape == (2, 2)
    payload = result.to_dict()
    assert payload["host_callback"] is False
    assert payload["method"] == "jax_native_registered_phase_qnode_transform_audit"
    assert payload["claim_boundary"] == "registered_phase_qnode_jax_native_transform_lowering"


def test_phase_jax_registered_qnode_pytree_transform_audit_uses_no_callback() -> None:
    """Registered Phase-QNode PyTrees should lower through native JAX transforms."""
    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("ry", (0,), 0),
            ("rx", (1,), 1),
            ("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    params_pytree = {
        "parameter_0": np.array([0.37], dtype=float),
        "parameter_1": (np.array([-0.21], dtype=float),),
    }
    flat_params = np.array([0.37, -0.21], dtype=float)

    result = jax_phase_qnode_pytree_transform_audit(
        circuit,
        params_pytree,
        tangent=np.array([0.25, -0.15], dtype=float),
        batch_offsets=np.array([[0.0, 0.0], [0.03, -0.01]], dtype=float),
        tolerance=5e-5,
    )
    reference = parameter_shift_phase_qnode_gradient(circuit, flat_params)

    assert isinstance(result, PhaseJAXPhaseQNodePyTreeTransformResult)
    assert result.passed
    assert result.native_framework_autodiff
    assert not result.host_callback
    assert result.jit_value_and_grad
    assert result.vmap_value_and_grad
    assert result.leaf_shapes == ((1,), (1,))
    assert set(result.transform_names) == {
        "grad",
        "value_and_grad",
        "jacfwd",
        "jacrev",
        "hessian",
        "jvp",
        "vjp",
        "vmap",
        "jit",
    }
    np.testing.assert_allclose(result.value, reference.value, atol=5e-5)
    np.testing.assert_allclose(result.gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.parameter_vector, flat_params, atol=5e-5)
    np.testing.assert_allclose(result.value_and_grad_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.jacfwd_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.jacrev_gradient, reference.gradient, atol=5e-5)
    np.testing.assert_allclose(result.vjp_cotangent_gradient, reference.gradient, atol=5e-5)
    assert result.hessian.shape == (2, 2)
    np.testing.assert_allclose(result.hessian, result.hessian.T, atol=5e-5)
    assert result.max_abs_hessian_symmetry_error <= 5e-5
    assert result.batch_params.shape == (2, 2)
    assert result.vmap_gradients.shape == (2, 2)
    payload = result.to_dict()
    assert payload["host_callback"] is False
    np.testing.assert_allclose(cast(Any, payload["hessian"]), result.hessian, atol=5e-5)
    assert cast(float, payload["max_abs_hessian_symmetry_error"]) <= 5e-5
    assert payload["leaf_shapes"] == [[1], [1]]
    assert payload["method"] == "jax_native_registered_phase_qnode_pytree_transform_audit"
    assert payload["claim_boundary"] == "registered_phase_qnode_jax_pytree_transform_lowering"


def test_phase_jax_registered_qnode_sharding_transform_audit_uses_no_callback() -> None:
    """Registered Phase-QNode batches should lower through native JAX pmap."""
    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    import jax

    local_device_count = int(jax.local_device_count())
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("ry", (0,), 0),
            ("rx", (1,), 1),
            ("cnot", (0, 1)),
        ),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )
    base_params = np.array([0.37, -0.21], dtype=float)
    offsets = np.arange(local_device_count, dtype=float)[:, None] * np.array(
        [[0.01, -0.015]],
        dtype=float,
    )
    params_batch = base_params[None, :] + offsets

    result = jax_bridge.jax_phase_qnode_sharding_transform_audit(
        circuit,
        params_batch,
        tolerance=5e-5,
    )
    reference_gradients = np.vstack(
        [parameter_shift_phase_qnode_gradient(circuit, row).gradient for row in params_batch]
    )

    assert result.passed
    assert result.native_framework_autodiff
    assert not result.host_callback
    assert result.pmapped
    assert result.batch_size == local_device_count
    assert result.local_device_count == local_device_count
    assert result.sharding_mode in {"single_device_pmap_smoke", "multi_device_pmap"}
    assert result.values.shape == (local_device_count,)
    assert result.gradients.shape == (local_device_count, 2)
    np.testing.assert_allclose(result.gradients, reference_gradients, atol=5e-5)
    payload = result.to_dict()
    assert payload["host_callback"] is False
    assert payload["pmapped"] is True
    assert payload["local_device_count"] == local_device_count
    assert payload["claim_boundary"] == "registered_phase_qnode_jax_pmap_sharding_lowering"


def test_phase_jax_registered_qnode_native_transform_audit_fails_closed_without_transforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial JAX modules should not pass as native-transform lowering."""

    class _MissingTransforms(_FakeJAX):
        grad: Any = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_MissingTransforms(), np))
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )

    with pytest.raises(RuntimeError, match="grad"):
        jax_phase_qnode_native_transform_audit(circuit, np.array([0.2], dtype=float))


def test_phase_jax_registered_qnode_pytree_transform_audit_fails_closed_without_tree_util(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial JAX modules should not pass as PyTree transform lowering."""

    class _MissingTreeUtil(_FakeJAX):
        tree_util: ClassVar[Any] = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_MissingTreeUtil(), np))
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )

    with pytest.raises(RuntimeError, match="tree_util"):
        jax_phase_qnode_pytree_transform_audit(
            circuit,
            {"theta": np.array([0.2], dtype=float)},
        )


def test_phase_jax_registered_qnode_sharding_transform_audit_fails_closed_without_pmap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Partial JAX modules should not pass as PMAP sharding lowering."""

    class _MissingPMAP(_FakeJAX):
        pmap: Any = None

    monkeypatch.setattr(jax_bridge, "_load_jax", lambda: (_MissingPMAP(), np))
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0),),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )

    with pytest.raises(RuntimeError, match="pmap"):
        jax_bridge.jax_phase_qnode_sharding_transform_audit(
            circuit,
            np.array([[0.2]], dtype=float),
        )


def test_phase_jax_registered_qnode_statevector_lowering_matches_scpn_reference() -> None:
    """Native statevector autodiff should match the canonical SCPN reference."""
    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(
            ("h", (0,)),
            ("ry", (1,), 0),
            ("rx", (2,), 1),
            ("cnot", (1, 2)),
            ("crz", (0, 1), 2),
            ("rxx", (0, 2), 3),
            ("rzz", (1, 2), 4),
            ("ccnot", (0, 1, 2)),
        ),
        observable=SparsePauliHamiltonian(
            (
                PauliTerm(0.5, ((0, "z"),)),
                PauliTerm(-0.25, ((1, "x"), (2, "z"))),
                PauliTerm(0.75, ((0, "y"), (2, "y"))),
            )
        ),
    )
    params = np.array([0.21, -0.32, 0.43, -0.54, 0.65], dtype=float)

    result = jax_phase_qnode_value_and_grad(circuit, params, tolerance=2e-5)
    scpn_value = execute_phase_qnode_circuit(circuit, params).value
    scpn_gradient = parameter_shift_phase_qnode_gradient(circuit, params).gradient

    assert isinstance(result, PhaseJAXPhaseQNodeStatevectorResult)
    assert result.passed
    assert result.native_framework_autodiff
    assert not result.host_callback
    assert not result.jitted
    assert result.method == "jax_native_registered_phase_qnode_statevector_value_and_grad"
    np.testing.assert_allclose(result.value, scpn_value, atol=2e-5)
    np.testing.assert_allclose(result.gradient, scpn_gradient, atol=2e-5)
    np.testing.assert_allclose(result.parameter_shift_gradient, scpn_gradient, atol=1e-12)
    np.testing.assert_allclose(np.vdot(result.state, result.state).real, 1.0, atol=2e-5)
    assert result.to_dict()["host_callback"] is False


def test_phase_jax_registered_qnode_statevector_lowering_jits_without_callback() -> None:
    """Jitted statevector lowering should remain native and callback-free."""
    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("ry", (0,), 0), ("cnot", (0, 1)), ("rz", (1,), 1)),
        observable=PauliTerm(1.0, ((0, "z"), (1, "z"))),
    )

    result = jax_phase_qnode_value_and_grad(
        circuit,
        np.array([0.17, -0.23], dtype=float),
        tolerance=2e-5,
        jit=True,
    )

    assert result.passed
    assert result.jit_requested
    assert result.jitted
    assert result.native_framework_autodiff
    assert not result.host_callback


def test_phase_jax_registered_qnode_lowering_covers_gate_and_observable_family() -> None:
    """Native lowering should execute the complete registered gate vocabulary."""
    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    params = np.linspace(0.11, 0.91, 10)
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(
            ("h", (0,)),
            ("x", (1,)),
            ("y", (2,)),
            ("z", (0,)),
            ("s", (1,)),
            ("t", (2,)),
            ("sx", (0,)),
            ("rx", (0,), 0),
            ("ry", (1,), 1),
            ("rz", (2,), 2),
            ("phase", (0,), 3),
            ("cnot", (0, 1)),
            ("cz", (1, 2)),
            ("cy", (2, 0)),
            ("swap", (0, 2)),
            ("ch", (0, 1)),
            ("cs", (1, 2)),
            ("ct", (2, 0)),
            ("crx", (0, 1), 4),
            ("cry", (1, 2), 5),
            ("crz", (2, 0), 6),
            ("rxx", (0, 1), 7),
            ("ryy", (1, 2), 8),
            ("rzz", (0, 2), 9),
            ("ccnot", (0, 1, 2)),
            ("ccz", (0, 1, 2)),
            ("cswap", (0, 1, 2)),
        ),
        observable=SparsePauliHamiltonian(
            (
                PauliTerm(0.5, ((0, "x"),)),
                PauliTerm(-0.25, ((1, "y"), (2, "z"))),
                PauliTerm(0.75, ((0, "z"), (1, "z"), (2, "z"))),
            )
        ),
    )

    result = jax_phase_qnode_value_and_grad(circuit, params, tolerance=5e-6)
    reference = parameter_shift_phase_qnode_gradient(circuit, params)

    assert result.passed
    assert not result.host_callback
    np.testing.assert_allclose(result.value, reference.value, atol=5e-6)
    np.testing.assert_allclose(result.gradient, reference.gradient, atol=5e-6)


def test_phase_jax_registered_qnode_lowering_matches_dense_and_covariance_observables() -> None:
    """Dense and covariance observables should match parameter-shift gradients."""
    if not is_phase_jax_available():
        pytest.skip("JAX optional dependency is not installed")
    dense_params = np.array([0.31, -0.17], dtype=float)
    covariance_params = np.array([0.23, -0.41], dtype=float)
    dense_circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0), ("rz", (0,), 1)),
        observable=DenseHermitianObservable(np.array([[1.0, 0.2], [0.2, -0.5]], dtype=float)),
    )
    covariance_circuit = PhaseQNodeCircuit(
        n_qubits=2,
        operations=(("ry", (0,), 0), ("cnot", (0, 1)), ("rx", (1,), 1)),
        observable=PauliCovarianceObservable(
            PauliTerm(1.0, ((0, "z"),)),
            PauliTerm(1.0, ((1, "x"),)),
        ),
    )

    dense = jax_phase_qnode_value_and_grad(
        dense_circuit,
        dense_params,
        tolerance=5e-6,
    )
    covariance = jax_phase_qnode_value_and_grad(
        covariance_circuit,
        covariance_params,
        tolerance=5e-6,
    )

    assert dense.passed
    assert covariance.passed
    np.testing.assert_allclose(
        dense.gradient,
        parameter_shift_phase_qnode_gradient(dense_circuit, dense_params).gradient,
        atol=5e-6,
    )
    np.testing.assert_allclose(
        covariance.gradient,
        parameter_shift_phase_qnode_gradient(covariance_circuit, covariance_params).gradient,
        atol=5e-6,
    )
