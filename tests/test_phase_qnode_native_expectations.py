# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native observable arithmetic tests
"""Exercise complex quantum moments, real native derivatives and public consumers."""

from __future__ import annotations

import importlib
import math
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import (
    DenseHermitianObservable,
    PauliCovarianceObservable,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    SparsePauliHamiltonian,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)
from scpn_quantum_control.phase_qnode_native_expectations import (
    covariance_from_moments,
    real_quadratic_form,
    real_symmetrized_overlap,
    weighted_complex_overlap,
    weighted_real_overlap,
)

pytestmark = pytest.mark.framework_imports


def _native_moments(namespace: Any, state: Any, matrix: Any, x: Any, y: Any) -> tuple[Any, ...]:
    dense = real_quadratic_form(namespace, state, matrix)
    left_mean = weighted_real_overlap(namespace, state, x @ state, -0.3)
    right_mean = weighted_real_overlap(namespace, state, y @ state, 0.8)
    left_right = weighted_complex_overlap(namespace, state, x @ (y @ state), -0.3, 0.8)
    right_left = weighted_complex_overlap(namespace, state, y @ (x @ state), 0.8, -0.3)
    symmetrized = real_symmetrized_overlap(namespace, left_right, right_left)
    covariance = covariance_from_moments(symmetrized, left_mean, right_mean)
    return dense, left_mean, right_mean, left_right, right_left, symmetrized, covariance


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_native_moments_keep_complex_products_and_second_derivatives(backend: str) -> None:
    """Check moments and dense Hessians against independent trigonometric formulas.

    Parameters
    ----------
    backend
        Actual native CPU runtime used for the complex state and differentiation.

    """
    angle, phase = 0.37, 0.31
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    y = np.array([[0.0, -1j], [1j, 0.0]], dtype=np.complex128)
    z = np.diag([1.0, -1.0])
    matrix = 0.3 * np.eye(2) + 0.4 * x - 0.7 * y + 0.2 * z
    if backend == "jax":
        jax = importlib.import_module("jax")
        jax.config.update("jax_enable_x64", True)
        namespace = importlib.import_module("jax.numpy")
        xn, yn, mn = (namespace.asarray(a) for a in (x, y, matrix))

        def evaluate(theta: Any) -> tuple[Any, ...]:
            """Build a traced complex state and evaluate supplied operator moments."""
            state = namespace.asarray(
                [namespace.cos(theta), namespace.exp(1j * phase) * namespace.sin(theta)]
            )
            return _native_moments(namespace, state, mn, xn, yn)

        def objective(theta: Any) -> Any:
            """Differentiate the actual dense observable expectation."""
            return evaluate(theta)[0]

        actual = jax.jit(evaluate)(angle)
        gradient = jax.jit(jax.grad(objective))(angle)
        hessian = jax.jit(jax.grad(jax.grad(objective)))(angle)
    else:
        namespace = importlib.import_module("torch")
        xn, yn, mn = (namespace.as_tensor(a) for a in (x, y, matrix))
        theta = namespace.tensor(angle, dtype=namespace.float64, requires_grad=True)
        state = namespace.stack(
            (
                namespace.cos(theta).to(namespace.complex128),
                namespace.exp(namespace.tensor(1j * phase, dtype=namespace.complex128))
                * namespace.sin(theta),
            )
        )
        moments = _native_moments(namespace, state, mn, xn, yn)
        gradient_obj = namespace.autograd.grad(moments[0], theta, create_graph=True)[0]
        hessian_obj = namespace.autograd.grad(gradient_obj, theta)[0]
        actual = tuple(a.detach() for a in moments)
        gradient, hessian = gradient_obj.detach(), hessian_obj.detach()
    sx = math.sin(2 * angle) * math.cos(phase)
    sy = math.sin(2 * angle) * math.sin(phase)
    sz = math.cos(2 * angle)
    amplitude = 0.4 * math.cos(phase) - 0.7 * math.sin(phase)
    expected = [
        0.3 + 0.4 * sx - 0.7 * sy + 0.2 * sz,
        -0.3 * sx,
        0.8 * sy,
        -0.24j * sz,
        0.24j * sz,
        0.0,
        0.24 * sx * sy,
    ]
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(
        gradient, 2 * amplitude * sz - 0.4 * math.sin(2 * angle), rtol=0.0, atol=2e-15
    )
    np.testing.assert_allclose(
        hessian, -4 * amplitude * math.sin(2 * angle) - 0.8 * sz, rtol=0.0, atol=2e-15
    )
    assert abs(complex(actual[3]).imag) > 0.1


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize("kind", ["term", "zero", "sparse", "dense", "commuting", "anticommuting"])
def test_original_public_observable_routes_match_statevector_and_parameter_shift(
    backend: str, kind: str
) -> None:
    """Execute original public observable selection and its lower moment arithmetic.

    Parameters
    ----------
    backend
        Native registered-circuit bridge.
    kind
        Original observable family, including commuting and anticommuting products.

    """
    observable: (
        PauliTerm | SparsePauliHamiltonian | DenseHermitianObservable | PauliCovarianceObservable
    )
    if kind == "term":
        observable = PauliTerm(-0.7, ((2, "y"), (0, "x")))
    elif kind == "zero":
        observable = PauliTerm(0.0, ((1, "z"),))
    elif kind == "sparse":
        observable = SparsePauliHamiltonian(
            (
                PauliTerm(0.2, ((2, "z"),)),
                PauliTerm(-0.7, ((0, "x"),)),
                PauliTerm(0.4, ((1, "y"), (2, "z"))),
            )
        )
    elif kind == "dense":
        numbers = np.arange(64, dtype=np.float64).reshape(8, 8)
        matrix = numbers + numbers.T + 1j * (numbers - numbers.T)
        observable = DenseHermitianObservable(np.asarray(matrix / 80, dtype=np.complex128))
    elif kind == "commuting":
        observable = PauliCovarianceObservable(
            PauliTerm(0.4, ((0, "z"),)), PauliTerm(-0.6, ((1, "z"),))
        )
    else:
        observable = PauliCovarianceObservable(
            PauliTerm(0.4, ((0, "x"), (2, "y"))),
            PauliTerm(-0.6, ((0, "z"), (1, "x"))),
        )
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("ry", (2,), parameter_index=2),
            PhaseQNodeOperation("cnot", (0, 2)),
            PhaseQNodeOperation("cy", (2, 1)),
        ),
        observable=observable,
    )
    params = np.array([0.37, -0.21, 0.61])
    expected = execute_phase_qnode_circuit(circuit, params)
    reference = parameter_shift_phase_qnode_gradient(circuit, params)
    bridge = importlib.import_module("scpn_quantum_control.phase." + backend + "_bridge")
    actual = getattr(bridge, backend + "_phase_qnode_value_and_grad")(circuit, params)
    assert actual.passed
    np.testing.assert_allclose(actual.state, expected.state, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual.value, expected.value, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual.gradient, reference.gradient, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_dense_matrix_conversion_and_native_callable_resolution_remain_ordered(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rebind real SDK operations during actual matrix conversion and conjugate dot.

    Parameters
    ----------
    backend
        Original native registered-circuit route.
    monkeypatch
        Restore runtime operations and original compatibility globals afterwards.

    """
    observable = DenseHermitianObservable(np.array([[0.2, 0.3j], [-0.3j, -0.7]]))
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(PhaseQNodeOperation("ry", (0,), parameter_index=0),),
        observable=observable,
    )
    owner = importlib.import_module("scpn_quantum_control.phase." + backend + "_qnode_transforms")
    namespace = importlib.import_module("jax.numpy" if backend == "jax" else "torch")
    original_real, original_vdot = namespace.real, namespace.vdot
    events: list[str] = []

    def late_real(value: Any) -> Any:
        """Preserve real projection while recording a late attribute lookup."""
        events.append("late-real")
        return original_real(value)

    def active_real(value: Any) -> Any:
        """Run the exact real projection resolved before the conjugate dot."""
        events.append("active-real")
        return original_real(value)

    def active_vdot(left: Any, right: Any) -> Any:
        """Rebind real projection during genuine native dot execution."""
        events.append("vdot")
        monkeypatch.setattr(namespace, "real", late_real)
        return original_vdot(left, right)

    def observe_conversion() -> None:
        events.append("matrix")
        monkeypatch.setattr(namespace, "real", active_real)
        monkeypatch.setattr(namespace, "vdot", active_vdot)

    if backend == "jax":
        original_array = namespace.asarray

        def array(values: object, *args: Any, **kwargs: Any) -> Any:
            """Keep the real JAX constructor and observe the actual dense matrix."""
            result = original_array(values, *args, **kwargs)
            if values is observable.matrix:
                observe_conversion()
            return result

        monkeypatch.setattr(namespace, "asarray", array)
    else:
        original_tensor = owner._torch_complex_tensor

        def tensor(runtime: Any, values: object) -> Any:
            """Keep the original complex constructor and observe the dense matrix."""
            result = original_tensor(runtime, values)
            if values is observable.matrix:
                observe_conversion()
            return result

        monkeypatch.setitem(vars(owner), "_torch_complex_tensor", tensor)
    bridge = importlib.import_module("scpn_quantum_control.phase." + backend + "_bridge")
    actual = getattr(bridge, backend + "_phase_qnode_value_and_grad")(circuit, np.array([0.37]))
    assert actual.passed
    first = events.index("matrix")
    assert events[first : first + 3] == ["matrix", "vdot", "active-real"]


def test_original_pauli_observable_rejects_empty_identity_factors() -> None:
    """Keep the original constructor's explicit refusal of an empty Pauli product."""
    with pytest.raises(ValueError, match="PauliTerm factors must be non-empty"):
        PauliTerm(0.4, ())
