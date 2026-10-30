# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native registered gate numerical tests
"""Exercise every registered native gate and observable numerical factory behavior."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.phase.qnode_circuit import (
    DenseHermitianObservable,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)
from scpn_quantum_control.phase_qnode_gate_numerics import pauli_rotation
from scpn_quantum_control.phase_qnode_native_operators import (
    phase_operator_entries,
    two_qubit_operator_entries,
)

pytestmark = pytest.mark.framework_imports


@pytest.mark.parametrize("backend", ["jax", "torch"])
@pytest.mark.parametrize(
    "gate,width,parameterized",
    [
        ("h", 1, False),
        ("x", 1, False),
        ("y", 1, False),
        ("z", 1, False),
        ("s", 1, False),
        ("t", 1, False),
        ("sx", 1, False),
        ("rx", 1, True),
        ("ry", 1, True),
        ("rz", 1, True),
        ("phase", 1, True),
        ("cnot", 2, False),
        ("cz", 2, False),
        ("cy", 2, False),
        ("swap", 2, False),
        ("ch", 2, False),
        ("cs", 2, False),
        ("ct", 2, False),
        ("ccnot", 3, False),
        ("ccz", 3, False),
        ("cswap", 3, False),
        ("crx", 2, True),
        ("cry", 2, True),
        ("crz", 2, True),
        ("rxx", 2, True),
        ("ryy", 2, True),
        ("rzz", 2, True),
    ],
)
def test_every_original_registered_gate_retains_state_and_gradient(
    backend: str, gate: str, width: int, parameterized: bool
) -> None:
    """Execute every original gate on reordered wires through the real public bridge.

    Parameters
    ----------
    backend
        Actual CPU framework used by the original registered-circuit entry point.
    gate
        Original gate name, with no added alias or changed support policy.
    width
        Actual gate width used to select reordered circuit wires.
    parameterized
        Whether the supplied fourth parameter controls this gate.

    """
    qubits = {1: (2,), 2: (2, 0), 3: (2, 0, 1)}[width]
    values = np.arange(64, dtype=np.float64).reshape(8, 8)
    observable = DenseHermitianObservable(
        np.asarray((values + values.T + 1j * (values - values.T)) / 80.0, dtype=np.complex128)
    )
    circuit = PhaseQNodeCircuit(
        n_qubits=3,
        operations=(
            PhaseQNodeOperation("ry", (0,), parameter_index=0),
            PhaseQNodeOperation("rx", (1,), parameter_index=1),
            PhaseQNodeOperation("ry", (2,), parameter_index=2),
            PhaseQNodeOperation(gate, qubits, parameter_index=3 if parameterized else None),
        ),
        observable=observable,
    )
    params = np.array([0.37, -0.21, 0.61, -0.43])
    expected = execute_phase_qnode_circuit(circuit, params)
    reference = parameter_shift_phase_qnode_gradient(circuit, params)
    bridge = importlib.import_module("scpn_quantum_control.phase." + backend + "_bridge")
    kwargs = {"jit": True} if backend == "jax" else {}
    actual = getattr(bridge, backend + "_phase_qnode_value_and_grad")(
        circuit, params, tolerance=1e-10, **kwargs
    )
    assert actual.passed
    np.testing.assert_allclose(actual.state, expected.state, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual.value, expected.value, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(actual.gradient, reference.gradient, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_native_rotation_retains_second_derivatives_and_graph_capture(backend: str) -> None:
    """Compare actual native transformations and graph capture with analytic derivatives.

    Parameters
    ----------
    backend
        Actual native differentiation and capture runtime.

    """
    angle = 0.37
    namespace = importlib.import_module("jax.numpy" if backend == "jax" else "torch")
    if backend == "jax":
        runtime = importlib.import_module("jax")
        runtime.config.update("jax_enable_x64", True)
        array = namespace.asarray
    else:
        runtime = namespace
        array = namespace.as_tensor

    def objective(theta: Any) -> Any:
        """Read the off-diagonal amplitude of the actual supplied Pauli rotation."""
        matrix = pauli_rotation(
            namespace,
            theta,
            array(1j, dtype=namespace.complex128),
            identity=lambda: namespace.eye(2, dtype=namespace.complex128),
            operator=lambda: array([[0.0, 1.0], [1.0, 0.0]], dtype=namespace.complex128),
        )
        return namespace.real(1j * matrix[0, 1])

    if backend == "jax":
        value = runtime.jit(objective)(angle)
        gradient = runtime.jit(runtime.grad(objective))(angle)
        hessian = runtime.jit(runtime.grad(runtime.grad(objective)))(angle)
    else:
        theta = runtime.tensor(angle, dtype=runtime.float64)
        value = runtime.compile(objective, backend="eager", fullgraph=True)(theta)
        gradient = runtime.func.jacrev(objective)(theta)
        hessian = runtime.func.jacrev(runtime.func.jacrev(objective))(theta)
    np.testing.assert_allclose(
        [value, gradient, hessian],
        [math.sin(angle / 2), 0.5 * math.cos(angle / 2), -0.25 * math.sin(angle / 2)],
        rtol=0.0,
        atol=2e-15,
    )


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_original_t_gate_keeps_late_numpy_namespace_reads(
    backend: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Rebind the original NumPy namespace between exp resolution and pi lookup.

    Parameters
    ----------
    backend
        Actual native gate construction runtime.
    monkeypatch
        Restore the original compatibility module's numerical namespace.

    """
    owner = importlib.import_module("scpn_quantum_control.phase." + backend + "_qnode_transforms")
    namespace = importlib.import_module("jax.numpy" if backend == "jax" else "torch")
    events: list[str] = []

    class PiNamespace:
        """Supply a real numerical pi value after the namespace rebinding."""

        @property
        def pi(self) -> float:
            """Changed scalar constant, read without replacing native arithmetic."""
            events.append("pi")
            return float(np.pi / 2.0)

    class ExpNamespace:
        """Return the genuine NumPy exponential while changing the next namespace read."""

        @property
        def exp(self) -> Callable[[complex], Any]:
            """Resolve the real exponential before the subsequent pi access."""
            events.append("exp")
            monkeypatch.setitem(vars(owner), "np", PiNamespace())
            return np.exp

    monkeypatch.setitem(vars(owner), "np", ExpNamespace())
    actual = getattr(owner, "_" + backend + "_gate_matrix")(namespace, "t", 0.37)
    np.testing.assert_allclose(
        np.asarray(actual), np.diag([1.0, np.exp(1j * np.pi / 8.0)]), rtol=0.0, atol=2e-15
    )
    assert events == ["exp", "pi"]


@pytest.mark.parametrize("backend", ["jax", "torch"])
def test_original_unknown_gate_error_remains_at_its_original_path(backend: str) -> None:
    """Keep the original framework-specific unsupported-gate error contracts.

    Parameters
    ----------
    backend
        Actual native numerical runtime used by the original helper alias.

    """
    owner = importlib.import_module("scpn_quantum_control.phase." + backend + "_qnode_transforms")
    namespace = importlib.import_module("jax.numpy" if backend == "jax" else "torch")
    error = KeyError if backend == "jax" else ValueError
    with pytest.raises(error, match="unknown_gate"):
        getattr(owner, "_" + backend + "_gate_matrix")(namespace, "unknown_gate", 0.37)


def test_fixed_gate_entries_keep_basis_action_phase_and_fresh_rows() -> None:
    """Check fixed operators through basis action, scalar identity and mutation isolation."""
    expected: dict[str, list[tuple[int, int | complex]]] = {
        "cnot": [(0, 1), (1, 1), (3, 1), (2, 1)],
        "cy": [(0, 1), (1, 1), (3, 1j), (2, -1j)],
        "swap": [(0, 1), (2, 1), (1, 1), (3, 1)],
    }
    for name, columns in expected.items():
        matrix = np.asarray(two_qubit_operator_entries(name))
        for source, (destination, amplitude) in enumerate(columns):
            vector = np.zeros(4, dtype=np.complex128)
            vector[destination] = amplitude
            np.testing.assert_array_equal(matrix[:, source], vector)
        rows = two_qubit_operator_entries(name)
        rows[0][0] = 7
        np.testing.assert_array_equal(two_qubit_operator_entries(name), matrix)
    phase = complex(0.2, 0.3)
    first = phase_operator_entries(phase)
    assert first[1][1] is phase
    first[0][0] = 8.0
    np.testing.assert_array_equal(phase_operator_entries(phase), [[1.0, 0.0], [0.0, phase]])
    with pytest.raises(ValueError, match="unsupported fixed two-qubit gate: phase"):
        two_qubit_operator_entries("phase")
