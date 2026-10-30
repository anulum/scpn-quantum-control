# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native Phase-QNode statevector engine tests
"""Check the PyTorch and JAX statevector engines against the NumPy executor."""

from __future__ import annotations

from typing import Any, cast

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
from scpn_quantum_control.phase_qnode_native_statevector import (
    jax_statevector_kernels,
    jax_statevector_value_and_state,
    torch_statevector_kernels,
    torch_statevector_value_and_state,
)
from scpn_quantum_control.phase_qnode_statevector_engine import (
    observable_expectation,
    statevector_value_and_state,
)

_PARAMS = np.array([0.31, -0.72, 1.05, 0.44, -0.18, 0.93, 0.27], dtype=np.float64)

_OPERATIONS = (
    PhaseQNodeOperation("h", (0,)),
    PhaseQNodeOperation("rx", (0,), 0),
    PhaseQNodeOperation("ry", (1,), 1),
    PhaseQNodeOperation("cnot", (0, 1)),
    PhaseQNodeOperation("rz", (2,), 2),
    PhaseQNodeOperation("rxx", (1, 2), 3),
    PhaseQNodeOperation("ccnot", (0, 1, 2)),
    PhaseQNodeOperation("ryy", (0, 2), 4),
    PhaseQNodeOperation("sx", (1,)),
    PhaseQNodeOperation("rzz", (0, 1), 5),
    PhaseQNodeOperation("phase", (2,), 6),
    PhaseQNodeOperation("cswap", (2, 0, 1)),
)

_DENSE = (
    np.diag(np.arange(8, dtype=np.float64)) + 0.25 * (np.eye(8, k=1) + np.eye(8, k=-1))
).astype(np.complex128)

_Observable = (
    PauliTerm | SparsePauliHamiltonian | PauliCovarianceObservable | DenseHermitianObservable
)

_OBSERVABLES: dict[str, _Observable] = {
    "pauli_product": PauliTerm(0.7, ((0, "z"), (2, "x"))),
    "hamiltonian": SparsePauliHamiltonian(
        (PauliTerm(1.0, ((0, "z"), (1, "z"))), PauliTerm(-0.4, ((2, "y"),)))
    ),
    "covariance": PauliCovarianceObservable(
        PauliTerm(1.0, ((0, "x"),)), PauliTerm(0.5, ((1, "y"), (2, "z")))
    ),
    "dense": DenseHermitianObservable(_DENSE),
}


def _circuit(observable: _Observable) -> PhaseQNodeCircuit:
    return PhaseQNodeCircuit(3, _OPERATIONS, observable)


def _jax_numpy() -> Any:
    jnp = pytest.importorskip("jax.numpy")
    if jnp.asarray(0.0).dtype != np.float64:
        pytest.skip("JAX statevector parity requires JAX_ENABLE_X64=true")
    return jnp


@pytest.mark.parametrize("name", sorted(_OBSERVABLES))
def test_torch_engine_matches_executor_value_state_and_parameter_shift(name: str) -> None:
    """Match the NumPy executor's value and state and the parameter-shift gradient.

    Parameters
    ----------
    name
        Observable family under test.

    """
    torch = pytest.importorskip("torch")
    circuit = _circuit(_OBSERVABLES[name])
    reference = execute_phase_qnode_circuit(circuit, _PARAMS)
    params = torch.tensor(_PARAMS, dtype=torch.float64, requires_grad=True)
    value, state = torch_statevector_value_and_state(torch, circuit, params)
    assert state.dtype == torch.complex128
    np.testing.assert_allclose(state.detach().numpy(), reference.state, rtol=0, atol=1e-12)
    assert abs(float(value.detach()) - reference.value) < 1e-12
    value.backward()
    shift = parameter_shift_phase_qnode_gradient(circuit, _PARAMS)
    np.testing.assert_allclose(params.grad.numpy(), shift.gradient, rtol=0, atol=1e-10)


@pytest.mark.parametrize("name", sorted(_OBSERVABLES))
def test_jax_engine_matches_executor_value_state_and_parameter_shift(name: str) -> None:
    """Match the NumPy executor and the parameter-shift gradient under ``jax.grad``.

    Parameters
    ----------
    name
        Observable family under test.

    """
    jnp = _jax_numpy()
    jax = pytest.importorskip("jax")
    circuit = _circuit(_OBSERVABLES[name])
    reference = execute_phase_qnode_circuit(circuit, _PARAMS)
    value, state = jax_statevector_value_and_state(jnp, circuit, jnp.asarray(_PARAMS))
    assert state.dtype == np.complex128
    np.testing.assert_allclose(np.asarray(state), reference.state, rtol=0, atol=1e-12)
    assert abs(float(value) - reference.value) < 1e-12
    gradient = jax.grad(lambda angles: jax_statevector_value_and_state(jnp, circuit, angles)[0])(
        jnp.asarray(_PARAMS)
    )
    shift = parameter_shift_phase_qnode_gradient(circuit, _PARAMS)
    np.testing.assert_allclose(np.asarray(gradient), shift.gradient, rtol=0, atol=1e-10)


def test_torch_engine_uses_a_supplied_dense_matrix() -> None:
    """Measure a dense observable through a caller's pre-converted native matrix."""
    torch = pytest.importorskip("torch")
    circuit = _circuit(_OBSERVABLES["dense"])
    reference = execute_phase_qnode_circuit(circuit, _PARAMS)
    supplied = torch.as_tensor(2.0 * _DENSE)
    value, _state = torch_statevector_value_and_state(
        torch, circuit, torch.as_tensor(_PARAMS), dense_matrix=supplied
    )
    assert abs(float(value) - 2.0 * reference.value) < 1e-12


@pytest.mark.parametrize("backend", ["torch", "jax"])
def test_engines_refuse_unregistered_pauli_labels_and_observables(backend: str) -> None:
    """Refuse labels and observables outside the registered family.

    Parameters
    ----------
    backend
        Native runtime under test.

    """
    if backend == "torch":
        runtime: Any = pytest.importorskip("torch")
        kernels = torch_statevector_kernels(runtime)
        state = runtime.zeros(2, dtype=runtime.complex128)
        label = "PyTorch"
    else:
        runtime = _jax_numpy()
        kernels = jax_statevector_kernels(runtime)
        state = runtime.zeros(2, dtype=runtime.complex128)
        label = "JAX"
    with pytest.raises(ValueError, match=f"unsupported {label} Pauli label: w"):
        kernels.pauli_operator("w")
    with pytest.raises(ValueError, match=f"unsupported {label} Phase-QNode observable: bogus"):
        observable_expectation(kernels, state, 1, "bogus")


class _RecordingCircuit:
    """Circuit stand-in recording when the engine reads each operand."""

    def __init__(self, circuit: PhaseQNodeCircuit, log: list[str]) -> None:
        """Wrap ``circuit`` and append every operand read to ``log``.

        Parameters
        ----------
        circuit
            Admitted circuit whose operands are returned.
        log
            Shared event log.

        """
        self._circuit = circuit
        self._log = log

    def __getattr__(self, name: str) -> Any:
        """Record and return one circuit operand.

        Parameters
        ----------
        name
            Operand name.

        Returns
        -------
        Any
            The wrapped circuit's operand.

        """
        self._log.append(f"circuit.{name}")
        return getattr(self._circuit, name)


class _RecordingNamespace:
    """PyTorch namespace stand-in recording the zero-state allocation lookup."""

    def __init__(self, namespace: Any, log: list[str]) -> None:
        """Wrap ``namespace`` and append the ``zeros`` lookup to ``log``.

        Parameters
        ----------
        namespace
            Real PyTorch module.
        log
            Shared event log.

        """
        self._namespace = namespace
        self._log = log

    def __getattr__(self, name: str) -> Any:
        """Record the ``zeros`` lookup and return the real attribute.

        Parameters
        ----------
        name
            Attribute name.

        Returns
        -------
        Any
            The real PyTorch attribute.

        """
        if name == "zeros":
            self._log.append("namespace.zeros")
        return getattr(self._namespace, name)


def test_circuit_operands_are_read_after_the_zero_state_allocation() -> None:
    """Defer every circuit operand read until the native zeros callable is resolved."""
    torch = pytest.importorskip("torch")
    log: list[str] = []
    circuit = _circuit(_OBSERVABLES["pauli_product"])
    kernels = torch_statevector_kernels(_RecordingNamespace(torch, log))
    value, _state = statevector_value_and_state(
        kernels,
        cast(PhaseQNodeCircuit, _RecordingCircuit(circuit, log)),
        torch.as_tensor(_PARAMS),
    )
    assert log[:3] == ["namespace.zeros", "circuit.n_qubits", "circuit.operations"]
    assert log[-2:] == ["circuit.n_qubits", "circuit.observable"]
    assert abs(float(value) - execute_phase_qnode_circuit(circuit, _PARAMS).value) < 1e-12
