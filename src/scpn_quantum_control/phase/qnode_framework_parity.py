# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase QNode Framework Parity
"""Real optional-framework parity checks for a bounded Phase-QNode circuit."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias, cast

import numpy as np
from numpy.typing import NDArray

from .qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    SparsePauliHamiltonian,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)

FloatArray: TypeAlias = NDArray[np.float64]
FrameworkStatus = Literal["passed", "dependency_missing", "failed"]
ParityScenario = Literal[
    "single_qubit_ry_rx_pauli_z",
    "registered_two_qubit_entangling_statevector",
]
FailureClass = Literal[
    "none",
    "dependency_missing",
    "value_mismatch",
    "gradient_mismatch",
    "runtime_error",
]

CLAIM_BOUNDARY = (
    "local framework parity for one registered Phase-QNode circuit; no provider "
    "execution, no hardware gradients, and no arbitrary simulator-autodiff claim"
)
TWO_QUBIT_CLAIM_BOUNDARY = (
    "local framework parity for one registered two-qubit entangling Phase-QNode "
    "statevector circuit; no provider execution, no hardware gradients, no "
    "finite-shot claim, and no unrestricted arbitrary simulator-autodiff claim"
)


@dataclass(frozen=True)
class PhaseQNodeFrameworkParityRecord:
    """One framework parity row."""

    framework: str
    status: FrameworkStatus
    failure_class: FailureClass
    value: float | None
    gradient: FloatArray | None
    value_abs_error: float | None
    gradient_max_abs_error: float | None
    dtype: str
    device: str
    failure_reason: str

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready parity row evidence."""
        return {
            "framework": self.framework,
            "status": self.status,
            "failure_class": self.failure_class,
            "value": self.value,
            "gradient": None if self.gradient is None else self.gradient.tolist(),
            "value_abs_error": self.value_abs_error,
            "gradient_max_abs_error": self.gradient_max_abs_error,
            "dtype": self.dtype,
            "device": self.device,
            "failure_reason": self.failure_reason,
        }


@dataclass(frozen=True)
class PhaseQNodeFrameworkParitySuiteResult:
    """Parity evidence across SCPN plus installed optional ML frameworks."""

    records: tuple[PhaseQNodeFrameworkParityRecord, ...]
    reference_value: float
    reference_gradient: FloatArray
    tolerance: float
    scenario: ParityScenario = "single_qubit_ry_rx_pauli_z"
    claim_boundary: str = CLAIM_BOUNDARY
    hardware_execution: bool = False

    @property
    def frameworks(self) -> tuple[str, ...]:
        """Return frameworks in evaluation order."""
        return tuple(record.framework for record in self.records)

    @property
    def record_count(self) -> int:
        """Return row count."""
        return len(self.records)

    @property
    def dependency_sparse(self) -> bool:
        """Return true when at least one optional dependency was unavailable."""
        return any(record.status == "dependency_missing" for record in self.records)

    @property
    def passed(self) -> bool:
        """Return true when every installed framework agrees with SCPN."""
        return all(record.status in {"passed", "dependency_missing"} for record in self.records)

    def record_by_framework(self, framework: str) -> PhaseQNodeFrameworkParityRecord:
        """Return a parity row by framework name."""
        for record in self.records:
            if record.framework == framework:
                return record
        raise KeyError(f"unknown framework parity row: {framework}")

    def to_dict(self) -> dict[str, object]:
        """Return JSON-ready suite evidence."""
        return {
            "passed": self.passed,
            "dependency_sparse": self.dependency_sparse,
            "frameworks": list(self.frameworks),
            "scenario": self.scenario,
            "reference_value": self.reference_value,
            "reference_gradient": self.reference_gradient.tolist(),
            "tolerance": self.tolerance,
            "records": [record.to_dict() for record in self.records],
            "claim_boundary": self.claim_boundary,
            "hardware_execution": self.hardware_execution,
        }


def run_phase_qnode_framework_parity_suite(
    *,
    params: FloatArray | None = None,
    tolerance: float = 1.0e-7,
    scenario: ParityScenario = "single_qubit_ry_rx_pauli_z",
) -> PhaseQNodeFrameworkParitySuiteResult:
    """Run parity checks against installed JAX, PyTorch, TensorFlow, and PennyLane."""
    scenario_value = _as_scenario(scenario)
    values = (
        _default_params(scenario_value) if params is None else _as_params(params, scenario_value)
    )
    circuit = _scenario_circuit(scenario_value)
    reference_value = execute_phase_qnode_circuit(circuit, values).value
    reference_gradient = parameter_shift_phase_qnode_gradient(circuit, values).gradient
    runners = (
        ("scpn", lambda: (reference_value, reference_gradient, "float64", "cpu")),
        ("jax", lambda: _run_jax(values, scenario_value)),
        ("torch", lambda: _run_torch(values, scenario_value)),
        ("tensorflow", lambda: _run_tensorflow(values, scenario_value)),
        ("pennylane", lambda: _run_pennylane(values, scenario_value)),
    )
    records = tuple(
        _run_framework_record(
            framework,
            runner,
            reference_value=reference_value,
            reference_gradient=reference_gradient,
            tolerance=tolerance,
        )
        for framework, runner in runners
    )
    return PhaseQNodeFrameworkParitySuiteResult(
        records=records,
        reference_value=reference_value,
        reference_gradient=reference_gradient,
        tolerance=float(tolerance),
        scenario=scenario_value,
        claim_boundary=_scenario_claim_boundary(scenario_value),
    )


def _as_scenario(scenario: str) -> ParityScenario:
    if scenario in {"single_qubit_ry_rx_pauli_z", "registered_two_qubit_entangling_statevector"}:
        return cast(ParityScenario, scenario)
    raise ValueError(f"unsupported Phase-QNode framework parity scenario: {scenario!r}")


def _default_params(scenario: ParityScenario) -> FloatArray:
    if scenario == "single_qubit_ry_rx_pauli_z":
        return np.array([0.37, -0.29], dtype=np.float64)
    return np.array([0.37, -0.29, 0.23], dtype=np.float64)


def _as_params(params: FloatArray, scenario: ParityScenario) -> FloatArray:
    values = np.asarray(params, dtype=np.float64)
    expected_shape = _default_params(scenario).shape
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise ValueError(f"params must be a finite vector with shape {expected_shape}")
    return cast(FloatArray, values.copy())


def _scenario_circuit(scenario: ParityScenario) -> PhaseQNodeCircuit:
    if scenario == "single_qubit_ry_rx_pauli_z":
        return PhaseQNodeCircuit(
            n_qubits=1,
            operations=(("ry", (0,), 0), ("rx", (0,), 1)),
            observable=PauliTerm(1.0, ((0, "z"),)),
        )
    return PhaseQNodeCircuit(
        n_qubits=2,
        operations=(
            ("ry", (0,), 0),
            ("rx", (1,), 1),
            ("cnot", (0, 1)),
            ("rzz", (0, 1), 2),
            ("ry", (1,), 1),
        ),
        observable=SparsePauliHamiltonian(
            (
                PauliTerm(0.7, ((0, "z"),)),
                PauliTerm(-0.3, ((1, "x"),)),
                PauliTerm(0.5, ((0, "z"), (1, "z"))),
            )
        ),
    )


def _scenario_claim_boundary(scenario: ParityScenario) -> str:
    if scenario == "single_qubit_ry_rx_pauli_z":
        return CLAIM_BOUNDARY
    return TWO_QUBIT_CLAIM_BOUNDARY


def _objective_numpy(values: FloatArray) -> float:
    return float(np.cos(values[0]) * np.cos(values[1]))


def _run_framework_record(
    framework: str,
    runner: Any,
    *,
    reference_value: float,
    reference_gradient: FloatArray,
    tolerance: float,
) -> PhaseQNodeFrameworkParityRecord:
    try:
        value, gradient, dtype, device = runner()
    except ImportError as exc:
        return PhaseQNodeFrameworkParityRecord(
            framework=framework,
            status="dependency_missing",
            failure_class="dependency_missing",
            value=None,
            gradient=None,
            value_abs_error=None,
            gradient_max_abs_error=None,
            dtype="unavailable",
            device="unavailable",
            failure_reason=str(exc),
        )
    except Exception as exc:  # pragma: no cover - defensive framework boundary
        return PhaseQNodeFrameworkParityRecord(
            framework=framework,
            status="failed",
            failure_class="runtime_error",
            value=None,
            gradient=None,
            value_abs_error=None,
            gradient_max_abs_error=None,
            dtype="unknown",
            device="unknown",
            failure_reason=str(exc),
        )
    gradient_vector = np.asarray(gradient, dtype=np.float64)
    if gradient_vector.shape != reference_gradient.shape:
        return PhaseQNodeFrameworkParityRecord(
            framework=framework,
            status="failed",
            failure_class="gradient_mismatch",
            value=float(value),
            gradient=gradient_vector,
            value_abs_error=abs(float(value) - reference_value),
            gradient_max_abs_error=None,
            dtype=dtype,
            device=device,
            failure_reason=(
                f"gradient shape {gradient_vector.shape} does not match "
                f"reference {reference_gradient.shape}"
            ),
        )
    value_error = abs(float(value) - reference_value)
    gradient_error = float(np.max(np.abs(gradient_vector - reference_gradient)))
    failure_class: FailureClass = "none"
    if value_error > tolerance:
        failure_class = "value_mismatch"
    elif gradient_error > tolerance:
        failure_class = "gradient_mismatch"
    return PhaseQNodeFrameworkParityRecord(
        framework=framework,
        status="passed" if failure_class == "none" else "failed",
        failure_class=failure_class,
        value=float(value),
        gradient=gradient_vector,
        value_abs_error=value_error,
        gradient_max_abs_error=gradient_error,
        dtype=dtype,
        device=device,
        failure_reason="" if failure_class == "none" else failure_class,
    )


def _run_jax(values: FloatArray, scenario: ParityScenario) -> tuple[float, FloatArray, str, str]:
    jax = importlib.import_module("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = importlib.import_module("jax.numpy")

    def objective(x: Any) -> Any:
        if scenario == "registered_two_qubit_entangling_statevector":
            return _registered_two_qubit_jax_objective(jnp, x)
        return jnp.cos(x[0]) * jnp.cos(x[1])

    array = jnp.asarray(values, dtype=jnp.float64)
    value, gradient = jax.value_and_grad(objective)(array)
    return (
        float(value),
        np.asarray(gradient, dtype=np.float64),
        str(array.dtype),
        str(array.device),
    )


def _run_torch(values: FloatArray, scenario: ParityScenario) -> tuple[float, FloatArray, str, str]:
    torch = importlib.import_module("torch")
    tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    if scenario == "registered_two_qubit_entangling_statevector":
        value = _registered_two_qubit_torch_objective(torch, tensor)
    else:
        value = torch.cos(tensor[0]) * torch.cos(tensor[1])
    value.backward()
    return (
        float(value.detach().cpu().item()),
        np.asarray(tensor.grad.detach().cpu().numpy(), dtype=np.float64),
        str(tensor.dtype).replace("torch.", ""),
        str(tensor.device),
    )


def _run_tensorflow(
    values: FloatArray, scenario: ParityScenario
) -> tuple[float, FloatArray, str, str]:
    tf = importlib.import_module("tensorflow")
    tensor = tf.Variable(values, dtype=tf.float64)
    with tf.GradientTape() as tape:
        if scenario == "registered_two_qubit_entangling_statevector":
            value = _registered_two_qubit_tensorflow_objective(tf, tensor)
        else:
            value = tf.cos(tensor[0]) * tf.cos(tensor[1])
    gradient = tape.gradient(value, tensor)
    return (
        float(value.numpy()),
        np.asarray(gradient.numpy(), dtype=np.float64),
        tensor.dtype.name,
        "cpu",
    )


def _run_pennylane(
    values: FloatArray, scenario: ParityScenario
) -> tuple[float, FloatArray, str, str]:
    qml = importlib.import_module("pennylane")
    pnp = importlib.import_module("pennylane.numpy")
    if scenario == "registered_two_qubit_entangling_statevector":
        device = qml.device("default.qubit", wires=2)

        @qml.qnode(device)
        def two_qubit_circuit(theta: Any) -> Any:
            qml.RY(theta[0], wires=0)
            qml.RX(theta[1], wires=1)
            qml.CNOT(wires=(0, 1))
            qml.IsingZZ(theta[2], wires=(0, 1))
            qml.RY(theta[1], wires=1)
            observable = (
                0.7 * qml.PauliZ(0) - 0.3 * qml.PauliX(1) + 0.5 * qml.PauliZ(0) @ qml.PauliZ(1)
            )
            return qml.expval(observable)

        gradient_fn = qml.grad(two_qubit_circuit)
        theta = pnp.array(values, dtype=np.float64, requires_grad=True)
        value = two_qubit_circuit(theta)
        gradient = gradient_fn(theta)
        return float(value), np.asarray(gradient, dtype=np.float64), "float64", "default.qubit"

    device = qml.device("default.qubit", wires=1)

    @qml.qnode(device)
    def circuit(theta: Any) -> Any:
        qml.RY(theta[0], wires=0)
        qml.RX(theta[1], wires=0)
        return qml.expval(qml.PauliZ(0))

    gradient_fn = qml.grad(circuit)
    theta = pnp.array(values, dtype=np.float64, requires_grad=True)
    value = circuit(theta)
    gradient = gradient_fn(theta)
    return float(value), np.asarray(gradient, dtype=np.float64), "float64", "default.qubit"


def _registered_two_qubit_jax_objective(jnp: Any, theta: Any) -> Any:
    complex_dtype = jnp.complex128
    real_dtype = jnp.float64
    eye = jnp.eye(2, dtype=complex_dtype)
    x = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex_dtype)
    z = jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex_dtype)
    state = jnp.asarray([1.0, 0.0, 0.0, 0.0], dtype=complex_dtype)
    cnot = jnp.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex_dtype,
    )
    state = jnp.kron(_jax_ry(jnp, theta[0]), eye) @ state
    state = jnp.kron(eye, _jax_rx(jnp, theta[1])) @ state
    state = cnot @ state
    state = _jax_rzz(jnp, theta[2]) @ state
    state = jnp.kron(eye, _jax_ry(jnp, theta[1])) @ state
    observable = (
        jnp.asarray(0.7, dtype=real_dtype) * jnp.kron(z, eye)
        - jnp.asarray(0.3, dtype=real_dtype) * jnp.kron(eye, x)
        + jnp.asarray(0.5, dtype=real_dtype) * jnp.kron(z, z)
    )
    return jnp.real(jnp.vdot(state, observable @ state))


def _jax_rx(jnp: Any, theta: Any) -> Any:
    return jnp.asarray(
        [
            [jnp.cos(theta / 2.0), -1.0j * jnp.sin(theta / 2.0)],
            [-1.0j * jnp.sin(theta / 2.0), jnp.cos(theta / 2.0)],
        ],
        dtype=jnp.complex128,
    )


def _jax_ry(jnp: Any, theta: Any) -> Any:
    return jnp.asarray(
        [
            [jnp.cos(theta / 2.0), -jnp.sin(theta / 2.0)],
            [jnp.sin(theta / 2.0), jnp.cos(theta / 2.0)],
        ],
        dtype=jnp.complex128,
    )


def _jax_rzz(jnp: Any, theta: Any) -> Any:
    phases = jnp.asarray(
        [
            jnp.exp(-0.5j * theta),
            jnp.exp(0.5j * theta),
            jnp.exp(0.5j * theta),
            jnp.exp(-0.5j * theta),
        ],
        dtype=jnp.complex128,
    )
    return jnp.diag(phases)


def _registered_two_qubit_torch_objective(torch: Any, theta: Any) -> Any:
    complex_dtype = torch.complex128
    eye = torch.eye(2, dtype=complex_dtype, device=theta.device)
    x = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=complex_dtype, device=theta.device)
    z = torch.tensor([[1.0, 0.0], [0.0, -1.0]], dtype=complex_dtype, device=theta.device)
    state = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=complex_dtype, device=theta.device)
    cnot = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex_dtype,
        device=theta.device,
    )
    state = torch.kron(_torch_ry(torch, theta[0]), eye) @ state
    state = torch.kron(eye, _torch_rx(torch, theta[1])) @ state
    state = cnot @ state
    state = _torch_rzz(torch, theta[2]) @ state
    state = torch.kron(eye, _torch_ry(torch, theta[1])) @ state
    observable = 0.7 * torch.kron(z, eye) - 0.3 * torch.kron(eye, x) + 0.5 * torch.kron(z, z)
    return torch.vdot(state, observable @ state).real


def _torch_rx(torch: Any, theta: Any) -> Any:
    zero = torch.zeros((), dtype=torch.float64, device=theta.device)
    return torch.stack(
        (
            torch.stack((torch.cos(theta / 2.0), torch.complex(zero, -torch.sin(theta / 2.0)))),
            torch.stack((torch.complex(zero, -torch.sin(theta / 2.0)), torch.cos(theta / 2.0))),
        )
    ).to(torch.complex128)


def _torch_ry(torch: Any, theta: Any) -> Any:
    return torch.stack(
        (
            torch.stack((torch.cos(theta / 2.0), -torch.sin(theta / 2.0))),
            torch.stack((torch.sin(theta / 2.0), torch.cos(theta / 2.0))),
        )
    ).to(torch.complex128)


def _torch_rzz(torch: Any, theta: Any) -> Any:
    phases = torch.stack(
        (
            torch.exp(torch.complex(torch.zeros_like(theta), -0.5 * theta)),
            torch.exp(torch.complex(torch.zeros_like(theta), 0.5 * theta)),
            torch.exp(torch.complex(torch.zeros_like(theta), 0.5 * theta)),
            torch.exp(torch.complex(torch.zeros_like(theta), -0.5 * theta)),
        )
    )
    return torch.diag(phases.to(torch.complex128))


def _registered_two_qubit_tensorflow_objective(tf: Any, theta: Any) -> Any:
    complex_dtype = tf.complex128
    eye = tf.eye(2, dtype=complex_dtype)
    x = tf.constant([[0.0, 1.0], [1.0, 0.0]], dtype=complex_dtype)
    z = tf.constant([[1.0, 0.0], [0.0, -1.0]], dtype=complex_dtype)
    state = tf.constant([1.0, 0.0, 0.0, 0.0], dtype=complex_dtype)
    cnot = tf.constant(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
        dtype=complex_dtype,
    )
    state = tf.linalg.matvec(_tf_kron(tf, _tf_ry(tf, theta[0]), eye), state)
    state = tf.linalg.matvec(_tf_kron(tf, eye, _tf_rx(tf, theta[1])), state)
    state = tf.linalg.matvec(cnot, state)
    state = tf.linalg.matvec(_tf_rzz(tf, theta[2]), state)
    state = tf.linalg.matvec(_tf_kron(tf, eye, _tf_ry(tf, theta[1])), state)
    observable = (
        tf.cast(0.7, complex_dtype) * _tf_kron(tf, z, eye)
        - tf.cast(0.3, complex_dtype) * _tf_kron(tf, eye, x)
        + tf.cast(0.5, complex_dtype) * _tf_kron(tf, z, z)
    )
    return tf.math.real(tf.tensordot(tf.math.conj(state), tf.linalg.matvec(observable, state), 1))


def _tf_kron(tf: Any, left: Any, right: Any) -> Any:
    left_shape = tf.shape(left)
    right_shape = tf.shape(right)
    product = left[:, None, :, None] * right[None, :, None, :]
    return tf.reshape(product, (left_shape[0] * right_shape[0], left_shape[1] * right_shape[1]))


def _tf_rx(tf: Any, theta: Any) -> Any:
    zero = tf.zeros((), dtype=tf.float64)
    cosine = tf.complex(tf.cos(theta / 2.0), zero)
    minus_i_sine = tf.complex(zero, -tf.sin(theta / 2.0))
    return tf.stack(
        (
            tf.stack((cosine, minus_i_sine)),
            tf.stack((minus_i_sine, cosine)),
        )
    )


def _tf_ry(tf: Any, theta: Any) -> Any:
    zero = tf.zeros((), dtype=tf.float64)
    cosine = tf.complex(tf.cos(theta / 2.0), zero)
    sine = tf.complex(tf.sin(theta / 2.0), zero)
    return tf.stack(
        (
            tf.stack((cosine, -sine)),
            tf.stack((sine, cosine)),
        )
    )


def _tf_rzz(tf: Any, theta: Any) -> Any:
    phases = tf.stack(
        (
            tf.exp(tf.complex(tf.zeros_like(theta), -0.5 * theta)),
            tf.exp(tf.complex(tf.zeros_like(theta), 0.5 * theta)),
            tf.exp(tf.complex(tf.zeros_like(theta), 0.5 * theta)),
            tf.exp(tf.complex(tf.zeros_like(theta), -0.5 * theta)),
        )
    )
    return tf.linalg.diag(tf.cast(phases, tf.complex128))


__all__ = [
    "PhaseQNodeFrameworkParityRecord",
    "PhaseQNodeFrameworkParitySuiteResult",
    "ParityScenario",
    "run_phase_qnode_framework_parity_suite",
]
