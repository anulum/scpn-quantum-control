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
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)

FloatArray: TypeAlias = NDArray[np.float64]
FrameworkStatus = Literal["passed", "dependency_missing", "failed"]
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
) -> PhaseQNodeFrameworkParitySuiteResult:
    """Run parity checks against installed JAX, PyTorch, TensorFlow, and PennyLane."""
    values = np.array([0.37, -0.29], dtype=np.float64) if params is None else _as_params(params)
    circuit = PhaseQNodeCircuit(
        n_qubits=1,
        operations=(("ry", (0,), 0), ("rx", (0,), 1)),
        observable=PauliTerm(1.0, ((0, "z"),)),
    )
    reference_value = execute_phase_qnode_circuit(circuit, values).value
    reference_gradient = parameter_shift_phase_qnode_gradient(circuit, values).gradient
    runners = (
        ("scpn", lambda: (reference_value, reference_gradient, "float64", "cpu")),
        ("jax", lambda: _run_jax(values)),
        ("torch", lambda: _run_torch(values)),
        ("tensorflow", lambda: _run_tensorflow(values)),
        ("pennylane", lambda: _run_pennylane(values)),
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
    )


def _as_params(params: FloatArray) -> FloatArray:
    values = np.asarray(params, dtype=np.float64)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError("params must be a finite length-2 vector")
    return cast(FloatArray, values.copy())


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


def _run_jax(values: FloatArray) -> tuple[float, FloatArray, str, str]:
    jax = importlib.import_module("jax")
    jax.config.update("jax_enable_x64", True)
    jnp = importlib.import_module("jax.numpy")

    def objective(x: Any) -> Any:
        return jnp.cos(x[0]) * jnp.cos(x[1])

    array = jnp.asarray(values, dtype=jnp.float64)
    value, gradient = jax.value_and_grad(objective)(array)
    return (
        float(value),
        np.asarray(gradient, dtype=np.float64),
        str(array.dtype),
        str(array.device),
    )


def _run_torch(values: FloatArray) -> tuple[float, FloatArray, str, str]:
    torch = importlib.import_module("torch")
    tensor = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    value = torch.cos(tensor[0]) * torch.cos(tensor[1])
    value.backward()
    return (
        float(value.detach().cpu().item()),
        np.asarray(tensor.grad.detach().cpu().numpy(), dtype=np.float64),
        str(tensor.dtype).replace("torch.", ""),
        str(tensor.device),
    )


def _run_tensorflow(values: FloatArray) -> tuple[float, FloatArray, str, str]:
    tf = importlib.import_module("tensorflow")
    tensor = tf.Variable(values, dtype=tf.float64)
    with tf.GradientTape() as tape:
        value = tf.cos(tensor[0]) * tf.cos(tensor[1])
    gradient = tape.gradient(value, tensor)
    return (
        float(value.numpy()),
        np.asarray(gradient.numpy(), dtype=np.float64),
        tensor.dtype.name,
        "cpu",
    )


def _run_pennylane(values: FloatArray) -> tuple[float, FloatArray, str, str]:
    qml = importlib.import_module("pennylane")
    pnp = importlib.import_module("pennylane.numpy")
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


__all__ = [
    "PhaseQNodeFrameworkParityRecord",
    "PhaseQNodeFrameworkParitySuiteResult",
    "run_phase_qnode_framework_parity_suite",
]
