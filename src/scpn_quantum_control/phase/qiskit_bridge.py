# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase Qiskit Bridge
"""Qiskit parameter-shift circuit generation and local gradient execution."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.quantum_info import Statevector

from ..differentiable import ParameterShiftRule

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class QiskitParameterShiftRecord:
    """One Qiskit plus/minus shifted-circuit pair."""

    parameter_index: int
    parameter_name: str
    plus_values: FloatArray
    minus_values: FloatArray
    plus_circuit: QuantumCircuit
    minus_circuit: QuantumCircuit

    def __post_init__(self) -> None:
        if isinstance(self.parameter_index, bool) or self.parameter_index < 0:
            raise ValueError("parameter_index must be a non-negative integer")
        if not self.parameter_name:
            raise ValueError("parameter_name must be non-empty")
        plus_values = _as_finite_vector("plus_values", self.plus_values)
        minus_values = _as_finite_vector("minus_values", self.minus_values)
        if plus_values.shape != minus_values.shape:
            raise ValueError("plus_values and minus_values must have matching shapes")
        if self.plus_circuit.num_parameters != 0 or self.minus_circuit.num_parameters != 0:
            raise ValueError("shifted Qiskit circuits must be fully bound")
        object.__setattr__(self, "plus_values", plus_values)
        object.__setattr__(self, "minus_values", minus_values)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible shifted-circuit metadata."""
        return {
            "parameter_index": self.parameter_index,
            "parameter_name": self.parameter_name,
            "plus_values": self.plus_values.tolist(),
            "minus_values": self.minus_values.tolist(),
            "plus_depth": self.plus_circuit.depth(),
            "minus_depth": self.minus_circuit.depth(),
            "plus_size": self.plus_circuit.size(),
            "minus_size": self.minus_circuit.size(),
        }


@dataclass(frozen=True)
class QiskitParameterShiftGradientResult:
    """Local Qiskit Statevector parameter-shift gradient result."""

    value: float
    gradient: FloatArray
    records: tuple[QiskitParameterShiftRecord, ...]
    method: str
    evaluations: int
    claim_boundary: str

    def __post_init__(self) -> None:
        value = _as_finite_scalar("value", self.value)
        gradient = _as_finite_vector("gradient", self.gradient)
        if self.evaluations <= 0:
            raise ValueError("evaluations must be positive")
        if not self.claim_boundary:
            raise ValueError("claim_boundary must be non-empty")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "gradient", gradient)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-compatible Qiskit gradient metadata."""
        return {
            "value": self.value,
            "gradient": self.gradient.tolist(),
            "records": [record.to_dict() for record in self.records],
            "method": self.method,
            "evaluations": self.evaluations,
            "claim_boundary": self.claim_boundary,
        }


def generate_qiskit_parameter_shift_circuits(
    circuit: QuantumCircuit,
    parameters: Sequence[Parameter],
    values: ArrayLike,
    *,
    rule: ParameterShiftRule | None = None,
    shift: float = float(np.pi / 2.0),
) -> tuple[QiskitParameterShiftRecord, ...]:
    """Generate fully bound Qiskit plus/minus circuits for parameter shift."""
    parameter_tuple = _normalise_parameters(parameters)
    values_vector = _as_finite_vector("values", values, width=len(parameter_tuple))
    shift_value, _coefficient = _parameter_shift_rule(rule, shift)
    _validate_circuit_parameters(circuit, parameter_tuple)

    records: list[QiskitParameterShiftRecord] = []
    for index, parameter in enumerate(parameter_tuple):
        plus_values = values_vector.copy()
        minus_values = values_vector.copy()
        plus_values[index] += shift_value
        minus_values[index] -= shift_value
        records.append(
            QiskitParameterShiftRecord(
                parameter_index=index,
                parameter_name=parameter.name,
                plus_values=plus_values,
                minus_values=minus_values,
                plus_circuit=_bind_circuit(circuit, parameter_tuple, plus_values),
                minus_circuit=_bind_circuit(circuit, parameter_tuple, minus_values),
            )
        )
    return tuple(records)


def execute_qiskit_statevector_parameter_shift(
    circuit: QuantumCircuit,
    observable: object,
    parameters: Sequence[Parameter],
    values: ArrayLike,
    *,
    rule: ParameterShiftRule | None = None,
    shift: float = float(np.pi / 2.0),
) -> QiskitParameterShiftGradientResult:
    """Evaluate a local Qiskit Statevector value and parameter-shift gradient."""
    parameter_tuple = _normalise_parameters(parameters)
    values_vector = _as_finite_vector("values", values, width=len(parameter_tuple))
    _shift_value, coefficient = _parameter_shift_rule(rule, shift)
    _validate_circuit_parameters(circuit, parameter_tuple)
    base_circuit = _bind_circuit(circuit, parameter_tuple, values_vector)
    value = _expectation(base_circuit, observable)
    records = generate_qiskit_parameter_shift_circuits(
        circuit,
        parameter_tuple,
        values_vector,
        rule=rule,
        shift=shift,
    )
    gradient = np.array(
        [
            coefficient
            * (
                _expectation(record.plus_circuit, observable)
                - _expectation(record.minus_circuit, observable)
            )
            for record in records
        ],
        dtype=np.float64,
    )
    return QiskitParameterShiftGradientResult(
        value=value,
        gradient=gradient,
        records=records,
        method="qiskit_statevector_parameter_shift",
        evaluations=1 + 2 * len(records),
        claim_boundary=(
            "local Qiskit Statevector parameter-shift execution for fully bound circuits; "
            "not hardware execution, provider submission, or finite-shot evidence"
        ),
    )


def _expectation(circuit: QuantumCircuit, observable: object) -> float:
    state = Statevector.from_instruction(circuit)
    value = state.expectation_value(observable)
    return _as_finite_scalar("Qiskit expectation value", np.real(value))


def _bind_circuit(
    circuit: QuantumCircuit,
    parameters: tuple[Parameter, ...],
    values: FloatArray,
) -> QuantumCircuit:
    mapping = {parameter: float(values[index]) for index, parameter in enumerate(parameters)}
    bound = circuit.assign_parameters(mapping, inplace=False)
    if bound.num_parameters != 0:
        raise ValueError("all circuit parameters must be supplied before Qiskit execution")
    return bound


def _validate_circuit_parameters(
    circuit: QuantumCircuit,
    parameters: tuple[Parameter, ...],
) -> None:
    supplied = set(parameters)
    circuit_parameters = set(circuit.parameters)
    if not circuit_parameters:
        raise ValueError("circuit must contain at least one trainable parameter")
    if not circuit_parameters.issubset(supplied):
        raise ValueError("all circuit parameters must be listed in parameters")


def _normalise_parameters(parameters: Sequence[Parameter]) -> tuple[Parameter, ...]:
    parameter_tuple = tuple(parameters)
    if not parameter_tuple:
        raise ValueError("parameters must contain at least one Qiskit Parameter")
    if len(set(parameter_tuple)) != len(parameter_tuple):
        raise ValueError("parameters must not contain duplicates")
    for parameter in parameter_tuple:
        if not isinstance(parameter, Parameter):
            raise ValueError("parameters must contain Qiskit Parameter objects")
    return parameter_tuple


def _parameter_shift_rule(
    rule: ParameterShiftRule | None,
    shift: float,
) -> tuple[float, float]:
    if rule is not None:
        return _as_positive_scalar("rule.shift", rule.shift), _as_finite_scalar(
            "rule.coefficient",
            rule.coefficient,
        )
    shift_value = _as_positive_scalar("shift", shift)
    denominator = 2.0 * np.sin(shift_value)
    if abs(denominator) <= 1.0e-15:
        raise ValueError("shift must not make the parameter-shift denominator singular")
    return shift_value, float(1.0 / denominator)


def _as_finite_vector(
    name: str,
    value: ArrayLike,
    *,
    width: int | None = None,
) -> FloatArray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if width is not None and vector.shape != (width,):
        raise ValueError(f"{name} must have shape ({width},), got {vector.shape}")
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain only finite values")
    return vector.astype(np.float64, copy=True)


def _as_finite_scalar(name: str, value: object) -> float:
    raw = np.asarray(value)
    if raw.shape != () or raw.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"{name} must be a real numeric scalar")
    scalar = float(raw)
    if not np.isfinite(scalar):
        raise ValueError(f"{name} must be finite")
    return scalar


def _as_positive_scalar(name: str, value: object) -> float:
    scalar = _as_finite_scalar(name, value)
    if scalar <= 0.0:
        raise ValueError(f"{name} must be positive")
    return scalar


__all__ = [
    "QiskitParameterShiftGradientResult",
    "QiskitParameterShiftRecord",
    "execute_qiskit_statevector_parameter_shift",
    "generate_qiskit_parameter_shift_circuits",
]
