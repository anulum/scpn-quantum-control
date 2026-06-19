# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — import a PennyLane circuit into a registered Phase-QNode
"""Convert a PennyLane quantum tape into a registered Phase-QNode circuit.

The export bridge in :mod:`scpn_quantum_control.phase.pennylane_bridge` turns a
registered :class:`PhaseQNodeCircuit` into a PennyLane QNode. This module is the
inverse: it reads a ``pennylane.tape.QuantumScript`` and builds the equivalent
registered circuit, mapping the supported PennyLane gate set and a Pauli-word
expectation observable. Every gate parameter becomes a Phase-QNode parameter in
tape order.

Import is fail-closed: gates outside the registered set, multi-parameter gates,
non-integer or non-contiguous wires, multiple measurements, non-expectation
measurements, and non-Pauli or identity observables are rejected. The
:func:`check_pennylane_phase_qnode_import_round_trip` helper verifies the
imported circuit reproduces the source value and parameter-shift gradient.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .pennylane_bridge import _PENNYLANE_OPERATION_NAMES, _load_pennylane
from .qnode_circuit import (
    FloatArray,
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    SparsePauliHamiltonian,
    execute_phase_qnode_circuit,
    parameter_shift_phase_qnode_gradient,
)

# PennyLane operation name -> registered Phase-QNode gate.
_PENNYLANE_TO_REGISTERED: dict[str, str] = {
    name: gate for gate, name in _PENNYLANE_OPERATION_NAMES.items()
}


@dataclass(frozen=True)
class PennyLaneImportResult:
    """A registered circuit imported from a PennyLane tape."""

    circuit: PhaseQNodeCircuit
    parameter_values: NDArray[np.float64]
    n_qubits: int
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PennyLaneImportRoundTripResult:
    """Value and gradient agreement between the source tape and the import."""

    value_match: bool
    gradient_match: bool
    phase_value: float
    pennylane_value: float
    max_value_difference: float
    max_gradient_difference: float
    n_parameters: int


def is_pennylane_import_available() -> bool:
    """Return whether the optional PennyLane import bridge can import PennyLane."""
    try:
        _load_pennylane()
    except ImportError:
        return False
    return True


def _wires_to_indices(operation: Any) -> tuple[int, ...]:
    indices = []
    for wire in operation.wires:
        if isinstance(wire, bool) or not isinstance(wire, (int, np.integer)):
            raise ValueError(
                f"gate {operation.name!r} uses non-integer wire {wire!r}; "
                "only integer wires 0..n-1 are supported"
            )
        indices.append(int(wire))
    return tuple(indices)


def _as_finite_gate_parameter(operation: Any) -> float:
    raw_parameter = np.asarray(operation.parameters[0])
    if raw_parameter.shape != () or raw_parameter.dtype.kind in {"b", "O", "S", "U", "c"}:
        raise ValueError(f"PennyLane gate {operation.name!r} parameter must be a real scalar")
    value = float(raw_parameter)
    if not np.isfinite(value):
        raise ValueError(f"PennyLane gate {operation.name!r} parameter must be finite")
    return value


def _as_non_negative_tolerance(field_name: str, value: float) -> float:
    tolerance = float(value)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(f"{field_name} must be finite and non-negative")
    return tolerance


def _import_observable(
    qml: Any, measurement: Any, n_qubits: int
) -> PauliTerm | SparsePauliHamiltonian:
    if type(measurement).__name__ != "ExpectationMP":
        raise ValueError("only a single qml.expval measurement is supported")
    observable = measurement.obs
    if observable is None:
        raise ValueError("expectation measurement has no observable")
    try:
        sentence = qml.pauli.pauli_sentence(observable)
    except (ValueError, TypeError, AttributeError) as error:
        raise ValueError(
            f"observable {observable} is not a supported Pauli-word expectation"
        ) from error

    terms: list[PauliTerm] = []
    for word, coefficient in sentence.items():
        if len(word) == 0:
            raise ValueError("identity-only observable terms are not supported")
        if abs(float(np.imag(coefficient))) > 1e-12:
            raise ValueError("observable coefficients must be real")
        real_coefficient = float(np.real(coefficient))
        if not np.isfinite(real_coefficient):
            raise ValueError("observable coefficients must be finite")
        factors = tuple(sorted(((int(wire), str(pauli).lower()) for wire, pauli in word.items())))
        for wire, _pauli in factors:
            if not 0 <= wire < n_qubits:
                raise ValueError(f"observable wire {wire} is outside 0..{n_qubits - 1}")
        terms.append(PauliTerm(real_coefficient, factors))

    if not terms:
        raise ValueError("observable reduced to an empty Pauli sentence")
    if len(terms) == 1:
        return terms[0]
    return SparsePauliHamiltonian(tuple(terms))


def import_phase_qnode_from_pennylane(tape: Any) -> PennyLaneImportResult:
    """Import a PennyLane ``QuantumScript`` into a registered Phase-QNode circuit.

    Args:
        tape: a ``pennylane.tape.QuantumScript`` with registered-supported gates
            and a single Pauli-word expectation measurement.

    Returns:
        A :class:`PennyLaneImportResult` with the circuit, the tape-order
        parameter vector, the qubit count, and a provenance record.
    """
    qml = _load_pennylane()
    if not hasattr(tape, "operations") or not hasattr(tape, "measurements"):
        raise ValueError("tape must be a PennyLane QuantumScript / QuantumTape")

    wire_labels = [w for w in tape.wires]
    if any(isinstance(w, bool) or not isinstance(w, (int, np.integer)) for w in wire_labels):
        raise ValueError("only integer wires are supported")
    used = sorted(int(w) for w in wire_labels)
    n_qubits = len(used)
    if used and used != list(range(n_qubits)):
        raise ValueError("wires must form a contiguous range 0..n-1")
    if n_qubits == 0:
        raise ValueError("tape must act on at least one wire")

    operations: list[PhaseQNodeOperation] = []
    values: list[float] = []
    parameter_index = 0
    for operation in tape.operations:
        gate = _PENNYLANE_TO_REGISTERED.get(operation.name)
        if gate is None:
            raise ValueError(f"PennyLane gate {operation.name!r} has no registered import")
        wires = _wires_to_indices(operation)
        if operation.num_params == 0:
            operations.append(PhaseQNodeOperation(gate, wires))
        elif operation.num_params == 1:
            operations.append(PhaseQNodeOperation(gate, wires, parameter_index=parameter_index))
            values.append(_as_finite_gate_parameter(operation))
            parameter_index += 1
        else:
            raise ValueError(
                f"PennyLane gate {operation.name!r} has {operation.num_params} parameters; "
                "only single-parameter registered gates are importable"
            )

    if len(tape.measurements) != 1:
        raise ValueError("exactly one expectation measurement is supported")
    observable = _import_observable(qml, tape.measurements[0], n_qubits)

    circuit = PhaseQNodeCircuit(
        n_qubits=n_qubits, operations=tuple(operations), observable=observable
    )
    provenance = {
        "source": "pennylane.tape.QuantumScript",
        "n_qubits": n_qubits,
        "n_operations": len(operations),
        "n_parameters": len(values),
        "claim_boundary": (
            "bounded import of the registered local gate family and a Pauli-word "
            "expectation; no mid-circuit measurement, channel, template, or "
            "non-Pauli observable import, and no provider or hardware claim"
        ),
    }
    return PennyLaneImportResult(
        circuit=circuit,
        parameter_values=np.array(values, dtype=np.float64),
        n_qubits=n_qubits,
        provenance=provenance,
    )


def _pennylane_value_and_gradient(
    qml: Any, tape: Any, n_qubits: int, n_gate_parameters: int
) -> tuple[float, NDArray[np.float64]]:
    device = qml.device("default.qubit", wires=n_qubits)
    value = float(qml.execute([tape], device, diff_method=None)[0])
    if n_gate_parameters == 0:
        return value, np.zeros(0, dtype=np.float64)
    # Differentiate only the gate parameters; observable (e.g. Hamiltonian)
    # coefficients are fixed in the imported circuit and must not be shifted.
    gradient_tape = tape.copy()
    gradient_tape.trainable_params = list(range(n_gate_parameters))
    gradient_tapes, processing = qml.gradients.param_shift(gradient_tape)
    if not gradient_tapes:
        return value, np.zeros(0, dtype=np.float64)
    results = qml.execute(gradient_tapes, device, diff_method=None)
    gradient = np.atleast_1d(np.asarray(processing(results), dtype=np.float64))
    return value, gradient


def check_pennylane_phase_qnode_import_round_trip(
    tape: Any,
    *,
    value_tolerance: float = 1e-6,
    gradient_tolerance: float = 1e-6,
) -> PennyLaneImportRoundTripResult:
    """Verify an imported circuit reproduces the source value and gradient.

    Executes the source PennyLane tape and the imported Phase-QNode circuit and
    compares their expectation values and parameter-shift gradients.
    """
    qml = _load_pennylane()
    value_tol = _as_non_negative_tolerance("value_tolerance", value_tolerance)
    gradient_tol = _as_non_negative_tolerance("gradient_tolerance", gradient_tolerance)
    imported = import_phase_qnode_from_pennylane(tape)
    values: FloatArray = imported.parameter_values

    phase_value = float(execute_phase_qnode_circuit(imported.circuit, values).value)
    phase_gradient = np.asarray(
        parameter_shift_phase_qnode_gradient(imported.circuit, values).gradient,
        dtype=np.float64,
    )
    pennylane_value, pennylane_gradient = _pennylane_value_and_gradient(
        qml, tape, imported.n_qubits, int(values.size)
    )

    max_value_difference = abs(phase_value - pennylane_value)
    if phase_gradient.shape != pennylane_gradient.shape:
        max_gradient_difference = float("inf")
    elif phase_gradient.size == 0:
        max_gradient_difference = 0.0
    else:
        max_gradient_difference = float(np.max(np.abs(phase_gradient - pennylane_gradient)))

    return PennyLaneImportRoundTripResult(
        value_match=max_value_difference <= value_tol,
        gradient_match=max_gradient_difference <= gradient_tol,
        phase_value=phase_value,
        pennylane_value=pennylane_value,
        max_value_difference=max_value_difference,
        max_gradient_difference=max_gradient_difference,
        n_parameters=int(values.size),
    )
