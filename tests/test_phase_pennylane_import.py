# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the PennyLane import-from bridge
"""Round-trip and fail-closed tests for importing PennyLane tapes into Phase-QNode."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, cast

import numpy as np
import pytest
from numpy.typing import NDArray

qml: Any = pytest.importorskip("pennylane")

from scpn_quantum_control.phase.pennylane_bridge import run_pennylane_maturity_audit
from scpn_quantum_control.phase.pennylane_bridge import build_pennylane_qnode_from_phase_qnode
from scpn_quantum_control.phase.pennylane_import import (
    PennyLaneImportResult,
    PennyLaneImportRoundTripResult,
    check_pennylane_phase_qnode_import_round_trip,
    import_phase_qnode_from_pennylane,
    is_pennylane_import_available,
)
from scpn_quantum_control.phase.qnode_circuit import (
    PauliTerm,
    PhaseQNodeCircuit,
    PhaseQNodeOperation,
    SparsePauliHamiltonian,
)

FloatArray = NDArray[np.float64]
TapeFactory = Callable[[], object]


def _script(ops: Sequence[object], measurements: Sequence[object]) -> object:
    return qml.tape.QuantumScript(ops, measurements)


def test_availability() -> None:
    assert is_pennylane_import_available() is True


# --------------------------------------------------------------------------- #
# Import structure
# --------------------------------------------------------------------------- #
def test_import_structure_and_parameter_order() -> None:
    tape = _script(
        [
            qml.Hadamard(0),
            qml.RX(0.6, wires=0),
            qml.CNOT([0, 1]),
            qml.CRY(0.9, wires=[0, 1]),
            qml.RZ(0.3, wires=1),
        ],
        [qml.expval(qml.PauliZ(0) @ qml.PauliX(1))],
    )
    result = import_phase_qnode_from_pennylane(tape)
    assert isinstance(result, PennyLaneImportResult)
    assert result.n_qubits == 2
    # Parameters follow tape order: RX, CRY, RZ.
    assert np.allclose(result.parameter_values, [0.6, 0.9, 0.3])
    operations = tuple(cast(PhaseQNodeOperation, op) for op in result.circuit.operations)
    gates = [op.gate for op in operations]
    assert gates == ["h", "rx", "cnot", "cry", "rz"]
    assert isinstance(result.circuit.observable, PauliTerm)
    assert "claim_boundary" in result.provenance


def test_import_hamiltonian_observable() -> None:
    tape = _script(
        [qml.RY(0.4, wires=0), qml.RY(1.1, wires=1)],
        [qml.expval(qml.Hamiltonian([0.5, -0.25], [qml.PauliZ(0), qml.PauliX(1)]))],
    )
    result = import_phase_qnode_from_pennylane(tape)
    assert isinstance(result.circuit.observable, SparsePauliHamiltonian)
    assert len(result.circuit.observable.terms) == 2


# --------------------------------------------------------------------------- #
# Round-trip value and gradient
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "ops, obs",
    [
        (
            [qml.Hadamard(0), qml.RX(0.6, wires=0), qml.CNOT([0, 1]), qml.RZ(0.3, wires=1)],
            qml.PauliZ(0) @ qml.PauliX(1),
        ),
        (
            [qml.RY(0.4, wires=0), qml.RY(1.1, wires=1), qml.IsingXX(0.7, wires=[0, 1])],
            qml.Hamiltonian(
                [0.5, -0.25, 0.75], [qml.PauliZ(0), qml.PauliX(1), qml.PauliZ(0) @ qml.PauliZ(1)]
            ),
        ),
        # Controlled rotations with an observable on the control qubit exercise
        # the four-term shift rule against PennyLane's own gradient.
        (
            [
                qml.Hadamard(0),
                qml.Hadamard(1),
                qml.CRZ(1.2, wires=[0, 1]),
                qml.CRX(0.5, wires=[1, 0]),
            ],
            qml.PauliX(0) @ qml.PauliZ(1),
        ),
        ([qml.CRY(0.8, wires=[0, 1]), qml.Hadamard(0)], qml.PauliX(0) @ qml.PauliY(1)),
    ],
)
def test_import_round_trip_value_and_gradient(ops: Sequence[object], obs: object) -> None:
    tape = _script(ops, [qml.expval(obs)])
    result = check_pennylane_phase_qnode_import_round_trip(tape)
    assert result.value_match
    assert result.gradient_match
    assert result.max_value_difference < 1e-6
    assert result.max_gradient_difference < 1e-6


def test_import_round_trip_parameterless() -> None:
    tape = _script([qml.Hadamard(0), qml.CNOT([0, 1])], [qml.expval(qml.PauliZ(1))])
    result = check_pennylane_phase_qnode_import_round_trip(tape)
    assert result.value_match
    assert result.gradient_match
    assert result.n_parameters == 0


def test_generated_phase_qnode_export_import_round_trip_preserves_value_and_gradient() -> None:
    """Generated PennyLane QNodes import back into equivalent Phase-QNode circuits."""

    circuit = PhaseQNodeCircuit(
        2,
        (("ry", (0,), 0), ("cnot", (0, 1)), ("rzz", (0, 1), 1)),
        SparsePauliHamiltonian(
            (
                PauliTerm(0.5, ((0, "z"),)),
                PauliTerm(-0.25, ((0, "x"), (1, "x"))),
            )
        ),
    )
    params = np.array([0.41, -0.23], dtype=float)
    conversion = build_pennylane_qnode_from_phase_qnode(circuit)
    trainable = qml.numpy.array(params, requires_grad=True)

    exported_tape = qml.workflow.construct_tape(conversion.qnode)(trainable)
    imported = import_phase_qnode_from_pennylane(exported_tape)
    round_trip = check_pennylane_phase_qnode_import_round_trip(exported_tape)

    assert imported.n_qubits == circuit.n_qubits
    assert np.allclose(imported.parameter_values, params)
    assert tuple(cast(PhaseQNodeOperation, op).gate for op in imported.circuit.operations) == (
        "ry",
        "cnot",
        "rzz",
    )
    assert isinstance(imported.circuit.observable, SparsePauliHamiltonian)
    assert imported.provenance["source"] == "pennylane.tape.QuantumScript"
    assert round_trip.value_match
    assert round_trip.gradient_match
    assert round_trip.n_parameters == params.size


def test_pennylane_maturity_audit_records_live_import_round_trip() -> None:
    tape = _script(
        [qml.RY(0.4, wires=0), qml.RX(-0.2, wires=0)],
        [qml.expval(qml.PauliZ(0))],
    )
    circuit = PhaseQNodeCircuit(
        1,
        (("ry", (0,), 0), ("rx", (0,), 1)),
        PauliTerm(1.0, ((0, "z"),)),
    )

    def objective(values: FloatArray) -> float:
        return float(np.cos(values[0]) + 0.25 * np.sin(values[1]))

    def gradient(values: FloatArray) -> FloatArray:
        return np.array([-np.sin(values[0]), 0.25 * np.cos(values[1])], dtype=float)

    result = run_pennylane_maturity_audit(
        objective=objective,
        pennylane_objective=objective,
        pennylane_gradient=gradient,
        values=np.array([0.2, -0.4], dtype=float),
        circuit=circuit,
        phase_qnode_values=np.array([0.4, -0.2], dtype=float),
        import_tape=tape,
        value_tolerance=1e-6,
        gradient_tolerance=1e-6,
    )

    assert result.identical_circuit_ready
    assert not result.ready_for_provider_exceedance
    assert result.required_capabilities["phase_qnode_import_round_trip"] == "passed"
    assert result.required_capabilities["pennylane_plugin_matrix"] == "passed"
    assert "provider_plugin_execution" in result.open_gaps
    imported = cast(
        PennyLaneImportRoundTripResult, result.evidence["phase_qnode_import_round_trip"]
    )
    assert imported.value_match
    assert imported.gradient_match
    assert result.promotion_metadata["phase_qnode_parameter_shift_evaluations"] == 4


# --------------------------------------------------------------------------- #
# Fail-closed diagnostics
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "tape_factory",
    [
        lambda: _script([qml.Rot(0.1, 0.2, 0.3, wires=0)], [qml.expval(qml.PauliZ(0))]),
        lambda: _script([qml.ISWAP([0, 1])], [qml.expval(qml.PauliZ(0))]),
        lambda: _script([qml.RX(0.3, wires=0)], [qml.var(qml.PauliZ(0))]),
        lambda: _script(
            [qml.RX(0.3, wires=0)], [qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliX(0))]
        ),
        lambda: _script([qml.RX(0.3, wires=0), qml.RX(0.2, wires=2)], [qml.expval(qml.PauliZ(0))]),
    ],
)
def test_import_rejects_unsupported(tape_factory: TapeFactory) -> None:
    with pytest.raises(ValueError):
        import_phase_qnode_from_pennylane(tape_factory())


@pytest.mark.parametrize("parameter", [float("nan"), float("inf"), -float("inf")])
def test_import_rejects_non_finite_gate_parameters(parameter: float) -> None:
    tape = _script(
        [qml.RX(parameter, wires=0)],
        [qml.expval(qml.PauliZ(0))],
    )

    with pytest.raises(ValueError, match="parameter must be finite"):
        import_phase_qnode_from_pennylane(tape)


@pytest.mark.parametrize(
    ("value_tolerance", "gradient_tolerance", "match"),
    [
        (-1e-6, 1e-6, "value_tolerance"),
        (1e-6, -1e-6, "gradient_tolerance"),
        (float("nan"), 1e-6, "value_tolerance"),
    ],
)
def test_import_round_trip_rejects_invalid_tolerances(
    value_tolerance: float,
    gradient_tolerance: float,
    match: str,
) -> None:
    tape = _script([qml.RY(0.4, wires=0)], [qml.expval(qml.PauliZ(0))])

    with pytest.raises(ValueError, match=match):
        check_pennylane_phase_qnode_import_round_trip(
            tape,
            value_tolerance=value_tolerance,
            gradient_tolerance=gradient_tolerance,
        )


def test_import_rejects_non_tape() -> None:
    with pytest.raises(ValueError):
        import_phase_qnode_from_pennylane(object())
