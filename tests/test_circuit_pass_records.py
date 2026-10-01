# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Immutable circuit pass record tests
"""Native provenance records preserve source binding and deep immutability."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, replace

import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.compiler import (
    CircuitDiagnostic,
    CircuitPassRefused,
    SourceSpan,
    qualify_circuit_pass,
    snapshot_circuit,
)

malformed_replace: Callable[..., object] = replace


@pytest.mark.parametrize(
    "changes",
    [
        {"start": -1},
        {"end": 0},
        {"line": 0},
        {"column": 0},
        {"start": False},
        {"end": 1.5},
    ],
)
def test_source_span_requires_exact_offsets_and_coordinates(changes: dict[str, object]) -> None:
    """Malformed provenance spans cannot masquerade as original source offsets."""
    with pytest.raises(ValueError, match="source span"):
        malformed_replace(SourceSpan(0, 1, 1, 1), **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"code": ""},
        {"message": None},
        {"line": 0},
        {"column": False},
    ],
)
def test_located_diagnostic_contract_is_strict(changes: dict[str, object]) -> None:
    """Malformed diagnostic records cannot claim a valid caller-facing location."""
    with pytest.raises(ValueError, match="diagnostic"):
        malformed_replace(
            CircuitDiagnostic("unsupported_operation", "unsupported operation"), **changes
        )


def test_refusal_retains_authored_structured_diagnostic() -> None:
    """The public exception retains the exact immutable authored explanation."""
    diagnostic = CircuitDiagnostic("unsupported_operation", "unsupported operation", 7, 3)
    error = CircuitPassRefused(diagnostic)
    assert error.diagnostic is diagnostic
    assert str(error) == diagnostic.message


@pytest.mark.parametrize(
    "changes",
    [
        {"name": ""},
        {"parameters": [0.2]},
        {"parameters": (float("nan"),)},
        {"parameters": (1,)},
        {"qubits": [0]},
        {"qubits": (-1,)},
        {"clbits": (True,)},
        {"source_span": None},
    ],
)
def test_native_operation_fields_cannot_be_mutable_or_nonfinite(
    changes: dict[str, object],
) -> None:
    """A public operation record rejects invalid persisted operands/parameters."""
    circuit = QuantumCircuit(1)
    circuit.ry(0.2, 0)
    operation = snapshot_circuit(circuit).operations[0]
    with pytest.raises(ValueError):
        malformed_replace(operation, **changes)


def test_measurement_requires_one_native_operand_pair() -> None:
    """Malformed measurement records cannot crash the observable-map consumer."""
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    operation = snapshot_circuit(circuit).operations[0]
    changes_list: list[dict[str, object]] = [
        {"qubits": ()},
        {"clbits": ()},
        {"parameters": (0.1,)},
    ]
    for changes in changes_list:
        with pytest.raises(ValueError, match="measurement"):
            malformed_replace(operation, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        {"source": ""},
        {"source": None},
        {"num_qubits": False},
        {"num_qubits": 9},
        {"num_clbits": -1},
        {"num_clbits": 65},
        {"global_phase": float("inf")},
        {"global_phase": 0},
        {"operations": []},
        {"operations": (None,)},
        {"quantum_registers": []},
        {"quantum_registers": (["q", (0,)],)},
        {"quantum_registers": (("q",),)},
        {"quantum_registers": (("", (0,)),)},
        {"quantum_registers": (("q", [0]),)},
        {"quantum_registers": (("q", (1,)),)},
        {"classical_registers": (("c", (True,)),)},
    ],
)
def test_circuit_ir_contract_rejects_corrupt_fields(changes: dict[str, object]) -> None:
    """Public snapshots cannot silently retain mutable or misindexed data."""
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    with pytest.raises(ValueError):
        malformed_replace(snapshot_circuit(circuit), **changes)


def test_ir_source_spans_and_operand_bounds_are_verified() -> None:
    """Source tampering or out-of-width operands invalidates the native snapshot."""
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    original = snapshot_circuit(circuit)
    operation = original.operations[0]
    bad_spans = [
        SourceSpan(0, len(original.source) + 1, 1, 1),
        replace(operation.source_span, line=1),
    ]
    for span in bad_spans:
        with pytest.raises(ValueError, match="source"):
            replace(original, operations=(replace(operation, source_span=span),))
    for bad_operation in [replace(operation, qubits=(1,)), replace(operation, clbits=(1,))]:
        with pytest.raises(ValueError, match="operand"):
            replace(original, operations=(bad_operation,))


def test_circuit_operation_limit_is_part_of_immutable_contract() -> None:
    """Reconstructed records retain the same bounded operation-count contract."""
    circuit = QuantumCircuit(1)
    circuit.x(0)
    original = snapshot_circuit(circuit)
    with pytest.raises(ValueError, match="bounded"):
        replace(original, operations=original.operations * 4097)


@pytest.mark.parametrize(
    "changes",
    [
        {"pass_name": ""},
        {"input_ir": None},
        {"output_ir": None},
        {"input_layout": [0]},
        {"output_layout": (False,)},
        {"output_layout": (1,)},
        {"observable_map": []},
        {"observable_map": ((0, 0, 0),)},
        {"observable_map": ((0, 0, 0, True),)},
        {"observable_map": ((0, 0, 0, -1),)},
        {"allow_global_phase": 1},
        {"operator_error": float("nan")},
        {"operator_error": 0},
        {"operator_error": -1e-12},
        {"tolerance": 0.0},
        {"tolerance": 1e-10},
        {"reference_backend": "unexecuted"},
        {"basis_convention": "big_endian"},
        {"schema": "circuit_pass.v2"},
    ],
)
def test_pass_record_requires_consistent_immutable_qualification(
    changes: dict[str, object],
) -> None:
    """A stored admission cannot silently change its original qualification gate."""
    circuit = QuantumCircuit(1, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    record = qualify_circuit_pass(circuit, circuit, pass_name="identity")
    with pytest.raises(ValueError):
        malformed_replace(record, **changes)


def test_pass_record_maps_and_widths_are_bound_to_ir() -> None:
    """The observable correspondence is derived from both source snapshots."""
    circuit = QuantumCircuit(1, 1)
    circuit.measure(0, 0)
    record = qualify_circuit_pass(circuit, circuit, pass_name="identity")
    with pytest.raises(ValueError, match="correspondence"):
        replace(record, observable_map=())
    with pytest.raises(ValueError, match="widths"):
        replace(record, output_ir=snapshot_circuit(QuantumCircuit(2, 1)))


def test_pass_digest_binds_phase_source_and_reference_policy() -> None:
    """Immutable provenance includes actual source text and equivalence policy."""
    circuit = QuantumCircuit(1)
    circuit.ry(0.2, 0)
    record = qualify_circuit_pass(circuit, circuit, pass_name="identity")
    assert record.sha256 == qualify_circuit_pass(circuit, circuit, pass_name="identity").sha256
    assert replace(record, allow_global_phase=False).sha256 != record.sha256
    assert replace(record, global_phase_delta=0.31).sha256 != record.sha256
    assert (
        replace(
            record, input_ir=replace(record.input_ir, source=record.input_ir.source + "\n")
        ).sha256
        != record.sha256
    )
    with pytest.raises(FrozenInstanceError):
        field_name = "global_phase"
        setattr(record.input_ir, field_name, 0.31)
