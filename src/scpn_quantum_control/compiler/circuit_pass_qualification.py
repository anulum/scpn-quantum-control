# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Native circuit pass qualification
"""Qualify native circuit transforms before exporting textual MLIR provenance."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from threading import Event

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import PermutationGate
from qiskit.quantum_info import Operator

from ..execution_memory import ExecutionBuffer, ExecutionMemoryPlan
from ..execution_reservations import reserve_execution_memory
from .circuit_pass_records import (
    CircuitDiagnostic,
    CircuitPassRecord,
    CircuitPassRefused,
)
from .circuit_source import import_circuit_source, snapshot_circuit
from .mlir_records import MLIRModule

_BASIS_LOWERING_ID = "qiskit_basis_lowering"


@dataclass(frozen=True, slots=True)
class QualifiedCircuitCompilation:
    """Textual export, immutable pass evidence and a native circuit copy.

    Parameters
    ----------
    mlir_module
        Textual interchange module, never an executable-machine-code claim.
    pass_record
        Immutable snapshots and the actual local unitary qualification result.
    output_circuit
        Native transformed circuit. Later mutation does not alter the record or
        qualify that later circuit; qualification applies only to the snapshot.

    """

    mlir_module: MLIRModule
    pass_record: CircuitPassRecord
    output_circuit: QuantumCircuit


def qualify_circuit_pass(
    circuit: QuantumCircuit,
    transformed: QuantumCircuit,
    *,
    pass_name: str,
    input_layout: tuple[int, ...] | None = None,
    output_layout: tuple[int, ...] | None = None,
    output_classical_layout: tuple[int, ...] | None = None,
    allow_global_phase: bool = True,
    tolerance: float = 1e-12,
    source: str | None = None,
    deadline_monotonic: float | None = None,
    cancelled: Event | None = None,
) -> CircuitPassRecord:
    """Check every unitary input basis and its mapped terminal observables.

    Parameters
    ----------
    circuit, transformed
        Bound native input/output circuits, each with at most eight qubits.
    pass_name
        Non-empty descriptive transformation identity.
    input_layout, output_layout
        Logical-to-physical qubit bijections; omitted layouts are identity.
    output_classical_layout
        Logical input clbit to physical output clbit bijection, identity by default.
    allow_global_phase
        Whether the full operators may differ by one constant phase.
    tolerance
        Absolute complex128 entry-error bound, positive and at most ``1e-12``.
    source
        Exact original input source, or native canonical export when omitted.
    deadline_monotonic, cancelled
        Existing reservation deadline/cancellation guards before and after the
        synchronous native reference call; no hard interruption of Qiskit occurs.

    Returns
    -------
    CircuitPassRecord
        Source-bound snapshots, observable correspondence and reference error.

    Raises
    ------
    CircuitPassRefused
        Invalid layout/configuration, unsupported source, readout or unitary change.
    DenseAllocationError
        Existing whole-buffer memory admission refuses the native reference.

    """
    if not isinstance(pass_name, str) or not pass_name.strip():
        raise CircuitPassRefused(CircuitDiagnostic("invalid_pass", "pass name must be non-empty"))
    if (
        type(allow_global_phase) is not bool
        or isinstance(tolerance, bool)
        or not isinstance(tolerance, (float, int))
        or not 0 < tolerance <= 1e-12
    ):
        raise CircuitPassRefused(
            CircuitDiagnostic(
                "invalid_tolerance",
                "qualification requires an explicit phase policy and tolerance at most 1e-12",
            )
        )
    original = snapshot_circuit(circuit, source=source)
    output = snapshot_circuit(transformed)
    if (original.num_qubits, original.num_clbits) != (output.num_qubits, output.num_clbits):
        raise CircuitPassRefused(
            CircuitDiagnostic("width_mismatch", "transformation must preserve circuit widths")
        )
    before_layout = _layout(input_layout, original.num_qubits)
    after_layout = _layout(output_layout, output.num_qubits)
    classical_layout = _layout(output_classical_layout, output.num_clbits)
    before_inverse = tuple(before_layout.index(i) for i in range(original.num_qubits))
    after_inverse = tuple(after_layout.index(i) for i in range(output.num_qubits))
    classical_inverse = tuple(classical_layout.index(i) for i in range(output.num_clbits))
    before_readout = {c: before_inverse[q] for q, c in original.measurements}
    after_readout = {classical_inverse[c]: after_inverse[q] for q, c in output.measurements}
    if before_readout != after_readout:
        raise CircuitPassRefused(
            CircuitDiagnostic(
                "measurement_mismatch", "transformation changes the measurement mapping"
            )
        )
    observable_map = tuple(
        (before_layout[q], c, after_layout[q], classical_layout[c])
        for c, q in before_readout.items()
    )
    plan = ExecutionMemoryPlan(
        (
            ExecutionBuffer.hilbert(
                "unitary_reference", "intermediate", original.num_qubits, rank=2, count=8
            ),
        )
    )
    with reserve_execution_memory(
        plan, deadline_monotonic=deadline_monotonic, cancelled=cancelled
    ) as reservation:
        reservation.checkpoint()
        before = Operator(_logical_body(circuit, before_inverse)).data
        after = Operator(_logical_body(transformed, after_inverse)).data
        reservation.checkpoint()
        overlap = complex(np.vdot(before, after)) / before.shape[0]
        delta = math.atan2(overlap.imag, overlap.real) if allow_global_phase else 0.0
        error = float(np.max(np.abs(after * np.exp(-1j * delta) - before)))
        reservation.checkpoint()
    if not math.isfinite(error) or error > tolerance:
        raise CircuitPassRefused(
            CircuitDiagnostic("unitary_mismatch", "transformation changes unitary semantics")
        )
    return CircuitPassRecord(
        pass_name,
        original,
        output,
        before_layout,
        after_layout,
        classical_layout,
        observable_map,
        delta,
        error,
        float(tolerance),
        allow_global_phase,
    )


def compile_circuit_to_mlir(
    circuit: QuantumCircuit | str,
    *,
    optimisation_level: int = 1,
    allow_global_phase: bool = True,
) -> QualifiedCircuitCompilation:
    """Lower through native Qiskit, qualify semantics and export textual IR.

    Parameters
    ----------
    circuit
        Bound native circuit or the supported static OpenQASM 2 source.
    optimisation_level
        Native optimisation level 0 through 3; deterministic seed zero.
    allow_global_phase
        Whether one constant unitary phase is an admitted equivalence.

    Returns
    -------
    QualifiedCircuitCompilation
        Native ``rx/ry/rz/cx`` lowering, immutable qualification and textual IR.
        No MLIR execution, physical backend mapping or provider submission occurs.

    Raises
    ------
    CircuitPassRefused
        Invalid configuration, unsupported circuit/source or failed equivalence.

    """
    if type(optimisation_level) is not int or optimisation_level not in range(4):
        raise CircuitPassRefused(
            CircuitDiagnostic(
                "invalid_optimisation", "optimisation level must be an integer from 0 through 3"
            )
        )
    source = circuit if isinstance(circuit, str) else None
    native = import_circuit_source(circuit) if isinstance(circuit, str) else circuit
    snapshot_circuit(native, source=source)
    lowered_body = transpile(
        _logical_body(native, tuple(range(native.num_qubits))),
        basis_gates=["rx", "ry", "rz", "cx"],
        optimization_level=optimisation_level,
        seed_transpiler=0,
        qubits_initially_zero=False,
    )
    native_layout = lowered_body.layout
    permutation = (
        tuple(native_layout.final_index_layout())
        if native_layout is not None
        else tuple(range(native.num_qubits))
    )
    if permutation != tuple(range(native.num_qubits)):
        restored = QuantumCircuit(native.num_qubits).compose(lowered_body)
        restored.append(PermutationGate(list(permutation)), range(native.num_qubits))
        lowered_body = transpile(
            restored,
            basis_gates=["rx", "ry", "rz", "cx"],
            optimization_level=0,
            seed_transpiler=0,
            qubits_initially_zero=False,
        )
    lowered = native.copy_empty_like()
    lowered.global_phase = 0.0
    lowered.compose(lowered_body, inplace=True)
    trailing = list(native.data)
    readout = []
    while trailing and trailing[-1].operation.name in {"measure", "barrier"}:
        readout.append(trailing.pop())
    for instruction in reversed(readout):
        lowered.append(
            instruction.operation.copy(),
            [native.find_bit(q).index for q in instruction.qubits],
            [native.find_bit(c).index for c in instruction.clbits],
        )
    record = qualify_circuit_pass(
        native,
        lowered,
        pass_name=_BASIS_LOWERING_ID,
        source=source,
        allow_global_phase=allow_global_phase,
    )
    lines = [
        "module {",
        f'  "scpn_circuit.layout"() {{qubits = {record.output_ir.num_qubits} : i64, classical_bits = {record.output_ir.num_clbits} : i64, global_phase = {record.output_ir.global_phase:.17e} : f64}} : () -> ()',
    ]
    for operation in record.output_ir.operations:
        operands = json.dumps(
            {
                "qubits": operation.qubits,
                "clbits": operation.clbits,
                "parameters": operation.parameters,
            },
            separators=(",", ":"),
        )
        lines.append(
            f'  "scpn_circuit.{operation.name}"() {{operands = {json.dumps(operands)}}} : () -> () loc("circuit-{record.output_ir.source_sha256}.qasm":{operation.source_span.line}:{operation.source_span.column})'
        )
    lines.append("}")
    text = "\n".join(lines) + "\n"
    module = MLIRModule(
        text,
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "scpn_circuit",
        {
            "qubits": record.output_ir.num_qubits,
            "classical_bits": record.output_ir.num_clbits,
            "operations": len(record.output_ir.operations),
        },
        {
            "execution_status": "textual_ir",
            "pass_sha256": record.sha256,
            "reference_backend": record.reference_backend,
            "input_source_sha256": record.input_ir.source_sha256,
            "output_source_sha256": record.output_ir.source_sha256,
            "claim_boundary": "Textual circuit IR; bounded native Qiskit unitary and readout equivalence only. No executed MLIR, hardware mapping or provider evidence.",
        },
    )
    return QualifiedCircuitCompilation(module, record, lowered.copy())


def _layout(value: tuple[int, ...] | None, width: int) -> tuple[int, ...]:
    if value is None:
        return tuple(range(width))
    if (
        not isinstance(value, tuple)
        or any(type(i) is not int for i in value)
        or sorted(value) != list(range(width))
    ):
        raise CircuitPassRefused(
            CircuitDiagnostic(
                "invalid_layout", "layout must be a complete logical-to-physical bit bijection"
            )
        )
    return value


def _logical_body(circuit: QuantumCircuit, inverse: tuple[int, ...]) -> QuantumCircuit:
    body = QuantumCircuit(circuit.num_qubits, global_phase=circuit.global_phase)
    for instruction in circuit.data:
        if instruction.operation.name not in {"measure", "barrier"}:
            body.append(
                instruction.operation.copy(),
                [inverse[circuit.find_bit(q).index] for q in instruction.qubits],
            )
    return body
