# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Compiler observable preservation
"""Compiler transformations through the original public MLIR facade."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector

from scpn_quantum_control.compiler import mlir
from scpn_quantum_control.mitigation.zne import gate_fold_circuit


def test_compiler_observable_preservation_01() -> None:
    """Admitted lowering retains entanglement and swapped classical outputs."""
    circuit = QuantumCircuit(2, 2)
    circuit.ry(np.pi / 3, 0)
    circuit.cx(0, 1)
    circuit.rz(0.37, 1)
    circuit.measure([0, 1], [1, 0])
    compiled = mlir.compile_circuit_to_mlir(circuit, optimisation_level=2)
    record = compiled.pass_record
    assert record.input_ir.measurements == ((0, 1), (1, 0))
    assert record.output_ir.measurements == ((0, 1), (1, 0))
    assert record.operator_error < 1e-12
    assert compiled.mlir_module.metadata["execution_status"] == "textual_ir"
    assert record.reference_backend == "qiskit.quantum_info.Operator"
    assert record.input_ir.operations[0].source_span.line > 1
    state = Statevector.from_instruction(
        compiled.output_circuit.remove_final_measurements(inplace=False)
    )
    np.testing.assert_allclose(state.probabilities(), [0.75, 0, 0, 0.25], atol=1e-12)
    assert state.expectation_value(np.diag([1, -1, 1, -1])).real == pytest.approx(0.5)


def test_compiler_observable_preservation_02() -> None:
    """A smaller circuit that deletes a measured gate fails semantic admission."""
    circuit = QuantumCircuit(2, 2)
    circuit.x(0)
    circuit.h(1)
    circuit.measure([0, 1], [1, 0])
    changed = QuantumCircuit(2, 2)
    changed.h(1)
    changed.measure([0, 1], [1, 0])
    assert changed.depth() <= circuit.depth()
    before = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
    after = Statevector.from_instruction(changed.remove_final_measurements(inplace=False))
    assert not np.allclose(before.probabilities(), after.probabilities())
    with pytest.raises(mlir.CircuitPassRefused, match="unitary semantics"):
        mlir.qualify_circuit_pass(circuit, changed, pass_name="gate_deletion")
    assert circuit.count_ops()["x"] == 1


def test_compiler_observable_preservation_03() -> None:
    """Unsupported source syntax identifies its real line and column."""
    source = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[2];\n  reset q[0];\n'
    with pytest.raises(mlir.CircuitPassRefused) as caught:
        mlir.compile_circuit_to_mlir(source)
    diagnostic = caught.value.diagnostic
    assert (diagnostic.code, diagnostic.line, diagnostic.column) == ("unsupported_operation", 4, 3)
    assert diagnostic.message == "operation is outside the static unitary subset"


@pytest.mark.parametrize("scale", [1, 3, 5, 7])
def test_folded_partial_multiregister_readout_is_qualified(scale: int) -> None:
    """Original ZNE folding preserves measured bits, phase and logical meaning."""
    q = QuantumRegister(3, "q")
    alpha = ClassicalRegister(2, "alpha")
    beta = ClassicalRegister(1, "beta")
    circuit = QuantumCircuit(q, alpha, beta, global_phase=0.37)
    circuit.x(0)
    circuit.ry(np.pi / 3, 2)
    circuit.cx(2, 1)
    circuit.measure(q[2], alpha[0])
    circuit.barrier()
    circuit.measure(q[0], beta[0])
    folded = gate_fold_circuit(circuit, scale)
    record = mlir.qualify_circuit_pass(circuit, folded, pass_name="global_unitary_folding")
    assert record.input_ir.measurements == record.output_ir.measurements == ((2, 0), (0, 2))
    assert record.input_ir.classical_registers == (("alpha", (0, 1)), ("beta", (2,)))
    assert record.operator_error < 1e-12
    assert record.global_phase_delta == pytest.approx(0, abs=1e-12)


def test_declared_qubit_layout_transforms_readout_observables() -> None:
    """Physical bit permutation carries its logical observable correspondence."""
    original = QuantumCircuit(2, 2)
    original.ry(0.41, 0)
    original.cx(0, 1)
    original.measure([0, 1], [1, 0])
    physical = QuantumCircuit(2, 2)
    physical.ry(0.41, 1)
    physical.cx(1, 0)
    physical.measure([1, 0], [1, 0])
    record = mlir.qualify_circuit_pass(
        original, physical, pass_name="layout_permutation", output_layout=(1, 0)
    )
    assert record.observable_map == ((0, 1, 1, 1), (1, 0, 0, 0))
    assert record.output_layout == (1, 0)
    with pytest.raises(mlir.CircuitPassRefused, match="measurement mapping"):
        mlir.qualify_circuit_pass(original, physical, pass_name="undeclared_layout")


def test_last_classical_assignment_defines_the_terminal_observable() -> None:
    """Earlier overwritten readout does not replace the final classical meaning."""
    original = QuantumCircuit(2, 1)
    original.x(0)
    original.measure(0, 0)
    original.measure(1, 0)
    equivalent = QuantumCircuit(2, 1)
    equivalent.x(0)
    equivalent.measure(1, 0)
    record = mlir.qualify_circuit_pass(
        original, equivalent, pass_name="remove_overwritten_readout"
    )
    assert record.input_ir.measurements == ((0, 0), (1, 0))
    assert record.output_ir.measurements == ((1, 0),)
    assert record.observable_map == ((1, 0, 1, 0),)
    changed = QuantumCircuit(2, 1)
    changed.x(0)
    changed.measure(0, 0)
    with pytest.raises(mlir.CircuitPassRefused, match="measurement mapping"):
        mlir.qualify_circuit_pass(original, changed, pass_name="change_last_assignment")
