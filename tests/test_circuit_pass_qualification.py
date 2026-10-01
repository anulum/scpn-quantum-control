# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Circuit pass qualification tests
"""Native all-input equivalence, readout layout and compiler integration."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError
from threading import Event

import numpy as np
import pytest
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27, knm_to_ansatz
from scpn_quantum_control.compiler import (
    CircuitPassRecord,
    CircuitPassRefused,
    QualifiedCircuitCompilation,
    compile_circuit_to_mlir,
    qualify_circuit_pass,
)
from scpn_quantum_control.execution_reservations import ExecutionCancelledError


def test_global_phase_is_explicit_and_can_be_refused() -> None:
    """A declared global equivalence differs from exact unitary preservation."""
    before = QuantumCircuit(1)
    before.h(0)
    after = before.copy()
    after.global_phase = 0.31
    record = qualify_circuit_pass(before, after, pass_name="phase_change")
    assert record.global_phase_delta == pytest.approx(0.31, abs=1e-12)
    assert record.allow_global_phase is True
    assert record.basis_convention == "qiskit_little_endian"
    with pytest.raises(CircuitPassRefused, match="unitary semantics"):
        qualify_circuit_pass(before, after, pass_name="exact_phase", allow_global_phase=False)
    exact = qualify_circuit_pass(before, before, pass_name="exact_phase", allow_global_phase=False)
    assert exact.allow_global_phase is False
    assert exact.sha256 != record.sha256


def test_every_input_basis_is_checked() -> None:
    """Deleting a CX that is invisible on zero input still fails admission."""
    before = QuantumCircuit(2)
    before.cx(0, 1)
    after = QuantumCircuit(2)
    np.testing.assert_array_equal(
        Statevector.from_instruction(before).data, Statevector.from_instruction(after).data
    )
    with pytest.raises(CircuitPassRefused, match="unitary semantics"):
        qualify_circuit_pass(before, after, pass_name="delete_controlled_gate")


@pytest.mark.parametrize("angle", [1e-20, -1e-20, 1e20, np.pi / 3])
def test_exact_native_angles_round_trip_through_strict_source(angle: float) -> None:
    """Finite native parameters retain their bits in strict OpenQASM literals."""
    circuit = QuantumCircuit(1)
    circuit.rx(angle, 0)
    compiled = compile_circuit_to_mlir(circuit, optimisation_level=0)
    assert compiled.pass_record.input_ir.operations[0].parameters == (float(angle),)
    restored = qasm2.loads(compiled.pass_record.input_ir.source, include_path=(), strict=True)
    assert restored.data[0].operation.params[0] == angle
    assert compiled.pass_record.operator_error < 1e-12


def test_classical_bit_permutation_is_explicit() -> None:
    """An explicit classical layout preserves logical output interpretation."""
    before = QuantumCircuit(2, 2)
    before.ry(0.3, 0)
    before.measure([0, 1], [0, 1])
    after = QuantumCircuit(2, 2)
    after.ry(0.3, 0)
    after.measure([0, 1], [1, 0])
    with pytest.raises(CircuitPassRefused, match="measurement mapping"):
        qualify_circuit_pass(before, after, pass_name="undeclared_classical_permutation")
    record = qualify_circuit_pass(
        before, after, pass_name="classical_permutation", output_classical_layout=(1, 0)
    )
    assert record.observable_map == ((0, 0, 0, 1), (1, 1, 1, 0))


def test_both_physical_layouts_are_normalised() -> None:
    """The same logical circuit can have distinct input and output numbering."""
    before = QuantumCircuit(2, 1)
    before.x(1)
    before.measure(1, 0)
    after = QuantumCircuit(2, 1)
    after.x(0)
    after.measure(0, 0)
    record = qualify_circuit_pass(before, after, pass_name="physical_layouts", input_layout=(1, 0))
    assert record.observable_map == ((1, 0, 0, 0),)


@pytest.mark.parametrize("value", [(0, 0), (1,), (False, 1), (0, 2), [0, 1]])
def test_invalid_layout_cannot_certify_a_pass(value: object) -> None:
    """Malformed physical layouts fail before native reference allocation."""
    circuit = QuantumCircuit(2)
    boundary: Callable[..., CircuitPassRecord] = qualify_circuit_pass
    with pytest.raises(CircuitPassRefused, match="bijection"):
        boundary(circuit, circuit, pass_name="bad_layout", output_layout=value)


@pytest.mark.parametrize("value", [True, -1, 4, 1.0, "1"])
def test_native_lowering_configuration_is_strict(value: object) -> None:
    """Invalid optimisation configuration is refused without input mutation."""
    circuit = QuantumCircuit(1)
    boundary: Callable[..., QualifiedCircuitCompilation] = compile_circuit_to_mlir
    with pytest.raises(CircuitPassRefused, match="optimisation level"):
        boundary(circuit, optimisation_level=value)
    assert len(circuit.data) == 0


@pytest.mark.parametrize("value", [True, 0, -1e-12, 1e-10, float("nan"), "small"])
def test_tolerance_cannot_weaken_original_unitary_gate(value: object) -> None:
    """Admission refuses nonfinite, lossy or wider error budgets."""
    circuit = QuantumCircuit(1)
    boundary: Callable[..., CircuitPassRecord] = qualify_circuit_pass
    with pytest.raises(CircuitPassRefused, match="tolerance"):
        boundary(circuit, circuit, pass_name="bad_tolerance", tolerance=value)


@pytest.mark.parametrize("value", ["", " ", None])
def test_pass_name_is_required(value: object) -> None:
    """Pass provenance requires a descriptive non-empty identity."""
    circuit = QuantumCircuit(1)
    boundary: Callable[..., CircuitPassRecord] = qualify_circuit_pass
    with pytest.raises(CircuitPassRefused, match="pass name"):
        boundary(circuit, circuit, pass_name=value)


def test_changed_widths_are_not_implicit_layouts() -> None:
    """Ancilla insertion and classical width changes require another contract."""
    for after in [QuantumCircuit(2), QuantumCircuit(1, 1)]:
        with pytest.raises(CircuitPassRefused, match="widths"):
            qualify_circuit_pass(QuantumCircuit(1), after, pass_name="width_change")


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_source_compilation_preserves_exact_text_and_finite_phase(level: int) -> None:
    """Native lowering produces source-bound text independently of execution."""
    circuit = QuantumCircuit(2, 1, global_phase=0.23)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(0.37, 1)
    circuit.measure(1, 0)
    source = "// original source α\n" + qasm2.dumps(circuit)
    compiled = compile_circuit_to_mlir(source, optimisation_level=level)
    assert compiled.pass_record.input_ir.source == source
    assert compiled.mlir_module.metadata["pass_sha256"] == compiled.pass_record.sha256
    assert compiled.pass_record.input_ir.global_phase == 0
    direct = compile_circuit_to_mlir(circuit, optimisation_level=level)
    assert direct.pass_record.input_ir.global_phase == pytest.approx(0.23)
    assert direct.pass_record.operator_error < 1e-12
    assert direct.output_circuit.count_ops()["measure"] == 1
    assert circuit.count_ops()["rz"] == 1


def test_native_hamiltonian_ansatz_reaches_original_compiler_facade() -> None:
    """An actual canonical Hamiltonian consumer supplies the qualified circuit."""
    coupling = build_knm_paper27(L=3)
    ansatz = knm_to_ansatz(coupling, reps=1)
    bound = ansatz.assign_parameters(np.linspace(0.1, 0.6, ansatz.num_parameters))
    compiled = compile_circuit_to_mlir(bound, optimisation_level=2)
    np.testing.assert_allclose(
        Statevector.from_instruction(bound).probabilities(),
        Statevector.from_instruction(compiled.output_circuit).probabilities(),
        atol=1e-12,
    )
    assert compiled.pass_record.input_ir.num_qubits == 3
    assert "cz" not in compiled.output_circuit.count_ops()


@pytest.mark.parametrize(
    "apply",
    [
        lambda c: c.swap(0, 1),
        lambda c: c.u(0.2, 0.3, 0.4, 0),
        lambda c: c.p(0.3, 0),
        lambda c: c.id(0),
        lambda c: c.sx(0),
    ],
    ids=["swap", "u", "p", "id", "sx"],
)
def test_native_standard_gate_exports_preserve_sdk_semantics(
    apply: Callable[[QuantumCircuit], object],
) -> None:
    """Native exporter vocabulary round-trips through the SDK compatibility registry."""
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.ry(0.37, 1)
    apply(circuit)
    compiled = compile_circuit_to_mlir(circuit, optimisation_level=2)
    assert compiled.pass_record.input_ir.operations[-1].name == circuit.data[-1].operation.name
    np.testing.assert_allclose(
        Statevector.from_instruction(compiled.output_circuit).probabilities(),
        Statevector.from_instruction(circuit).probabilities(),
        atol=1e-12,
    )
    assert compiled.pass_record.operator_error < 1e-12


@pytest.mark.parametrize("level", [2, 3])
def test_native_cyclic_permutation_retains_partial_readout(level: int) -> None:
    """Virtual three-qubit permutations become physical basis gates before readout."""
    circuit = QuantumCircuit(3, 2)
    circuit.x(0)
    circuit.ry(0.37, 1)
    circuit.swap(0, 1)
    circuit.swap(1, 2)
    circuit.measure([0, 2], [1, 0])
    compiled = compile_circuit_to_mlir(circuit, optimisation_level=level)
    assert compiled.pass_record.input_layout == compiled.pass_record.output_layout == (0, 1, 2)
    assert (
        compiled.pass_record.input_ir.measurements == compiled.pass_record.output_ir.measurements
    )
    np.testing.assert_allclose(
        Statevector.from_instruction(
            compiled.output_circuit.remove_final_measurements(inplace=False)
        ).probabilities(),
        Statevector.from_instruction(
            circuit.remove_final_measurements(inplace=False)
        ).probabilities(),
        atol=1e-12,
    )
    assert compiled.pass_record.operator_error < 1e-12


def test_snapshot_does_not_follow_later_native_mutation() -> None:
    """Saved pass evidence remains immutable after caller circuit edits."""
    circuit = QuantumCircuit(1)
    circuit.h(0)
    compiled = compile_circuit_to_mlir(circuit)
    digest = compiled.pass_record.sha256
    circuit.x(0)
    compiled.output_circuit.x(0)
    assert compiled.pass_record.sha256 == digest
    assert len(compiled.pass_record.input_ir.operations) == 1
    with pytest.raises(FrozenInstanceError):
        field_name = "pass_name"
        setattr(compiled.pass_record, field_name, "changed")


def test_cancelled_reference_does_not_return_qualification() -> None:
    """Existing admission cancellation refuses native reference work."""
    event = Event()
    event.set()
    circuit = QuantumCircuit(1)
    with pytest.raises(ExecutionCancelledError, match="cancel"):
        qualify_circuit_pass(circuit, circuit, pass_name="cancelled", cancelled=event)
