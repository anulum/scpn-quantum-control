# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU compiled-cell proof
"""Prove that one compiled cell uses the target basis, reads its layout qubits and keeps the logical distribution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..contracts.experiment import PlannedExperiment
from ..contracts.measurement import MeasurementPlan
from ..contracts.plan_set import CircuitPlanSet
from ..contracts.sampled import cell_probabilities, experiment_circuits
from ..contracts.target import TargetSnapshot

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


def check_compiled_cell(
    compiled: QuantumCircuit,
    operations: tuple[tuple[str, tuple[int, ...], str], ...],
    measurement: MeasurementPlan,
    target: TargetSnapshot,
    layout: tuple[int, ...],
    tolerance: float,
    *,
    allowed: frozenset[str] | None = None,
) -> None:
    """Require allowed gates only, each logical wire read from its layout qubit, and the logical distribution.

    ``allowed`` defaults to the target basis; a provider adapter that names its native gates differently
    at the Qiskit level (IQM's ``prx`` is Qiskit's ``r``) passes the names its own target declares.
    """
    from qiskit.quantum_info import Statevector

    names = set(target.basis_gates) if allowed is None else set(allowed)
    outside = sorted({item.operation.name for item in compiled.data} - names)
    if outside:
        raise ValueError(f"compiled gates outside the target basis: {', '.join(outside)}")
    measured = sorted(
        (compiled.find_bit(item.qubits[0]).index, compiled.find_bit(item.clbits[0]).index)
        for item in compiled.data
        if item.operation.name == "measure"
    )
    if measured != sorted(zip(layout, measurement.logical_to_clbit, strict=True)):
        raise ValueError("a compiled cell does not read each logical wire from its layout qubit")
    probabilities = Statevector.from_instruction(_reduced(compiled, layout)).probabilities()
    expected = cell_probabilities(operations, measurement)
    if max(abs(a - b) for a, b in zip(probabilities, expected, strict=True)) > tolerance:
        raise ValueError("a compiled cell differs from the independent statevector")


def _reduced(compiled: QuantumCircuit, layout: tuple[int, ...]) -> QuantumCircuit:
    """Return the compiled unitary part on the layout qubits, logical wire w as qubit w."""
    from qiskit import QuantumCircuit

    index = {physical: logical for logical, physical in enumerate(layout)}
    reduced = QuantumCircuit(len(layout), global_phase=compiled.global_phase)
    for item in compiled.data:
        if item.operation.name in ("measure", "barrier"):
            continue
        qubits = [compiled.find_bit(qubit).index for qubit in item.qubits]
        if any(qubit not in index for qubit in qubits):
            raise ValueError("a compiled cell acts on a qubit outside its layout")
        reduced.append(item.operation, [index[qubit] for qubit in qubits])
    return reduced


def _check_all(
    compiled: tuple[QuantumCircuit, ...],
    experiment: PlannedExperiment,
    plan_set: CircuitPlanSet,
    target: TargetSnapshot,
    layout: tuple[int, ...],
    tolerance: float,
    *,
    allowed: frozenset[str] | None = None,
) -> None:
    """Prove every compiled cell against its measurement and its logical gates."""
    operations = experiment_circuits(experiment, plan_set)
    for circuit, measurement in zip(compiled, experiment.measurements, strict=True):
        check_compiled_cell(
            circuit,
            operations[measurement.circuit_plan_digest],
            measurement,
            target,
            layout,
            tolerance,
            allowed=allowed,
        )
