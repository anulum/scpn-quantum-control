# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU measurement compiler
"""Render planned cells as measured Qiskit circuits whose count keys decode back to the logical bits."""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from typing import TYPE_CHECKING

from ..contracts.experiment import PlannedExperiment
from ..contracts.measurement import MeasurementPlan, _observable_ids
from ..contracts.plan_set import CircuitPlanSet
from ..contracts.sampled import cell_probabilities, experiment_circuits
from ..contracts.wire import canonical_bytes

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

_Operations = tuple[tuple[str, tuple[int, ...], str], ...]


def render_measured_cell(operations: _Operations, measurement: MeasurementPlan) -> QuantumCircuit:
    """Return the cell's gates, its readout rotations and one measurement per wire into its clbit."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    qubits = QuantumRegister(measurement.n_qubits, "q")
    clbits = ClassicalRegister(measurement.n_qubits, measurement.register_name)
    circuit = QuantumCircuit(qubits, clbits)
    for gate, wires, parameter in operations:
        getattr(circuit, gate)(float.fromhex(parameter), *(qubits[wire] for wire in wires))
    for gate, wire in measurement.readout_rotations():
        getattr(circuit, gate)(qubits[wire])
    for wire, clbit in enumerate(measurement.logical_to_clbit):
        circuit.measure(qubits[wire], clbits[clbit])
    return circuit


def check_bit_custody(measurement: MeasurementPlan) -> int:
    """Prepare every logical basis state, read it through Qiskit and require the declared decoding.

    Returns the number of basis states checked. A register whose display order does not describe
    the sampler's count keys is refused before any of its counts can be interpreted.
    """
    from qiskit.primitives import StatevectorSampler

    width = measurement.n_qubits
    z_basis = replace(measurement, basis_id="Z", observable_ids=_observable_ids("Z", width))
    circuits = []
    for state in range(1 << width):
        flips = tuple(("ry", (wire,), math.pi.hex()) for wire in range(width) if state >> wire & 1)
        circuits.append(render_measured_cell(flips, z_basis))
    results = StatevectorSampler(seed=0).run(circuits, shots=1).result()
    virtual = replace(measurement, physical_qubits=tuple(range(width)))
    for state, result in enumerate(results):
        (key,) = getattr(result.data, measurement.register_name).get_counts()
        if virtual.logical_bits(key) != tuple(state >> wire & 1 for wire in range(width)):
            raise ValueError(
                "the register's display order does not match the sampler's count keys"
            )
    return 1 << width


def render_measured_experiment(
    experiment: PlannedExperiment, plan_set: CircuitPlanSet, *, tolerance: float = 1e-10
) -> tuple[QuantumCircuit, ...]:
    """Render every planned cell after its register passes bit custody and its gates match the contract."""
    from qiskit.quantum_info import Statevector

    if (
        experiment.plan_set_digest
        != hashlib.sha256(canonical_bytes(plan_set.to_wire())).hexdigest()
    ):
        raise ValueError("the experiment names a different plan set")
    circuits = experiment_circuits(experiment, plan_set)
    checked: set[tuple[object, ...]] = set()
    rendered = []
    for measurement in experiment.measurements:
        register = (
            measurement.n_qubits,
            measurement.logical_to_clbit,
            measurement.display_order_clbits,
        )
        if register not in checked:
            check_bit_custody(measurement)
            checked.add(register)
        operations = circuits[measurement.circuit_plan_digest]
        circuit = render_measured_cell(operations, measurement)
        reference = Statevector.from_instruction(circuit.remove_final_measurements(inplace=False))
        expected = cell_probabilities(operations, measurement)
        if (
            max(abs(a - b) for a, b in zip(reference.probabilities(), expected, strict=True))
            > tolerance
        ):
            raise ValueError("a rendered cell differs from the independent statevector")
        rendered.append(circuit)
    return tuple(rendered)
