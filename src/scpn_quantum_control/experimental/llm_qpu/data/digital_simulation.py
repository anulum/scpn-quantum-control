# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU ideal digital simulation
"""Exact expectations and seeded counts of every plan, computed with the Qiskit statevector."""

from __future__ import annotations

import hashlib
from typing import Any

from ..contracts.exact import ExactFeatureSet, ExactRow, validate_exact_feature_set
from ..contracts.experiment import PlannedExperiment
from ..contracts.measurement import MeasurementPlan
from ..contracts.plan_set import CircuitPlanSet
from ..contracts.sampled import (
    SAMPLER_ID,
    SimulatedCell,
    SimulatedCountSet,
    cell_probabilities,
    draw_counts,
    experiment_circuits,
    validate_simulated_count_set,
)
from ..contracts.static import StaticCircuitPlan
from ..contracts.wire import (
    EXACT_FEATURE_SET_SCHEMA,
    SIMULATED_COUNT_SET_SCHEMA,
    ArtifactHeader,
    canonical_bytes,
)


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _qiskit_expectations(
    plan: StaticCircuitPlan, n_qubits: int, bases: tuple[str, ...]
) -> tuple[tuple[str, tuple[float, ...]], ...]:
    """Render the gate list into a Qiskit circuit and read every same-axis expectation."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Pauli, Statevector

    circuit = QuantumCircuit(n_qubits)
    for gate, wires, parameter in plan.operations:
        getattr(circuit, gate)(float.fromhex(parameter), *wires)
    state = Statevector.from_instruction(circuit)

    def value(basis: str, wires: tuple[int, ...]) -> float:
        label = ["I"] * n_qubits
        for wire in wires:
            label[n_qubits - 1 - wire] = basis
        return float(state.expectation_value(Pauli("".join(label))).real)

    singles = [(wire,) for wire in range(n_qubits)]
    pairs = [(wire, wire + 1) for wire in range(n_qubits - 1)]
    return tuple(
        (basis, tuple(value(basis, wires) for wires in singles + pairs)) for basis in bases
    )


def simulate_exact(
    plan_set: CircuitPlanSet,
    n_qubits: int,
    *,
    bases: tuple[str, ...],
    revisions: tuple[str, str],
    tolerance: float = 1e-10,
) -> ExactFeatureSet:
    """Return the exact features of every plan and check them against the pure-Python statevector."""
    import qiskit

    rows = tuple(
        ExactRow(
            plan.row_index,
            plan.sample_id,
            plan.arm_id,
            _sha(plan.to_wire()),
            _qiskit_expectations(plan, n_qubits, bases),
        )
        for plan in plan_set.plans
    )
    fields: dict[str, Any] = {
        "plan_set_digest": _sha(plan_set.to_wire()),
        "simulator": "qiskit_statevector",
        "simulator_version": qiskit.__version__,
        "rows": rows,
        "tolerance_hex": tolerance.hex(),
    }
    content = {
        "schema": EXACT_FEATURE_SET_SCHEMA,
        "object_kind": "exact_feature_set",
        **fields,
        "rows": [row.to_wire() for row in rows],
    }
    header = ArtifactHeader(
        "exact_feature_set",
        _sha(content),
        (fields["plan_set_digest"],),
        revisions[0],
        revisions[1],
        "offline_design",
        plan_set.header.data_origin,
        "design_only",
    )
    features = ExactFeatureSet(**fields, header=header)
    validate_exact_feature_set(features, plan_set, n_qubits)
    return features


def _qiskit_probabilities(
    operations: tuple[tuple[str, tuple[int, ...], str], ...], measurement: MeasurementPlan
) -> list[float]:
    """Render one cell with its readout rotations and return Qiskit's outcome probabilities."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    circuit = QuantumCircuit(measurement.n_qubits)
    for gate, wires, parameter in operations:
        getattr(circuit, gate)(float.fromhex(parameter), *wires)
    for gate, wire in measurement.readout_rotations():
        getattr(circuit, gate)(wire)
    return [float(p) for p in Statevector.from_instruction(circuit).probabilities()]


def simulate_sampled(
    experiment: PlannedExperiment,
    plan_set: CircuitPlanSet,
    *,
    seed_hex: str,
    revisions: tuple[str, str],
    tolerance: float = 1e-10,
) -> SimulatedCountSet:
    """Return seeded counts for every planned cell after checking each cell's probabilities in Qiskit."""
    import qiskit

    circuits = experiment_circuits(experiment, plan_set)
    cells = []
    for ordinal, (planned, measurement) in enumerate(
        zip(experiment.cells, experiment.measurements, strict=True)
    ):
        operations = circuits[measurement.circuit_plan_digest]
        probabilities = cell_probabilities(operations, measurement)
        reference = _qiskit_probabilities(operations, measurement)
        if max(abs(a - b) for a, b in zip(probabilities, reference, strict=True)) > tolerance:
            raise ValueError("Qiskit and the independent statevector disagree on a cell")
        counts = draw_counts(probabilities, measurement, seed_hex=seed_hex, ordinal=ordinal)
        cells.append(
            SimulatedCell(
                ordinal, _sha(planned.to_wire()), measurement.request_digest, planned.shots, counts
            )
        )
    fields: dict[str, Any] = {
        "planned_experiment_digest": _sha(experiment.to_wire()),
        "sampler": SAMPLER_ID,
        "seed_hex": seed_hex,
        "simulator": "qiskit_statevector",
        "simulator_version": qiskit.__version__,
        "cells": tuple(cells),
    }
    content = {
        "schema": SIMULATED_COUNT_SET_SCHEMA,
        "object_kind": "simulated_count_set",
        **fields,
        "cells": [cell.to_wire() for cell in cells],
    }
    header = ArtifactHeader(
        "simulated_count_set",
        _sha(content),
        (fields["planned_experiment_digest"],),
        revisions[0],
        revisions[1],
        "offline_design",
        experiment.header.data_origin,
        "design_only",
    )
    simulated = SimulatedCountSet(**fields, header=header)
    validate_simulated_count_set(simulated, experiment, plan_set)
    return simulated
