# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU ideal digital simulation
"""Exact ideal-digital expectations of every plan, computed with the Qiskit statevector."""

from __future__ import annotations

import hashlib
from typing import Any

from ..contracts.exact import ExactFeatureSet, ExactRow, validate_exact_feature_set
from ..contracts.plan_set import CircuitPlanSet
from ..contracts.static import StaticCircuitPlan
from ..contracts.wire import EXACT_FEATURE_SET_SCHEMA, ArtifactHeader, canonical_bytes


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
