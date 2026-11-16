# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental measurement compiler
"""Measured circuits whose Qiskit count keys decode back to the logical bits of every cell."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any

import pytest
from _llm_qpu_dev_chain import _ARMS, _planned, _sha, _spec
from qiskit.primitives import StatevectorSampler

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    MeasurementPlan,
    cell_probabilities,
    estimate_sampled_basis,
    exact_expectations,
)
from scpn_quantum_control.experimental.llm_qpu.contracts.measurement import _observable_ids
from scpn_quantum_control.experimental.llm_qpu.contracts.sampled import experiment_circuits
from scpn_quantum_control.experimental.llm_qpu.data import measurement_compiler
from scpn_quantum_control.experimental.llm_qpu.data.measurement_compiler import (
    check_bit_custody,
    render_measured_cell,
    render_measured_experiment,
)

_SHUFFLED = (1, 2, 3, 0)


def _measures(circuit: Any) -> list[tuple[int, int]]:
    """Return each measurement as (qubit index, clbit index) in circuit order."""
    return [
        (circuit.find_bit(item.qubits[0]).index, circuit.find_bit(item.clbits[0]).index)
        for item in circuit.data
        if item.operation.name == "measure"
    ]


def _z_plan(width: int, logical: tuple[int, ...], display: tuple[int, ...]) -> MeasurementPlan:
    """Return an unbound Z measurement of the given register maps."""
    return MeasurementPlan(
        "a" * 64,
        width,
        "Z",
        1,
        _observable_ids("Z", width),
        logical,
        display,
        "readout",
        None,
        "logical_q0_low_index_v1",
    )


def test_rendered_cells_measure_each_logical_wire_into_its_declared_clbit() -> None:
    """Every cell is rendered once, with its rotations, into the named register under the declared map."""
    planned = _planned(spec=_spec(logical_to_clbit=_SHUFFLED))
    circuits = render_measured_experiment(planned.experiment, planned.plan_set)
    assert len(circuits) == len(planned.experiment.cells) == 16
    for circuit, measurement in zip(circuits, planned.experiment.measurements, strict=True):
        assert [register.name for register in circuit.cregs] == ["readout"]
        assert circuit.num_clbits == 4
        assert _measures(circuit) == list(enumerate(_SHUFFLED))
        names = [item.operation.name for item in circuit.data]
        rotations = [name for name in names if name in ("h", "sdg")]
        assert rotations == [gate for gate, _ in measurement.readout_rotations()]
        assert names[-4:] == ["measure"] * 4


def test_bit_custody_accepts_any_logical_map_and_refuses_a_foreign_display_order() -> None:
    """Qiskit writes the highest clbit first, so only a descending display order describes its keys."""
    assert check_bit_custody(_z_plan(4, (0, 1, 2, 3), (3, 2, 1, 0))) == 16
    assert check_bit_custody(_z_plan(4, _SHUFFLED, (3, 2, 1, 0))) == 16
    assert check_bit_custody(_z_plan(8, (7, 0, 6, 1, 5, 2, 4, 3), tuple(range(7, -1, -1)))) == 256
    for display in ((0, 1, 2, 3), (3, 1, 2, 0)):
        with pytest.raises(ValueError, match="^the register's display order does not match"):
            check_bit_custody(_z_plan(4, _SHUFFLED, display))
    refused = _planned(spec=_spec(display_order_clbits=(0, 1, 2, 3)))
    with pytest.raises(ValueError, match="^the register's display order does not match"):
        render_measured_experiment(refused.experiment, refused.plan_set)


def test_qiskit_counts_of_rendered_cells_decode_to_the_exact_expectations() -> None:
    """Sampled through Qiskit and decoded by the plan, every estimate lies within five sigma of exact."""
    planned = _planned(spec=_spec(logical_to_clbit=_SHUFFLED, data_shots=20_000))
    circuits = render_measured_experiment(planned.experiment, planned.plan_set)
    plans = {_sha(plan.to_wire()): plan for plan in planned.plan_set.plans}
    operations = experiment_circuits(planned.experiment, planned.plan_set)
    data = [
        (circuit, measurement)
        for circuit, measurement, cell in zip(
            circuits, planned.experiment.measurements, planned.experiment.cells, strict=True
        )
        if cell.role == "data"
    ]
    results = StatevectorSampler(seed=11).run([c for c, _ in data], shots=20_000).result()
    gaps = []
    for (_, measurement), result in zip(data, results, strict=True):
        counts = getattr(result.data, measurement.register_name).get_counts()
        virtual = replace(measurement, physical_qubits=(0, 1, 2, 3))
        probabilities = cell_probabilities(
            operations[measurement.circuit_plan_digest], measurement
        )
        for key in counts:
            bits = virtual.logical_bits(key)
            assert probabilities[sum(bit << wire for wire, bit in enumerate(bits))] > 0.0
        estimate = estimate_sampled_basis(virtual, counts, origin="digital_sampled")
        exact = exact_expectations(plans[measurement.circuit_plan_digest], 4, measurement.basis_id)
        covariance = estimate.covariance_of_mean
        assert covariance is not None
        for index, (value, ideal) in enumerate(zip(estimate.expectations, exact, strict=True)):
            gaps.append(abs(value - ideal) / math.sqrt(covariance[index][index]))
    assert len(gaps) == 4 * 3 * 7
    assert max(gaps) < 5.0


def test_rendering_refuses_another_plan_set_and_a_cell_off_the_independent_statevector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The experiment must name the plan set, and each rendered cell must match the contract."""
    planned = _planned()
    other = _planned(arms=(_ARMS[1],))
    with pytest.raises(ValueError, match="^the experiment names a different plan set$"):
        render_measured_experiment(planned.experiment, other.plan_set)
    original = cell_probabilities

    def drifted(operations: Any, measurement: MeasurementPlan) -> tuple[float, ...]:
        """Return the contract's probabilities with 1e-9 moved between the first two outcomes."""
        values = original(operations, measurement)
        return (values[0] + 1e-9, values[1] - 1e-9, *values[2:])

    monkeypatch.setattr(measurement_compiler, "cell_probabilities", drifted)
    with pytest.raises(
        ValueError, match="^a rendered cell differs from the independent statevector$"
    ):
        render_measured_experiment(planned.experiment, planned.plan_set)
    tolerant = render_measured_experiment(planned.experiment, planned.plan_set, tolerance=1e-8)
    assert len(tolerant) == len(planned.experiment.cells)


def test_a_single_rendered_cell_carries_its_gates_before_the_readout() -> None:
    """The plan's gates come first, in order, with their exact hex parameters."""
    planned = _planned()
    plan = planned.plan_set.plans[1]
    measurement = next(
        m for m in planned.experiment.measurements if m.circuit_plan_digest == _sha(plan.to_wire())
    )
    circuit = render_measured_cell(plan.operations, measurement)
    gates = [
        (
            item.operation.name,
            tuple(circuit.find_bit(q).index for q in item.qubits),
            item.operation.params,
        )
        for item in circuit.data[: len(plan.operations)]
    ]
    assert gates == [
        (gate, wires, [float.fromhex(parameter)]) for gate, wires, parameter in plan.operations
    ]
