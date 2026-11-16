# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Seeded simulated counts for every planned cell, reproducible from the record alone."""

from __future__ import annotations

import bisect
import hashlib
import itertools
import math
from dataclasses import dataclass

from .exact import _prepared_state
from .experiment import PlannedExperiment
from .measurement import MeasurementPlan
from .plan_set import CircuitPlanSet
from .wire import (
    SIMULATED_COUNT_SET_SCHEMA,
    ArtifactHeader,
    _digest,
    _strict_json,
    _text,
    canonical_bytes,
)

SAMPLER_ID = "sha256_counter_inverse_cdf_v1"
_DOMAIN = b"scpn.experimental.llm_qpu.simulated_counts.v1\x00"
_SIMULATORS = ("qiskit_statevector",)
_HEX = frozenset("0123456789abcdef")
_Operations = tuple[tuple[str, tuple[int, ...], str], ...]


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _rotate(state: list[complex], gate: str, wire: int) -> list[complex]:
    """Apply one H or S-dagger readout rotation on one wire."""
    out = list(state)
    bit = 1 << wire
    for index, amplitude in enumerate(state):
        if gate == "sdg":
            out[index] = -1j * amplitude if index & bit else amplitude
        elif index & bit:
            out[index] = (state[index ^ bit] - amplitude) / math.sqrt(2.0)
        else:
            out[index] = (amplitude + state[index | bit]) / math.sqrt(2.0)
    return out


def cell_probabilities(operations: _Operations, measurement: MeasurementPlan) -> tuple[float, ...]:
    """Return the outcome probabilities of one cell, indexed with logical qubit w as bit w."""
    state = _prepared_state(operations, measurement.n_qubits)
    for gate, wire in measurement.readout_rotations():
        state = _rotate(state, gate, wire)
    return tuple(abs(amplitude) ** 2 for amplitude in state)


def _register_key(outcome: int, measurement: MeasurementPlan) -> str:
    """Write one logical outcome as a raw key in the register's display order."""
    logical = {clbit: wire for wire, clbit in enumerate(measurement.logical_to_clbit)}
    return "".join(
        str(outcome >> logical[clbit] & 1) for clbit in measurement.display_order_clbits
    )


def draw_counts(
    probabilities: tuple[float, ...], measurement: MeasurementPlan, *, seed_hex: str, ordinal: int
) -> tuple[tuple[str, int], ...]:
    """Draw one cell's shots by inverse CDF over SHA-256 counter uniforms and return sorted counts.

    Only outcomes of nonzero probability are searched, and a uniform past the rounded cumulative
    sum lands on the last of them, so no shot ever falls on an outcome of zero probability.
    """
    support = [index for index, p in enumerate(probabilities) if p > 0.0]
    cumulative = list(itertools.accumulate(probabilities[index] for index in support))
    last = len(cumulative) - 1
    prefix = _DOMAIN + bytes.fromhex(seed_hex) + ordinal.to_bytes(4, "big")
    tallies: dict[int, int] = {}
    for shot in range(measurement.shots):
        block = hashlib.sha256(prefix + shot.to_bytes(4, "big")).digest()
        uniform = (int.from_bytes(block[:8], "big") >> 11) * 2.0**-53
        outcome = support[bisect.bisect_right(cumulative, uniform, hi=last)]
        tallies[outcome] = tallies.get(outcome, 0) + 1
    return tuple(
        sorted((_register_key(outcome, measurement), count) for outcome, count in tallies.items())
    )


@dataclass(frozen=True, slots=True)
class SimulatedCell:
    """One planned cell's seeded counts, keyed in the register's display order."""

    cell_ordinal: int
    planned_cell_digest: str
    measurement_request_digest: str
    shots: int
    counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        """Require sorted unique n4 or n8 keys and positive counts summing to the shots."""
        if type(self.cell_ordinal) is not int or not 0 <= self.cell_ordinal < 4096:
            raise ValueError("cell ordinal must be a bounded non-negative integer")
        for name in ("planned_cell_digest", "measurement_request_digest"):
            _digest(getattr(self, name), name=name.replace("_", " "))
        if type(self.shots) is not int or not 0 < self.shots <= 1_000_000:
            raise ValueError("shots must be a positive bounded integer")
        pairs = self.counts
        if (
            type(pairs) is not tuple
            or not pairs
            or any(type(p) is not tuple or len(p) != 2 for p in pairs)
        ):
            raise ValueError("counts must be (key, count) pairs")
        keys = [key for key, _ in pairs]
        widths = {len(key) if type(key) is str else 0 for key in keys}
        if len(widths) != 1 or widths - {4, 8} or any(set(key) - {"0", "1"} for key in keys):
            raise ValueError("count keys must be n4 or n8 bitstrings of one width")
        if keys != sorted(set(keys)):
            raise ValueError("count keys must be sorted and unique")
        if any(type(count) is not int or count <= 0 for _, count in pairs):
            raise ValueError("every count must be a positive integer")
        if sum(count for _, count in pairs) != self.shots:
            raise ValueError("counts must sum to the cell's shots")

    def to_wire(self) -> dict[str, object]:
        """Return the simulated cell with list-encoded counts."""
        return {
            "cell_ordinal": self.cell_ordinal,
            "planned_cell_digest": self.planned_cell_digest,
            "measurement_request_digest": self.measurement_request_digest,
            "shots": self.shots,
            "counts": [[key, count] for key, count in self.counts],
        }


@dataclass(frozen=True, slots=True)
class SimulatedCountSet:
    """Design-only seeded counts for every cell of one planned experiment; never provider results."""

    planned_experiment_digest: str
    sampler: str
    seed_hex: str
    simulator: str
    simulator_version: str
    cells: tuple[SimulatedCell, ...]
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse an unknown sampler or simulator, a malformed seed and unordered cells."""
        _digest(self.planned_experiment_digest, name="planned experiment digest")
        if self.sampler != SAMPLER_ID:
            raise ValueError("unknown simulated-count sampler")
        seed = self.seed_hex
        if type(seed) is not str or len(seed) != 64 or set(seed) - _HEX:
            raise ValueError("seed must be 64 lowercase hex characters")
        if self.simulator not in _SIMULATORS:
            raise ValueError("unknown ideal-digital simulator")
        _text(self.simulator_version, name="simulator version")
        cells = self.cells
        if (
            type(cells) is not tuple
            or not cells
            or any(type(c) is not SimulatedCell for c in cells)
        ):
            raise ValueError("cells must be a nonempty tuple of SimulatedCell records")
        if [cell.cell_ordinal for cell in cells] != list(range(len(cells))):
            raise ValueError("cells must follow the experiment's cell order")
        if (
            type(self.header) is not ArtifactHeader
            or self.header.object_kind != "simulated_count_set"
        ):
            raise ValueError("SimulatedCountSet requires its simulated artifact header")
        self.header.validate_content(
            self._scientific_wire(), parents=(self.planned_experiment_digest,)
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return the count-set content hashed by its header."""
        return {
            "schema": SIMULATED_COUNT_SET_SCHEMA,
            "object_kind": "simulated_count_set",
            "planned_experiment_digest": self.planned_experiment_digest,
            "sampler": self.sampler,
            "seed_hex": self.seed_hex,
            "simulator": self.simulator,
            "simulator_version": self.simulator_version,
            "cells": [cell.to_wire() for cell in self.cells],
        }

    def to_wire(self) -> dict[str, object]:
        """Return the detached versioned count set."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> SimulatedCountSet:
        """Reject unknown fields and restore cells and counts."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("SimulatedCountSet fields mismatch")
        if (
            value["schema"] != SIMULATED_COUNT_SET_SCHEMA
            or value["object_kind"] != "simulated_count_set"
        ):
            raise ValueError("unknown SimulatedCountSet schema")
        if type(value["cells"]) is not list:
            raise ValueError("cells must be a list on wire")
        return cls(
            planned_experiment_digest=value["planned_experiment_digest"],
            sampler=value["sampler"],
            seed_hex=value["seed_hex"],
            simulator=value["simulator"],
            simulator_version=value["simulator_version"],
            cells=tuple(_cell_from_wire(cell) for cell in value["cells"]),
            header=ArtifactHeader.from_wire(value["header"]),
        )


def _cell_from_wire(value: object) -> SimulatedCell:
    """Decode one simulated cell."""
    fields = {
        "cell_ordinal",
        "planned_cell_digest",
        "measurement_request_digest",
        "shots",
        "counts",
    }
    if type(value) is not dict or set(value) != fields or type(value["counts"]) is not list:
        raise ValueError("SimulatedCell fields mismatch")
    if any(type(pair) is not list or len(pair) != 2 for pair in value["counts"]):
        raise ValueError("counts must be [key, count] pairs on wire")
    return SimulatedCell(
        value["cell_ordinal"],
        value["planned_cell_digest"],
        value["measurement_request_digest"],
        value["shots"],
        tuple((pair[0], pair[1]) for pair in value["counts"]),
    )


def experiment_circuits(
    experiment: PlannedExperiment, plan_set: CircuitPlanSet
) -> dict[str, _Operations]:
    """Map every data plan and sentinel digest to its gate list."""
    circuits: dict[str, _Operations] = {
        _sha(plan.to_wire()): plan.operations for plan in plan_set.plans
    }
    for sentinel in experiment.sentinels:
        turn = (("ry", (wire,), math.pi.hex()) for wire in range(sentinel.n_qubits))
        circuits[_sha(sentinel.to_wire())] = tuple(turn) if sentinel.state_id == "one" else ()
    return circuits


def validate_simulated_count_set(
    counts: SimulatedCountSet, experiment: PlannedExperiment, plan_set: CircuitPlanSet
) -> None:
    """Refuse another experiment or plan set, misaligned cells or counts the seed does not give."""
    if type(counts) is not SimulatedCountSet:
        raise ValueError("simulated-count validation needs a SimulatedCountSet")
    if counts.planned_experiment_digest != _sha(experiment.to_wire()):
        raise ValueError("the counts name a different planned experiment")
    if experiment.plan_set_digest != _sha(plan_set.to_wire()):
        raise ValueError("the experiment names a different plan set")
    if counts.header.data_origin != experiment.header.data_origin:
        raise ValueError("count origin differs from the planned experiment")
    if len(counts.cells) != len(experiment.cells):
        raise ValueError("every planned cell needs exactly one simulated cell")
    circuits = experiment_circuits(experiment, plan_set)
    for cell, planned, measurement in zip(
        counts.cells, experiment.cells, experiment.measurements, strict=True
    ):
        lineage = (_sha(planned.to_wire()), measurement.request_digest, planned.shots)
        if (cell.planned_cell_digest, cell.measurement_request_digest, cell.shots) != lineage:
            raise ValueError("a simulated cell does not follow its planned cell")
        operations = circuits.get(measurement.circuit_plan_digest)
        if operations is None:
            raise ValueError("a planned cell names no plan or sentinel of the experiment")
        probabilities = cell_probabilities(operations, measurement)
        expected = draw_counts(
            probabilities, measurement, seed_hex=counts.seed_hex, ordinal=cell.cell_ordinal
        )
        if cell.counts != expected:
            raise ValueError("simulated counts differ from the seeded sampler")


def decode_simulated_count_set(raw: bytes) -> SimulatedCountSet:
    """Decode one simulated count set from strict canonical wire bytes."""
    return SimulatedCountSet.from_wire(_strict_json(raw))
