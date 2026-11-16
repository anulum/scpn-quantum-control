# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Planned experiments: one cell per plan and basis plus readout sentinels, rebuilt exactly."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

from .cells import CellKey, PlannedCell
from .measurement import MeasurementPlan, _observable_ids, build_measurement_plan
from .plan_set import CircuitPlanSet
from .static import ReservoirSpec
from .wire import (
    MEASUREMENT_SPEC_SCHEMA,
    PLANNED_EXPERIMENT_SCHEMA,
    SENTINEL_SCHEMA,
    ArtifactHeader,
    _digest,
    _positive_int,
    _strict_json,
    _text,
    canonical_bytes,
)

_BASES = ("X", "Y", "Z")
_STATES = ("one", "zero")


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _sorted_unique(values: object, *, name: str, allowed: tuple[str, ...] | None = None) -> None:
    """Require a nonempty sorted unique tuple of text, optionally from a fixed vocabulary."""
    if type(values) is not tuple or not values or any(type(value) is not str for value in values):
        raise ValueError(f"{name} must be a nonempty sorted unique tuple")
    if values != tuple(sorted(set(values))):
        raise ValueError(f"{name} must be a nonempty sorted unique tuple")
    for value in values:
        _text(value, name=name)
        if allowed is not None and value not in allowed:
            raise ValueError(f"unknown {name}: {value}")


def _permutation(values: object, width: int, *, name: str) -> None:
    """Require a complete permutation of the classical bits."""
    if type(values) is not tuple or sorted(values) != list(range(width)):
        raise ValueError(f"{name} must be a complete permutation")
    if any(type(value) is not int for value in values):
        raise ValueError(f"{name} must be a complete permutation")


@dataclass(frozen=True, slots=True)
class SentinelPreparation:
    """A computational-basis readout sentinel: all zeros, or RY(pi) on every wire."""

    state_id: str
    n_qubits: int

    def __post_init__(self) -> None:
        """Refuse any state but the two basis extremes and unsupported widths."""
        if self.state_id not in _STATES:
            raise ValueError("a sentinel prepares the zero or the one state")
        if type(self.n_qubits) is not int or self.n_qubits not in (4, 8):
            raise ValueError("a sentinel needs an n4 or n8 register")

    def to_wire(self) -> dict[str, object]:
        """Return the exact preparation, read out in Z for sanity, never for mitigation."""
        gates = (
            []
            if self.state_id == "zero"
            else [
                {"gate": "ry", "wire": wire, "angle_hex": math.pi.hex()}
                for wire in range(self.n_qubits)
            ]
        )
        return {
            "schema": SENTINEL_SCHEMA,
            "object_kind": "sentinel_preparation",
            "state_id": self.state_id,
            "n_qubits": self.n_qubits,
            "initial_state": "zero",
            "operations": gates,
            "basis_id": "Z",
            "purpose": "bounded_readout_sanity_not_mitigation",
        }


@dataclass(frozen=True, slots=True)
class MeasurementSpec:
    """How every plan of a set is read out, and which sentinels accompany it."""

    experiment_id: str
    bases: tuple[str, ...]
    data_shots: int
    logical_to_clbit: tuple[int, ...]
    display_order_clbits: tuple[int, ...]
    register_name: str
    sentinel_states: tuple[str, ...]
    sentinel_slots: tuple[str, ...]
    sentinel_shots: int
    replicate_id: str
    planned_epoch_id: str

    def __post_init__(self) -> None:
        """Refuse unknown bases or sentinel states, unbounded shots and partial bit maps."""
        for name in ("experiment_id", "register_name", "replicate_id", "planned_epoch_id"):
            object.__setattr__(self, name, _text(getattr(self, name), name=name))
        _sorted_unique(self.bases, name="basis", allowed=_BASES)
        _sorted_unique(self.sentinel_states, name="sentinel state", allowed=_STATES)
        _sorted_unique(self.sentinel_slots, name="sentinel slot")
        for name in ("data_shots", "sentinel_shots"):
            _positive_int(getattr(self, name), name=name, maximum=1_000_000)
        width = len(self.logical_to_clbit) if type(self.logical_to_clbit) is tuple else 0
        if width not in (4, 8):
            raise ValueError("bit maps must cover an n4 or n8 register")
        _permutation(self.logical_to_clbit, width, name="logical-to-classical map")
        _permutation(self.display_order_clbits, width, name="display bit order")

    def to_wire(self) -> dict[str, object]:
        """Return the exact readout specification."""
        return {
            "schema": MEASUREMENT_SPEC_SCHEMA,
            "object_kind": "measurement_spec",
            **{
                name: list(value) if type(value) is tuple else value
                for name in self.__dataclass_fields__
                for value in (getattr(self, name),)
            },
        }

    @classmethod
    def from_wire(cls, value: object) -> MeasurementSpec:
        """Reject unknown fields and restore tuples."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("MeasurementSpec fields mismatch")
        if (
            value["schema"] != MEASUREMENT_SPEC_SCHEMA
            or value["object_kind"] != "measurement_spec"
        ):
            raise ValueError("unknown MeasurementSpec schema")
        restored: dict[str, Any] = {
            name: tuple(value[name]) if type(value[name]) is list else value[name]
            for name in cls.__dataclass_fields__
        }
        return cls(**restored)


def _data_cells(
    plan_set: CircuitPlanSet, reservoir: ReservoirSpec, spec: MeasurementSpec
) -> list[tuple[PlannedCell, MeasurementPlan]]:
    """Read every plan in every basis, keyed by sample, arm, basis and circuit."""
    pairs = []
    for plan in plan_set.plans:
        circuit = _sha(plan.to_wire())
        for basis in spec.bases:
            measurement = build_measurement_plan(
                plan,
                reservoir,
                basis_id=basis,
                shots=spec.data_shots,
                logical_to_clbit=spec.logical_to_clbit,
                display_order_clbits=spec.display_order_clbits,
                register_name=spec.register_name,
                physical_qubits=None,
            )
            key = CellKey(
                spec.experiment_id,
                plan.sample_id,
                "contextual_single",
                plan.arm_id,
                basis,
                spec.replicate_id,
                spec.planned_epoch_id,
                circuit,
            )
            pairs.append(
                (
                    _cell(key, circuit, spec.data_shots, reservoir.n_qubits, "data", measurement),
                    measurement,
                )
            )
    return pairs


def _cell(
    key: CellKey, circuit: str, shots: int, width: int, role: str, measurement: MeasurementPlan
) -> PlannedCell:
    """Return one planned cell bound to its measurement request."""
    return PlannedCell.create(
        key=key,
        circuit_plan_digest=circuit,
        shots=shots,
        measurement_width=width,
        role=role,
        metadata={"measurement_request_digest": measurement.request_digest},
    )


def _sentinel_cells(
    spec: MeasurementSpec, width: int
) -> tuple[tuple[SentinelPreparation, ...], list[tuple[PlannedCell, MeasurementPlan]]]:
    """Read each sentinel in Z once per slot."""
    sentinels = tuple(SentinelPreparation(state, width) for state in spec.sentinel_states)
    pairs = []
    for sentinel in sentinels:
        circuit = _sha(sentinel.to_wire())
        for slot in spec.sentinel_slots:
            measurement = MeasurementPlan(
                circuit,
                width,
                "Z",
                spec.sentinel_shots,
                _observable_ids("Z", width),
                spec.logical_to_clbit,
                spec.display_order_clbits,
                spec.register_name,
                None,
                "logical_q0_low_index_v1",
            )
            key = CellKey(
                spec.experiment_id,
                f"sentinel_{sentinel.state_id}",
                "none",
                "sentinel",
                "Z",
                f"slot_{slot}",
                spec.planned_epoch_id,
                circuit,
            )
            pairs.append(
                (
                    _cell(key, circuit, spec.sentinel_shots, width, "sentinel", measurement),
                    measurement,
                )
            )
    return sentinels, pairs


@dataclass(frozen=True, slots=True)
class PlannedExperiment:
    """Design-only cell inventory of one plan set under one measurement spec."""

    plan_set_digest: str
    measurement_spec_digest: str
    sentinels: tuple[SentinelPreparation, ...]
    cells: tuple[PlannedCell, ...]
    measurements: tuple[MeasurementPlan, ...]
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse duplicate keys and any cell not bound to its own measurement request."""
        for name in ("plan_set_digest", "measurement_spec_digest"):
            _digest(getattr(self, name), name=name)
        if type(self.sentinels) is not tuple or any(
            type(s) is not SentinelPreparation for s in self.sentinels
        ):
            raise ValueError("sentinels must be a tuple of SentinelPreparation records")
        cells, measurements = self.cells, self.measurements
        if type(cells) is not tuple or not cells or any(type(c) is not PlannedCell for c in cells):
            raise ValueError("cells must be a nonempty tuple of PlannedCell records")
        if type(measurements) is not tuple or any(
            type(m) is not MeasurementPlan for m in measurements
        ):
            raise ValueError("measurements must be a tuple of MeasurementPlan records")
        if len(measurements) != len(cells):
            raise ValueError("every cell needs exactly one measurement plan")
        if len({canonical_bytes(cell.key.to_wire()) for cell in cells}) != len(cells):
            raise ValueError("cell keys must be unique")
        for cell, measurement in zip(cells, measurements, strict=True):
            bound = _strict_json(cell.metadata_blob)
            if (cell.circuit_plan_digest, cell.shots) != (
                measurement.circuit_plan_digest,
                measurement.shots,
            ) or bound != {"measurement_request_digest": measurement.request_digest}:
                raise ValueError("a cell is not bound to its own measurement request")
        if (
            type(self.header) is not ArtifactHeader
            or self.header.object_kind != "planned_experiment"
        ):
            raise ValueError("PlannedExperiment requires its exact artifact header")
        self.header.validate_content(
            self._scientific_wire(), parents=(self.plan_set_digest, self.measurement_spec_digest)
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return the experiment content hashed by its header."""
        return {
            "schema": PLANNED_EXPERIMENT_SCHEMA,
            "object_kind": "planned_experiment",
            "plan_set_digest": self.plan_set_digest,
            "measurement_spec_digest": self.measurement_spec_digest,
            "sentinels": [sentinel.to_wire() for sentinel in self.sentinels],
            "cells": [cell.to_wire() for cell in self.cells],
            "measurements": [measurement.to_wire() for measurement in self.measurements],
        }

    def to_wire(self) -> dict[str, object]:
        """Return the detached versioned experiment."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> PlannedExperiment:
        """Reject unknown fields, restore records and refuse a noncanonical sentinel."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("PlannedExperiment fields mismatch")
        if (
            value["schema"] != PLANNED_EXPERIMENT_SCHEMA
            or value["object_kind"] != "planned_experiment"
        ):
            raise ValueError("unknown PlannedExperiment schema")
        if any(type(value[name]) is not list for name in ("sentinels", "cells", "measurements")):
            raise ValueError("sentinels, cells and measurements must be lists on wire")
        sentinels = []
        for item in value["sentinels"]:
            if type(item) is not dict or not {"state_id", "n_qubits"} <= set(item):
                raise ValueError("SentinelPreparation fields mismatch")
            sentinel = SentinelPreparation(item["state_id"], item["n_qubits"])
            if sentinel.to_wire() != item:
                raise ValueError("a sentinel preparation differs from its canonical form")
            sentinels.append(sentinel)
        return cls(
            plan_set_digest=value["plan_set_digest"],
            measurement_spec_digest=value["measurement_spec_digest"],
            sentinels=tuple(sentinels),
            cells=tuple(PlannedCell.from_wire(cell) for cell in value["cells"]),
            measurements=tuple(MeasurementPlan.from_wire(m) for m in value["measurements"]),
            header=ArtifactHeader.from_wire(value["header"]),
        )

    def total_shots(self) -> int:
        """Return the requested shots over every data and sentinel cell."""
        return sum(cell.shots for cell in self.cells)


def plan_measurements(
    plan_set: CircuitPlanSet,
    reservoir: ReservoirSpec,
    spec: MeasurementSpec,
    *,
    revisions: tuple[str, str],
) -> PlannedExperiment:
    """Plan every data cell and sentinel; ``revisions`` are base and implementation."""
    if type(spec) is not MeasurementSpec or len(spec.logical_to_clbit) != reservoir.n_qubits:
        raise ValueError("the measurement spec must read the reservoir's register")
    data = _data_cells(plan_set, reservoir, spec)
    sentinels, extra = _sentinel_cells(spec, reservoir.n_qubits)
    pairs = data + extra
    fields: dict[str, Any] = {
        "plan_set_digest": _sha(plan_set.to_wire()),
        "measurement_spec_digest": _sha(spec.to_wire()),
        "sentinels": sentinels,
        "cells": tuple(cell for cell, _ in pairs),
        "measurements": tuple(measurement for _, measurement in pairs),
    }
    content = {
        "schema": PLANNED_EXPERIMENT_SCHEMA,
        "object_kind": "planned_experiment",
        "plan_set_digest": fields["plan_set_digest"],
        "measurement_spec_digest": fields["measurement_spec_digest"],
        "sentinels": [s.to_wire() for s in sentinels],
        "cells": [c.to_wire() for c in fields["cells"]],
        "measurements": [m.to_wire() for m in fields["measurements"]],
    }
    header = ArtifactHeader(
        "planned_experiment",
        _sha(content),
        tuple(sorted((fields["plan_set_digest"], fields["measurement_spec_digest"]))),
        revisions[0],
        revisions[1],
        "offline_design",
        plan_set.header.data_origin,
        "design_only",
    )
    return PlannedExperiment(**fields, header=header)


def validate_planned_experiment(
    experiment: PlannedExperiment,
    plan_set: CircuitPlanSet,
    reservoir: ReservoirSpec,
    spec: MeasurementSpec,
) -> None:
    """Refuse an experiment for another plan set or spec, or any drifted cell."""
    if type(experiment) is not PlannedExperiment:
        raise ValueError("experiment validation needs a PlannedExperiment")
    if experiment.plan_set_digest != _sha(plan_set.to_wire()):
        raise ValueError("the experiment names a different plan set")
    if experiment.measurement_spec_digest != _sha(spec.to_wire()):
        raise ValueError("the experiment names a different measurement spec")
    if experiment.header.data_origin != plan_set.header.data_origin:
        raise ValueError("experiment origin differs from the plan set")
    expected = plan_measurements(
        plan_set,
        reservoir,
        spec,
        revisions=(experiment.header.base_repo_commit, experiment.header.implementation_revision),
    )
    if (experiment.sentinels, experiment.cells, experiment.measurements) != (
        expected.sentinels,
        expected.cells,
        expected.measurements,
    ):
        raise ValueError("a cell, measurement or sentinel differs from its exact plan")


def decode_planned_experiment(raw: bytes) -> PlannedExperiment:
    """Decode one planned experiment from strict canonical wire bytes."""
    return PlannedExperiment.from_wire(_strict_json(raw))
