# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Sampled features of seeded simulated counts, kept apart from provider-result features."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

from .cells import CellKey
from .experiment import PlannedExperiment
from .features import FeatureRow
from .measurement import MeasurementPlan, estimate_sampled_basis
from .sampled import SimulatedCell, SimulatedCountSet
from .wire import (
    SIMULATED_FEATURE_BATCH_SCHEMA,
    ArtifactHeader,
    _digest,
    _strict_json,
    _text,
    canonical_bytes,
)

_ESTIMATORS = ("sampled_mean_v1",)


def _sha(value: object) -> str:
    """Return the SHA-256 of a value's canonical wire bytes."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class SimulatedFeatureBatch:
    """Design-only [N, F] features of one basis from a simulated count set; never hardware features."""

    simulated_count_set_digest: str
    estimator: str
    basis_id: str
    observable_ids: tuple[str, ...]
    rows: tuple[FeatureRow, ...]
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse unknown estimators, missing rows, reordered rows and ragged widths."""
        _digest(self.simulated_count_set_digest, name="simulated count set digest")
        if self.estimator not in _ESTIMATORS:
            raise ValueError("unknown feature estimator")
        if self.basis_id not in ("X", "Y", "Z"):
            raise ValueError("feature basis must be X, Y or Z")
        names = self.observable_ids
        if type(names) is not tuple or not 0 < len(names) <= 64:
            raise ValueError("observable IDs must be a nonempty tuple of at most 64")
        if len({_text(name, name="observable ID") for name in names}) != len(names):
            raise ValueError("observable IDs must be unique")
        rows = self.rows
        if type(rows) is not tuple or not rows or any(type(row) is not FeatureRow for row in rows):
            raise ValueError("feature rows must be a nonempty tuple of FeatureRow records")
        if any(row.values is None for row in rows):
            raise ValueError("a simulated cell is never missing")
        ordinals = [row.cell_ordinal for row in rows]
        if ordinals != sorted(set(ordinals)):
            raise ValueError("feature rows must follow the experiment's cell order")
        if any(len(row.values or ()) != len(names) for row in rows):
            raise ValueError("every row has one value per observable")
        if (
            type(self.header) is not ArtifactHeader
            or self.header.object_kind != "simulated_feature_batch"
        ):
            raise ValueError("SimulatedFeatureBatch requires its simulated artifact header")
        self.header.validate_content(
            self._scientific_wire(), parents=(self.simulated_count_set_digest,)
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return the batch content hashed by its header."""
        return {
            "schema": SIMULATED_FEATURE_BATCH_SCHEMA,
            "object_kind": "simulated_feature_batch",
            "simulated_count_set_digest": self.simulated_count_set_digest,
            "estimator": self.estimator,
            "basis_id": self.basis_id,
            "observable_ids": list(self.observable_ids),
            "rows": [row.to_wire() for row in self.rows],
        }

    def to_wire(self) -> dict[str, object]:
        """Return the detached versioned batch."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> SimulatedFeatureBatch:
        """Reject unknown fields and restore the frozen batch."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("SimulatedFeatureBatch fields mismatch")
        if (
            value["schema"] != SIMULATED_FEATURE_BATCH_SCHEMA
            or value["object_kind"] != "simulated_feature_batch"
        ):
            raise ValueError("unknown SimulatedFeatureBatch schema")
        if type(value["observable_ids"]) is not list or type(value["rows"]) is not list:
            raise ValueError("observable IDs and rows must be lists on wire")
        return cls(
            simulated_count_set_digest=value["simulated_count_set_digest"],
            estimator=value["estimator"],
            basis_id=value["basis_id"],
            observable_ids=tuple(value["observable_ids"]),
            rows=tuple(FeatureRow.from_wire(row) for row in value["rows"]),
            header=ArtifactHeader.from_wire(value["header"]),
        )


def _row(
    ordinal: int, cell: SimulatedCell, planned_key: CellKey, measurement: MeasurementPlan
) -> FeatureRow:
    """Estimate one cell under the simulator's virtual identity layout; keys are already in display order."""
    virtual = replace(measurement, physical_qubits=tuple(range(measurement.n_qubits)))
    estimate = estimate_sampled_basis(virtual, dict(cell.counts), origin="digital_sampled")
    return FeatureRow(
        ordinal,
        planned_key,
        measurement.request_digest,
        estimate.shots,
        estimate.expectations,
        estimate.covariance_of_mean,
        None,
    )


def estimate_simulated_features(
    counts: SimulatedCountSet,
    experiment: PlannedExperiment,
    *,
    basis_id: str,
    revisions: tuple[str, str],
) -> SimulatedFeatureBatch:
    """Return the sampled features of every cell measured in one basis, in experiment order."""
    if counts.planned_experiment_digest != _sha(experiment.to_wire()):
        raise ValueError("the counts name a different planned experiment")
    chosen = [
        index
        for index, measurement in enumerate(experiment.measurements)
        if measurement.basis_id == basis_id
    ]
    if not chosen:
        raise ValueError("the experiment measures no cell in this basis")
    rows = tuple(
        _row(
            index, counts.cells[index], experiment.cells[index].key, experiment.measurements[index]
        )
        for index in chosen
    )
    observable_ids = experiment.measurements[chosen[0]].observable_ids
    count_digest = _sha(counts.to_wire())
    content = {
        "schema": SIMULATED_FEATURE_BATCH_SCHEMA,
        "object_kind": "simulated_feature_batch",
        "simulated_count_set_digest": count_digest,
        "estimator": "sampled_mean_v1",
        "basis_id": basis_id,
        "observable_ids": list(observable_ids),
        "rows": [row.to_wire() for row in rows],
    }
    header = ArtifactHeader(
        "simulated_feature_batch",
        _sha(content),
        (count_digest,),
        revisions[0],
        revisions[1],
        "offline_design",
        counts.header.data_origin,
        "design_only",
    )
    return SimulatedFeatureBatch(
        count_digest, "sampled_mean_v1", basis_id, observable_ids, rows, header
    )


def validate_simulated_feature_batch(
    batch: SimulatedFeatureBatch, counts: SimulatedCountSet, experiment: PlannedExperiment
) -> None:
    """Refuse features of other counts, dropped or reordered rows, foreign columns or altered values."""
    if type(batch) is not SimulatedFeatureBatch:
        raise ValueError("feature validation needs a SimulatedFeatureBatch")
    if batch.simulated_count_set_digest != _sha(counts.to_wire()):
        raise ValueError("features name a different simulated count set")
    if counts.planned_experiment_digest != _sha(experiment.to_wire()):
        raise ValueError("the counts name a different planned experiment")
    if batch.header.data_origin != counts.header.data_origin:
        raise ValueError("feature origin differs from the simulated counts")
    expected = [
        index
        for index, measurement in enumerate(experiment.measurements)
        if measurement.basis_id == batch.basis_id
    ]
    if [row.cell_ordinal for row in batch.rows] != expected:
        raise ValueError("feature rows must cover every cell measured in this basis, in order")
    for row in batch.rows:
        index = row.cell_ordinal
        measurement = experiment.measurements[index]
        if measurement.observable_ids != batch.observable_ids:
            raise ValueError("feature columns differ from the cell's observables")
        if row != _row(index, counts.cells[index], experiment.cells[index].key, measurement):
            raise ValueError("a feature row differs from its simulated counts")


def decode_simulated_feature_batch(raw: bytes) -> SimulatedFeatureBatch:
    """Decode one simulated feature batch from strict canonical wire bytes."""
    return SimulatedFeatureBatch.from_wire(_strict_json(raw))
