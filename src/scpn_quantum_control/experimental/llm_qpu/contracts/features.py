# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Feature batches recomputed from raw counts; a missing cell is never a zero row."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

from .cells import CellKey, PlannedCell
from .compiled import CompiledBundle
from .measurement import MeasurementPlan, estimate_sampled_basis
from .results import CellResult, RawResultSet, _check_planned
from .wire import (
    FEATURE_BATCH_SCHEMA,
    ArtifactHeader,
    _digest,
    _positive_int,
    _strict_json,
    _text,
    canonical_bytes,
    f64,
)

_QUASI = "quasi_probability_v1"
_ESTIMATORS = ("sampled_mean_v1",)
_ESTIMATE_ORIGIN = {"provider_result": "hardware_raw", "synthetic_classical": "digital_sampled"}


def _float(value: object, *, name: str) -> float:
    """Return one finite float, refusing integers, booleans and NaN."""
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{name} must be finite floats")
    return value


def _from_f64(value: object, *, name: str) -> float:
    """Decode one f64 wire tag."""
    if type(value) is not dict or set(value) != {"$f64"} or type(value["$f64"]) is not str:
        raise ValueError(f"{name} must be f64 tags on wire")
    return float.fromhex(value["$f64"])


@dataclass(frozen=True, slots=True)
class FeatureRow:
    """One cell's sampled feature means and covariance, or an explicit missing row."""

    cell_ordinal: int
    key: CellKey
    measurement_request_digest: str
    shots: int | None
    values: tuple[float, ...] | None
    covariance: tuple[tuple[float, ...], ...] | None
    missing_reason: str | None

    def __post_init__(self) -> None:
        """Refuse values outside [-1, 1], asymmetric covariance and unexplained gaps."""
        if type(self.cell_ordinal) is not int or not 0 <= self.cell_ordinal < 10_000_000:
            raise ValueError("cell ordinal must be a bounded non-negative integer")
        if type(self.key) is not CellKey:
            raise ValueError("a feature row needs its CellKey")
        _digest(self.measurement_request_digest, name="measurement request digest")
        if self.values is None:
            if self.shots is not None or self.covariance is not None:
                raise ValueError("a missing row carries no shots or covariance")
            object.__setattr__(
                self, "missing_reason", _text(self.missing_reason, name="missing reason")
            )
            return
        if self.missing_reason is not None:
            raise ValueError("a measured row cannot carry a missing reason")
        _positive_int(self.shots, name="feature shots", maximum=10_000_000)
        values = self.values
        if type(values) is not tuple or not values or len(values) > 64:
            raise ValueError("feature values must be a nonempty tuple of at most 64")
        for value in values:
            if not -1.0 <= _float(value, name="feature values") <= 1.0:
                raise ValueError(
                    "a sampled mean of a ±1 observable lies in [-1, 1]; "
                    "quasi-probability estimates are not counts"
                )
        self._check_covariance()

    def _check_covariance(self) -> None:
        """Require a symmetric square covariance, absent only for a single shot."""
        covariance, width = self.covariance, len(self.values or ())
        if self.shots == 1:
            if covariance is not None:
                raise ValueError("one shot has no sampling covariance")
            return
        if type(covariance) is not tuple or len(covariance) != width:
            raise ValueError("covariance must be a square tuple matching the features")
        for row in covariance:
            if type(row) is not tuple or len(row) != width:
                raise ValueError("covariance must be a square tuple matching the features")
            for value in row:
                _float(value, name="covariance entries")
        if any(covariance[i][j] != covariance[j][i] for i in range(width) for j in range(i)):
            raise ValueError("covariance must be symmetric")

    def to_wire(self) -> dict[str, object]:
        """Return the exact row with f64-tagged values."""
        return {
            "cell_ordinal": self.cell_ordinal,
            "key": self.key.to_wire(),
            "measurement_request_digest": self.measurement_request_digest,
            "shots": self.shots,
            "values": None if self.values is None else [f64(value) for value in self.values],
            "covariance": None
            if self.covariance is None
            else [[f64(value) for value in row] for row in self.covariance],
            "missing_reason": self.missing_reason,
        }

    @classmethod
    def from_wire(cls, value: object) -> FeatureRow:
        """Reject missing or extra fields and decode f64 tags."""
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("FeatureRow fields mismatch")
        values, covariance = value["values"], value["covariance"]
        if values is not None and type(values) is not list:
            raise ValueError("feature values must be a list or null on wire")
        if covariance is not None and (
            type(covariance) is not list or any(type(row) is not list for row in covariance)
        ):
            raise ValueError("covariance must be a list of lists or null on wire")
        return cls(
            cell_ordinal=value["cell_ordinal"],
            key=CellKey.from_wire(value["key"]),
            measurement_request_digest=value["measurement_request_digest"],
            shots=value["shots"],
            values=None
            if values is None
            else tuple(_from_f64(item, name="feature values") for item in values),
            covariance=None
            if covariance is None
            else tuple(
                tuple(_from_f64(item, name="covariance entries") for item in row)
                for row in covariance
            ),
            missing_reason=value["missing_reason"],
        )


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    """Design-only [N, F] sampled features of one basis, in result-set cell order."""

    raw_result_set_digest: str
    estimator: str
    basis_id: str
    observable_ids: tuple[str, ...]
    rows: tuple[FeatureRow, ...]
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse quasi-probabilities as counts, reordered rows and ragged widths."""
        _digest(self.raw_result_set_digest, name="raw result set digest")
        if self.estimator == _QUASI:
            raise ValueError("quasi-probabilities are not raw counts; they need their own source")
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
        ordinals = [row.cell_ordinal for row in rows]
        if ordinals != sorted(set(ordinals)):
            raise ValueError("feature rows must follow the result set's cell order")
        if any(row.values is not None and len(row.values) != len(names) for row in rows):
            raise ValueError("every measured row has one value per observable")
        if type(self.header) is not ArtifactHeader or self.header.object_kind != "feature_batch":
            raise ValueError("FeatureBatch requires its exact artifact header")
        if self.header.data_origin not in _ESTIMATE_ORIGIN:
            raise ValueError("a feature batch derives from provider results or a fixture")
        self.header.validate_content(
            self._scientific_wire(), parents=(self.raw_result_set_digest,)
        )

    def missing(self) -> tuple[tuple[int, str], ...]:
        """Return each missing row with its reason."""
        return tuple(
            (row.cell_ordinal, str(row.missing_reason)) for row in self.rows if row.values is None
        )

    def _scientific_wire(self) -> dict[str, object]:
        """Return the batch content hashed by its header."""
        return {
            "schema": FEATURE_BATCH_SCHEMA,
            "object_kind": "feature_batch",
            "raw_result_set_digest": self.raw_result_set_digest,
            "estimator": self.estimator,
            "basis_id": self.basis_id,
            "observable_ids": list(self.observable_ids),
            "rows": [row.to_wire() for row in self.rows],
        }

    def to_wire(self) -> dict[str, object]:
        """Return the detached versioned batch."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> FeatureBatch:
        """Reject unknown fields and restore the frozen batch."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("FeatureBatch fields mismatch")
        if value["schema"] != FEATURE_BATCH_SCHEMA or value["object_kind"] != "feature_batch":
            raise ValueError("unknown FeatureBatch schema")
        if type(value["observable_ids"]) is not list or type(value["rows"]) is not list:
            raise ValueError("observable IDs and rows must be lists on wire")
        return cls(
            raw_result_set_digest=value["raw_result_set_digest"],
            estimator=value["estimator"],
            basis_id=value["basis_id"],
            observable_ids=tuple(value["observable_ids"]),
            rows=tuple(FeatureRow.from_wire(row) for row in value["rows"]),
            header=ArtifactHeader.from_wire(value["header"]),
        )


def _display_counts(result: CellResult, plan: MeasurementPlan) -> dict[str, int]:
    """Rewrite physical-qubit bitstrings into the plan's display bit order."""
    physical = plan.physical_qubits
    if physical is None or sorted(physical) != sorted(result.measured_qubits):
        raise ValueError("measurement plan layout differs from the measured qubits")
    position = {qubit: index for index, qubit in enumerate(result.measured_qubits)}
    order = [
        position[physical[plan.logical_to_clbit.index(clbit)]]
        for clbit in plan.display_order_clbits
    ]
    return {"".join(bits[index] for index in order): count for bits, count in result.counts}


def _check_plans(plans: tuple[MeasurementPlan, ...], planned: tuple[PlannedCell, ...]) -> None:
    """Require one measurement plan per planned cell, measuring that cell's circuit."""
    if type(plans) is not tuple or len(plans) != len(planned):
        raise ValueError("validation needs one measurement plan per compiled cell")
    for plan, cell in zip(plans, planned, strict=True):
        if type(plan) is not MeasurementPlan:
            raise ValueError("measurement plans must be MeasurementPlan records")
        if plan.circuit_plan_digest != cell.circuit_plan_digest:
            raise ValueError("measurement plan does not measure the planned circuit")


def _check_row(
    row: FeatureRow,
    batch: FeatureBatch,
    plan: MeasurementPlan,
    planned: PlannedCell,
    result: CellResult | None,
) -> None:
    """Tie one row to its plan and planned cell and recompute it from raw counts."""
    if row.key != planned.key:
        raise ValueError("feature row key differs from the planned cell")
    if plan.observable_ids != batch.observable_ids:
        raise ValueError("feature columns differ from the cell's observables")
    if row.measurement_request_digest != plan.request_digest:
        raise ValueError("feature row names another measurement request")
    if result is None:
        if row.values is not None:
            raise ValueError("a cell without a result must stay missing, never zero-filled")
        return
    if row.values is None:
        raise ValueError("a measured cell cannot be reported missing")
    origin = _ESTIMATE_ORIGIN[batch.header.data_origin]
    estimate = estimate_sampled_basis(plan, _display_counts(result, plan), origin=origin)
    recomputed = (estimate.shots, estimate.expectations, estimate.covariance_of_mean)
    if (row.shots, row.values, row.covariance) != recomputed:
        raise ValueError("feature values differ from the raw counts")


def validate_feature_batch(
    batch: FeatureBatch,
    result_set: RawResultSet,
    bundle: CompiledBundle,
    planned: tuple[PlannedCell, ...],
    plans: tuple[MeasurementPlan, ...],
) -> None:
    """Refuse features for other results, dropped or reordered rows, or altered values."""
    if type(batch) is not FeatureBatch:
        raise ValueError("feature validation needs a FeatureBatch")
    if (
        batch.raw_result_set_digest
        != hashlib.sha256(canonical_bytes(result_set.to_wire())).hexdigest()
    ):
        raise ValueError("features name a different raw result set")
    if (
        result_set.compiled_bundle_digest
        != hashlib.sha256(canonical_bytes(bundle.to_wire())).hexdigest()
    ):
        raise ValueError("the raw result set belongs to another compiled bundle")
    if batch.header.data_origin != result_set.header.data_origin:
        raise ValueError("feature origin differs from the raw results")
    _check_planned(planned, bundle)
    _check_plans(plans, planned)
    expected = [index for index, plan in enumerate(plans) if plan.basis_id == batch.basis_id]
    if [row.cell_ordinal for row in batch.rows] != expected:
        raise ValueError("feature rows must cover every cell measured in this basis, in order")
    results = {result.cell_ordinal: result for result in result_set.results}
    for row in batch.rows:
        index = row.cell_ordinal
        _check_row(row, batch, plans[index], planned[index], results.get(index))


def decode_feature_batch(raw: bytes) -> FeatureBatch:
    """Decode one feature batch from strict canonical wire bytes."""
    return FeatureBatch.from_wire(_strict_json(raw))
