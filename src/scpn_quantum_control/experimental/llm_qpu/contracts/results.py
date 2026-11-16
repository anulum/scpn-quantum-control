# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU contracts
"""Raw per-cell results: exact shots, sparse integer counts, no silently missing cell."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from uuid import UUID

from .cells import CellKey, PlannedCell
from .compiled import CompiledBundle
from .receipt import ProviderBatchReceipt
from .wire import (
    RAW_RESULT_SET_SCHEMA,
    ArtifactHeader,
    _digest,
    _positive_int,
    _strict_json,
    _text,
    canonical_bytes,
)

_RESULT_ORIGIN = {
    "provider_metadata": "provider_result",
    "synthetic_classical": "synthetic_classical",
}
_GAP_STATES = ("missing", "quarantined")


def _bounded_index(value: object, *, name: str, maximum: int) -> int:
    """Return a non-negative integer below the bound or refuse."""
    if type(value) is not int or not 0 <= value < maximum:
        raise ValueError(f"{name} must be a bounded non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class CellResult:
    """One cell's raw provider outcome; bitstring position i is measured_qubits[i]."""

    cell_ordinal: int
    key: CellKey
    job_ordinal: int
    provider_job_id: str
    pub_index: int
    binding_index: int
    measured_qubits: tuple[int, ...]
    requested_shots: int
    actual_shots: int
    counts: tuple[tuple[str, int], ...]
    raw_payload_sha256: str

    def __post_init__(self) -> None:
        """Refuse other shots, float or boolean counts and zero-padded support."""
        _bounded_index(self.cell_ordinal, name="cell ordinal", maximum=10_000_000)
        if type(self.key) is not CellKey:
            raise ValueError("a cell result needs its CellKey")
        _bounded_index(self.job_ordinal, name="job ordinal", maximum=100_000)
        object.__setattr__(
            self, "provider_job_id", _text(self.provider_job_id, name="provider job ID")
        )
        _bounded_index(self.pub_index, name="PUB index", maximum=10_000_000)
        _bounded_index(self.binding_index, name="binding index", maximum=1_000_000)
        qubits = self.measured_qubits
        if type(qubits) is not tuple or not 0 < len(qubits) <= 64:
            raise ValueError("measured qubits must be a tuple of one to 64 qubits")
        for qubit in qubits:
            _bounded_index(qubit, name="measured qubit", maximum=100_000)
        if len(set(qubits)) != len(qubits):
            raise ValueError("a qubit cannot be measured into two bits")
        _positive_int(self.requested_shots, name="requested shots", maximum=10_000_000)
        if type(self.actual_shots) is not int or self.actual_shots != self.requested_shots:
            raise ValueError("actual shots differ from the requested shots; quarantine the cell")
        self._check_counts()
        _digest(self.raw_payload_sha256, name="raw payload digest")

    def _check_counts(self) -> None:
        """Require sorted sparse bitstring counts that sum to the actual shots."""
        counts = self.counts
        if type(counts) is not tuple or not 0 < len(counts) <= 4096:
            raise ValueError("raw counts must be a nonempty tuple of at most 4096 outcomes")
        width = len(self.measured_qubits)
        total = 0
        for entry in counts:
            if type(entry) is not tuple or len(entry) != 2:
                raise ValueError("each raw count is one bitstring and one count")
            bits, count = entry
            if type(bits) is not str or len(bits) != width or set(bits) - {"0", "1"}:
                raise ValueError("raw bitstring width differs from the measured qubits")
            if type(count) is not int:
                raise ValueError("raw counts must be integers, not floats or booleans")
            if count < 1:
                raise ValueError("a zero or negative count pads sparse data to full support")
            total += count
        names = [bits for bits, _ in counts]
        if names != sorted(set(names)):
            raise ValueError("raw counts must be sorted by unique bitstring")
        if total != self.actual_shots:
            raise ValueError("raw counts do not sum to the actual shots")

    def to_wire(self) -> dict[str, object]:
        """Return the exact cell result."""
        return {
            "cell_ordinal": self.cell_ordinal,
            "key": self.key.to_wire(),
            "job_ordinal": self.job_ordinal,
            "provider_job_id": self.provider_job_id,
            "pub_index": self.pub_index,
            "binding_index": self.binding_index,
            "measured_qubits": list(self.measured_qubits),
            "requested_shots": self.requested_shots,
            "actual_shots": self.actual_shots,
            "counts": [[bits, count] for bits, count in self.counts],
            "raw_payload_sha256": self.raw_payload_sha256,
        }

    @classmethod
    def from_wire(cls, value: object) -> CellResult:
        """Reject missing or extra fields and restore tuples."""
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("CellResult fields mismatch")
        qubits, counts = value["measured_qubits"], value["counts"]
        if type(qubits) is not list or type(counts) is not list:
            raise ValueError("measured qubits and counts must be lists on wire")
        if any(type(entry) is not list for entry in counts):
            raise ValueError("each raw count must be a list on wire")
        fields = {name: value[name] for name in cls.__dataclass_fields__}
        return cls(
            **{
                **fields,
                "key": CellKey.from_wire(value["key"]),
                "measured_qubits": tuple(qubits),
                "counts": tuple(tuple(entry) for entry in counts),
            }
        )


@dataclass(frozen=True, slots=True)
class CellGap:
    """An expected cell without a usable result: missing, or quarantined with its raw data."""

    cell_ordinal: int
    status: str
    reason: str
    raw_payload_sha256: str | None

    def __post_init__(self) -> None:
        """Require a reason; quarantine keeps the raw payload, missing has none."""
        _bounded_index(self.cell_ordinal, name="cell ordinal", maximum=10_000_000)
        if self.status not in _GAP_STATES:
            raise ValueError("a cell gap is missing or quarantined")
        object.__setattr__(self, "reason", _text(self.reason, name="gap reason"))
        if self.status == "quarantined":
            _digest(self.raw_payload_sha256, name="quarantined raw payload digest")
        elif self.raw_payload_sha256 is not None:
            raise ValueError("a missing cell has no raw payload")

    def to_wire(self) -> dict[str, object]:
        """Return the exact gap record."""
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_wire(cls, value: object) -> CellGap:
        """Reject a gap record with missing or extra fields."""
        if type(value) is not dict or set(value) != set(cls.__dataclass_fields__):
            raise ValueError("CellGap fields mismatch")
        return cls(**{name: value[name] for name in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class RawResultSet:
    """Design-only inventory of every expected cell of one receipt: result or recorded gap."""

    attempt_id: str
    provider_batch_receipt_digest: str
    compiled_bundle_digest: str
    expected_cell_count: int
    results: tuple[CellResult, ...]
    gaps: tuple[CellGap, ...]
    header: ArtifactHeader

    def __post_init__(self) -> None:
        """Refuse silently missing cells, duplicates and merged provider result units."""
        if type(self.attempt_id) is not str or str(_uuid(self.attempt_id)) != self.attempt_id:
            raise ValueError("attempt ID must be canonical UUID text")
        for name in ("provider_batch_receipt_digest", "compiled_bundle_digest"):
            _digest(getattr(self, name), name=name)
        _positive_int(self.expected_cell_count, name="expected cell count", maximum=10_000_000)
        self._check_inventory()
        if type(self.header) is not ArtifactHeader or self.header.object_kind != "raw_result_set":
            raise ValueError("RawResultSet requires its exact artifact header")
        if self.header.data_origin not in ("provider_result", "synthetic_classical"):
            raise ValueError("a raw result set derives from provider results or a fixture")
        parents = (self.provider_batch_receipt_digest, self.compiled_bundle_digest)
        self.header.validate_content(self._scientific_wire(), parents=parents)

    def _check_inventory(self) -> None:
        """Account for each expected cell exactly once, in order, with unique result units."""
        for name, kind in (("results", CellResult), ("gaps", CellGap)):
            records = getattr(self, name)
            if type(records) is not tuple or any(type(record) is not kind for record in records):
                raise ValueError(f"{name} must be a tuple of {kind.__name__} records")
            ordinals = [record.cell_ordinal for record in records]
            if ordinals != sorted(set(ordinals)):
                raise ValueError(f"{name} must be ordered by unique cell ordinal")
        accounted = sorted(
            [result.cell_ordinal for result in self.results]
            + [gap.cell_ordinal for gap in self.gaps]
        )
        if accounted != list(range(self.expected_cell_count)):
            raise ValueError("every expected cell needs one result or one recorded gap")
        units = {(r.provider_job_id, r.pub_index, r.binding_index) for r in self.results}
        if len(units) != len(self.results):
            raise ValueError("two cells cannot share one provider result unit (merged PUB)")

    def complete(self) -> bool:
        """Return whether every expected cell has a usable result."""
        return not self.gaps

    def _scientific_wire(self) -> dict[str, object]:
        """Return the result-set content hashed by its header."""
        return {
            "schema": RAW_RESULT_SET_SCHEMA,
            "object_kind": "raw_result_set",
            "attempt_id": self.attempt_id,
            "provider_batch_receipt_digest": self.provider_batch_receipt_digest,
            "compiled_bundle_digest": self.compiled_bundle_digest,
            "expected_cell_count": self.expected_cell_count,
            "results": [result.to_wire() for result in self.results],
            "gaps": [gap.to_wire() for gap in self.gaps],
        }

    def to_wire(self) -> dict[str, object]:
        """Return the detached versioned result set."""
        return {**self._scientific_wire(), "header": self.header.to_wire()}

    @classmethod
    def from_wire(cls, value: object) -> RawResultSet:
        """Reject unknown fields and restore the frozen result set."""
        fields = set(cls.__dataclass_fields__) | {"schema", "object_kind"}
        if type(value) is not dict or set(value) != fields:
            raise ValueError("RawResultSet fields mismatch")
        if value["schema"] != RAW_RESULT_SET_SCHEMA or value["object_kind"] != "raw_result_set":
            raise ValueError("unknown RawResultSet schema")
        if type(value["results"]) is not list or type(value["gaps"]) is not list:
            raise ValueError("results and gaps must be lists on wire")
        return cls(
            attempt_id=value["attempt_id"],
            provider_batch_receipt_digest=value["provider_batch_receipt_digest"],
            compiled_bundle_digest=value["compiled_bundle_digest"],
            expected_cell_count=value["expected_cell_count"],
            results=tuple(CellResult.from_wire(result) for result in value["results"]),
            gaps=tuple(CellGap.from_wire(gap) for gap in value["gaps"]),
            header=ArtifactHeader.from_wire(value["header"]),
        )


def _uuid(value: str) -> UUID:
    """Parse UUID text, refusing anything that is not one."""
    try:
        return UUID(value)
    except ValueError as exc:
        raise ValueError("attempt ID must be canonical UUID text") from exc


def _check_planned(planned: tuple[PlannedCell, ...], bundle: CompiledBundle) -> None:
    """Require one planned cell per compiled cell, with its digest and its shots."""
    if type(planned) is not tuple or len(planned) != len(bundle.cells):
        raise ValueError("validation needs one planned cell per compiled cell")
    for cell, compiled in zip(planned, bundle.cells, strict=True):
        if type(cell) is not PlannedCell:
            raise ValueError("planned cells must be PlannedCell records")
        digest = hashlib.sha256(canonical_bytes(cell.to_wire())).hexdigest()
        if digest != compiled.planned_cell_digest:
            raise ValueError("planned cell differs from the compiled bundle")
        if cell.shots != compiled.shots:
            raise ValueError("compiled shots differ from the planned cell")


def _check_result(
    result: CellResult,
    planned: PlannedCell,
    receipt: ProviderBatchReceipt,
    layout: frozenset[int],
) -> None:
    """Tie one result to its planned cell, its receipt job and its circuit position."""
    if result.key != planned.key:
        raise ValueError("result key differs from the planned cell")
    if result.requested_shots != planned.shots:
        raise ValueError("requested shots differ from the planned cell")
    if len(result.measured_qubits) != planned.measurement_width:
        raise ValueError("measured width differs from the planned cell")
    if not layout.issuperset(result.measured_qubits):
        raise ValueError("a measured qubit is outside the compiled layout")
    if result.job_ordinal >= len(receipt.jobs):
        raise ValueError("result names a job the receipt does not hold")
    job = receipt.jobs[result.job_ordinal]
    if job.state != "submitted":
        raise ValueError("a result cannot come from an ambiguous job")
    if job.provider_job_id != result.provider_job_id:
        raise ValueError("result job ID differs from the receipt")
    if result.cell_ordinal not in job.cell_ordinals:
        raise ValueError("result cell is not in its receipt job")
    if job.cell_ordinals.index(result.cell_ordinal) != result.pub_index:
        raise ValueError("result is not at its receipt circuit position")
    if result.binding_index != 0:
        raise ValueError("v1 maps one cell per PUB, so the binding index is zero")


def validate_raw_result_set(
    result_set: RawResultSet,
    receipt: ProviderBatchReceipt,
    bundle: CompiledBundle,
    planned: tuple[PlannedCell, ...],
) -> None:
    """Refuse results for another receipt or bundle, a fixture as hardware, or a mismatch."""
    if type(result_set) is not RawResultSet:
        raise ValueError("result validation needs a RawResultSet")
    if (
        result_set.provider_batch_receipt_digest
        != hashlib.sha256(canonical_bytes(receipt.to_wire())).hexdigest()
    ):
        raise ValueError("results name a different provider receipt")
    bundle_digest = hashlib.sha256(canonical_bytes(bundle.to_wire())).hexdigest()
    if bundle_digest != result_set.compiled_bundle_digest or (
        receipt.compiled_bundle_digest != bundle_digest
    ):
        raise ValueError("results, receipt and bundle name different compiled bundles")
    if result_set.attempt_id != receipt.attempt_id:
        raise ValueError("results belong to another attempt")
    if result_set.expected_cell_count != len(bundle.cells):
        raise ValueError("expected cell count differs from the compiled bundle")
    if result_set.header.data_origin != _RESULT_ORIGIN[receipt.header.data_origin]:
        raise ValueError("result origin does not match the receipt; a fixture is never hardware")
    _check_planned(planned, bundle)
    layout = frozenset(bundle.layout)
    for result in result_set.results:
        _check_result(result, planned[result.cell_ordinal], receipt, layout)


def decode_raw_result_set(raw: bytes) -> RawResultSet:
    """Decode one raw result set from strict canonical wire bytes."""
    return RawResultSet.from_wire(_strict_json(raw))
