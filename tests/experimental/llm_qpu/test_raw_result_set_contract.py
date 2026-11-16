# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental raw result set contract
"""Exercise per-cell raw results against their planned cells, receipt, bundle and worker."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    COMPILED_BUNDLE_SCHEMA,
    COST_QUOTE_SCHEMA,
    OWNER_APPROVAL_SCHEMA,
    PROVIDER_BATCH_RECEIPT_SCHEMA,
    RAW_RESULT_SET_SCHEMA,
    TARGET_SNAPSHOT_SCHEMA,
    ApprovalSignature,
    ArtifactHeader,
    CellGap,
    CellKey,
    CellResult,
    CompiledBundle,
    CompiledCell,
    CostQuote,
    OwnerApproval,
    PlannedCell,
    ProviderBatchReceipt,
    QuotedCount,
    RawResultSet,
    ReportedValue,
    SubmittedJob,
    TargetSnapshot,
    canonical_bytes,
    decode_contract,
    decode_raw_result_set,
    validate_raw_result_set,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _REPO_ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"
_ATTEMPT = "6f1c2d3e-4b5a-4c6d-8e7f-901a2b3c4d5e"
_JOB_IDS = ("019f8639-96be-77d1-8f70-109a70e6fb33", "019f8639-980e-7c21-bb38-f26396a883e3")
_RESULT_ORIGIN = {
    "synthetic_classical": "synthetic_classical",
    "provider_metadata": "provider_result",
}
_COUNTS = (
    (("00", 600), ("11", 400)),
    (("000", 1500), ("101", 500)),
)
_QUBITS = ((2, 0), (0, 1, 2))
_RAW = ("e" * 64, "f" * 64)


def _header(
    kind: str, content: dict[str, object], parents: tuple[str, ...], origin: str
) -> ArtifactHeader:
    """Return a design-only header that hashes exactly the given content."""
    return ArtifactHeader(
        object_kind=kind,
        content_digest=hashlib.sha256(canonical_bytes(content)).hexdigest(),
        parents=tuple(sorted(parents)),
        base_repo_commit=_BASE,
        implementation_revision=_REVISION,
        execution_origin="offline_design",
        data_origin=origin,
        claim_scope="design_only",
    )


def _digest_of(value: Any) -> str:
    """Return the SHA-256 of one artifact wire."""
    return hashlib.sha256(canonical_bytes(value.to_wire())).hexdigest()


def _key(sample: str) -> CellKey:
    """Return the CellKey of one fixture sample."""
    return CellKey("exp-1", sample, "seq-1", "arm-a", "zz", "rep-0", "epoch-0", "bind-0")


def _planned() -> tuple[PlannedCell, ...]:
    """Return two planned cells: two measured bits at 1000 shots, three at 2000."""
    return tuple(
        PlannedCell.create(
            key=_key(f"sample-{index}"),
            circuit_plan_digest=str(index + 1) * 64,
            shots=shots,
            measurement_width=width,
            role="data",
            metadata={},
        )
        for index, (shots, width) in enumerate(((1000, 2), (2000, 3)))
    )


def _target(provenance: str) -> TargetSnapshot:
    """Build a three-qubit IQM fixture target with one circuit per job."""
    known: dict[str, Any] = {
        "calibration_id": ReportedValue("known", "cal-2026-11-15-0800", None),
        "calibration_time_utc": ReportedValue("known", "2026-11-15T08:00:00Z", None),
        "max_shots": ReportedValue("known", 100_000, None),
        "max_circuits_per_job": ReportedValue("known", 2, None),
    }
    plain: dict[str, Any] = {
        "provider": "iqm",
        "account_route": "iqm-resonance/owner-project",
        "device_name": "garnet",
        "sdk_name": "fixture-sdk",
        "sdk_version": "1.0.0",
        "sdk_target_digest": "a" * 64,
        "capability_provenance": provenance,
        "num_qubits": 3,
    }
    origin = {"synthetic_fixture": "synthetic_classical"}.get(provenance, "provider_metadata")
    content: dict[str, object] = {
        "schema": TARGET_SNAPSHOT_SCHEMA,
        "object_kind": "target_snapshot",
        **plain,
        "basis_gates": ["cz", "measure", "prx"],
        "coupling_edges": [[0, 1], [1, 2]],
        **{name: value.to_wire() for name, value in known.items()},
    }
    return TargetSnapshot(
        **plain,
        basis_gates=("cz", "measure", "prx"),
        coupling_edges=((0, 1), (1, 2)),
        **known,
        header=_header("target_snapshot", content, ("a" * 64,), origin),
    )


def _bundle(target: TargetSnapshot, shots: tuple[int, int] = (1000, 2000)) -> CompiledBundle:
    """Build a bundle whose cells carry the planned cells' digests."""
    cells = tuple(
        CompiledCell(_digest_of(planned), "2" * 64, "3" * 64, "4" * 64, count)
        for planned, count in zip(_planned(), shots, strict=True)
    )
    payload: dict[str, object] = {
        "cells": [cell.to_wire() for cell in cells],
        "layout": [2, 0, 1],
        "logical_gates": ["cx", "measure", "rz"],
        "compiled_gate_counts": [["cz", 4], ["measure", 3], ["prx", 10]],
        "substitutions": [["cx", ["cz", "prx"]], ["rz", ["prx"]]],
        "compiled_depth": 12,
        "provider_options": [],
    }
    identity: dict[str, Any] = {
        "target_snapshot_digest": _digest_of(target),
        "provider": "iqm",
        "device_name": "garnet",
        "sdk_lock_digest": "b" * 64,
        "compile_log_digest": "c" * 64,
        "payload_digest": hashlib.sha256(canonical_bytes(payload)).hexdigest(),
    }
    content = {"schema": COMPILED_BUNDLE_SCHEMA, "object_kind": "compiled_bundle"}
    parents = (identity["target_snapshot_digest"], "b" * 64, "c" * 64)
    return CompiledBundle(
        **identity,
        cells=cells,
        layout=(2, 0, 1),
        logical_gates=("cx", "measure", "rz"),
        compiled_gate_counts=(("cz", 4), ("measure", 3), ("prx", 10)),
        substitutions=(("cx", ("cz", "prx")), ("rz", ("prx",))),
        compiled_depth=12,
        provider_options=(),
        header=_header(
            "compiled_bundle",
            {**content, **identity, **payload},
            parents,
            target.header.data_origin,
        ),
    )


def _quote(bundle: CompiledBundle) -> CostQuote:
    """Build the bundle's one-job quote (two circuits per job)."""
    counts: dict[str, Any] = {
        "job_count": QuotedCount("known", 1, None),
        "overhead_executions": QuotedCount("known", 0, None),
        "resource_units": QuotedCount("known", 45, None),
        "price_minor_units": QuotedCount("known", 4500, None),
    }
    fields: dict[str, Any] = {
        "compiled_bundle_digest": _digest_of(bundle),
        "bundle_payload_digest": bundle.payload_digest,
        "circuit_count": 2,
        "total_shots": sum(cell.shots for cell in bundle.cells),
        "resource_unit": "provider_credit",
        "resource_derivation": "published_rate_card",
        "currency": "EUR",
        "pricing_source": "fixture price list",
        "priced_at_utc": "2026-11-15T09:00:00Z",
    }
    content = {
        "schema": COST_QUOTE_SCHEMA,
        "object_kind": "cost_quote",
        **fields,
        "quoted_options": [],
        **{name: count.to_wire() for name, count in counts.items()},
    }
    header = _header(
        "cost_quote", content, (fields["compiled_bundle_digest"],), bundle.header.data_origin
    )
    return CostQuote(**fields, quoted_options=(), **counts, header=header)


def _approval(quote: CostQuote, bundle: CompiledBundle, environment: str) -> OwnerApproval:
    """Approve the quote; result sets never verify signatures, so the digest is a placeholder."""
    scope: dict[str, Any] = {
        "approval_id": "approval-2026-11-16-001",
        "compiled_bundle_digest": _digest_of(bundle),
        "bundle_payload_digest": bundle.payload_digest,
        "cost_quote_digest": _digest_of(quote),
        "provider": "iqm",
        "account_route": "iqm-resonance/owner-project",
        "device_name": "garnet",
        "max_shots": quote.total_shots,
        "max_jobs": 2,
        "max_resource_units": 45,
        "max_price_minor_units": 4500,
        "currency": "EUR",
        "valid_from_utc": "2026-11-16T00:00:00Z",
        "expires_at_utc": "2026-11-17T00:00:00Z",
        "approved_egress": "value_free",
    }
    envelope = ApprovalSignature("ml_dsa_65", "9" * 64, environment, "d" * 64)
    content = {
        "schema": OWNER_APPROVAL_SCHEMA,
        "object_kind": "owner_approval",
        **scope,
        "signature": envelope.to_wire(),
    }
    origin = {"live": "owner_decision", "test": "synthetic_classical"}[environment]
    parents = (scope["compiled_bundle_digest"], scope["cost_quote_digest"])
    header = _header("owner_approval", content, parents, origin)
    return OwnerApproval(**scope, signature=envelope, header=header)


def _job(ordinal: int, cells: tuple[int, ...], *, ambiguous: bool = False) -> SubmittedJob:
    """Build one submitted job, or one whose submit outcome is unknown."""
    return SubmittedJob(
        ordinal=ordinal,
        name=f"phase_{ordinal}",
        payload_sha256=hashlib.sha256(f"request-{ordinal}".encode()).hexdigest(),
        cell_ordinals=cells,
        state="recovery_required" if ambiguous else "submitted",
        provider_job_id=None if ambiguous else _JOB_IDS[ordinal],
        id_source=None if ambiguous else "submit_response",
        boundary_at_utc="2026-11-16T09:00:00Z",
        provider_created_at_utc=None,
        blocker="the submit response was lost" if ambiguous else None,
    )


def _receipt(
    approval: OwnerApproval, bundle: CompiledBundle, jobs: tuple[SubmittedJob, ...]
) -> ProviderBatchReceipt:
    """Build the receipt for the given jobs, hashing exactly its content."""
    fields: dict[str, Any] = {
        "attempt_id": _ATTEMPT,
        "compiled_bundle_digest": _digest_of(bundle),
        "bundle_payload_digest": bundle.payload_digest,
        "owner_approval_digest": _digest_of(approval),
        "provider": "iqm",
        "account_route": "iqm-resonance/owner-project",
        "device_name": "garnet",
    }
    content = {
        "schema": PROVIDER_BATCH_RECEIPT_SCHEMA,
        "object_kind": "provider_batch_receipt",
        **fields,
        "jobs": [job.to_wire() for job in jobs],
    }
    parents = (fields["compiled_bundle_digest"], fields["owner_approval_digest"])
    header = _header("provider_batch_receipt", content, parents, bundle.header.data_origin)
    return ProviderBatchReceipt(**fields, jobs=jobs, header=header)


@dataclass(frozen=True)
class _Chain:
    """One consistent target, bundle, quote, approval, receipt and planned cells."""

    target: TargetSnapshot
    bundle: CompiledBundle
    quote: CostQuote
    approval: OwnerApproval
    receipt: ProviderBatchReceipt
    planned: tuple[PlannedCell, ...]


def _chain(
    jobs: tuple[SubmittedJob, ...] | None = None,
    *,
    provenance: str = "synthetic_fixture",
    shots: tuple[int, int] = (1000, 2000),
) -> _Chain:
    """Return one consistent chain; by default one job per cell."""
    target = _target(provenance)
    bundle = _bundle(target, shots)
    quote = _quote(bundle)
    environment = "test" if provenance == "synthetic_fixture" else "live"
    approval = _approval(quote, bundle, environment)
    jobs = jobs if jobs is not None else (_job(0, (0,)), _job(1, (1,)))
    receipt = _receipt(approval, bundle, jobs)
    return _Chain(target, bundle, quote, approval, receipt, _planned())


def _result(ordinal: int, **changes: Any) -> CellResult:
    """Return the fixture result of one cell, from its own job at circuit position zero."""
    shots = (1000, 2000)[ordinal]
    fields: dict[str, Any] = {
        "cell_ordinal": ordinal,
        "key": _key(f"sample-{ordinal}"),
        "job_ordinal": ordinal,
        "provider_job_id": _JOB_IDS[ordinal],
        "pub_index": 0,
        "binding_index": 0,
        "measured_qubits": _QUBITS[ordinal],
        "requested_shots": shots,
        "actual_shots": shots,
        "counts": _COUNTS[ordinal],
        "raw_payload_sha256": _RAW[ordinal],
        **changes,
    }
    return CellResult(**fields)


def _set(
    chain: _Chain,
    results: tuple[CellResult, ...] | None = None,
    gaps: tuple[CellGap, ...] = (),
    *,
    origin: str | None = None,
    **changes: Any,
) -> RawResultSet:
    """Build the chain's result set, hashing exactly its content."""
    fields: dict[str, Any] = {
        "attempt_id": _ATTEMPT,
        "provider_batch_receipt_digest": _digest_of(chain.receipt),
        "compiled_bundle_digest": _digest_of(chain.bundle),
        "expected_cell_count": 2,
        **changes,
    }
    results = results if results is not None else (_result(0), _result(1))
    content = {
        "schema": RAW_RESULT_SET_SCHEMA,
        "object_kind": "raw_result_set",
        **fields,
        "results": [result.to_wire() for result in results],
        "gaps": [gap.to_wire() for gap in gaps],
    }
    parents = (fields["provider_batch_receipt_digest"], fields["compiled_bundle_digest"])
    data_origin = origin or _RESULT_ORIGIN[chain.receipt.header.data_origin]
    header = _header("raw_result_set", content, parents, data_origin)
    return RawResultSet(**fields, results=results, gaps=gaps, header=header)


def _validate(chain: _Chain, result_set: RawResultSet) -> None:
    """Validate the result set against its chain."""
    validate_raw_result_set(result_set, chain.receipt, chain.bundle, chain.planned)


def _worker(request: dict[str, object], tmp_path: Path) -> subprocess.CompletedProcess[bytes]:
    """Run the standalone protocol worker on one request with poisoned provider tokens."""
    return subprocess.run(
        [sys.executable, "-S", str(_WORKER)],
        input=json.dumps(request).encode(),
        cwd=tmp_path,
        env={"PYTHONPATH": str(tmp_path), "IQM_TOKEN": "poison", "IBM_QUANTUM_TOKEN": "poison"},
        capture_output=True,
        timeout=15,
        check=False,
    )


def _request(chain: _Chain, results: object) -> dict[str, object]:
    """Return one worker result-set request."""
    return {
        "op": "roundtrip_raw_result_set",
        "target": chain.target.to_wire(),
        "bundle": chain.bundle.to_wire(),
        "quote": chain.quote.to_wire(),
        "approval": chain.approval.to_wire(),
        "receipt": chain.receipt.to_wire(),
        "planned": [cell.to_wire() for cell in chain.planned],
        "results": results,
    }


def _quarantined(ordinal: int) -> CellGap:
    """Return a quarantined gap that keeps its raw payload."""
    return CellGap(ordinal, "quarantined", "999 of 1000 shots returned", _RAW[ordinal])


def _missing(ordinal: int) -> CellGap:
    """Return a missing gap with no raw payload."""
    return CellGap(ordinal, "missing", "the job's submit outcome is unknown", None)


def test_result_sets_validate_complete_quarantined_and_missing_cells(tmp_path: Path) -> None:
    """Complete, quarantined, missing and multi-circuit results validate and round-trip."""
    swapped = _chain((_job(0, (1, 0)),))
    ambiguous = _chain((_job(0, (0,)), _job(1, (1,), ambiguous=True)))
    queried = _chain(provenance="provider_target_query")
    cases = (
        (_chain(), _set(_chain())),
        (_chain(), _set(_chain(), (_result(0),), (_quarantined(1),))),
        (ambiguous, _set(ambiguous, (_result(0),), (_missing(1),))),
        (_chain(), _set(_chain(), (), (_missing(0), _quarantined(1)))),
        (
            swapped,
            _set(
                swapped,
                (
                    _result(0, job_ordinal=0, provider_job_id=_JOB_IDS[0], pub_index=1),
                    _result(1, job_ordinal=0, provider_job_id=_JOB_IDS[0], pub_index=0),
                ),
            ),
        ),
        (queried, _set(queried)),
    )
    for chain, result_set in cases:
        _validate(chain, result_set)
        assert decode_raw_result_set(canonical_bytes(result_set.to_wire())) == result_set
        response = _worker(_request(chain, result_set.to_wire()), tmp_path)
        assert response.returncode == 0, response.stdout.decode()
        payload = json.loads(response.stdout)
        assert payload["status"] == "validated_roundtrip_no_compute"
        assert payload["hardware_submission_enabled"] is False
        assert payload["results_sha256"] == _digest_of(result_set)
        assert decode_raw_result_set(canonical_bytes(payload["results"])) == result_set
    assert cases[0][1].complete() is True
    assert cases[1][1].complete() is False
    assert queried.receipt.header.data_origin == "provider_metadata"
    assert cases[-1][1].header.data_origin == "provider_result"


def test_cell_result_refuses_other_shots_coerced_counts_and_padded_support() -> None:
    """Counts are sparse positive integers over the measured bits and sum to exact shots."""
    for changes, reason in (
        ({"actual_shots": 999}, "^actual shots differ from the requested shots; quarantine"),
        ({"actual_shots": True}, "^actual shots differ from the requested shots"),
        ({"counts": (("00", 600.0), ("11", 400))}, "^raw counts must be integers, not floats"),
        ({"counts": (("00", True), ("11", 999))}, "^raw counts must be integers, not floats"),
        ({"counts": (("00", 600), ("01", 0), ("11", 400))}, "^a zero or negative count pads"),
        ({"counts": (("00", 1100), ("11", -100))}, "^a zero or negative count pads"),
        (
            {"counts": (("11", 400), ("00", 600))},
            "^raw counts must be sorted by unique bitstring$",
        ),
        (
            {"counts": (("00", 600), ("00", 400))},
            "^raw counts must be sorted by unique bitstring$",
        ),
        ({"counts": (("000", 600), ("11", 400))}, "^raw bitstring width differs from the"),
        ({"counts": (("02", 600), ("11", 400))}, "^raw bitstring width differs from the"),
        ({"counts": ((0, 600), ("11", 400))}, "^raw bitstring width differs from the"),
        ({"counts": (("00", 600), ("11", 399))}, "^raw counts do not sum to the actual shots$"),
        ({"counts": (("00", 600, 1), ("11", 400))}, "^each raw count is one bitstring and one"),
        ({"counts": (["00", 1000],)}, "^each raw count is one bitstring and one count$"),
        ({"counts": ()}, "^raw counts must be a nonempty tuple of at most 4096 outcomes$"),
        ({"counts": [("00", 1000)]}, "^raw counts must be a nonempty tuple of at most 4096"),
        ({"measured_qubits": ()}, "^measured qubits must be a tuple of one to 64 qubits$"),
        ({"measured_qubits": [2, 0]}, "^measured qubits must be a tuple of one to 64 qubits$"),
        ({"measured_qubits": tuple(range(65))}, "^measured qubits must be a tuple of one to 64"),
        ({"measured_qubits": (0, 0)}, "^a qubit cannot be measured into two bits$"),
        ({"measured_qubits": (-1, 0)}, "^measured qubit must be a bounded non-negative integer$"),
        ({"measured_qubits": (100_000, 0)}, "^measured qubit must be a bounded non-negative"),
        ({"key": "sample-0"}, "^a cell result needs its CellKey$"),
        ({"provider_job_id": ""}, "provider job ID must be bounded nonempty text"),
        ({"cell_ordinal": 10_000_000}, "^cell ordinal must be a bounded non-negative integer$"),
        ({"job_ordinal": 100_000}, "^job ordinal must be a bounded non-negative integer$"),
        ({"pub_index": -1}, "^PUB index must be a bounded non-negative integer$"),
        ({"binding_index": 1_000_000}, "^binding index must be a bounded non-negative integer$"),
        ({"requested_shots": 0, "actual_shots": 0}, "requested shots must be a bounded positive"),
        ({"raw_payload_sha256": "E" * 64}, "^raw payload digest must be a lowercase SHA-256"),
    ):
        with pytest.raises(ValueError, match=reason):
            _result(0, **changes)
    wide = tuple((format(value, "012b"), 1) for value in range(4097))
    with pytest.raises(ValueError, match="^raw counts must be a nonempty tuple of at most 4096"):
        _result(
            0,
            measured_qubits=tuple(range(12)),
            requested_shots=4097,
            actual_shots=4097,
            counts=wide,
        )
    full = tuple((format(value, "012b"), 1) for value in range(4096))
    assert _result(
        0, measured_qubits=tuple(range(12)), requested_shots=4096, actual_shots=4096, counts=full
    )
    for status, reason_text, raw, reason in (
        ("dropped", "gone", None, "^a cell gap is missing or quarantined$"),
        ("missing", "", None, "gap reason must be bounded nonempty text"),
        ("quarantined", "short", None, "^quarantined raw payload digest must be a lowercase"),
        ("missing", "lost", "e" * 64, "^a missing cell has no raw payload$"),
    ):
        with pytest.raises(ValueError, match=reason):
            CellGap(0, status, reason_text, raw)
    with pytest.raises(ValueError, match="^cell ordinal must be a bounded non-negative integer$"):
        CellGap(-1, "missing", "lost", None)


def test_result_set_refuses_silently_missing_cells_merged_pubs_and_bad_records() -> None:
    """Every expected cell is accounted for once; one provider unit backs one cell."""
    chain = _chain()
    valid = _set(chain)
    merged = _result(1, job_ordinal=0, provider_job_id=_JOB_IDS[0])
    stray: Any = (_result(0), "result")
    for changes, reason in (
        ({"results": (_result(0),)}, "^every expected cell needs one result or one recorded gap$"),
        (
            {"results": (_result(0), _result(1)), "gaps": (_missing(1),)},
            "^every expected cell needs one result or one recorded gap$",
        ),
        ({"results": (_result(1), _result(0))}, "^results must be ordered by unique cell"),
        ({"results": (_result(0), _result(0))}, "^results must be ordered by unique cell"),
        ({"gaps": (_missing(1), _missing(0)), "results": ()}, "^gaps must be ordered by unique"),
        ({"results": [_result(0), _result(1)]}, "^results must be a tuple of CellResult records$"),
        ({"results": stray}, "^results must be a tuple of CellResult records$"),
        ({"results": (_result(0), merged)}, r"^two cells cannot share one provider result unit"),
        ({"attempt_id": _ATTEMPT.upper()}, "^attempt ID must be canonical UUID text$"),
        ({"attempt_id": "attempt-1"}, "^attempt ID must be canonical UUID text$"),
        ({"compiled_bundle_digest": "E" * 64}, "^compiled_bundle_digest must be a lowercase"),
        ({"expected_cell_count": 0}, "expected cell count must be a bounded positive integer"),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(valid, **changes)
    with pytest.raises(ValueError, match="^RawResultSet requires its exact artifact header$"):
        replace(valid, header=replace(valid.header, object_kind="provider_batch_receipt"))
    with pytest.raises(ValueError, match="^a raw result set derives from provider results or a"):
        replace(valid, header=replace(valid.header, data_origin="provider_metadata"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(valid, results=(_result(0), _result(1, raw_payload_sha256="d" * 64)))


def test_result_set_validation_refuses_another_chain_or_a_fixture_as_hardware() -> None:
    """Results name this receipt, bundle and attempt; origins follow the receipt."""
    chain = _chain()
    other = _chain((_job(0, (0, 1)),))
    queried = _chain(provenance="provider_target_query")
    for result_set, reason in (
        (_set(other), "^results name a different provider receipt$"),
        (
            _set(chain, compiled_bundle_digest=_digest_of(queried.bundle)),
            "^results, receipt and bundle name different compiled bundles$",
        ),
        (
            _set(chain, attempt_id="7f1c2d3e-4b5a-4c6d-8e7f-901a2b3c4d5e"),
            "^results belong to another attempt$",
        ),
        (
            _set(chain, gaps=(_missing(2),), expected_cell_count=3),
            "^expected cell count differs from the compiled bundle$",
        ),
        (
            _set(chain, origin="provider_result"),
            "^result origin does not match the receipt; a fixture is never hardware$",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            _validate(chain, result_set)
    with pytest.raises(ValueError, match="^result origin does not match the receipt"):
        _validate(queried, _set(queried, origin="synthetic_classical"))
    with pytest.raises(ValueError, match="^results, receipt and bundle name different compiled"):
        validate_raw_result_set(_set(chain), chain.receipt, queried.bundle, chain.planned)
    not_a_set: Any = "results"
    with pytest.raises(ValueError, match="^result validation needs a RawResultSet$"):
        validate_raw_result_set(not_a_set, chain.receipt, chain.bundle, chain.planned)


def test_result_set_validation_ties_each_result_to_its_plan_and_receipt_position() -> None:
    """Keys, shots, widths, qubits, jobs, IDs and circuit positions must all match."""
    chain = _chain()
    swapped = _chain((_job(0, (1, 0)),))
    ambiguous = _chain((_job(0, (0,)), _job(1, (1,), ambiguous=True)))
    other_key = _key("sample-9")
    short = (("00", 600), ("11", 399))
    for used, results, reason in (
        (chain, (_result(0, key=other_key), _result(1)), "^result key differs from the planned"),
        (
            chain,
            (_result(0, requested_shots=999, actual_shots=999, counts=short), _result(1)),
            "^requested shots differ from the planned cell$",
        ),
        (
            chain,
            (_result(0, measured_qubits=(2, 0, 1), counts=(("000", 1000),)), _result(1)),
            "^measured width differs from the planned cell$",
        ),
        (
            chain,
            (_result(0, measured_qubits=(5, 0)), _result(1)),
            "^a measured qubit is outside the compiled layout$",
        ),
        (chain, (_result(0, job_ordinal=2), _result(1)), "^result names a job the receipt does"),
        (ambiguous, (_result(0), _result(1)), "^a result cannot come from an ambiguous job$"),
        (
            chain,
            (_result(0, provider_job_id=_JOB_IDS[1]), _result(1, provider_job_id=_JOB_IDS[0])),
            "^result job ID differs from the receipt$",
        ),
        (
            chain,
            (_result(0, job_ordinal=1, provider_job_id=_JOB_IDS[1], pub_index=1), _result(1)),
            "^result cell is not in its receipt job$",
        ),
        (
            swapped,
            (
                _result(0, job_ordinal=0, provider_job_id=_JOB_IDS[0], pub_index=0),
                _result(1, job_ordinal=0, provider_job_id=_JOB_IDS[0], pub_index=1),
            ),
            "^result is not at its receipt circuit position$",
        ),
        (
            chain,
            (_result(0, binding_index=1), _result(1)),
            "^v1 maps one cell per PUB, so the binding index is zero$",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            _validate(used, _set(used, results))
    result_set = _set(chain)
    planned = chain.planned
    other_plan = replace(planned[0], circuit_plan_digest="9" * 64)
    stray: Any = (planned[0], "plan")
    as_list: Any = list(planned)
    for cells, reason in (
        (planned[:1], "^validation needs one planned cell per compiled cell$"),
        (as_list, "^validation needs one planned cell per compiled cell$"),
        (stray, "^planned cells must be PlannedCell records$"),
        ((other_plan, planned[1]), "^planned cell differs from the compiled bundle$"),
    ):
        with pytest.raises(ValueError, match=reason):
            validate_raw_result_set(result_set, chain.receipt, chain.bundle, cells)
    reshot = _chain(shots=(1500, 2000))
    with pytest.raises(ValueError, match="^compiled shots differ from the planned cell$"):
        _validate(reshot, _set(reshot, (), (_missing(0), _missing(1))))


def _forge(wire: dict[str, Any], path: tuple[str | int, ...], value: object) -> dict[str, Any]:
    """Return the wire with one nested field replaced."""
    holder: Any = wire
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return wire


_WORKER_FORGERIES: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
    (("unexpected",), True, "RawResultSet fields mismatch"),
    (("schema",), "scpn.experimental.llm_qpu.raw_result_set.v2", "unknown RawResultSet schema"),
    (("attempt_id",), "attempt-1", "attempt ID must be canonical UUID text"),
    (("compiled_bundle_digest",), "E" * 64, "invalid compiled_bundle_digest"),
    (("expected_cell_count",), 0, "expected cell count must be a bounded positive integer"),
    (("results",), {"first": 1}, "results and gaps must be lists"),
    (("results",), [], "every expected cell needs one result or one recorded gap"),
    (("results", 0, "unexpected"), True, "CellResult fields mismatch"),
    (("results", 0, "pub_index"), -1, "pub_index must be a bounded non-negative integer"),
    (("results", 0, "key"), {"experiment_id": "x"}, "CellKey fields mismatch"),
    (("results", 0, "provider_job_id"), "", "provider job ID must be bounded normalized text"),
    (("results", 0, "measured_qubits"), [], "measured qubits must be a list of one to 64"),
    (("results", 0, "measured_qubits"), [2, 2], "a qubit cannot be measured into two bits"),
    (("results", 0, "measured_qubits"), [True, 0], "measured qubit must be a bounded"),
    (("results", 0, "requested_shots"), 0, "requested shots must be a bounded positive"),
    (("results", 0, "actual_shots"), 999, "actual shots differ from the requested shots"),
    (("results", 0, "counts"), [], "raw counts must be a nonempty list of at most 4096"),
    (("results", 0, "counts", 0), ["00", 600, 1], "each raw count is one bitstring and one"),
    (("results", 0, "counts", 0), ["000", 600], "raw bitstring width differs from the measured"),
    (("results", 0, "counts", 0), ["00", 0], "raw counts must be positive integers; zeros pad"),
    (("results", 0, "counts", 0), ["00", True], "raw counts must be positive integers"),
    (("results", 0, "counts", 0), ["00", 599], "sorted, unique and sum to the actual shots"),
    (("results", 0, "counts", 0), ["11", 600], "sorted, unique and sum to the actual shots"),
    (("results", 0, "raw_payload_sha256"), "E" * 64, "invalid raw payload digest"),
    (("results", 1, "cell_ordinal"), 0, "ordered by unique cell ordinal"),
    (("results", 1, "pub_index"), 5, "artifact content digest mismatch"),
    (("gaps",), [{"cell_ordinal": 0}], "CellGap fields mismatch"),
    (("header", "data_origin"), "provider_metadata", "derives from provider results or a"),
    (("header", "content_digest"), "f" * 64, "artifact content digest mismatch"),
)


def test_result_set_wire_is_exact_here_and_in_the_worker(tmp_path: Path) -> None:
    """Edited wires are refused by the decoder and the worker; decode_contract stays closed."""
    chain = _chain()
    result_set = _set(chain)
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(result_set.to_wire()))
    decoder_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "^RawResultSet fields mismatch$"),
        (("schema",), "scpn.experimental.llm_qpu.raw_result_set.v2", "^unknown RawResultSet"),
        (("object_kind",), "provider_batch_receipt", "^unknown RawResultSet schema$"),
        (("results",), {"first": 1}, "^results and gaps must be lists on wire$"),
        (("gaps",), {"first": 1}, "^results and gaps must be lists on wire$"),
        (("results", 0, "unexpected"), True, "^CellResult fields mismatch$"),
        (("results", 0, "counts"), {"first": 1}, "^measured qubits and counts must be lists"),
        (("results", 0, "measured_qubits"), "20", "^measured qubits and counts must be lists"),
        (("results", 0, "counts", 0), "00", "^each raw count must be a list on wire$"),
    )
    for path, value, reason in decoder_cases:
        with pytest.raises(ValueError, match=reason):
            decode_raw_result_set(canonical_bytes(_forge(result_set.to_wire(), path, value)))
    gap_wire: dict[str, Any] = _set(chain, (_result(0),), (_missing(1),)).to_wire()
    gap_wire["gaps"][0]["unexpected"] = True
    with pytest.raises(ValueError, match="^CellGap fields mismatch$"):
        decode_raw_result_set(canonical_bytes(gap_wire))
    for path, forgery, refusal in _WORKER_FORGERIES:
        forged = _forge(result_set.to_wire(), path, forgery)
        response = _worker(_request(chain, forged), tmp_path)
        assert response.returncode == 2, path
        assert refusal in json.loads(response.stdout)["reason"], path
    for status, raw, refusal in (
        ("quarantined", None, "invalid quarantined raw payload digest"),
        ("dropped", None, "missing without raw data or quarantined with it"),
        ("missing", "e" * 64, "missing without raw data or quarantined with it"),
    ):
        wire = _set(chain, (_result(0),), (_missing(1),)).to_wire()
        forged = _forge(
            _forge(wire, ("gaps", 0, "status"), status), ("gaps", 0, "raw_payload_sha256"), raw
        )
        response = _worker(_request(chain, forged), tmp_path)
        assert response.returncode == 2, status
        assert refusal in json.loads(response.stdout)["reason"], status
    merged = _forge(result_set.to_wire(), ("results", 1, "provider_job_id"), _JOB_IDS[0])
    response = _worker(_request(chain, _forge(merged, ("results", 1, "job_ordinal"), 0)), tmp_path)
    assert response.returncode == 2
    assert "merged PUB" in json.loads(response.stdout)["reason"]
    swapped = _chain((_job(0, (1, 0)),))
    ambiguous = _chain((_job(0, (0,)), _job(1, (1,), ambiguous=True)))
    queried = _chain(provenance="provider_target_query")
    other = _chain((_job(0, (0, 1)),))
    short = (("00", 600), ("11", 399))
    consistent: tuple[tuple[_Chain, RawResultSet, str], ...] = (
        (chain, _set(other), "results name a different provider receipt"),
        (
            chain,
            _set(chain, compiled_bundle_digest=_digest_of(queried.bundle)),
            "name different compiled bundles",
        ),
        (
            chain,
            _set(chain, attempt_id="7f1c2d3e-4b5a-4c6d-8e7f-901a2b3c4d5e"),
            "results belong to another attempt",
        ),
        (
            chain,
            _set(chain, gaps=(_missing(2),), expected_cell_count=3),
            "expected cell count differs from the compiled bundle",
        ),
        (chain, _set(chain, origin="provider_result"), "a fixture is never hardware"),
        (queried, _set(queried, origin="synthetic_classical"), "a fixture is never hardware"),
        (chain, _set(chain, (_result(0, key=_key("s-9")), _result(1))), "result key or requested"),
        (
            chain,
            _set(
                chain,
                (_result(0, requested_shots=999, actual_shots=999, counts=short), _result(1)),
            ),
            "result key or requested shots differ",
        ),
        (
            chain,
            _set(
                chain, (_result(0, measured_qubits=(2, 0, 1), counts=(("000", 1000),)), _result(1))
            ),
            "measured width differs",
        ),
        (chain, _set(chain, (_result(0, measured_qubits=(5, 0)), _result(1))), "outside the"),
        (chain, _set(chain, (_result(0, job_ordinal=2), _result(1))), "a job the receipt does"),
        (ambiguous, _set(ambiguous), "result job is ambiguous or its ID differs"),
        (
            chain,
            _set(
                chain,
                (_result(0, provider_job_id=_JOB_IDS[1]), _result(1, provider_job_id=_JOB_IDS[0])),
            ),
            "result job is ambiguous or its ID differs",
        ),
        (
            chain,
            _set(
                chain,
                (_result(0, job_ordinal=1, provider_job_id=_JOB_IDS[1], pub_index=1), _result(1)),
            ),
            "result cell is not in its receipt job",
        ),
        (
            swapped,
            _set(
                swapped,
                (
                    _result(0, job_ordinal=0, provider_job_id=_JOB_IDS[0], pub_index=0),
                    _result(1, job_ordinal=0, provider_job_id=_JOB_IDS[0], pub_index=1),
                ),
            ),
            "not at its receipt circuit position",
        ),
        (chain, _set(chain, (_result(0, binding_index=1), _result(1))), "binding zero"),
    )
    for used, edited, refusal in consistent:
        response = _worker(_request(used, edited.to_wire()), tmp_path)
        assert response.returncode == 2, refusal
        assert refusal in json.loads(response.stdout)["reason"], refusal
    planned = [cell.to_wire() for cell in chain.planned]
    for cells, refusal in (
        (planned[:1], "one planned cell per compiled cell"),
        ([planned[0], {**planned[0], "unexpected": True}], "planned cell fields mismatch"),
        (
            [{**planned[0], "circuit_plan_digest": "9" * 64}, planned[1]],
            "planned cell differs from the compiled bundle",
        ),
    ):
        request = _request(chain, result_set.to_wire())
        request["planned"] = cells
        response = _worker(request, tmp_path)
        assert response.returncode == 2, refusal
        assert refusal in json.loads(response.stdout)["reason"], refusal
    reshot = _chain(shots=(1500, 2000))
    response = _worker(
        _request(reshot, _set(reshot, (), (_missing(0), _missing(1))).to_wire()), tmp_path
    )
    assert response.returncode == 2
    assert "compiled shots differ from the planned cell" in json.loads(response.stdout)["reason"]
    missing = _request(chain, result_set.to_wire())
    del missing["planned"]
    response = _worker(missing, tmp_path)
    assert response.returncode == 2
    assert json.loads(response.stdout)["reason"] == "result request fields mismatch"
