# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental feature batch contract
"""Exercise feature batches recomputed from raw counts against the chain and the worker."""

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
    FEATURE_BATCH_SCHEMA,
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
    FeatureBatch,
    FeatureRow,
    MeasurementPlan,
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
    decode_feature_batch,
    estimate_sampled_basis,
    validate_feature_batch,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _REPO_ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"
_ATTEMPT = "6f1c2d3e-4b5a-4c6d-8e7f-901a2b3c4d5e"
_JOB_IDS = ("019f8639-96be-77d1-8f70-109a70e6fb33", "019f8639-980e-7c21-bb38-f26396a883e3")
_LAYOUT = (3, 0, 1, 2)
_MEASURED = ((0, 1, 2, 3), (2, 3, 0, 1))
_COUNTS = (
    (("0001", 500), ("0110", 300), ("1101", 200)),
    (("0010", 1200), ("1011", 800)),
)
_LOGICAL_TO_CLBIT = (1, 2, 3, 0)
_DISPLAY = (3, 2, 1, 0)


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


def _planned(shots: tuple[int, int]) -> tuple[PlannedCell, ...]:
    """Return two four-qubit planned cells with their own circuits."""
    return tuple(
        PlannedCell.create(
            key=_key(f"sample-{index}"),
            circuit_plan_digest=str(index + 1) * 64,
            shots=count,
            measurement_width=4,
            role="data",
            metadata={},
        )
        for index, count in enumerate(shots)
    )


def _plan(index: int, shots: int, basis: str = "Z", **changes: Any) -> MeasurementPlan:
    """Return the measurement plan of one cell's circuit on the compiled layout."""
    singles = tuple(f"{basis}{wire}" for wire in range(4))
    pairs = tuple(f"{basis}{wire}{basis}{wire + 1}" for wire in range(3))
    fields: dict[str, Any] = {
        "circuit_plan_digest": str(index + 1) * 64,
        "n_qubits": 4,
        "basis_id": basis,
        "shots": shots,
        "observable_ids": singles + pairs,
        "logical_to_clbit": _LOGICAL_TO_CLBIT,
        "display_order_clbits": _DISPLAY,
        "register_name": "meas",
        "physical_qubits": _LAYOUT,
        "pauli_label_convention": "logical_q0_low_index_v1",
        **changes,
    }
    return MeasurementPlan(**fields)


def _target(provenance: str) -> TargetSnapshot:
    """Build a five-qubit line fixture target with two circuits per job."""
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
        "num_qubits": 5,
    }
    edges = ((0, 1), (1, 2), (2, 3), (3, 4))
    origin = {"synthetic_fixture": "synthetic_classical"}.get(provenance, "provider_metadata")
    content: dict[str, object] = {
        "schema": TARGET_SNAPSHOT_SCHEMA,
        "object_kind": "target_snapshot",
        **plain,
        "basis_gates": ["cz", "measure", "prx"],
        "coupling_edges": [list(edge) for edge in edges],
        **{name: value.to_wire() for name, value in known.items()},
    }
    return TargetSnapshot(
        **plain,
        basis_gates=("cz", "measure", "prx"),
        coupling_edges=edges,
        **known,
        header=_header("target_snapshot", content, ("a" * 64,), origin),
    )


def _bundle(target: TargetSnapshot, planned: tuple[PlannedCell, ...]) -> CompiledBundle:
    """Build a bundle whose cells carry the planned cells' digests and shots."""
    cells = tuple(
        CompiledCell(_digest_of(cell), "2" * 64, "3" * 64, "4" * 64, cell.shots)
        for cell in planned
    )
    payload: dict[str, object] = {
        "cells": [cell.to_wire() for cell in cells],
        "layout": list(_LAYOUT),
        "logical_gates": ["cx", "measure", "rz"],
        "compiled_gate_counts": [["cz", 6], ["measure", 4], ["prx", 12]],
        "substitutions": [["cx", ["cz", "prx"]], ["rz", ["prx"]]],
        "compiled_depth": 14,
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
        layout=_LAYOUT,
        logical_gates=("cx", "measure", "rz"),
        compiled_gate_counts=(("cz", 6), ("measure", 4), ("prx", 12)),
        substitutions=(("cx", ("cz", "prx")), ("rz", ("prx",))),
        compiled_depth=14,
        provider_options=(),
        header=_header(
            "compiled_bundle",
            {**content, **identity, **payload},
            parents,
            target.header.data_origin,
        ),
    )


def _quote(bundle: CompiledBundle) -> CostQuote:
    """Build the bundle's one-job quote."""
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
    """Approve the quote with a placeholder signature digest."""
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
    return OwnerApproval(
        **scope, signature=envelope, header=_header("owner_approval", content, parents, origin)
    )


def _receipt(approval: OwnerApproval, bundle: CompiledBundle) -> ProviderBatchReceipt:
    """Build the receipt for one job per cell."""
    jobs = tuple(
        SubmittedJob(
            ordinal=ordinal,
            name=f"phase_{ordinal}",
            payload_sha256=hashlib.sha256(f"request-{ordinal}".encode()).hexdigest(),
            cell_ordinals=(ordinal,),
            state="submitted",
            provider_job_id=_JOB_IDS[ordinal],
            id_source="submit_response",
            boundary_at_utc="2026-11-16T09:00:00Z",
            provider_created_at_utc=None,
            blocker=None,
        )
        for ordinal in range(2)
    )
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


def _result(ordinal: int, counts: tuple[tuple[str, int], ...] | None = None) -> CellResult:
    """Return one cell's raw counts from its own job."""
    chosen = counts or _COUNTS[ordinal]
    shots = sum(count for _, count in chosen)
    return CellResult(
        cell_ordinal=ordinal,
        key=_key(f"sample-{ordinal}"),
        job_ordinal=ordinal,
        provider_job_id=_JOB_IDS[ordinal],
        pub_index=0,
        binding_index=0,
        measured_qubits=_MEASURED[ordinal],
        requested_shots=shots,
        actual_shots=shots,
        counts=chosen,
        raw_payload_sha256=("e", "f")[ordinal] * 64,
    )


def _result_set(
    receipt: ProviderBatchReceipt,
    bundle: CompiledBundle,
    results: tuple[CellResult, ...],
    gaps: tuple[CellGap, ...],
) -> RawResultSet:
    """Build the result set for the given results and gaps."""
    fields: dict[str, Any] = {
        "attempt_id": _ATTEMPT,
        "provider_batch_receipt_digest": _digest_of(receipt),
        "compiled_bundle_digest": _digest_of(bundle),
        "expected_cell_count": 2,
    }
    content = {
        "schema": RAW_RESULT_SET_SCHEMA,
        "object_kind": "raw_result_set",
        **fields,
        "results": [result.to_wire() for result in results],
        "gaps": [gap.to_wire() for gap in gaps],
    }
    parents = (fields["provider_batch_receipt_digest"], fields["compiled_bundle_digest"])
    origin = {"synthetic_classical": "synthetic_classical"}.get(
        receipt.header.data_origin, "provider_result"
    )
    header = _header("raw_result_set", content, parents, origin)
    return RawResultSet(**fields, results=results, gaps=gaps, header=header)


@dataclass(frozen=True)
class _Chain:
    """One consistent chain from target to raw results, with planned cells and plans."""

    target: TargetSnapshot
    bundle: CompiledBundle
    quote: CostQuote
    approval: OwnerApproval
    receipt: ProviderBatchReceipt
    planned: tuple[PlannedCell, ...]
    results: RawResultSet
    plans: tuple[MeasurementPlan, ...]


def _chain(
    *,
    provenance: str = "synthetic_fixture",
    counts: tuple[tuple[tuple[str, int], ...], ...] = _COUNTS,
    missing: bool = False,
    bases: tuple[str, str] = ("Z", "Z"),
) -> _Chain:
    """Return one consistent chain; ``missing`` records cell 1 as a gap."""
    shots = tuple(sum(count for _, count in cell) for cell in counts)
    planned = _planned((shots[0], shots[1]))
    target = _target(provenance)
    bundle = _bundle(target, planned)
    quote = _quote(bundle)
    approval = _approval(quote, bundle, "test" if provenance == "synthetic_fixture" else "live")
    receipt = _receipt(approval, bundle)
    if missing:
        found: tuple[CellResult, ...] = (_result(0, counts[0]),)
        gaps: tuple[CellGap, ...] = (CellGap(1, "missing", "no data returned", None),)
    else:
        found, gaps = (_result(0, counts[0]), _result(1, counts[1])), ()
    plans = tuple(_plan(index, shots[index], bases[index]) for index in range(2))
    results = _result_set(receipt, bundle, found, gaps)
    return _Chain(target, bundle, quote, approval, receipt, planned, results, plans)


def _display_key(bits: str, measured: tuple[int, ...], plan: MeasurementPlan) -> str:
    """Write one physical-order bitstring in the plan's display order (test oracle)."""
    physical = plan.physical_qubits or ()
    logical = {
        clbit: bits[measured.index(physical[wire])]
        for wire, clbit in enumerate(plan.logical_to_clbit)
    }
    return "".join(logical[clbit] for clbit in plan.display_order_clbits)


def _row(chain: _Chain, index: int, **changes: Any) -> FeatureRow:
    """Return the row the oracle computes for one cell, or a missing row without a result."""
    plan = chain.plans[index]
    found = {result.cell_ordinal: result for result in chain.results.results}.get(index)
    fields: dict[str, Any] = {
        "cell_ordinal": index,
        "key": _key(f"sample-{index}"),
        "measurement_request_digest": plan.request_digest,
        "shots": None,
        "values": None,
        "covariance": None,
        "missing_reason": "no data returned",
    }
    if found is not None:
        counts = {_display_key(bits, found.measured_qubits, plan): n for bits, n in found.counts}
        estimate = estimate_sampled_basis(plan, counts, origin="digital_sampled")
        fields.update(
            shots=estimate.shots,
            values=estimate.expectations,
            covariance=estimate.covariance_of_mean,
            missing_reason=None,
        )
    return FeatureRow(**{**fields, **changes})


def _batch(
    chain: _Chain,
    rows: tuple[FeatureRow, ...] | None = None,
    *,
    basis: str = "Z",
    origin: str | None = None,
    **changes: Any,
) -> FeatureBatch:
    """Build the chain's feature batch for one basis, hashing exactly its content."""
    selected = [index for index, plan in enumerate(chain.plans) if plan.basis_id == basis]
    rows = rows if rows is not None else tuple(_row(chain, index) for index in selected)
    fields: dict[str, Any] = {
        "raw_result_set_digest": _digest_of(chain.results),
        "estimator": "sampled_mean_v1",
        "basis_id": basis,
        "observable_ids": _plan(0, 1, basis).observable_ids,
        **changes,
    }
    content = {
        "schema": FEATURE_BATCH_SCHEMA,
        "object_kind": "feature_batch",
        **fields,
        "observable_ids": list(fields["observable_ids"]),
        "rows": [row.to_wire() for row in rows],
    }
    data_origin = origin or chain.results.header.data_origin
    header = _header("feature_batch", content, (fields["raw_result_set_digest"],), data_origin)
    return FeatureBatch(**fields, rows=rows, header=header)


def _validate(chain: _Chain, batch: FeatureBatch) -> None:
    """Validate the batch against its chain."""
    validate_feature_batch(batch, chain.results, chain.bundle, chain.planned, chain.plans)


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


def _request(chain: _Chain, features: object) -> dict[str, object]:
    """Return one worker feature request."""
    return {
        "op": "roundtrip_feature_batch",
        "target": chain.target.to_wire(),
        "bundle": chain.bundle.to_wire(),
        "quote": chain.quote.to_wire(),
        "approval": chain.approval.to_wire(),
        "receipt": chain.receipt.to_wire(),
        "planned": [cell.to_wire() for cell in chain.planned],
        "results": chain.results.to_wire(),
        "plans": [plan.to_wire() for plan in chain.plans],
        "features": features,
    }


def test_feature_batches_recompute_from_raw_counts_and_roundtrip(tmp_path: Path) -> None:
    """Complete, missing, mixed-basis, single-shot and provider batches validate everywhere."""
    mixed = _chain(bases=("Z", "X"))
    single = _chain(counts=((("0110", 1),), (("1001", 1),)))
    cases = (
        (_chain(), _batch(_chain())),
        (_chain(missing=True), _batch(_chain(missing=True))),
        (mixed, _batch(mixed)),
        (mixed, _batch(mixed, basis="X")),
        (single, _batch(single)),
        (
            _chain(provenance="provider_target_query"),
            _batch(_chain(provenance="provider_target_query")),
        ),
    )
    for chain, batch in cases:
        _validate(chain, batch)
        assert decode_feature_batch(canonical_bytes(batch.to_wire())) == batch
        response = _worker(_request(chain, batch.to_wire()), tmp_path)
        assert response.returncode == 0, response.stdout.decode()
        payload = json.loads(response.stdout)
        assert payload["status"] == "validated_roundtrip_no_compute"
        assert payload["hardware_submission_enabled"] is False
        assert payload["features_sha256"] == _digest_of(batch)
        assert decode_feature_batch(canonical_bytes(payload["features"])) == batch
    complete = cases[0][1]
    assert complete.missing() == ()
    assert cases[1][1].missing() == ((1, "no data returned"),)
    assert [row.cell_ordinal for row in cases[2][1].rows] == [0]
    assert [row.cell_ordinal for row in cases[3][1].rows] == [1]
    assert cases[4][1].rows[0].covariance is None
    assert cases[5][1].header.data_origin == "provider_result"
    first = complete.rows[0].values or ()
    assert len(first) == 7 and all(-1.0 <= value <= 1.0 for value in first)


def test_feature_row_refuses_quasi_values_coercions_and_unexplained_gaps() -> None:
    """Values are finite floats in [-1, 1]; covariance is square and symmetric."""
    chain = _chain()
    row = _row(chain, 0)
    values = row.values or ()
    matrix = [list(line) for line in row.covariance or ()]
    matrix[0][1] = matrix[0][1] + 1e-9
    asymmetric = tuple(tuple(line) for line in matrix)
    for changes, reason in (
        (
            {"values": (1.5, *values[1:])},
            "lies in \\[-1, 1\\]; quasi-probability estimates are not",
        ),
        ({"values": (1, *values[1:])}, "^feature values must be finite floats$"),
        ({"values": (float("nan"), *values[1:])}, "^feature values must be finite floats$"),
        ({"values": ()}, "^feature values must be a nonempty tuple of at most 64$"),
        ({"values": list(values)}, "^feature values must be a nonempty tuple of at most 64$"),
        ({"values": (0.0,) * 65}, "^feature values must be a nonempty tuple of at most 64$"),
        ({"covariance": asymmetric}, "^covariance must be symmetric$"),
        ({"covariance": None}, "^covariance must be a square tuple matching the features$"),
        ({"covariance": (row.covariance or ())[:3]}, "^covariance must be a square tuple"),
        ({"covariance": tuple(line[:3] for line in row.covariance or ())}, "^covariance must be"),
        ({"covariance": ((1,) * 7,) * 7}, "^covariance entries must be finite floats$"),
        ({"shots": 0}, "feature shots must be a bounded positive integer"),
        ({"shots": None}, "feature shots must be a bounded positive integer"),
        ({"missing_reason": "late"}, "^a measured row cannot carry a missing reason$"),
        ({"key": "sample-0"}, "^a feature row needs its CellKey$"),
        ({"measurement_request_digest": "E" * 64}, "^measurement request digest must be a"),
        ({"cell_ordinal": -1}, "^cell ordinal must be a bounded non-negative integer$"),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(row, **changes)
    with pytest.raises(ValueError, match="^one shot has no sampling covariance$"):
        replace(row, shots=1)
    gap = _row(_chain(missing=True), 1)
    gap_cases: tuple[tuple[dict[str, Any], str], ...] = (
        ({"shots": 1000}, "^a missing row carries no shots or covariance$"),
        ({"covariance": row.covariance}, "^a missing row carries no shots or covariance$"),
        ({"missing_reason": None}, "missing reason must be text"),
        ({"missing_reason": ""}, "missing reason must be bounded nonempty text"),
    )
    for changes, reason in gap_cases:
        with pytest.raises(ValueError, match=reason):
            replace(gap, **changes)


def test_feature_batch_refuses_quasi_estimators_reordered_rows_and_bad_headers() -> None:
    """Only sampled means of raw counts, in cell order, with one value per observable."""
    chain = _chain()
    valid = _batch(chain)
    rows = valid.rows
    names = valid.observable_ids
    stray: Any = (rows[0], "row")
    for changes, reason in (
        ({"estimator": "quasi_probability_v1"}, "^quasi-probabilities are not raw counts"),
        ({"estimator": "mitigated_mean_v1"}, "^unknown feature estimator$"),
        ({"basis_id": "W"}, "^feature basis must be X, Y or Z$"),
        ({"observable_ids": ()}, "^observable IDs must be a nonempty tuple of at most 64$"),
        ({"observable_ids": list(names)}, "^observable IDs must be a nonempty tuple of at most"),
        ({"observable_ids": (names[0], names[0], *names[2:])}, "^observable IDs must be unique$"),
        ({"observable_ids": names[:6]}, "^every measured row has one value per observable$"),
        ({"rows": ()}, "^feature rows must be a nonempty tuple of FeatureRow records$"),
        ({"rows": list(rows)}, "^feature rows must be a nonempty tuple of FeatureRow records$"),
        ({"rows": stray}, "^feature rows must be a nonempty tuple of FeatureRow records$"),
        ({"rows": (rows[1], rows[0])}, "^feature rows must follow the result set's cell order$"),
        ({"rows": (rows[0], rows[0])}, "^feature rows must follow the result set's cell order$"),
        ({"raw_result_set_digest": "E" * 64}, "^raw result set digest must be a lowercase"),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(valid, **changes)
    with pytest.raises(ValueError, match="^FeatureBatch requires its exact artifact header$"):
        replace(valid, header=replace(valid.header, object_kind="raw_result_set"))
    with pytest.raises(ValueError, match="^a feature batch derives from provider results or a"):
        replace(valid, header=replace(valid.header, data_origin="provider_metadata"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(valid, rows=rows[:1])


def test_feature_validation_refuses_other_lineage_dropped_rows_and_altered_values() -> None:
    """Rows cover every cell of the basis; values are exactly the raw-count estimates."""
    chain = _chain()
    gapped = _chain(missing=True)
    queried = _chain(provenance="provider_target_query")
    row = _row(chain, 0)
    values = row.values or ()
    altered = (values[0] + 2**-40, *values[1:])
    for used, batch, reason in (
        (chain, _batch(gapped), "^features name a different raw result set$"),
        (chain, _batch(chain, origin="provider_result"), "^feature origin differs from the raw"),
        (chain, _batch(chain, rows=(row,)), "^feature rows must cover every cell measured in"),
        (chain, _batch(chain, rows=(_row(chain, 1),)), "^feature rows must cover every cell"),
        (
            chain,
            _batch(chain, rows=(_row(chain, 0, key=_key("sample-9")), _row(chain, 1))),
            "^feature row key differs from the planned cell$",
        ),
        (
            chain,
            _batch(
                chain, rows=(_row(chain, 0, measurement_request_digest="9" * 64), _row(chain, 1))
            ),
            "^feature row names another measurement request$",
        ),
        (
            gapped,
            _batch(gapped, rows=(_row(gapped, 0), _row(chain, 1))),
            "^a cell without a result must stay missing, never zero-filled$",
        ),
        (
            chain,
            _batch(
                chain,
                rows=(
                    _row(
                        chain, 0, shots=None, values=None, covariance=None, missing_reason="lost"
                    ),
                    _row(chain, 1),
                ),
            ),
            "^a measured cell cannot be reported missing$",
        ),
        (
            chain,
            _batch(chain, rows=(_row(chain, 0, values=altered), _row(chain, 1))),
            "^feature values differ from the raw counts$",
        ),
        (
            chain,
            _batch(chain, rows=(_row(chain, 0, shots=999), _row(chain, 1))),
            "^feature values differ from the raw counts$",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            _validate(used, batch)
    zeroed = tuple(0.0 for _ in values)
    with pytest.raises(ValueError, match="^feature values differ from the raw counts$"):
        _validate(chain, _batch(chain, rows=(_row(chain, 0, values=zeroed), _row(chain, 1))))
    x_columns = _plan(0, 1, "X").observable_ids
    with pytest.raises(ValueError, match="^feature columns differ from the cell's observables$"):
        _validate(chain, _batch(chain, observable_ids=x_columns))
    batch = _batch(chain)
    with pytest.raises(ValueError, match="^the raw result set belongs to another compiled bundle"):
        validate_feature_batch(batch, chain.results, queried.bundle, chain.planned, chain.plans)
    with pytest.raises(ValueError, match="^validation needs one planned cell per compiled cell$"):
        validate_feature_batch(batch, chain.results, chain.bundle, chain.planned[:1], chain.plans)
    as_list: Any = list(chain.plans)
    stray: Any = (chain.plans[0], "plan")
    for plans, reason in (
        (chain.plans[:1], "^validation needs one measurement plan per compiled cell$"),
        (as_list, "^validation needs one measurement plan per compiled cell$"),
        (stray, "^measurement plans must be MeasurementPlan records$"),
        ((_plan(0, 1000, circuit_plan_digest="9" * 64), chain.plans[1]), "^measurement plan does"),
    ):
        with pytest.raises(ValueError, match=reason):
            validate_feature_batch(batch, chain.results, chain.bundle, chain.planned, plans)
    not_a_batch: Any = "features"
    with pytest.raises(ValueError, match="^feature validation needs a FeatureBatch$"):
        _validate(chain, not_a_batch)


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"physical_qubits": (0, 1, 2, 4)}, "^measurement plan layout differs from the measured"),
        ({"physical_qubits": None}, "^measurement plan layout differs from the measured qubits$"),
        ({"shots": 1001}, "^raw counts do not match requested shots$"),
        ({"shots": 999}, "^raw counts exceed requested shots$"),
    ],
)
def test_feature_validation_refuses_a_plan_that_does_not_fit_the_raw_counts(
    changes: dict[str, Any], reason: str
) -> None:
    """The plan's layout must be the measured qubits and its shots the counted shots."""
    chain = _chain()
    edits: dict[str, Any] = dict(changes)
    plans = (_plan(0, edits.pop("shots", 1000), **edits), chain.plans[1])
    edited = replace(chain, plans=plans)
    rows = (_row(chain, 0, measurement_request_digest=plans[0].request_digest), _row(chain, 1))
    with pytest.raises(ValueError, match=reason):
        _validate(edited, _batch(edited, rows=rows))


def _forge(wire: dict[str, Any], path: tuple[str | int, ...], value: object) -> dict[str, Any]:
    """Return the wire with one nested field replaced."""
    holder: Any = wire
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return wire


_WORKER_FORGERIES: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
    (("unexpected",), True, "FeatureBatch fields mismatch"),
    (("schema",), "scpn.experimental.llm_qpu.feature_batch.v2", "unknown FeatureBatch schema"),
    (("raw_result_set_digest",), "E" * 64, "invalid raw result set digest"),
    (("estimator",), "quasi_probability_v1", "quasi-probabilities are not"),
    (("basis_id",), "W", "feature basis must be X, Y or Z"),
    (("observable_ids",), [], "observable IDs must be a nonempty unique list"),
    (("observable_ids",), ["Z0", "Z0"], "observable IDs must be a nonempty unique list"),
    (("rows",), [], "feature rows must be a nonempty list"),
    (("rows", 0, "unexpected"), True, "FeatureRow fields mismatch"),
    (("rows", 0, "cell_ordinal"), -1, "cell_ordinal must be a bounded non-negative integer"),
    (("rows", 0, "key"), {"experiment_id": "x"}, "CellKey fields mismatch"),
    (("rows", 0, "measurement_request_digest"), "E" * 64, "invalid measurement request"),
    (("rows", 0, "shots"), 0, "a measured row has positive shots and no missing reason"),
    (("rows", 0, "missing_reason"), "late", "a measured row has positive shots and no"),
    (("rows", 0, "values"), [], "every measured row has one value per observable"),
    (("rows", 0, "values", 0), {"$f64": "0x1.8p+0"}, "lies in [-1, 1]"),
    (("rows", 0, "values", 0), 1, "feature numbers must be f64 tags"),
    (("rows", 0, "covariance"), None, "covariance must be a square list matching the features"),
    (("rows", 0, "covariance", 0), [], "covariance must be a square list matching the features"),
    (("rows", 1, "cell_ordinal"), 0, "feature rows must follow the result set's cell order"),
    (("rows", 0, "values", 0), {"$f64": "0x0.0p+0"}, "artifact content digest mismatch"),
    (("header", "data_origin"), "provider_metadata", "derives from provider results or a"),
    (("header", "content_digest"), "f" * 64, "artifact content digest mismatch"),
)


def test_feature_batch_wire_is_exact_here_and_in_the_worker(tmp_path: Path) -> None:
    """Edited wires are refused by the decoder and the worker; decode_contract stays closed."""
    chain = _chain()
    batch = _batch(chain)
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(batch.to_wire()))
    decoder_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "^FeatureBatch fields mismatch$"),
        (("schema",), "scpn.experimental.llm_qpu.feature_batch.v2", "^unknown FeatureBatch"),
        (("object_kind",), "raw_result_set", "^unknown FeatureBatch schema$"),
        (("rows",), {"first": 1}, "^observable IDs and rows must be lists on wire$"),
        (("observable_ids",), "Z0", "^observable IDs and rows must be lists on wire$"),
        (("rows", 0, "unexpected"), True, "^FeatureRow fields mismatch$"),
        (("rows", 0, "values"), "0.5", "^feature values must be a list or null on wire$"),
        (("rows", 0, "covariance"), [1], "^covariance must be a list of lists or null on wire$"),
        (("rows", 0, "covariance"), "x", "^covariance must be a list of lists or null on wire$"),
        (("rows", 0, "values", 0), 1, "^feature values must be f64 tags on wire$"),
        (("rows", 0, "covariance", 0, 0), "x", "^covariance entries must be f64 tags on wire$"),
    )
    for path, value, reason in decoder_cases:
        forged = _forge(batch.to_wire(), path, value)
        with pytest.raises(ValueError, match=reason):
            decode_feature_batch(canonical_bytes(forged))
    for path, forgery, refusal in _WORKER_FORGERIES:
        response = _worker(_request(chain, _forge(batch.to_wire(), path, forgery)), tmp_path)
        assert response.returncode == 2, path
        assert refusal in json.loads(response.stdout)["reason"], path
    for field, forgery, refusal in (
        ("shots", 1000, "a missing row carries no shots or covariance"),
        ("missing_reason", "", "missing reason must be bounded normalized text"),
    ):
        forged = _forge(_batch(_chain(missing=True)).to_wire(), ("rows", 1, field), forgery)
        response = _worker(_request(_chain(missing=True), forged), tmp_path)
        assert response.returncode == 2, field
        assert refusal in json.loads(response.stdout)["reason"], field
    single = _chain(counts=((("0110", 1),), (("1001", 1),)))
    forged = _forge(_batch(single).to_wire(), ("rows", 0, "covariance"), [[{"$f64": "0x0.0p+0"}]])
    response = _worker(_request(single, forged), tmp_path)
    assert response.returncode == 2
    assert "one shot has no sampling covariance" in json.loads(response.stdout)["reason"]
    gapped = _chain(missing=True)
    row = _row(chain, 0)
    values = row.values or ()
    altered = (values[0] + 2**-40, *values[1:])
    x_columns = _plan(0, 1, "X").observable_ids
    consistent: tuple[tuple[_Chain, FeatureBatch, str], ...] = (
        (chain, _batch(gapped), "features name a different raw result set"),
        (chain, _batch(chain, origin="provider_result"), "feature origin differs from the raw"),
        (chain, _batch(chain, rows=(row,)), "must cover every cell measured in this basis"),
        (
            chain,
            _batch(chain, rows=(_row(chain, 0, key=_key("sample-9")), _row(chain, 1))),
            "key or columns differ",
        ),
        (chain, _batch(chain, observable_ids=x_columns), "key or columns differ"),
        (
            chain,
            _batch(
                chain, rows=(_row(chain, 0, measurement_request_digest="9" * 64), _row(chain, 1))
            ),
            "names another measurement request",
        ),
        (
            gapped,
            _batch(gapped, rows=(_row(gapped, 0), _row(chain, 1))),
            "must stay missing, never zero-filled",
        ),
        (
            chain,
            _batch(
                chain,
                rows=(
                    _row(
                        chain, 0, shots=None, values=None, covariance=None, missing_reason="lost"
                    ),
                    _row(chain, 1),
                ),
            ),
            "a measured cell cannot be reported missing",
        ),
        (
            chain,
            _batch(chain, rows=(_row(chain, 0, values=altered), _row(chain, 1))),
            "feature values differ from the raw counts",
        ),
        (
            chain,
            _batch(chain, rows=(_row(chain, 0, shots=999), _row(chain, 1))),
            "feature values differ from the raw counts",
        ),
    )
    for used, edited, refusal in consistent:
        response = _worker(_request(used, edited.to_wire()), tmp_path)
        assert response.returncode == 2, refusal
        assert refusal in json.loads(response.stdout)["reason"], refusal
    for changes, refusal in (
        ({"physical_qubits": (0, 1, 2, 4)}, "layout differs from the measured qubits"),
        ({"physical_qubits": None}, "layout differs from the measured qubits"),
        ({"shots": 1001}, "raw counts do not match requested shots"),
        ({"shots": 999}, "raw counts do not match requested shots"),
        ({"circuit_plan_digest": "9" * 64}, "does not measure the planned circuit"),
    ):
        edits: dict[str, Any] = dict(changes)
        plans = (_plan(0, edits.pop("shots", 1000), **edits), chain.plans[1])
        edited_chain = replace(chain, plans=plans)
        rows = (_row(chain, 0, measurement_request_digest=plans[0].request_digest), _row(chain, 1))
        response = _worker(
            _request(edited_chain, _batch(edited_chain, rows=rows).to_wire()), tmp_path
        )
        assert response.returncode == 2, refusal
        assert refusal in json.loads(response.stdout)["reason"], refusal
    plan_wire = chain.plans[0].to_wire()
    for field, forgery, refusal in (
        ("unexpected", True, "measurement plan fields mismatch"),
        (
            "schema",
            "scpn.experimental.llm_qpu.measurement_plan.v2",
            "unsupported measurement plan",
        ),
        ("n_qubits", 5, "requires n4/n8 and an X, Y or Z basis"),
        ("shots", 0, "measurement shots must be a bounded positive integer"),
        ("observable_ids", ["Z0"], "measurement observable order or basis mismatch"),
        ("logical_to_clbit", [0, 0, 1, 2], "must be complete permutations"),
        ("display_order_clbits", [0.0, 1, 2, 3], "must be integer lists"),
        ("register_name", "", "measurement register must be bounded normalized text"),
        ("physical_qubits", [0, 0, 1, 2], "physical qubits must be explicit unique bounded"),
        ("pauli_label_convention", "big_endian", "unsupported Pauli label convention"),
    ):
        request = _request(chain, batch.to_wire())
        request["plans"] = [{**plan_wire, field: forgery}, chain.plans[1].to_wire()]
        response = _worker(request, tmp_path)
        assert response.returncode == 2, field
        assert refusal in json.loads(response.stdout)["reason"], field
    request = _request(chain, batch.to_wire())
    request["plans"] = [plan.to_wire() for plan in chain.plans[:1]]
    response = _worker(request, tmp_path)
    assert response.returncode == 2
    assert "one measurement plan per compiled cell" in json.loads(response.stdout)["reason"]
    missing = _request(chain, batch.to_wire())
    del missing["plans"]
    response = _worker(missing, tmp_path)
    assert response.returncode == 2
    assert json.loads(response.stdout)["reason"] == "feature request fields mismatch"
