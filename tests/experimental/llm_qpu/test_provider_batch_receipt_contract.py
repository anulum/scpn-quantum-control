# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental provider batch receipt contract
"""Exercise provider batch receipts against their approval, bundle and the worker."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.experimental.llm_qpu.contracts import (
    COMPILED_BUNDLE_SCHEMA,
    COST_QUOTE_SCHEMA,
    OWNER_APPROVAL_SCHEMA,
    PROVIDER_BATCH_RECEIPT_SCHEMA,
    TARGET_SNAPSHOT_SCHEMA,
    ApprovalSignature,
    ArtifactHeader,
    CompiledBundle,
    CompiledCell,
    CostQuote,
    OwnerApproval,
    ProviderBatchReceipt,
    QuotedCount,
    ReportedValue,
    SubmittedJob,
    TargetSnapshot,
    canonical_bytes,
    decode_contract,
    decode_provider_batch_receipt,
    validate_provider_batch_receipt,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _REPO_ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"
_ATTEMPT = "6f1c2d3e-4b5a-4c6d-8e7f-901a2b3c4d5e"
_JOB_IDS = {
    "iqm": ("019f8639-96be-77d1-8f70-109a70e6fb33", "019f8639-980e-7c21-bb38-f26396a883e3"),
    "ibm": ("d86g7h1789is738vkreg", "d86ggpis46sc73f6v170"),
}
_ROUTES = {
    "iqm": ("iqm-resonance/owner-project", "garnet"),
    "ibm": ("ibm-quantum/owner-instance", "ibm_kingston"),
}
_ORIGIN = {
    "synthetic_fixture": "synthetic_classical",
    "provider_target_query": "provider_metadata",
}
_HAL_HANDLE = "ibm-run-0f96961442e05a77"
_THIRD_JOB_ID = "019f8642-e162-7383-83e5-e60c15a57355"
_BLOCKER = "the submit response was lost after the network boundary; reservation retained"


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


def _target(provider: str = "iqm", provenance: str = "synthetic_fixture") -> TargetSnapshot:
    """Build a three-qubit line fixture target with one circuit per job."""
    account, device = _ROUTES[provider]
    known = {
        "calibration_id": ReportedValue("known", "cal-2026-11-15-0800", None),
        "calibration_time_utc": ReportedValue("known", "2026-11-15T08:00:00Z", None),
        "max_shots": ReportedValue("known", 100_000, None),
        "max_circuits_per_job": ReportedValue("known", 1, None),
    }
    plain: dict[str, Any] = {
        "provider": provider,
        "account_route": account,
        "device_name": device,
        "sdk_name": "fixture-sdk",
        "sdk_version": "1.0.0",
        "sdk_target_digest": "a" * 64,
        "capability_provenance": provenance,
        "num_qubits": 3,
    }
    content: dict[str, object] = {
        "schema": TARGET_SNAPSHOT_SCHEMA,
        "object_kind": "target_snapshot",
        **plain,
        "basis_gates": ["cz", "measure", "prx"],
        "coupling_edges": [[0, 1], [1, 2]],
        **{name: value.to_wire() for name, value in known.items()},
    }
    header = _header("target_snapshot", content, ("a" * 64,), _ORIGIN[provenance])
    return TargetSnapshot(
        **plain,
        basis_gates=("cz", "measure", "prx"),
        coupling_edges=((0, 1), (1, 2)),
        calibration_id=known["calibration_id"],
        calibration_time_utc=known["calibration_time_utc"],
        max_shots=known["max_shots"],
        max_circuits_per_job=known["max_circuits_per_job"],
        header=header,
    )


def _bundle(target: TargetSnapshot, *, compiled_depth: int = 12) -> CompiledBundle:
    """Build a two-cell bundle compiled into the target's cz and prx basis."""
    cells = (
        CompiledCell("1" * 64, "2" * 64, "3" * 64, "4" * 64, 1000),
        CompiledCell("5" * 64, "6" * 64, "7" * 64, "8" * 64, 2000),
    )
    payload: dict[str, object] = {
        "cells": [cell.to_wire() for cell in cells],
        "layout": [2, 0, 1],
        "logical_gates": ["cx", "measure", "rz"],
        "compiled_gate_counts": [["cz", 4], ["measure", 3], ["prx", 10]],
        "substitutions": [["cx", ["cz", "prx"]], ["rz", ["prx"]]],
        "compiled_depth": compiled_depth,
        "provider_options": [["layout_method", "sabre"]],
    }
    identity: dict[str, Any] = {
        "target_snapshot_digest": _digest_of(target),
        "provider": target.provider,
        "device_name": target.device_name,
        "sdk_lock_digest": "b" * 64,
        "compile_log_digest": "c" * 64,
        "payload_digest": hashlib.sha256(canonical_bytes(payload)).hexdigest(),
    }
    content = {"schema": COMPILED_BUNDLE_SCHEMA, "object_kind": "compiled_bundle"}
    parents = (identity["target_snapshot_digest"], "b" * 64, "c" * 64)
    header = _header(
        "compiled_bundle", {**content, **identity, **payload}, parents, target.header.data_origin
    )
    return CompiledBundle(
        **identity,
        cells=cells,
        layout=(2, 0, 1),
        logical_gates=("cx", "measure", "rz"),
        compiled_gate_counts=(("cz", 4), ("measure", 3), ("prx", 10)),
        substitutions=(("cx", ("cz", "prx")), ("rz", ("prx",))),
        compiled_depth=compiled_depth,
        provider_options=(("layout_method", "sabre"),),
        header=header,
    )


def _known(value: int) -> QuotedCount:
    """Return a known count."""
    return QuotedCount("known", value, None)


def _quote(bundle: CompiledBundle) -> CostQuote:
    """Build the bundle's two-job quote."""
    counts: dict[str, Any] = {
        "job_count": _known(2),
        "overhead_executions": _known(0),
        "resource_units": _known(45),
        "price_minor_units": _known(4500),
    }
    fields: dict[str, Any] = {
        "compiled_bundle_digest": _digest_of(bundle),
        "bundle_payload_digest": bundle.payload_digest,
        "circuit_count": 2,
        "total_shots": 3000,
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
        "quoted_options": ["layout_method"],
        **{name: count.to_wire() for name, count in counts.items()},
    }
    header = _header(
        "cost_quote", content, (fields["compiled_bundle_digest"],), bundle.header.data_origin
    )
    return CostQuote(**fields, quoted_options=("layout_method",), **counts, header=header)


def _approval(
    quote: CostQuote, bundle: CompiledBundle, target: TargetSnapshot, **changes: Any
) -> OwnerApproval:
    """Approve the quote; receipts do not verify signatures, so the digest is a placeholder."""
    environment = changes.pop("environment", "test")
    scope: dict[str, Any] = {
        "approval_id": "approval-2026-11-16-001",
        "compiled_bundle_digest": _digest_of(bundle),
        "bundle_payload_digest": bundle.payload_digest,
        "cost_quote_digest": _digest_of(quote),
        "provider": target.provider,
        "account_route": target.account_route,
        "device_name": target.device_name,
        "max_shots": 3000,
        "max_jobs": 2,
        "max_resource_units": 45,
        "max_price_minor_units": 4500,
        "currency": "EUR",
        "valid_from_utc": "2026-11-16T00:00:00Z",
        "expires_at_utc": "2026-11-17T00:00:00Z",
        "approved_egress": "value_free",
        **changes,
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


def _job(
    ordinal: int, cells: tuple[int, ...], provider: str = "iqm", **changes: Any
) -> SubmittedJob:
    """Build one submitted job with the provider's real-form job ID."""
    fields: dict[str, Any] = {
        "ordinal": ordinal,
        "name": f"phase_{ordinal}",
        "payload_sha256": hashlib.sha256(f"request-{ordinal}".encode()).hexdigest(),
        "cell_ordinals": cells,
        "state": "submitted",
        "provider_job_id": _JOB_IDS[provider][ordinal % 2],
        "id_source": "submit_response",
        "boundary_at_utc": "2026-11-16T09:00:00Z",
        "provider_created_at_utc": "2026-11-16T09:00:01Z",
        "blocker": None,
        **changes,
    }
    return SubmittedJob(**fields)


def _ambiguous(ordinal: int, cells: tuple[int, ...]) -> SubmittedJob:
    """Build one job whose submit outcome is unknown."""
    return _job(
        ordinal,
        cells,
        state="recovery_required",
        provider_job_id=None,
        id_source=None,
        provider_created_at_utc=None,
        blocker=_BLOCKER,
    )


def _receipt(
    approval: OwnerApproval,
    bundle: CompiledBundle,
    jobs: tuple[SubmittedJob, ...] | None = None,
    *,
    origin: str | None = None,
    **changes: Any,
) -> ProviderBatchReceipt:
    """Build the receipt for one cell per job, hashing exactly its content."""
    provider = changes.get("provider", approval.provider)
    fields: dict[str, Any] = {
        "attempt_id": _ATTEMPT,
        "compiled_bundle_digest": _digest_of(bundle),
        "bundle_payload_digest": bundle.payload_digest,
        "owner_approval_digest": _digest_of(approval),
        "provider": provider,
        "account_route": approval.account_route,
        "device_name": approval.device_name,
        **changes,
    }
    jobs = jobs if jobs is not None else (_job(0, (0,), provider), _job(1, (1,), provider))
    content = {
        "schema": PROVIDER_BATCH_RECEIPT_SCHEMA,
        "object_kind": "provider_batch_receipt",
        **fields,
        "jobs": [job.to_wire() for job in jobs],
    }
    parents = (fields["compiled_bundle_digest"], fields["owner_approval_digest"])
    data_origin = origin or bundle.header.data_origin
    header = _header("provider_batch_receipt", content, parents, data_origin)
    return ProviderBatchReceipt(**fields, jobs=jobs, header=header)


def _chain(
    provider: str = "iqm", provenance: str = "synthetic_fixture", **approval_changes: Any
) -> tuple[TargetSnapshot, CompiledBundle, CostQuote, OwnerApproval]:
    """Return one consistent target, bundle, quote and approval."""
    target = _target(provider, provenance)
    bundle = _bundle(target)
    quote = _quote(bundle)
    return target, bundle, quote, _approval(quote, bundle, target, **approval_changes)


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


def _request(
    chain: tuple[TargetSnapshot, CompiledBundle, CostQuote, OwnerApproval], receipt: object
) -> dict[str, object]:
    """Return one worker receipt request."""
    target, bundle, quote, approval = chain
    return {
        "op": "roundtrip_provider_batch_receipt",
        "target": target.to_wire(),
        "bundle": bundle.to_wire(),
        "quote": quote.to_wire(),
        "approval": approval.to_wire(),
        "receipt": receipt,
    }


@pytest.mark.parametrize("provider", ["iqm", "ibm"])
def test_receipt_maps_every_cell_to_a_real_job_and_roundtrips(
    provider: str, tmp_path: Path
) -> None:
    """Submitted, recovered and ambiguous jobs validate and survive the worker round trip."""
    chain = _chain(provider)
    _, bundle, _, approval = chain
    recovered = _job(1, (1,), provider, id_source="read_only_recovery")
    unknown_creation = _job(1, (1,), provider, provider_created_at_utc=None)
    for jobs in (
        (_job(0, (0,), provider), _job(1, (1,), provider)),
        (_job(0, (0,), provider), recovered),
        (_job(0, (0,), provider), unknown_creation),
        (_job(0, (1, 0), provider),),
        (_job(0, (0,), provider), _ambiguous(1, (1,))),
    ):
        receipt = _receipt(approval, bundle, jobs)
        validate_provider_batch_receipt(receipt, approval, bundle)
        assert decode_provider_batch_receipt(canonical_bytes(receipt.to_wire())) == receipt
        response = _worker(_request(chain, receipt.to_wire()), tmp_path)
        assert response.returncode == 0, response.stdout.decode()
        payload = json.loads(response.stdout)
        assert payload["status"] == "validated_roundtrip_no_compute"
        assert payload["hardware_submission_enabled"] is False
        assert payload["receipt_sha256"] == _digest_of(receipt)
        assert decode_provider_batch_receipt(canonical_bytes(payload["receipt"])) == receipt
    assert receipt.blockers() == ((1, _BLOCKER),)
    assert _receipt(approval, bundle).blockers() == ()


def test_receipt_refuses_local_handles_and_reused_job_ids() -> None:
    """Only the provider's own job ID form counts; one ID backs one job."""
    _, bundle, _, approval = _chain("iqm")
    _, ibm_bundle, _, ibm_approval = _chain("ibm")
    upper = _JOB_IDS["iqm"][0].upper()
    for chosen, jobs, reason in (
        (
            (approval, bundle),
            (_job(0, (0,), provider_job_id=_HAL_HANDLE), _job(1, (1,))),
            "^an IQM job ID is a canonical UUID; a local handle is not a job ID$",
        ),
        (
            (approval, bundle),
            (_job(0, (0,), provider_job_id=upper), _job(1, (1,))),
            "^an IQM job ID is a canonical UUID",
        ),
        (
            (ibm_approval, ibm_bundle),
            (_job(0, (0,), "ibm", provider_job_id=_HAL_HANDLE), _job(1, (1,), "ibm")),
            "^an IBM job ID is 20 lowercase alphanumerics; a local handle is not one$",
        ),
        (
            (ibm_approval, ibm_bundle),
            (_job(0, (0,), "ibm", provider_job_id=_JOB_IDS["iqm"][0]), _job(1, (1,), "ibm")),
            "^an IBM job ID is 20 lowercase alphanumerics",
        ),
        (
            (approval, bundle),
            (_job(0, (0,), provider_job_id=_ATTEMPT), _job(1, (1,))),
            "^the local attempt ID is not a provider job ID$",
        ),
        (
            (approval, bundle),
            (_job(0, (0,)), _job(1, (1,), provider_job_id=_JOB_IDS["iqm"][0])),
            "^one provider job ID cannot back two jobs$",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            _receipt(chosen[0], chosen[1], jobs)


def test_submitted_job_refuses_missing_ids_posing_as_failure_and_bad_fields() -> None:
    """A job is submitted with its ID or ambiguous with a blocker, never failed at zero cost."""
    submitted: dict[str, Any] = {**_job(0, (0,)).to_wire(), "cell_ordinals": (0,)}
    malformed_jobs: tuple[tuple[dict[str, Any], str], ...] = (
        ({"state": "failed"}, "a missing job ID is ambiguous, never a failed zero-cost job$"),
        ({"provider_job_id": None}, "^a submitted job needs its provider job ID and its source$"),
        ({"id_source": "hal_handle"}, "^a submitted job needs its provider job ID and its"),
        ({"blocker": "stale"}, "^a submitted job cannot carry a blocker$"),
        ({"provider_created_at_utc": "2026-11-16"}, "^provider creation time must be an ISO"),
        ({"boundary_at_utc": None}, "^submit boundary time must be an ISO-8601 UTC second$"),
        ({"ordinal": -1}, "^job ordinal must be a bounded non-negative integer$"),
        ({"ordinal": True}, "^job ordinal must be a bounded non-negative integer$"),
        ({"ordinal": 100_000}, "^job ordinal must be a bounded non-negative integer$"),
        ({"name": "Phase A"}, "^invalid provider job phase name$"),
        ({"payload_sha256": "F" * 64}, "^submitted payload digest must be a lowercase SHA-256"),
        ({"cell_ordinals": ()}, "^a job maps a nonempty tuple of compiled cells$"),
        ({"cell_ordinals": [0]}, "^a job maps a nonempty tuple of compiled cells$"),
        ({"cell_ordinals": (0, 0)}, "^a job cannot map one cell twice$"),
        ({"cell_ordinals": (-1,)}, "^cell ordinals must be bounded non-negative integers$"),
        ({"cell_ordinals": (True,)}, "^cell ordinals must be bounded non-negative integers$"),
        ({"cell_ordinals": (10_000_000,)}, "^cell ordinals must be bounded non-negative"),
    )
    for changes, reason in malformed_jobs:
        with pytest.raises(ValueError, match=reason):
            SubmittedJob(**{**submitted, **changes})
    assert _job(99_999, (9_999_999,)).ordinal == 99_999
    ambiguous: dict[str, Any] = _ambiguous(0, (0,)).to_wire()
    ambiguous["cell_ordinals"] = (0,)
    for changes, reason in (
        ({"provider_job_id": _JOB_IDS["iqm"][0]}, "^an ambiguous job has no provider job ID yet$"),
        ({"id_source": "submit_response"}, "^an ambiguous job has no provider job ID yet$"),
        ({"provider_created_at_utc": "2026-11-16T09:00:01Z"}, "^an ambiguous job has no provider"),
        ({"blocker": None}, "^recovery blocker must be text$"),
        ({"blocker": ""}, "^recovery blocker must be bounded nonempty text$"),
    ):
        with pytest.raises(ValueError, match=reason):
            SubmittedJob(**{**ambiguous, **changes})


def test_receipt_refuses_malformed_identity_order_and_header() -> None:
    """Attempt IDs are canonical, jobs are ordered records, the header is exact."""
    _, bundle, _, approval = _chain()
    identities: tuple[tuple[dict[str, Any], str], ...] = (
        ({"attempt_id": _ATTEMPT.upper()}, "^attempt ID must be canonical UUID text$"),
        ({"attempt_id": "attempt-1"}, "^attempt ID must be canonical UUID text$"),
        ({"owner_approval_digest": "E" * 64}, "^owner_approval_digest must be a lowercase"),
        ({"provider": "aws"}, "^receipt provider must be ibm or iqm$"),
    )
    for changes, reason in identities:
        with pytest.raises(ValueError, match=reason):
            replace(_receipt(approval, bundle), **changes)
    valid = _receipt(approval, bundle)
    not_jobs: Any = [_job(0, (0,))]
    stray: Any = (_job(0, (0,)), "job")
    for jobs, reason in (
        ((), "^a receipt needs a nonempty tuple of jobs$"),
        (not_jobs, "^a receipt needs a nonempty tuple of jobs$"),
        (stray, "^receipt jobs must be SubmittedJob records$"),
        ((_job(1, (0,)), _job(0, (1,))), "^receipt jobs must be ordered by ordinal from zero$"),
        ((_job(1, (0, 1)),), "^receipt jobs must be ordered by ordinal from zero$"),
    ):
        with pytest.raises(ValueError, match=reason):
            replace(valid, jobs=jobs)
    with pytest.raises(ValueError, match="account_route must be bounded nonempty text"):
        replace(valid, account_route="")
    with pytest.raises(ValueError, match="^ProviderBatchReceipt requires its exact artifact"):
        replace(valid, header=replace(valid.header, object_kind="owner_approval"))
    with pytest.raises(ValueError, match="^a receipt derives from provider responses or a"):
        replace(valid, header=replace(valid.header, data_origin="owner_decision"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(valid, device_name="emerald")


def test_receipt_validation_refuses_another_bundle_approval_scope_or_lost_cells() -> None:
    """The receipt must name this bundle and approval and map each cell exactly once."""
    target, bundle, quote, approval = _chain()
    other = _bundle(target, compiled_depth=13)
    other_approval = _approval(quote, bundle, target, approval_id="approval-2026-11-16-002")
    stranger = _approval(_quote(other), other, target)
    single_job = _approval(quote, bundle, target, max_jobs=1)
    for checked, receipt, reason in (
        (approval, _receipt(approval, other), "^receipt names a different compiled bundle$"),
        (
            approval,
            _receipt(approval, bundle, bundle_payload_digest="e" * 64),
            "^receipt names a different compiled payload$",
        ),
        (
            approval,
            _receipt(other_approval, bundle),
            "^receipt names a different owner approval$",
        ),
        (stranger, _receipt(stranger, bundle), "^the approval covers a different compiled"),
        (
            approval,
            _receipt(approval, bundle, account_route="iqm-resonance/other"),
            "^receipt route or device differs from the approval$",
        ),
        (
            approval,
            _receipt(approval, bundle, device_name="emerald"),
            "^receipt route or device differs from the approval$",
        ),
        (single_job, _receipt(single_job, bundle), "^receipt has more provider jobs than"),
        (
            approval,
            _receipt(
                approval,
                bundle,
                (_job(0, (0,)), _job(1, (1,)), _job(2, (0,), provider_job_id=_THIRD_JOB_ID)),
            ),
            "^receipt has more provider jobs than approved$",
        ),
        (
            approval,
            _receipt(approval, bundle, (_job(0, (0,)), _job(1, (0,)))),
            "^receipt does not map every compiled cell exactly once$",
        ),
        (
            approval,
            _receipt(approval, bundle, (_job(0, (0,)),)),
            "^receipt does not map every compiled cell exactly once$",
        ),
        (
            approval,
            _receipt(approval, bundle, (_job(0, (0,)), _job(1, (1, 0)))),
            "^receipt does not map every compiled cell exactly once$",
        ),
        (
            approval,
            _receipt(approval, bundle, (_job(0, (0, 1, 2)),)),
            "^receipt does not map every compiled cell exactly once$",
        ),
        (
            approval,
            _receipt(approval, bundle, origin="provider_metadata"),
            "^receipt and bundle data origins differ$",
        ),
    ):
        with pytest.raises(ValueError, match=reason):
            validate_provider_batch_receipt(receipt, checked, bundle)
    validate_provider_batch_receipt(
        _receipt(single_job, bundle, (_job(0, (0, 1)),)), single_job, bundle
    )
    _, queried, _, test_key = _chain(provenance="provider_target_query")
    with pytest.raises(ValueError, match="^a test-key approval cannot back a provider receipt$"):
        validate_provider_batch_receipt(_receipt(test_key, queried), test_key, queried)
    _, queried, _, live_key = _chain(provenance="provider_target_query", environment="live")
    validate_provider_batch_receipt(_receipt(live_key, queried), live_key, queried)
    not_a_receipt: Any = "receipt"
    with pytest.raises(ValueError, match="^receipt validation needs a ProviderBatchReceipt$"):
        validate_provider_batch_receipt(not_a_receipt, approval, bundle)


_WORKER_FORGERIES: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
    (("unexpected",), True, "ProviderBatchReceipt fields mismatch"),
    (("schema",), "scpn.experimental.llm_qpu.provider_batch_receipt.v2", "unknown"),
    (("attempt_id",), "attempt-1", "attempt ID must be canonical UUID text"),
    (("owner_approval_digest",), "E" * 64, "invalid owner_approval_digest"),
    (("provider",), "aws", "receipt provider must be ibm or iqm"),
    (("account_route",), "", "account_route must be bounded normalized text"),
    (("jobs",), [], "a receipt needs a nonempty list of jobs"),
    (("jobs", 0, "unexpected"), True, "SubmittedJob fields mismatch"),
    (("jobs", 0, "ordinal"), 1, "ordered by ordinal from zero"),
    (("jobs", 0, "name"), "Phase A", "invalid provider job phase name"),
    (("jobs", 0, "payload_sha256"), "F" * 64, "invalid submitted payload digest"),
    (("jobs", 0, "cell_ordinals"), [], "a job maps a nonempty list of distinct cells"),
    (("jobs", 0, "cell_ordinals"), [0, 0], "a job maps a nonempty list of distinct cells"),
    (("jobs", 0, "cell_ordinals"), [-1], "cell ordinals must be bounded non-negative"),
    (("jobs", 0, "cell_ordinals"), [True], "cell ordinals must be bounded non-negative"),
    (("jobs", 0, "boundary_at_utc"), "2026-11-16", "submit boundary time must be an ISO"),
    (("jobs", 0, "provider_created_at_utc"), "x", "provider creation time must be an ISO"),
    (("jobs", 0, "state"), "failed", "never a failed zero-cost job"),
    (("jobs", 0, "provider_job_id"), None, "needs its provider job ID and its source"),
    (("jobs", 0, "id_source"), "hal_handle", "needs its provider job ID and its source"),
    (("jobs", 0, "blocker"), "stale", "a submitted job cannot carry a blocker"),
    (("jobs", 0, "provider_job_id"), _HAL_HANDLE, "does not have the provider's form"),
    (("jobs", 0, "provider_job_id"), _ATTEMPT, "repeats or reuses the attempt ID"),
    (("jobs", 1, "provider_job_id"), _JOB_IDS["iqm"][0], "repeats or reuses the attempt ID"),
    (("jobs", 1, "state"), "recovery_required", "has no provider job ID or creation time"),
    (("header", "data_origin"), "owner_decision", "derives from provider responses or a"),
    (("header", "content_digest"), "f" * 64, "artifact content digest mismatch"),
)


def _forge(wire: dict[str, Any], path: tuple[str | int, ...], value: object) -> dict[str, Any]:
    """Return the wire with one nested field replaced."""
    holder: Any = wire
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return wire


def test_receipt_wire_is_exact_here_and_in_the_worker(tmp_path: Path) -> None:
    """Edited wires are refused by the decoder and the worker; decode_contract stays closed."""
    chain = _chain()
    target, bundle, quote, approval = chain
    receipt = _receipt(approval, bundle)
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(receipt.to_wire()))
    decoder_cases: tuple[tuple[tuple[str | int, ...], object, str], ...] = (
        (("unexpected",), True, "^ProviderBatchReceipt fields mismatch$"),
        (("schema",), "scpn.experimental.llm_qpu.provider_batch_receipt.v2", "^unknown"),
        (("object_kind",), "owner_approval", "^unknown ProviderBatchReceipt schema$"),
        (("jobs",), {"first": "job"}, "^receipt jobs must be a list on wire$"),
        (("jobs", 0, "unexpected"), True, "^SubmittedJob fields mismatch$"),
        (("jobs", 0, "cell_ordinals"), "0", "^cell ordinals must be a list on wire$"),
    )
    for path, value, reason in decoder_cases:
        with pytest.raises(ValueError, match=reason):
            decode_provider_batch_receipt(canonical_bytes(_forge(receipt.to_wire(), path, value)))
    for path, forgery, refusal in _WORKER_FORGERIES:
        response = _worker(_request(chain, _forge(receipt.to_wire(), path, forgery)), tmp_path)
        assert response.returncode == 2, path
        assert refusal in json.loads(response.stdout)["reason"], path
    other = _bundle(target, compiled_depth=13)
    other_approval = _approval(quote, bundle, target, approval_id="approval-2026-11-16-002")
    third = _job(2, (0,), provider_job_id=_THIRD_JOB_ID)
    _, queried, _, test_key = queried_chain = _chain(provenance="provider_target_query")
    consistent: tuple[tuple[Any, ProviderBatchReceipt, str], ...] = (
        (chain, _receipt(approval, other), "receipt names a different compiled bundle"),
        (
            chain,
            _receipt(approval, bundle, bundle_payload_digest="e" * 64),
            "receipt names a different compiled payload",
        ),
        (chain, _receipt(other_approval, bundle), "receipt names a different owner approval"),
        (
            chain,
            _receipt(approval, bundle, account_route="iqm-resonance/other"),
            "route or device differs from the approval",
        ),
        (
            chain,
            _receipt(approval, bundle, (_job(0, (0,)), _job(1, (1,)), third)),
            "more provider jobs than approved",
        ),
        (
            chain,
            _receipt(approval, bundle, (_job(0, (0,)), _job(1, (0,)))),
            "does not map every compiled cell exactly once",
        ),
        (
            chain,
            _receipt(approval, bundle, (_job(0, (0, 1, 2)),)),
            "does not map every compiled cell exactly once",
        ),
        (
            chain,
            _receipt(approval, bundle, origin="provider_metadata"),
            "receipt and bundle data origins differ",
        ),
        (queried_chain, _receipt(test_key, queried), "test-key approval cannot back a provider"),
    )
    for used, edited, reason in consistent:
        response = _worker(_request(used, edited.to_wire()), tmp_path)
        assert response.returncode == 2, reason
        assert reason in json.loads(response.stdout)["reason"], reason
    live_chain = _chain(provenance="provider_target_query", environment="live")
    response = _worker(
        _request(live_chain, _receipt(live_chain[3], live_chain[1]).to_wire()), tmp_path
    )
    assert response.returncode == 0, response.stdout.decode()
    missing = _request(chain, receipt.to_wire())
    del missing["receipt"]
    response = _worker(missing, tmp_path)
    assert response.returncode == 2
    assert json.loads(response.stdout)["reason"] == "receipt request fields mismatch"
