# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental owner approval contract
"""Exercise signed owner approvals against their quote, bundle, target and trust store."""

from __future__ import annotations

import functools
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from scpn_quantum_control.crypto.ml_dsa import key_gen, sign, verify
from scpn_quantum_control.experimental.llm_qpu.contracts import (
    COMPILED_BUNDLE_SCHEMA,
    COST_QUOTE_SCHEMA,
    OWNER_APPROVAL_SCHEMA,
    TARGET_SNAPSHOT_SCHEMA,
    ApprovalSignature,
    ArtifactHeader,
    CompiledBundle,
    CompiledCell,
    CostQuote,
    OwnerApproval,
    QuotedCount,
    ReportedValue,
    TargetSnapshot,
    canonical_bytes,
    decode_contract,
    decode_owner_approval,
    validate_owner_approval,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORKER = _REPO_ROOT / "experimental_workers/llm_qpu/protocol/worker.py"
_BASE = "de259e4837a92ecb09b63c8f4332dbcf3d21021c"
_REVISION = "0883d1e5204ffe4594bb2f5b6e7b6a5d0a915209"
_CONTEXT = b"scpn.experimental.llm_qpu.owner_approval.v1"
_LIVE_SEED = bytes(range(32))
_TEST_SEED = bytes(range(1, 33))
_ROGUE_SEED = bytes(range(2, 34))
_UNSIGNED = bytes(3309)
_DETACHED: dict[str, bytes] = {}
_NOW = "2026-11-16T12:00:00Z"
_ORIGIN = {"live": "owner_decision", "test": "synthetic_classical"}


@dataclass(frozen=True)
class _Key:
    """A fixture ML-DSA-65 key pair and its trust-store id."""

    public_key: bytes
    secret_key: bytes

    @property
    def key_id(self) -> str:
        """Return the SHA-256 fingerprint of the public key."""
        return hashlib.sha256(self.public_key).hexdigest()


@functools.cache
def _key(seed: bytes) -> _Key:
    """Derive one deterministic fixture key; offline fixtures only, never a live owner key."""
    pair = key_gen(seed, suppress_research_warning=True)
    return _Key(pair.public_key, pair.secret_key)


@functools.cache
def _signature(seed: bytes, message: bytes) -> bytes:
    """Sign one approval message under the approval domain context."""
    return sign(_key(seed).secret_key, message, context=_CONTEXT, suppress_research_warning=True)


class _TrustStore:
    """Fixture trust store: key ids bound to one environment and one public key."""

    def __init__(self) -> None:
        """Trust the live and test fixture keys, each for its own environment only."""
        self._entries = {
            _key(seed).key_id: (environment, _key(seed).public_key)
            for seed, environment in ((_LIVE_SEED, "live"), (_TEST_SEED, "test"))
        }
        self.calls: list[bytes] = []

    def __call__(
        self,
        *,
        scheme: str,
        key_id: str,
        key_environment: str,
        message: bytes,
        signature: bytes,
    ) -> bool:
        """Verify with the trusted key, refusing unknown ids and relabelled environments."""
        self.calls.append(message)
        entry = self._entries.get(key_id)
        if scheme != "ml_dsa_65" or entry is None or entry[0] != key_environment:
            return False
        return verify(entry[1], message, signature, context=_CONTEXT)


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


def _target() -> TargetSnapshot:
    """Build a synthetic three-qubit line target with one circuit per job."""
    known = {
        "calibration_id": ReportedValue("known", "cal-2026-11-15-0800", None),
        "calibration_time_utc": ReportedValue("known", "2026-11-15T08:00:00Z", None),
        "max_shots": ReportedValue("known", 100_000, None),
        "max_circuits_per_job": ReportedValue("known", 1, None),
    }
    plain: dict[str, Any] = {
        "provider": "iqm",
        "account_route": "iqm-resonance/owner-project",
        "device_name": "garnet",
        "sdk_name": "iqm-client",
        "sdk_version": "35.0.3",
        "sdk_target_digest": "a" * 64,
        "capability_provenance": "synthetic_fixture",
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
    header = _header("target_snapshot", content, ("a" * 64,), "synthetic_classical")
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
    """Build a two-cell bundle that compiles cx and rz into the target's cz and prx basis."""
    cells = (
        CompiledCell("1" * 64, "2" * 64, "3" * 64, "4" * 64, 1000),
        CompiledCell("5" * 64, "6" * 64, "7" * 64, "8" * 64, 2000),
    )
    fields: dict[str, Any] = {
        "cells": cells,
        "layout": (2, 0, 1),
        "logical_gates": ("cx", "measure", "rz"),
        "compiled_gate_counts": (("cz", 4), ("measure", 3), ("prx", 10)),
        "substitutions": (("cx", ("cz", "prx")), ("rz", ("prx",))),
        "compiled_depth": compiled_depth,
        "provider_options": (("layout_method", "sabre"), ("optimization_level", 1)),
    }
    payload: dict[str, object] = {
        "cells": [cell.to_wire() for cell in cells],
        "layout": [2, 0, 1],
        "logical_gates": ["cx", "measure", "rz"],
        "compiled_gate_counts": [["cz", 4], ["measure", 3], ["prx", 10]],
        "substitutions": [["cx", ["cz", "prx"]], ["rz", ["prx"]]],
        "compiled_depth": compiled_depth,
        "provider_options": [["layout_method", "sabre"], ["optimization_level", 1]],
    }
    identity: dict[str, Any] = {
        "target_snapshot_digest": hashlib.sha256(canonical_bytes(target.to_wire())).hexdigest(),
        "provider": target.provider,
        "device_name": target.device_name,
        "sdk_lock_digest": "b" * 64,
        "compile_log_digest": "c" * 64,
        "payload_digest": hashlib.sha256(canonical_bytes(payload)).hexdigest(),
    }
    content = {
        "schema": COMPILED_BUNDLE_SCHEMA,
        "object_kind": "compiled_bundle",
        **identity,
        **payload,
    }
    parents = (identity["target_snapshot_digest"], "b" * 64, "c" * 64)
    header = _header("compiled_bundle", content, parents, "synthetic_classical")
    return CompiledBundle(**identity, **fields, header=header)


def _known(value: int) -> QuotedCount:
    """Return a known count."""
    return QuotedCount("known", value, None)


def _quote(bundle: CompiledBundle, **changes: Any) -> CostQuote:
    """Build the bundle's quote: two jobs, 3000 shots, 45 credits, 4500 euro cents."""
    fields: dict[str, Any] = {
        "compiled_bundle_digest": hashlib.sha256(canonical_bytes(bundle.to_wire())).hexdigest(),
        "bundle_payload_digest": bundle.payload_digest,
        "circuit_count": 2,
        "total_shots": 3000,
        "quoted_options": ("layout_method", "optimization_level"),
        "job_count": _known(2),
        "overhead_executions": _known(0),
        "resource_unit": "iqm_credit",
        "resource_derivation": "published_rate_card",
        "resource_units": _known(45),
        "price_minor_units": _known(4500),
        "currency": "EUR",
        "pricing_source": "IQM Resonance price list, read 2026-11-15",
        "priced_at_utc": "2026-11-15T09:00:00Z",
        **changes,
    }
    content = {
        "schema": COST_QUOTE_SCHEMA,
        "object_kind": "cost_quote",
        **{
            name: value.to_wire()
            if type(value) is QuotedCount
            else list(value)
            if name == "quoted_options"
            else value
            for name, value in fields.items()
        },
    }
    header = _header(
        "cost_quote", content, (fields["compiled_bundle_digest"],), "synthetic_classical"
    )
    return CostQuote(**fields, header=header)


def _approval(
    quote: CostQuote,
    bundle: CompiledBundle,
    *,
    seed: bytes = _TEST_SEED,
    environment: str = "test",
    signed: dict[str, Any] | None = None,
    signature: bytes | None = None,
    **changes: Any,
) -> OwnerApproval:
    """Approve the quote at exactly its totals for one day; ``signed`` forges the signed scope.

    The detached signature is kept by its digest; ``_detached`` returns it.
    """
    scope: dict[str, Any] = {
        "approval_id": "approval-2026-11-16-001",
        "compiled_bundle_digest": hashlib.sha256(canonical_bytes(bundle.to_wire())).hexdigest(),
        "bundle_payload_digest": bundle.payload_digest,
        "cost_quote_digest": hashlib.sha256(canonical_bytes(quote.to_wire())).hexdigest(),
        "provider": "iqm",
        "account_route": "iqm-resonance/owner-project",
        "device_name": "garnet",
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
    signer = {"scheme": "ml_dsa_65", "key_id": _key(seed).key_id, "key_environment": environment}
    if signature is None:
        message = canonical_bytes(
            {
                "schema": OWNER_APPROVAL_SCHEMA,
                "object_kind": "owner_approval",
                **scope,
                **(signed or {}),
                "signer": signer,
            }
        )
        signature = _signature(seed, message)
    digest = hashlib.sha256(signature).hexdigest()
    _DETACHED[digest] = signature
    envelope = ApprovalSignature(**signer, signature_sha256=digest)
    content = {
        "schema": OWNER_APPROVAL_SCHEMA,
        "object_kind": "owner_approval",
        **scope,
        "signature": envelope.to_wire(),
    }
    parents = (scope["compiled_bundle_digest"], scope["cost_quote_digest"])
    header = _header("owner_approval", content, parents, _ORIGIN[environment])
    return OwnerApproval(**scope, signature=envelope, header=header)


def _detached(approval: OwnerApproval) -> bytes:
    """Return the detached signature the approval's envelope names."""
    return _DETACHED[approval.signature.signature_sha256]


def _validate(
    approval: OwnerApproval,
    quote: CostQuote,
    bundle: CompiledBundle,
    target: TargetSnapshot,
    *,
    verifier: object,
    now: object = _NOW,
    runner: str = "offline_fixture",
    signature: object = None,
) -> None:
    """Validate with the approval's own detached signature unless another is given."""
    detached: Any = _detached(approval) if signature is None else signature
    clock: Any = now
    trust: Any = verifier
    validate_owner_approval(
        approval,
        quote,
        bundle,
        target,
        now_utc=clock,
        runner_environment=runner,
        signature=detached,
        verifier=trust,
    )


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
    target: TargetSnapshot, bundle: CompiledBundle, quote: CostQuote, approval: object
) -> dict[str, object]:
    """Return one worker approval request."""
    return {
        "op": "roundtrip_owner_approval",
        "target": target.to_wire(),
        "bundle": bundle.to_wire(),
        "quote": quote.to_wire(),
        "approval": approval,
    }


@pytest.mark.parametrize(
    ("seed", "environment", "runner"),
    [
        (_LIVE_SEED, "live", "live"),
        (_LIVE_SEED, "live", "offline_fixture"),
        (_TEST_SEED, "test", "offline_fixture"),
    ],
)
def test_owner_approval_verifies_and_roundtrips(
    seed: bytes, environment: str, runner: str, tmp_path: Path
) -> None:
    """A signed approval at the quote's exact totals verifies, inside its window bounds."""
    target = _target()
    bundle = _bundle(target)
    quote = _quote(bundle)
    approval = _approval(quote, bundle, seed=seed, environment=environment)
    store = _TrustStore()

    for now in ("2026-11-16T00:00:00Z", _NOW, "2026-11-16T23:59:59Z"):
        _validate(approval, quote, bundle, target, verifier=store, now=now, runner=runner)
    scope = {
        name: value
        for name, value in approval.to_wire().items()
        if name not in ("signature", "header")
    }
    signer = {"scheme": "ml_dsa_65", "key_id": _key(seed).key_id, "key_environment": environment}
    assert approval.signed_message() == canonical_bytes({**scope, "signer": signer})
    assert store.calls == [approval.signed_message()] * 3
    assert approval.header.data_origin == _ORIGIN[environment]
    assert decode_owner_approval(canonical_bytes(approval.to_wire())) == approval

    response = _worker(_request(target, bundle, quote, approval.to_wire()), tmp_path)

    assert response.returncode == 0, response.stdout.decode()
    payload = json.loads(response.stdout)
    assert payload["status"] == "validated_roundtrip_no_compute"
    assert payload["signature_verified"] is False
    assert payload["hardware_submission_enabled"] is False
    expected = hashlib.sha256(canonical_bytes(approval.to_wire())).hexdigest()
    assert payload["approval_sha256"] == expected
    assert decode_owner_approval(canonical_bytes(payload["approval"])) == approval


def test_owner_approval_refuses_string_authority_test_keys_live_and_bad_signatures() -> None:
    """Only a detached signature by a key trusted for its own environment approves."""
    target = _target()
    bundle = _bundle(target)
    quote = _quote(bundle)
    approval = _approval(quote, bundle)
    string_authority: Any = "approved"
    with pytest.raises(
        ValueError, match="^an approval needs a structured owner signature, not a string$"
    ):
        replace(approval, signature=string_authority)
    wire: dict[str, Any] = approval.to_wire()
    wire["signature"] = "approved"
    with pytest.raises(ValueError, match="^ApprovalSignature fields mismatch$"):
        decode_owner_approval(canonical_bytes(wire))
    store = _TrustStore()
    with pytest.raises(ValueError, match="^a test key cannot approve a live run$"):
        _validate(approval, quote, bundle, target, verifier=store, runner="live")
    with pytest.raises(ValueError, match="^unknown runner environment$"):
        _validate(approval, quote, bundle, target, verifier=store, runner="staging")
    for detached in ("approved", b"", bytes(8193)):
        with pytest.raises(
            ValueError, match="^a detached approval signature must be bounded bytes$"
        ):
            _validate(approval, quote, bundle, target, verifier=store, signature=detached)
    with pytest.raises(
        ValueError, match="^detached signature differs from the approved signature digest$"
    ):
        _validate(approval, quote, bundle, target, verifier=store, signature=_UNSIGNED)
    assert store.calls == []
    for forged in (
        _approval(quote, bundle, seed=_TEST_SEED, environment="live"),
        _approval(quote, bundle, seed=_ROGUE_SEED, environment="live"),
        _approval(
            quote,
            bundle,
            seed=_LIVE_SEED,
            environment="live",
            signed={"max_price_minor_units": 9000},
        ),
        _approval(quote, bundle, seed=_LIVE_SEED, environment="live", signature=_UNSIGNED),
    ):
        with pytest.raises(
            ValueError, match="^approval signature does not verify against the trust store$"
        ):
            _validate(forged, quote, bundle, target, verifier=store, runner="live")
    assert len(store.calls) == 4

    def truthy(**_: object) -> object:
        """Return a truthy value that is not ``True``."""
        return 1

    with pytest.raises(
        ValueError, match="^approval signature does not verify against the trust store$"
    ):
        _validate(approval, quote, bundle, target, verifier=truthy)


def test_owner_approval_refuses_outside_its_validity_window() -> None:
    """The window starts at valid_from and ends before expiry; the clock is a UTC second."""
    target = _target()
    bundle = _bundle(target)
    quote = _quote(bundle)
    approval = _approval(quote, bundle, signature=_UNSIGNED)
    store = _TrustStore()
    for now in ("2026-11-15T23:59:59Z", "2026-11-17T00:00:00Z"):
        with pytest.raises(ValueError, match="^approval is outside its validity window$"):
            _validate(approval, quote, bundle, target, verifier=store, now=now)
    malformed: tuple[object, ...] = ("2026-11-16 12:00:00", 1_763_294_400)
    for clock in malformed:
        with pytest.raises(
            ValueError, match="^approval check time must be an ISO-8601 UTC second$"
        ):
            _validate(approval, quote, bundle, target, verifier=store, now=clock)
    assert store.calls == []


def _digest_of(value: CompiledBundle | CostQuote) -> str:
    """Return the SHA-256 of one artifact wire."""
    return hashlib.sha256(canonical_bytes(value.to_wire())).hexdigest()


def _scope_cases(
    target: TargetSnapshot, bundle: CompiledBundle
) -> tuple[tuple[CostQuote, dict[str, Any], str], ...]:
    """Return quotes, approval edits and the refusal each must raise."""
    quote = _quote(bundle)
    other_quote = _quote(bundle, pricing_source="IQM Resonance price list, read 2026-11-16")
    unknown_price = QuotedCount("unknown", None, "no price list covers this device")
    unpriced = _quote(bundle, price_minor_units=unknown_price, priced_at_utc=None)
    unknown_overhead = QuotedCount("unknown", None, "calibration circuits are not reported")
    open_overhead = _quote(bundle, overhead_executions=unknown_overhead)
    other_bundle = _digest_of(_bundle(target, compiled_depth=13))
    return (
        (quote, {"compiled_bundle_digest": other_bundle}, "covers a different compiled bundle"),
        (quote, {"bundle_payload_digest": "e" * 64}, "covers a different compiled payload"),
        (quote, {"cost_quote_digest": _digest_of(other_quote)}, "covers a different cost quote"),
        (quote, {"account_route": "iqm-resonance/other"}, "scope differs from the target"),
        (quote, {"device_name": "emerald"}, "scope differs from the target"),
        (quote, {"provider": "ibm"}, "scope differs from the target"),
        (quote, {"max_shots": 2999}, "quoted shots above the approved cap"),
        (quote, {"max_jobs": 1}, "quoted job_count above the approved cap"),
        (quote, {"max_resource_units": 44}, "quoted resource_units above the approved cap"),
        (
            quote,
            {"max_price_minor_units": 4499},
            "quoted price_minor_units above the approved cap",
        ),
        (quote, {"currency": "USD"}, "quote and approval currencies differ"),
        (unpriced, {}, "cannot cover an unknown price_minor_units"),
        (open_overhead, {}, "cannot cover an unknown overhead_executions"),
    )


def test_owner_approval_refuses_another_payload_scope_cap_or_unknown() -> None:
    """Every scope and cap check refuses before the trust store is consulted."""
    target = _target()
    bundle = _bundle(target)
    store = _TrustStore()
    for priced, changes, reason in _scope_cases(target, bundle):
        approval = _approval(priced, bundle, signature=_UNSIGNED, **changes)
        with pytest.raises(ValueError, match=reason):
            _validate(approval, priced, bundle, target, verifier=store)
    unknown_price = QuotedCount("unknown", None, "no price list covers this device")
    unpriced = _quote(bundle, price_minor_units=unknown_price, priced_at_utc=None)
    with pytest.raises(
        ValueError,
        match="^an approval cannot cover an unknown price_minor_units: "
        "no price list covers this device$",
    ):
        _validate(_approval(unpriced, bundle), unpriced, bundle, target, verifier=store)
    quote = _quote(bundle)
    approval = _approval(quote, bundle, signature=_UNSIGNED)
    stranger = _quote(_bundle(target, compiled_depth=13))
    with pytest.raises(ValueError, match="^quote prices a different compiled bundle$"):
        _validate(approval, stranger, bundle, target, verifier=store)
    not_an_approval: Any = "approved"
    with pytest.raises(ValueError, match="^approval validation needs an OwnerApproval$"):
        _validate(
            not_an_approval,
            quote,
            bundle,
            target,
            verifier=store,
            runner="live",
            signature=_UNSIGNED,
        )
    assert store.calls == []


def test_owner_approval_refuses_malformed_fields_and_envelopes() -> None:
    """Construction refuses unbounded caps, open windows, unknown egress and bad envelopes."""
    target = _target()
    bundle = _bundle(target)
    quote = _quote(bundle)
    valid = _approval(quote, bundle, signature=_UNSIGNED)
    malformed: tuple[tuple[dict[str, Any], str], ...] = (
        ({"approval_id": ""}, "^approval_id must be bounded nonempty text$"),
        ({"cost_quote_digest": "E" * 64}, "^cost_quote_digest must be a lowercase SHA-256"),
        ({"max_jobs": 0}, "^max_jobs must be a bounded positive integer$"),
        ({"max_shots": True}, "^max_shots must be a bounded positive integer$"),
        ({"max_price_minor_units": 10**15 + 1}, "^max_price_minor_units must be a bounded"),
        ({"currency": "eur"}, "^currency must be an ISO 4217 code$"),
        ({"valid_from_utc": "2026-11-16"}, "^valid_from_utc must be an ISO-8601 UTC second$"),
        ({"expires_at_utc": None}, "^expires_at_utc must be an ISO-8601 UTC second$"),
        ({"expires_at_utc": "2026-11-16T00:00:00Z"}, "^an approval must expire after it"),
        ({"approved_egress": "public"}, "^unknown approved data-egress scope$"),
    )
    for changes, reason in malformed:
        with pytest.raises(ValueError, match=reason):
            replace(valid, **changes)
    assert _approval(quote, bundle, signature=_UNSIGNED, max_resource_units=10**15)
    for egress in ("private_derived", "reviewed_synthetic", "value_free"):
        assert _approval(quote, bundle, signature=_UNSIGNED, approved_egress=egress)
    key_id = _key(_TEST_SEED).key_id
    for scheme, key, environment, digest, reason in (
        ("ed25519", key_id, "test", "d" * 64, "^unsupported approval signature scheme$"),
        ("ml_dsa_65", "owner", "test", "d" * 64, "^approval key id must be a lowercase"),
        ("ml_dsa_65", key_id, "prod", "d" * 64, "^approval key environment must be live or"),
        ("ml_dsa_65", key_id, "test", "D" * 64, "^approval signature digest must be a lower"),
    ):
        with pytest.raises(ValueError, match=reason):
            ApprovalSignature(scheme, key, environment, digest)
    approval = _approval(quote, bundle, signature=_UNSIGNED)
    with pytest.raises(ValueError, match="^OwnerApproval requires its exact artifact header$"):
        replace(approval, header=replace(approval.header, object_kind="cost_quote"))
    with pytest.raises(
        ValueError, match="^artifact data origin does not match the signing key environment$"
    ):
        replace(approval, header=replace(approval.header, data_origin="owner_decision"))
    with pytest.raises(ValueError, match="digest mismatch"):
        replace(approval, max_jobs=3)


def _forge(wire: dict[str, Any], path: tuple[str, ...], value: object) -> dict[str, Any]:
    """Return the wire with one nested field replaced."""
    holder = wire
    for key in path[:-1]:
        holder = holder[key]
    holder[path[-1]] = value
    return wire


_WORKER_FORGERIES: tuple[tuple[tuple[str, ...], object, str], ...] = (
    (("unexpected",), True, "OwnerApproval fields mismatch"),
    (("schema",), "scpn.experimental.llm_qpu.owner_approval.v2", "unknown OwnerApproval schema"),
    (("approval_id",), "", "approval_id must be bounded normalized text"),
    (("cost_quote_digest",), "E" * 64, "invalid cost_quote_digest"),
    (("max_jobs",), 0, "max_jobs must be a bounded positive integer"),
    (("max_shots",), 10**15 + 1, "max_shots must be a bounded positive integer"),
    (("currency",), "eur", "currency must be an ISO 4217 code"),
    (("valid_from_utc",), "2026-11-16", "validity bounds must be ISO-8601 UTC seconds"),
    (("expires_at_utc",), "2026-11-15T00:00:00Z", "must expire after it becomes valid"),
    (("expires_at_utc",), "2026-11-16T00:00:00Z", "must expire after it becomes valid"),
    (("approved_egress",), "public", "unknown approved data-egress scope"),
    (("signature", "scheme"), "ed25519", "unsupported approval signature scheme"),
    (("signature", "key_id"), "owner", "invalid approval key id"),
    (("signature", "key_environment"), "prod", "approval key environment must be live or test"),
    (("signature", "signature_sha256"), "D" * 64, "invalid approval signature digest"),
    (("signature",), "approved", "ApprovalSignature fields mismatch"),
    (("header", "data_origin"), "owner_decision", "artifact data origin mismatch"),
    (("header", "content_digest"), "f" * 64, "artifact content digest mismatch"),
)


def test_owner_approval_wire_is_exact_here_and_in_the_worker(tmp_path: Path) -> None:
    """Edited wires are refused by the decoder and the worker; decode_contract stays closed."""
    target = _target()
    bundle = _bundle(target)
    quote = _quote(bundle)
    approval = _approval(quote, bundle, signature=_UNSIGNED)
    with pytest.raises(ValueError, match="^unsupported contract schema$"):
        decode_contract(canonical_bytes(approval.to_wire()))
    decoder_cases: tuple[tuple[str, object, str], ...] = (
        ("unexpected", True, "^OwnerApproval fields mismatch$"),
        ("schema", "scpn.experimental.llm_qpu.owner_approval.v2", "^unknown OwnerApproval"),
        ("object_kind", "cost_quote", "^unknown OwnerApproval schema$"),
    )
    for name, value, reason in decoder_cases:
        with pytest.raises(ValueError, match=reason):
            decode_owner_approval(canonical_bytes(_forge(approval.to_wire(), (name,), value)))
    for path, forgery, refusal in _WORKER_FORGERIES:
        forged = _forge(approval.to_wire(), path, forgery)
        response = _worker(_request(target, bundle, quote, forged), tmp_path)
        assert response.returncode == 2, path
        assert refusal in json.loads(response.stdout)["reason"], path
    for priced, changes, reason in _scope_cases(target, bundle):
        edited = _approval(priced, bundle, signature=_UNSIGNED, **changes)
        response = _worker(_request(target, bundle, priced, edited.to_wire()), tmp_path)
        assert response.returncode == 2, reason
        assert reason in json.loads(response.stdout)["reason"], reason
    missing = _request(target, bundle, quote, approval.to_wire())
    del missing["approval"]
    response = _worker(missing, tmp_path)
    assert response.returncode == 2
    assert json.loads(response.stdout)["reason"] == "approval request fields mismatch"
    stranger = _quote(_bundle(target, compiled_depth=13))
    response = _worker(_request(target, bundle, stranger, approval.to_wire()), tmp_path)
    assert response.returncode == 2
    assert "quote prices a different compiled bundle" in json.loads(response.stdout)["reason"]
