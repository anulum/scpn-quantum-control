# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — attested result packs (BL-48)
"""Strip-resistant attested result-pack digests and local envelopes (BL-48).

Bind digests to **inputs and claim axes**, not a self-asserted “validated” badge.
Verification returns an explicit status:

* ``VERIFIED`` — envelope structure intact and content digest matches
* ``STRIPPED`` — required claim axes or digest fields were removed
* ``FORGED`` — digest present but does not match recomputed content
* ``UNGRADED`` — missing signature/keys when required, or insufficient signal

Unsigned local envelopes are first-class. Absent keys never become silent-validated.
This module does not invent PKI, HSM, or green “attested hardware” without digests.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

AttestationStatus = Literal["VERIFIED", "STRIPPED", "FORGED", "UNGRADED"]
"""Fail-closed attestation outcome labels."""

ATTESTED_RESULT_PACK_SCHEMA: Final[str] = "attested_result_pack.v1"
"""JSON schema identifier for serialised envelopes."""

ATTESTED_RESULT_PACK_CLAIM_BOUNDARY: Final[str] = (
    "attested result-pack digest contract only; digests bind inputs and claim "
    "axes, never a self-asserted validated badge; absent keys yield UNGRADED, "
    "never silent-validated hardware or marketing attestation claims"
)
"""Shared claim boundary for envelopes and verification results."""

_REQUIRED_AXIS_KEYS: Final[tuple[str, ...]] = (
    "claim_unit",
    "honesty_axes",
    "backend_class",
)


@dataclass(frozen=True, slots=True)
class AttestedEnvelope:
    """Local unsigned (or optionally tagged) attested result-pack envelope.

    Attributes
    ----------
    claim_id
        Stable claim unit identifier.
    claim_axes
        Honesty / claim axes mapping (must include required keys when verified).
    content
        Structured content being attested (inputs + numeric axes).
    content_digest
        Canonical SHA-256 hex digest of ``content`` under claim_axes binding.
    schema
        Envelope schema identifier.
    signature
        Optional signature token; empty means unsigned.
    claim_boundary
        Non-promotional claim boundary.

    """

    claim_id: str
    claim_axes: Mapping[str, object]
    content: Mapping[str, object]
    content_digest: str
    schema: str = ATTESTED_RESULT_PACK_SCHEMA
    signature: str = ""
    claim_boundary: str = ATTESTED_RESULT_PACK_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate envelope invariants."""
        if not self.claim_id or not self.claim_id.strip():
            raise ValueError("claim_id must be non-empty")
        if not isinstance(self.claim_axes, Mapping):
            raise TypeError("claim_axes must be a mapping")
        if not isinstance(self.content, Mapping):
            raise TypeError("content must be a mapping")
        digest = self.content_digest.strip().lower()
        if digest and (len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest)):
            raise ValueError("content_digest must be empty or 64-char lowercase hex")
        object.__setattr__(self, "content_digest", digest)
        object.__setattr__(self, "claim_id", self.claim_id.strip())

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this envelope."""
        return {
            "schema": self.schema,
            "claim_id": self.claim_id,
            "claim_axes": dict(self.claim_axes),
            "content": dict(self.content),
            "content_digest": self.content_digest,
            "signature": self.signature,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class AttestationVerdict:
    """Result of verifying an attested envelope.

    Attributes
    ----------
    status
        One of VERIFIED | STRIPPED | FORGED | UNGRADED.
    reason
        Human-readable decision reason.
    expected_digest
        Digest recomputed from content (may be empty if stripped).
    observed_digest
        Digest carried by the envelope.
    claim_id
        Claim unit under verification.

    """

    status: AttestationStatus
    reason: str
    expected_digest: str
    observed_digest: str
    claim_id: str
    claim_boundary: str = ATTESTED_RESULT_PACK_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate verdict invariants."""
        if self.status not in {"VERIFIED", "STRIPPED", "FORGED", "UNGRADED"}:
            raise ValueError(f"unknown attestation status: {self.status!r}")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if not self.claim_id or not self.claim_id.strip():
            raise ValueError("claim_id must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this verdict."""
        return {
            "status": self.status,
            "reason": self.reason,
            "expected_digest": self.expected_digest,
            "observed_digest": self.observed_digest,
            "claim_id": self.claim_id,
            "claim_boundary": self.claim_boundary,
        }


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    """Serialise a mapping with sorted keys for stable digests."""
    try:
        text = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"content is not JSON-canonicalisable: {exc}") from exc
    return text.encode("utf-8")


def canonical_content_digest(
    content: Mapping[str, object],
    claim_axes: Mapping[str, object],
) -> str:
    """Compute the canonical SHA-256 digest over content bound to claim axes.

    The digest covers both ``claim_axes`` and ``content`` so badges cannot be
    reattached to unrelated honesty axes without failing verification.

    Parameters
    ----------
    content
        Structured content mapping.
    claim_axes
        Claim / honesty axes mapping.

    Returns
    -------
    str
        Lowercase 64-character SHA-256 hex digest.

    Raises
    ------
    TypeError
        If either argument is not a mapping.
    ValueError
        If values are not JSON-serialisable under the canonical policy.

    """
    if not isinstance(content, Mapping):
        raise TypeError("content must be a mapping")
    if not isinstance(claim_axes, Mapping):
        raise TypeError("claim_axes must be a mapping")
    bound: dict[str, object] = {
        "claim_axes": dict(claim_axes),
        "content": dict(content),
    }
    return hashlib.sha256(_canonical_json_bytes(bound)).hexdigest()


def build_unsigned_envelope(
    *,
    claim_id: str,
    claim_axes: Mapping[str, object],
    content: Mapping[str, object],
) -> AttestedEnvelope:
    """Build a local unsigned envelope with a bound content digest.

    Parameters
    ----------
    claim_id
        Claim unit identifier.
    claim_axes
        Honesty axes; should include required keys for later VERIFIED status.
    content
        Structured inputs / numeric axes.

    Returns
    -------
    AttestedEnvelope
        Unsigned envelope with computed ``content_digest``.

    """
    if not claim_id or not str(claim_id).strip():
        raise ValueError("claim_id must be a non-empty string")
    digest = canonical_content_digest(content, claim_axes)
    return AttestedEnvelope(
        claim_id=str(claim_id).strip(),
        claim_axes=dict(claim_axes),
        content=dict(content),
        content_digest=digest,
        signature="",
    )


def _missing_required_axes(claim_axes: Mapping[str, object]) -> tuple[str, ...]:
    """Return required axis keys that are missing or empty."""
    missing: list[str] = []
    for key in _REQUIRED_AXIS_KEYS:
        if key not in claim_axes:
            missing.append(key)
            continue
        value = claim_axes[key]
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(key)
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 0:
            missing.append(key)
    return tuple(missing)


def verify_attested_envelope(
    envelope: AttestedEnvelope | Mapping[str, object],
    *,
    require_signature: bool = False,
) -> AttestationVerdict:
    """Verify an attested envelope with explicit fail-closed statuses.

    Parameters
    ----------
    envelope
        :class:`AttestedEnvelope` or mapping with the same fields.
    require_signature
        When True, missing/empty signature yields ``UNGRADED`` (never silent green).

    Returns
    -------
    AttestationVerdict
        Status and reason for the verification decision.

    Raises
    ------
    TypeError
        If ``envelope`` is neither an :class:`AttestedEnvelope` nor a mapping.

    """
    if isinstance(envelope, AttestedEnvelope):
        env = envelope
    elif isinstance(envelope, Mapping):
        raw_axes = envelope.get("claim_axes", {})
        raw_content = envelope.get("content", {})
        if not isinstance(raw_axes, Mapping) or not isinstance(raw_content, Mapping):
            raise TypeError("claim_axes and content must be mappings when present")
        env = AttestedEnvelope(
            claim_id=str(envelope.get("claim_id", "")),
            claim_axes={str(k): v for k, v in raw_axes.items()},
            content={str(k): v for k, v in raw_content.items()},
            content_digest=str(envelope.get("content_digest", "")),
            schema=str(envelope.get("schema", ATTESTED_RESULT_PACK_SCHEMA)),
            signature=str(envelope.get("signature", "")),
            claim_boundary=str(
                envelope.get("claim_boundary", ATTESTED_RESULT_PACK_CLAIM_BOUNDARY)
            ),
        )
    else:
        raise TypeError(f"envelope must be AttestedEnvelope or mapping (got {type(envelope)!r})")

    claim_id = env.claim_id
    observed = env.content_digest
    missing = _missing_required_axes(env.claim_axes)
    if missing:
        return AttestationVerdict(
            status="STRIPPED",
            reason=(
                "required claim axes missing or empty: "
                + ", ".join(missing)
                + "; refuse silent-validated badge"
            ),
            expected_digest="",
            observed_digest=observed,
            claim_id=claim_id,
        )

    if not observed:
        return AttestationVerdict(
            status="UNGRADED",
            reason=("content_digest absent; refuse invent-green attestation without digest"),
            expected_digest="",
            observed_digest="",
            claim_id=claim_id,
        )

    try:
        expected = canonical_content_digest(env.content, env.claim_axes)
    except (TypeError, ValueError) as exc:
        return AttestationVerdict(
            status="UNGRADED",
            reason=f"content not digestable: {exc}",
            expected_digest="",
            observed_digest=observed,
            claim_id=claim_id,
        )

    if expected != observed:
        return AttestationVerdict(
            status="FORGED",
            reason="content_digest does not match recomputed bound digest",
            expected_digest=expected,
            observed_digest=observed,
            claim_id=claim_id,
        )

    if require_signature and not (env.signature and env.signature.strip()):
        return AttestationVerdict(
            status="UNGRADED",
            reason=(
                "signature required but missing; absent keys yield UNGRADED, "
                "never silent-validated"
            ),
            expected_digest=expected,
            observed_digest=observed,
            claim_id=claim_id,
        )

    # Unsigned path is first-class: structural integrity + digest match.
    if env.signature and env.signature.strip():
        # Signature presence is recorded but not cryptographically verified in
        # this pure surface (optional signed path remains open / evidence-gated).
        return AttestationVerdict(
            status="UNGRADED",
            reason=(
                "signature field present but pure local verifier does not validate "
                "cryptographic signatures without a configured keyring; UNGRADED"
            ),
            expected_digest=expected,
            observed_digest=observed,
            claim_id=claim_id,
        )

    return AttestationVerdict(
        status="VERIFIED",
        reason="unsigned envelope intact: claim axes present and content digest matches",
        expected_digest=expected,
        observed_digest=observed,
        claim_id=claim_id,
    )


def refuse_invent_green_hardware_attestation(
    *,
    claim_id: str,
    has_content_digest: bool,
) -> AttestationVerdict:
    """Return a fail-closed verdict for invent-green hardware attestation claims.

    Parameters
    ----------
    claim_id
        Claim unit identifier.
    has_content_digest
        Whether a real content digest is present.

    Returns
    -------
    AttestationVerdict
        ``UNGRADED`` when digest is missing; otherwise a reason that hardware
        attestation still requires a verified envelope (never invent-green).

    """
    if not claim_id or not str(claim_id).strip():
        raise ValueError("claim_id must be a non-empty string")
    safe_id = str(claim_id).strip()
    if not has_content_digest:
        return AttestationVerdict(
            status="UNGRADED",
            reason=(
                "hardware attestation claim without content digest is UNGRADED; "
                "refuse invent-green"
            ),
            expected_digest="",
            observed_digest="",
            claim_id=safe_id,
        )
    return AttestationVerdict(
        status="UNGRADED",
        reason=(
            "hardware attestation requires verified envelope + owner-ticket evidence; "
            "digest alone is not invent-green hardware proof"
        ),
        expected_digest="",
        observed_digest="",
        claim_id=safe_id,
    )


def build_attestation_report(
    verdicts: Sequence[AttestationVerdict],
) -> dict[str, object]:
    """Aggregate verification verdicts into a small report payload.

    Parameters
    ----------
    verdicts
        Sequence of verification outcomes.

    Returns
    -------
    dict[str, object]
        Counts by status plus individual verdict mappings.

    """
    if not isinstance(verdicts, (list, tuple)):
        raise TypeError("verdicts must be a list or tuple of AttestationVerdict")
    counts: dict[str, int] = {
        "VERIFIED": 0,
        "STRIPPED": 0,
        "FORGED": 0,
        "UNGRADED": 0,
    }
    rows: list[dict[str, object]] = []
    for item in verdicts:
        if not isinstance(item, AttestationVerdict):
            raise TypeError(f"each verdict must be AttestationVerdict (got {type(item)!r})")
        counts[item.status] += 1
        rows.append(item.to_dict())
    return {
        "schema": ATTESTED_RESULT_PACK_SCHEMA,
        "claim_boundary": ATTESTED_RESULT_PACK_CLAIM_BOUNDARY,
        "verdict_count": len(rows),
        "status_counts": counts,
        "verdicts": rows,
    }


# Convenience builders for tests / docs with known claim axes.
def default_claim_axes(
    *,
    claim_unit: str = "local_result_pack",
    honesty_axes: Sequence[str] = ("inputs_digest", "backend_class"),
    backend_class: str = "simulator_local",
) -> dict[str, object]:
    """Return a minimal complete claim-axes mapping for local packs.

    Parameters
    ----------
    claim_unit
        Claim unit label.
    honesty_axes
        Honesty axis labels.
    backend_class
        Backend class label (must not invent QPU without evidence).

    Returns
    -------
    dict[str, object]
        Mapping suitable for :func:`build_unsigned_envelope`.

    """
    if not claim_unit or not str(claim_unit).strip():
        raise ValueError("claim_unit must be non-empty")
    if not backend_class or not str(backend_class).strip():
        raise ValueError("backend_class must be non-empty")
    axes = tuple(str(item).strip() for item in honesty_axes if str(item).strip())
    if not axes:
        raise ValueError("honesty_axes must contain at least one non-empty label")
    return {
        "claim_unit": str(claim_unit).strip(),
        "honesty_axes": list(axes),
        "backend_class": str(backend_class).strip(),
    }


def _json_ready(value: object) -> object:
    """Recursively convert mappings to plain dicts for typing helpers."""
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_ready(item) for item in value]
    return value


def envelope_from_mapping(payload: Mapping[str, Any]) -> AttestedEnvelope:
    """Construct an :class:`AttestedEnvelope` from a plain mapping.

    Parameters
    ----------
    payload
        Mapping with envelope fields.

    Returns
    -------
    AttestedEnvelope
        Validated envelope instance.

    """
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    axes = payload.get("claim_axes", {})
    content = payload.get("content", {})
    if not isinstance(axes, Mapping) or not isinstance(content, Mapping):
        raise TypeError("claim_axes and content must be mappings")
    return AttestedEnvelope(
        claim_id=str(payload.get("claim_id", "")),
        claim_axes={str(k): _json_ready(v) for k, v in axes.items()},
        content={str(k): _json_ready(v) for k, v in content.items()},
        content_digest=str(payload.get("content_digest", "")),
        schema=str(payload.get("schema", ATTESTED_RESULT_PACK_SCHEMA)),
        signature=str(payload.get("signature", "")),
        claim_boundary=str(payload.get("claim_boundary", ATTESTED_RESULT_PACK_CLAIM_BOUNDARY)),
    )


__all__ = [
    "ATTESTED_RESULT_PACK_CLAIM_BOUNDARY",
    "ATTESTED_RESULT_PACK_SCHEMA",
    "AttestationStatus",
    "AttestationVerdict",
    "AttestedEnvelope",
    "build_attestation_report",
    "build_unsigned_envelope",
    "canonical_content_digest",
    "default_claim_axes",
    "envelope_from_mapping",
    "refuse_invent_green_hardware_attestation",
    "verify_attested_envelope",
]
