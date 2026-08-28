# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — hermetic external reproduction kit
"""Hermetic external reproduction kit contract.

A skeptic-facing, versioned command-manifest catalogue with pure digest
verification. The **core** kit is local-first and must not require cloud GPUs or
QPUs. Digests attached to fixture payloads are derived with SHA-256 from the
payload bytes declared in this module (not invented).

This surface does not execute subprocesses, submit hardware jobs, or invent green
hardware-reproduction claims.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

KitEntryKind = Literal[
    "core_local",
    "optional_extra",
    "hardware_gated",
    "refuse_invent_green",
]
"""Classification of one kit catalogue entry."""

HERMETIC_REPRODUCTION_KIT_SCHEMA: Final[str] = "hermetic_reproduction_kit.v1"
"""JSON schema identifier for serialised kit payloads."""

HERMETIC_REPRODUCTION_CLAIM_BOUNDARY: Final[str] = (
    "hermetic reproduction kit contract only; core_local entries are offline/"
    "local-first command metadata with digests derived from declared fixture "
    "payloads; hardware_gated and refuse_invent_green rows never invent green "
    "QPU or cloud-GPU reproduction claims"
)
"""Shared claim boundary for kit rows and digest results."""

_SHA256_HEX_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

# Fixture payloads whose digests are computed at import time (verified-at-source).
_FIXTURE_TRANSFORM_SMOKE: Final[bytes] = b"scpn hermetic kit fixture: transform algebra smoke v1\n"
_FIXTURE_METAMORPHIC_LINEARITY: Final[bytes] = (
    b"scpn hermetic kit fixture: metamorphic linearity residual v1\n"
)
_FIXTURE_NO_ADVANTAGE_CERT: Final[bytes] = (
    b"scpn hermetic kit fixture: no-advantage certificate posture v1\n"
)

_DIGEST_TRANSFORM_SMOKE: Final[str] = hashlib.sha256(_FIXTURE_TRANSFORM_SMOKE).hexdigest()
_DIGEST_METAMORPHIC_LINEARITY: Final[str] = hashlib.sha256(
    _FIXTURE_METAMORPHIC_LINEARITY
).hexdigest()
_DIGEST_NO_ADVANTAGE_CERT: Final[str] = hashlib.sha256(_FIXTURE_NO_ADVANTAGE_CERT).hexdigest()


@dataclass(frozen=True, slots=True)
class KitDigestSpec:
    """One digestable artefact attached to a kit entry.

    Attributes
    ----------
    label
        Artefact label (fixture or relative path key).
    sha256_hex
        Lowercase 64-char SHA-256 hex digest of the declared payload/content.
    fixture_payload
        Optional exact payload bytes used to derive the digest for pure checks.

    """

    label: str
    sha256_hex: str
    fixture_payload: bytes | None = None

    def __post_init__(self) -> None:
        """Validate digest-spec invariants."""
        if not self.label or not self.label.strip():
            raise ValueError("digest label must be non-empty")
        digest = self.sha256_hex.strip().lower()
        if not _SHA256_HEX_RE.fullmatch(digest):
            raise ValueError("sha256_hex must be a 64-character lowercase hex SHA-256 digest")
        object.__setattr__(self, "sha256_hex", digest)
        if self.fixture_payload is not None:
            computed = hashlib.sha256(self.fixture_payload).hexdigest()
            if computed != digest:
                raise ValueError(f"fixture_payload digest mismatch for label={self.label!r}")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping (payload bytes omitted)."""
        return {
            "label": self.label,
            "sha256_hex": self.sha256_hex,
            "has_fixture_payload": self.fixture_payload is not None,
        }


@dataclass(frozen=True, slots=True)
class HermeticKitEntry:
    """One hermetic reproduction kit command-manifest entry.

    Attributes
    ----------
    entry_id
        Stable taxonomy key.
    kind
        Kit entry class.
    summary
        Short description.
    argv
        Command argv tokens (documentation/contract only; not executed here).
    cwd
        Working directory relative to repo root.
    requires_qpu
        Whether the entry requires a QPU (core kit must be False).
    digests
        Digest specs for skeptic verification.
    reason
        Required for non-core refuse/gated rows; empty for ``core_local``.
    claim_boundary
        Non-promotional claim boundary.

    """

    entry_id: str
    kind: KitEntryKind
    summary: str
    argv: tuple[str, ...]
    cwd: str
    requires_qpu: bool
    digests: tuple[KitDigestSpec, ...]
    reason: str = ""
    claim_boundary: str = HERMETIC_REPRODUCTION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate kit-entry invariants."""
        if not self.entry_id or not self.entry_id.strip():
            raise ValueError("entry_id must be non-empty")
        if self.kind not in {
            "core_local",
            "optional_extra",
            "hardware_gated",
            "refuse_invent_green",
        }:
            raise ValueError(f"unknown kit entry kind: {self.kind!r}")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if not self.argv or any(not token or not str(token).strip() for token in self.argv):
            raise ValueError("argv must be a non-empty sequence of non-empty tokens")
        if not self.cwd or not self.cwd.strip():
            raise ValueError("cwd must be non-empty")
        if self.kind == "core_local" and self.requires_qpu:
            raise ValueError("core_local kit entries must not require QPU")
        if self.kind == "core_local":
            if self.reason:
                raise ValueError("core_local entries must not carry a non-empty reason")
            if not self.digests:
                raise ValueError("core_local entries require at least one digest spec")
        elif not self.reason or not self.reason.strip():
            raise ValueError(
                f"non-core kit entries require a non-empty reason (entry_id={self.entry_id!r})"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this kit entry."""
        return {
            "entry_id": self.entry_id,
            "kind": self.kind,
            "summary": self.summary,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "requires_qpu": self.requires_qpu,
            "digests": [item.to_dict() for item in self.digests],
            "reason": self.reason,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class DigestCheckResult:
    """Result of a pure digest verification.

    Attributes
    ----------
    label
        Artefact label checked.
    matched
        Whether digests matched.
    expected_sha256_hex
        Expected digest (empty when blank-fail).
    actual_sha256_hex
        Computed digest of provided content (empty if content omitted).
    message
        Operator-facing decision message.
    refused
        True when blank/mismatch/invent-green paths refuse green.

    """

    label: str
    matched: bool
    expected_sha256_hex: str
    actual_sha256_hex: str
    message: str
    refused: bool = False
    claim_boundary: str = HERMETIC_REPRODUCTION_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate digest-check invariants."""
        if not self.label or not self.label.strip():
            raise ValueError("label must be non-empty")
        if not self.message or not self.message.strip():
            raise ValueError("message must be non-empty")
        if self.matched and self.refused:
            raise ValueError("a refused digest check cannot be marked matched")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this digest check."""
        return {
            "label": self.label,
            "matched": self.matched,
            "expected_sha256_hex": self.expected_sha256_hex,
            "actual_sha256_hex": self.actual_sha256_hex,
            "message": self.message,
            "refused": self.refused,
            "claim_boundary": self.claim_boundary,
        }


def _entry(
    entry_id: str,
    kind: KitEntryKind,
    summary: str,
    *,
    argv: Sequence[str],
    cwd: str,
    requires_qpu: bool,
    digests: Sequence[KitDigestSpec] = (),
    reason: str = "",
) -> HermeticKitEntry:
    """Build one validated kit catalogue row."""
    return HermeticKitEntry(
        entry_id=entry_id,
        kind=kind,
        summary=summary,
        argv=tuple(argv),
        cwd=cwd,
        requires_qpu=requires_qpu,
        digests=tuple(digests),
        reason=reason,
    )


_CANONICAL_ENTRIES: Final[tuple[HermeticKitEntry, ...]] = (
    _entry(
        "kit:core.transform_algebra_smoke",
        "core_local",
        "Local transform-algebra smoke fixture for skeptic digest verification.",
        argv=(
            "python",
            "-m",
            "pytest",
            "tests/test_metamorphic_ad_verification.py",
            "-q",
        ),
        cwd=".",
        requires_qpu=False,
        digests=(
            KitDigestSpec(
                label="fixture:transform_algebra_smoke",
                sha256_hex=_DIGEST_TRANSFORM_SMOKE,
                fixture_payload=_FIXTURE_TRANSFORM_SMOKE,
            ),
        ),
    ),
    _entry(
        "kit:core.metamorphic_linearity",
        "core_local",
        "Local metamorphic linearity residual fixture (pure check).",
        argv=(
            "python",
            "-c",
            "from scpn_quantum_control.metamorphic_ad_verification import "
            "evaluate_linearity_residual; "
            "assert evaluate_linearity_residual(1.0,2.0,3.0).passed",
        ),
        cwd=".",
        requires_qpu=False,
        digests=(
            KitDigestSpec(
                label="fixture:metamorphic_linearity",
                sha256_hex=_DIGEST_METAMORPHIC_LINEARITY,
                fixture_payload=_FIXTURE_METAMORPHIC_LINEARITY,
            ),
        ),
    ),
    _entry(
        "kit:core.no_advantage_certificate",
        "core_local",
        "Default no-advantage certificate posture fixture (BL-65).",
        argv=(
            "python",
            "-c",
            "from scpn_quantum_control.advantage_language_protocol import "
            "issue_no_advantage_certificate; "
            "c=issue_no_advantage_certificate(context='hermetic'); "
            "assert c.language_status=='no_advantage_default'",
        ),
        cwd=".",
        requires_qpu=False,
        digests=(
            KitDigestSpec(
                label="fixture:no_advantage_certificate",
                sha256_hex=_DIGEST_NO_ADVANTAGE_CERT,
                fixture_payload=_FIXTURE_NO_ADVANTAGE_CERT,
            ),
        ),
    ),
    _entry(
        "kit:optional.framework_parity_extra",
        "optional_extra",
        "Optional framework-parity extras (JAX/Torch/TF) — not required for core kit.",
        argv=("python", "-m", "pytest", "tests/test_phase_qnode_framework_parity.py", "-q"),
        cwd=".",
        requires_qpu=False,
        digests=(),
        reason=(
            "Optional framework extras may require paid/local extras; core kit remains "
            "valid without them"
        ),
    ),
    _entry(
        "kit:hardware.qpu_live_reproduction",
        "hardware_gated",
        "Live QPU reproduction is owner-ticket gated and not part of the core kit.",
        argv=("echo", "qpu-reproduction-requires-owner-ticket"),
        cwd=".",
        requires_qpu=True,
        digests=(),
        reason=(
            "Live QPU reproduction requires owner tickets and raw-count evidence chains; "
            "core kit must run without QPU"
        ),
    ),
    _entry(
        "kit:refuse.invent_green_qpu_digest",
        "refuse_invent_green",
        "Refuse invent-green hardware/QPU digests without verified artefacts.",
        argv=("false",),
        cwd=".",
        requires_qpu=True,
        digests=(),
        reason=(
            "Inventing green QPU or cloud-GPU reproduction digests is forbidden; "
            "blank or unverified digests fail closed"
        ),
    ),
)


def _catalogue_map() -> dict[str, HermeticKitEntry]:
    """Return the entry_id → record map for the canonical catalogue."""
    mapping = {row.entry_id: row for row in _CANONICAL_ENTRIES}
    if len(mapping) != len(_CANONICAL_ENTRIES):
        raise RuntimeError("duplicate entry_id in hermetic reproduction kit catalogue")
    return mapping


_ENTRY_BY_ID: Final[Mapping[str, HermeticKitEntry]] = _catalogue_map()


def list_hermetic_kit_entry_ids() -> tuple[str, ...]:
    """Return all canonical kit entry identifiers in stable catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered entry identifiers.

    """
    return tuple(row.entry_id for row in _CANONICAL_ENTRIES)


def get_hermetic_kit_entry(entry_id: str) -> HermeticKitEntry:
    """Return one catalogue row or raise for unknown identifiers.

    Parameters
    ----------
    entry_id
        Kit taxonomy key.

    Returns
    -------
    HermeticKitEntry
        Matching catalogue row.

    Raises
    ------
    ValueError
        If ``entry_id`` is blank or unknown.

    """
    if not entry_id or not str(entry_id).strip():
        raise ValueError("entry_id must be a non-empty string")
    key = str(entry_id).strip()
    try:
        return _ENTRY_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown hermetic kit entry_id {key!r}; refuse invent-green reproduction "
            f"(known_count={len(_ENTRY_BY_ID)})"
        ) from exc


def iter_hermetic_kit_entries(
    *,
    kind: KitEntryKind | None = None,
    core_only: bool = False,
) -> tuple[HermeticKitEntry, ...]:
    """Return filtered catalogue rows in stable order.

    Parameters
    ----------
    kind
        Optional kind filter.
    core_only
        When True, return only ``core_local`` entries.

    Returns
    -------
    tuple[HermeticKitEntry, ...]
        Matching rows.

    """
    rows: Iterable[HermeticKitEntry] = _CANONICAL_ENTRIES
    if core_only:
        rows = (row for row in rows if row.kind == "core_local")
    if kind is not None:
        rows = (row for row in rows if row.kind == kind)
    return tuple(rows)


def build_hermetic_reproduction_kit() -> dict[str, object]:
    """Build the full serialisable hermetic reproduction kit payload.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with every catalogue cell (no blanks).

    """
    rows = [row.to_dict() for row in _CANONICAL_ENTRIES]
    core = sum(1 for row in _CANONICAL_ENTRIES if row.kind == "core_local")
    qpu_core = sum(
        1 for row in _CANONICAL_ENTRIES if row.kind == "core_local" and row.requires_qpu
    )
    return {
        "schema": HERMETIC_REPRODUCTION_KIT_SCHEMA,
        "claim_boundary": HERMETIC_REPRODUCTION_CLAIM_BOUNDARY,
        "entry_count": len(rows),
        "core_local_count": core,
        "core_requires_qpu_count": qpu_core,
        "blank_entry_count": 0,
        "entries": rows,
    }


def sha256_hex_of(content: bytes) -> str:
    """Return the lowercase SHA-256 hex digest of ``content``.

    Parameters
    ----------
    content
        Exact bytes to hash.

    Returns
    -------
    str
        64-character lowercase hex digest.

    Raises
    ------
    TypeError
        If ``content`` is not :class:`bytes`.

    """
    if not isinstance(content, (bytes, bytearray)):
        raise TypeError("content must be bytes")
    return hashlib.sha256(bytes(content)).hexdigest()


def verify_digest(
    *,
    label: str,
    expected_sha256_hex: str,
    content: bytes | None,
) -> DigestCheckResult:
    """Verify content against an expected SHA-256 digest (fail-closed).

    Parameters
    ----------
    label
        Artefact label for the check message.
    expected_sha256_hex
        Expected lowercase hex digest; blank fails closed.
    content
        Bytes to hash; ``None`` fails closed without inventing a match.

    Returns
    -------
    DigestCheckResult
        Deterministic match/refuse result.

    """
    if not label or not str(label).strip():
        raise ValueError("label must be a non-empty string")
    safe_label = str(label).strip()
    expected = (expected_sha256_hex or "").strip().lower()
    if not expected:
        return DigestCheckResult(
            label=safe_label,
            matched=False,
            expected_sha256_hex="",
            actual_sha256_hex="",
            message=f"blank expected digest for {safe_label!r}; refuse invent-green",
            refused=True,
        )
    if not _SHA256_HEX_RE.fullmatch(expected):
        return DigestCheckResult(
            label=safe_label,
            matched=False,
            expected_sha256_hex=expected,
            actual_sha256_hex="",
            message=f"malformed expected digest for {safe_label!r}; refuse invent-green",
            refused=True,
        )
    if content is None:
        return DigestCheckResult(
            label=safe_label,
            matched=False,
            expected_sha256_hex=expected,
            actual_sha256_hex="",
            message=f"missing content for {safe_label!r}; refuse invent-green match",
            refused=True,
        )
    actual = sha256_hex_of(content)
    if actual == expected:
        return DigestCheckResult(
            label=safe_label,
            matched=True,
            expected_sha256_hex=expected,
            actual_sha256_hex=actual,
            message=f"digest matched for {safe_label!r}",
            refused=False,
        )
    return DigestCheckResult(
        label=safe_label,
        matched=False,
        expected_sha256_hex=expected,
        actual_sha256_hex=actual,
        message=f"digest mismatch for {safe_label!r}; refuse invent-green",
        refused=True,
    )


def verify_kit_entry_digests(
    entry_id: str,
    *,
    content_by_label: Mapping[str, bytes] | None = None,
    use_fixture_payloads: bool = True,
) -> tuple[DigestCheckResult, ...]:
    """Verify all digests for one kit entry.

    Parameters
    ----------
    entry_id
        Kit entry to verify.
    content_by_label
        Optional mapping of label → content bytes supplied by the caller.
    use_fixture_payloads
        When True (default), fall back to catalogue fixture payloads when the
        caller did not supply content for a label.

    Returns
    -------
    tuple[DigestCheckResult, ...]
        One result per digest spec (empty tuple for entries without digests).

    Raises
    ------
    ValueError
        If ``entry_id`` is unknown/blank, or the entry is a refuse-invent-green row
        that must not produce green digest success.

    """
    entry = get_hermetic_kit_entry(entry_id)
    if entry.kind == "refuse_invent_green":
        raise ValueError(f"entry {entry_id!r} is refuse_invent_green; refuse invent-green digests")
    supplied = dict(content_by_label or {})
    results: list[DigestCheckResult] = []
    for spec in entry.digests:
        content = supplied.get(spec.label)
        if content is None and use_fixture_payloads:
            content = spec.fixture_payload
        results.append(
            verify_digest(
                label=spec.label,
                expected_sha256_hex=spec.sha256_hex,
                content=content,
            )
        )
    return tuple(results)


def probe_hermetic_kit_entry(
    entry_id: str,
    *,
    unknown_policy: Literal["raise", "refuse"] = "raise",
) -> dict[str, object]:
    """Probe one kit entry for skeptic-facing metadata (no subprocess).

    Parameters
    ----------
    entry_id
        Kit taxonomy key.
    unknown_policy
        ``raise`` (default) rejects unknown IDs; ``refuse`` returns a refuse map.

    Returns
    -------
    dict[str, object]
        Deterministic probe payload.

    Raises
    ------
    ValueError
        If ``entry_id`` is blank or unknown under ``unknown_policy='raise'``.

    """
    if not entry_id or not str(entry_id).strip():
        raise ValueError("entry_id must be a non-empty string")
    key = str(entry_id).strip()
    entry = _ENTRY_BY_ID.get(key)
    if entry is None:
        if unknown_policy == "raise":
            raise ValueError(
                f"unknown hermetic kit entry_id {key!r}; refuse invent-green reproduction"
            )
        if unknown_policy != "refuse":
            raise ValueError(
                f"unknown_policy must be 'raise' or 'refuse' (got {unknown_policy!r})"
            )
        return {
            "entry_id": key,
            "found": False,
            "core_eligible": False,
            "refused": True,
            "message": (f"unknown entry_id {key!r}; refuse invent-green hermetic reproduction"),
            "claim_boundary": HERMETIC_REPRODUCTION_CLAIM_BOUNDARY,
        }

    core_eligible = entry.kind == "core_local" and not entry.requires_qpu
    refused = entry.kind in {"refuse_invent_green", "hardware_gated"}
    message = f"kind={entry.kind} requires_qpu={entry.requires_qpu} core_eligible={core_eligible}"
    if entry.reason:
        message = f"{message} — {entry.reason}"
    return {
        "entry_id": key,
        "found": True,
        "core_eligible": core_eligible,
        "refused": refused,
        "kind": entry.kind,
        "requires_qpu": entry.requires_qpu,
        "argv": list(entry.argv),
        "cwd": entry.cwd,
        "digest_count": len(entry.digests),
        "message": message,
        "claim_boundary": HERMETIC_REPRODUCTION_CLAIM_BOUNDARY,
    }


def assert_hermetic_kit_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the kit payload contains zero blank/core-QPU violations.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_hermetic_reproduction_kit`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If blank entries, core-QPU violations, or count drift are detected.

    """
    kit = dict(payload) if payload is not None else build_hermetic_reproduction_kit()
    entries = kit.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("hermetic kit must contain a non-empty entries list")
    blank = 0
    core_qpu = 0
    for index, row in enumerate(entries):
        if not isinstance(row, Mapping):
            raise ValueError(f"kit entry row {index} must be a mapping")
        entry_id = row.get("entry_id")
        kind = row.get("kind")
        if not entry_id:
            blank += 1
            continue
        if kind not in {
            "core_local",
            "optional_extra",
            "hardware_gated",
            "refuse_invent_green",
        }:
            blank += 1
            continue
        if kind == "core_local" and bool(row.get("requires_qpu")):
            core_qpu += 1
        if kind != "core_local" and not row.get("reason"):
            raise ValueError(f"entry {entry_id!r} is non-core without reason")
    if blank:
        raise ValueError(f"hermetic kit has {blank} blank or invalid entries; refuse invent-green")
    if core_qpu:
        raise ValueError(f"hermetic kit has {core_qpu} core_local entries requiring QPU; refuse")
    blank_entry_count = kit.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    entry_count = kit.get("entry_count", -1)
    if not isinstance(entry_count, int) or entry_count != len(entries):
        raise ValueError("entry_count does not match entries list length")
    core_requires_qpu_count = kit.get("core_requires_qpu_count", -1)
    if not isinstance(core_requires_qpu_count, int) or core_requires_qpu_count != 0:
        raise ValueError("core_requires_qpu_count must be 0")
    return kit


def fixture_payload(label: str) -> bytes:
    """Return the declared fixture payload bytes for a known fixture label.

    Parameters
    ----------
    label
        Fixture label (for example ``fixture:transform_algebra_smoke``).

    Returns
    -------
    bytes
        Exact payload used to derive the catalogue digest.

    Raises
    ------
    ValueError
        If the label is unknown or has no fixture payload.

    """
    if not label or not str(label).strip():
        raise ValueError("label must be a non-empty string")
    key = str(label).strip()
    for entry in _CANONICAL_ENTRIES:
        for spec in entry.digests:
            if spec.label == key and spec.fixture_payload is not None:
                return spec.fixture_payload
    raise ValueError(f"unknown fixture label {key!r}")


__all__ = [
    "HERMETIC_REPRODUCTION_CLAIM_BOUNDARY",
    "HERMETIC_REPRODUCTION_KIT_SCHEMA",
    "DigestCheckResult",
    "HermeticKitEntry",
    "KitDigestSpec",
    "KitEntryKind",
    "assert_hermetic_kit_integrity",
    "build_hermetic_reproduction_kit",
    "fixture_payload",
    "get_hermetic_kit_entry",
    "iter_hermetic_kit_entries",
    "list_hermetic_kit_entry_ids",
    "probe_hermetic_kit_entry",
    "sha256_hex_of",
    "verify_digest",
    "verify_kit_entry_digests",
]
