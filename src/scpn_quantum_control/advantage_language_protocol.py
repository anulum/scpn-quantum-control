# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — advantage / no-advantage language protocol
"""Fail-closed advantage-language governance and protocol catalogue.

Default posture is **no-advantage / research observation**. Decisive protocol
modules that already exist in-tree are catalogued with claim-gated statuses; this
surface never invents green quantum-advantage marketing language without an
explicit protocol identity and refuse path for ungoverned wording.

The module is pure and deterministic. It does not run QPU jobs, invent
benchmarks, or promote category leadership.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

ProtocolLanguageStatus = Literal[
    "no_advantage_default",
    "research_observation",
    "decisive_gated",
    "refuse_advantage_language",
]
"""Language-governance status for one protocol catalogue row."""

ADVANTAGE_LANGUAGE_PROTOCOL_SCHEMA: Final[str] = "advantage_language_protocol.v1"
"""JSON schema identifier for serialised registry payloads."""

ADVANTAGE_LANGUAGE_CLAIM_BOUNDARY: Final[str] = (
    "advantage-language governance only; default is no-advantage / research "
    "observation; decisive or advantage wording requires an explicit protocol "
    "identity and never invents green quantum-advantage marketing claims"
)
"""Shared claim boundary for certificates, probes, and catalogue rows."""

# Phrases that trigger the language gate when no protocol id is supplied.
_ADVANTAGE_TRIGGER_PHRASES: Final[tuple[str, ...]] = (
    "quantum advantage",
    "qpu advantage",
    "decides advantage",
    "decides_advantage",
    "demonstrates advantage",
    "proven advantage",
    "category leadership",
    "beats classical",
    "strictly faster than classical",
)


@dataclass(frozen=True, slots=True)
class AdvantageProtocolRecord:
    """One protocol catalogue entry under language governance.

    Attributes
    ----------
    protocol_id
        Stable protocol identifier.
    language_status
        Governance status for advantage-language use.
    summary
        Short human-readable description.
    evidence_modules
        In-tree module paths that implement related runners (compose, not fork).
    reason
        Required reason for refuse/gated rows; empty only for
        ``no_advantage_default``.
    claim_boundary
        Non-promotional claim boundary string.

    """

    protocol_id: str
    language_status: ProtocolLanguageStatus
    summary: str
    evidence_modules: tuple[str, ...]
    reason: str = ""
    claim_boundary: str = ADVANTAGE_LANGUAGE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate catalogue-entry invariants."""
        if not self.protocol_id or not self.protocol_id.strip():
            raise ValueError("protocol_id must be non-empty")
        if self.language_status not in {
            "no_advantage_default",
            "research_observation",
            "decisive_gated",
            "refuse_advantage_language",
        }:
            raise ValueError(f"unknown language_status: {self.language_status!r}")
        if not self.summary or not self.summary.strip():
            raise ValueError("summary must be non-empty")
        if any(not item or not item.strip() for item in self.evidence_modules):
            raise ValueError("evidence_modules must be non-empty strings")
        if self.language_status == "no_advantage_default":
            if self.reason:
                raise ValueError("no_advantage_default rows must not carry a non-empty reason")
        elif not self.reason or not self.reason.strip():
            raise ValueError(
                "non-default protocol rows require a non-empty reason "
                f"(protocol_id={self.protocol_id!r})"
            )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this protocol record."""
        return {
            "protocol_id": self.protocol_id,
            "language_status": self.language_status,
            "summary": self.summary,
            "evidence_modules": list(self.evidence_modules),
            "reason": self.reason,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class NoAdvantageCertificate:
    """Deterministic default no-advantage / research-observation certificate.

    Attributes
    ----------
    certificate_id
        Stable certificate identity.
    language_status
        Always ``no_advantage_default``.
    statement
        Non-promotional statement suitable for docs and logs.
    protocol_id
        Optional bound protocol when issued under a known id.
    claim_boundary
        Non-promotional claim boundary.

    """

    certificate_id: str
    language_status: ProtocolLanguageStatus
    statement: str
    protocol_id: str = "protocol:default.no_advantage"
    claim_boundary: str = ADVANTAGE_LANGUAGE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate certificate invariants."""
        if not self.certificate_id or not self.certificate_id.strip():
            raise ValueError("certificate_id must be non-empty")
        if self.language_status != "no_advantage_default":
            raise ValueError("NoAdvantageCertificate.language_status must be no_advantage_default")
        if not self.statement or not self.statement.strip():
            raise ValueError("statement must be non-empty")
        if not self.protocol_id or not self.protocol_id.strip():
            raise ValueError("protocol_id must be non-empty")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this certificate."""
        return {
            "certificate_id": self.certificate_id,
            "language_status": self.language_status,
            "statement": self.statement,
            "protocol_id": self.protocol_id,
            "claim_boundary": self.claim_boundary,
        }


@dataclass(frozen=True, slots=True)
class AdvantageLanguageProbeResult:
    """Result of probing a claim string under language governance.

    Attributes
    ----------
    claim_text
        Original claim text.
    allowed
        Whether the claim may proceed under the bound protocol.
    language_status
        Effective status applied to the claim.
    protocol_id
        Protocol used for the decision, or empty when none.
    triggers
        Trigger phrases found in the claim text.
    reason
        Human-readable decision reason.

    """

    claim_text: str
    allowed: bool
    language_status: ProtocolLanguageStatus
    protocol_id: str
    triggers: tuple[str, ...]
    reason: str
    claim_boundary: str = ADVANTAGE_LANGUAGE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        """Validate probe-result invariants."""
        if self.claim_text is None:
            raise ValueError("claim_text must not be None")
        if not self.reason or not self.reason.strip():
            raise ValueError("reason must be non-empty")
        if self.allowed and self.language_status == "refuse_advantage_language":
            raise ValueError("refuse_advantage_language results must not be allowed")
        if any(not item or not item.strip() for item in self.triggers):
            raise ValueError("triggers must be non-empty strings when present")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready mapping for this probe result."""
        return {
            "claim_text": self.claim_text,
            "allowed": self.allowed,
            "language_status": self.language_status,
            "protocol_id": self.protocol_id,
            "triggers": list(self.triggers),
            "reason": self.reason,
            "claim_boundary": self.claim_boundary,
        }


def _protocol(
    protocol_id: str,
    language_status: ProtocolLanguageStatus,
    summary: str,
    *,
    evidence_modules: Sequence[str],
    reason: str = "",
) -> AdvantageProtocolRecord:
    """Build one validated catalogue row."""
    return AdvantageProtocolRecord(
        protocol_id=protocol_id,
        language_status=language_status,
        summary=summary,
        evidence_modules=tuple(evidence_modules),
        reason=reason,
    )


_CANONICAL_PROTOCOLS: Final[tuple[AdvantageProtocolRecord, ...]] = (
    _protocol(
        "protocol:default.no_advantage",
        "no_advantage_default",
        "Default no-advantage / research-observation posture for all public claims.",
        evidence_modules=("scpn_quantum_control.advantage_language_protocol",),
    ),
    _protocol(
        "protocol:s2.scaling_matrix",
        "research_observation",
        "S2 size-by-baseline scaling matrix (claim-bounded research observation).",
        evidence_modules=("scpn_quantum_control.benchmarks.advantage_protocol",),
        reason=(
            "S2 scaling rows are research observation under explicit claim_boundary; "
            "they do not authorise marketing quantum-advantage language"
        ),
    ),
    _protocol(
        "protocol:decisive.kuramoto_xy",
        "decisive_gated",
        "Decisive Kuramoto-XY single-decision protocol (evidence-gated).",
        evidence_modules=("scpn_quantum_control.benchmarks.decisive_advantage_protocol",),
        reason=(
            "Decisive labels require populated matched-budget rows; default expected "
            "outcomes are classical_wins / crossover / inconclusive, never invent-green "
            "broad advantage"
        ),
    ),
    _protocol(
        "protocol:neural_operator.structural_surrogate",
        "research_observation",
        "Neural-operator surrogate structural advantage study (honest, non-QPU).",
        evidence_modules=("scpn_quantum_control.forecasting.neural_operator_advantage",),
        reason=(
            "Structural amortisation findings are classical surrogate research "
            "observation, not quantum advantage claims"
        ),
    ),
    _protocol(
        "protocol:entanglement.initial_state_observation",
        "research_observation",
        "Entanglement-sync initial-state coherence comparison with population-matched controls.",
        evidence_modules=(
            "scpn_quantum_control.analysis.entanglement_enhanced_sync",
            "scpn_quantum_control.analysis.entanglement_sync_evidence",
        ),
        reason=(
            "State-family and dephased-control differences are bounded simulation "
            "observations; they do not establish an entanglement-specific mechanism "
            "or quantum advantage"
        ),
    ),
    _protocol(
        "protocol:ungoverned.advantage_language",
        "refuse_advantage_language",
        "Catch-all refuse path for ungoverned advantage marketing language.",
        evidence_modules=(
            "scpn_quantum_control.advantage_language_protocol",
            "docs/internal/differentiable_programming/p3_strategic/"
            "bl65_advantage_no_advantage_protocol.md",
        ),
        reason=(
            "Advantage language without an explicit protocol identity is refused; "
            "default is no-advantage / research observation"
        ),
    ),
)


def _catalogue_map() -> dict[str, AdvantageProtocolRecord]:
    """Return the protocol_id → record map for the canonical catalogue."""
    mapping = {row.protocol_id: row for row in _CANONICAL_PROTOCOLS}
    if len(mapping) != len(_CANONICAL_PROTOCOLS):
        raise RuntimeError("duplicate protocol_id in advantage language catalogue")
    return mapping


_PROTOCOL_BY_ID: Final[Mapping[str, AdvantageProtocolRecord]] = _catalogue_map()


def list_advantage_protocol_ids() -> tuple[str, ...]:
    """Return all canonical protocol identifiers in stable catalogue order.

    Returns
    -------
    tuple[str, ...]
        Ordered protocol identifiers.

    """
    return tuple(row.protocol_id for row in _CANONICAL_PROTOCOLS)


def get_advantage_protocol(protocol_id: str) -> AdvantageProtocolRecord:
    """Return one catalogue row or raise for unknown identifiers.

    Parameters
    ----------
    protocol_id
        Protocol taxonomy key.

    Returns
    -------
    AdvantageProtocolRecord
        Matching catalogue row.

    Raises
    ------
    ValueError
        If ``protocol_id`` is blank or unknown.

    """
    if not protocol_id or not str(protocol_id).strip():
        raise ValueError("protocol_id must be a non-empty string")
    key = str(protocol_id).strip()
    try:
        return _PROTOCOL_BY_ID[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown advantage protocol_id {key!r}; refuse invent-green advantage "
            f"(known_count={len(_PROTOCOL_BY_ID)})"
        ) from exc


def iter_advantage_protocols(
    *,
    language_status: ProtocolLanguageStatus | None = None,
) -> tuple[AdvantageProtocolRecord, ...]:
    """Return filtered catalogue rows in stable order.

    Parameters
    ----------
    language_status
        Optional status filter.

    Returns
    -------
    tuple[AdvantageProtocolRecord, ...]
        Matching rows.

    """
    rows: Iterable[AdvantageProtocolRecord] = _CANONICAL_PROTOCOLS
    if language_status is not None:
        rows = (row for row in rows if row.language_status == language_status)
    return tuple(rows)


def build_advantage_language_registry() -> dict[str, object]:
    """Build the full serialisable advantage-language protocol registry.

    Returns
    -------
    dict[str, object]
        Schema-tagged payload with every catalogue cell (no blanks).

    """
    rows = [row.to_dict() for row in _CANONICAL_PROTOCOLS]
    counts: dict[str, int] = {
        "no_advantage_default": 0,
        "research_observation": 0,
        "decisive_gated": 0,
        "refuse_advantage_language": 0,
    }
    for row in _CANONICAL_PROTOCOLS:
        counts[row.language_status] += 1
    return {
        "schema": ADVANTAGE_LANGUAGE_PROTOCOL_SCHEMA,
        "claim_boundary": ADVANTAGE_LANGUAGE_CLAIM_BOUNDARY,
        "protocol_count": len(rows),
        "status_counts": counts,
        "blank_entry_count": 0,
        "protocols": rows,
    }


def issue_no_advantage_certificate(
    *,
    context: str = "public_claim",
    protocol_id: str = "protocol:default.no_advantage",
) -> NoAdvantageCertificate:
    """Issue the default no-advantage / research-observation certificate.

    Parameters
    ----------
    context
        Free-text context label included in the certificate id.
    protocol_id
        Optional known protocol id to bind (must exist and not be a refuse row).

    Returns
    -------
    NoAdvantageCertificate
        Deterministic non-promotional certificate.

    Raises
    ------
    ValueError
        If ``context`` is blank, or ``protocol_id`` is unknown / refuse-status.

    """
    if not context or not str(context).strip():
        raise ValueError("context must be a non-empty string")
    key = str(protocol_id).strip()
    record = get_advantage_protocol(key)
    if record.language_status == "refuse_advantage_language":
        raise ValueError(f"cannot issue no-advantage certificate under refuse protocol {key!r}")
    safe_context = str(context).strip().replace(" ", "_")
    statement = (
        "No quantum-advantage marketing claim is authorised under this certificate. "
        "Default language status is no-advantage / research observation. "
        f"Bound protocol={key!r}."
    )
    return NoAdvantageCertificate(
        certificate_id=f"no_advantage:{safe_context}:{key}",
        language_status="no_advantage_default",
        statement=statement,
        protocol_id=key,
    )


def find_advantage_language_triggers(claim_text: str) -> tuple[str, ...]:
    """Return advantage-language trigger phrases found in ``claim_text``.

    Parameters
    ----------
    claim_text
        Free-text claim to scan (case-insensitive).

    Returns
    -------
    tuple[str, ...]
        Trigger phrases found in catalogue order.

    """
    if claim_text is None:
        raise TypeError("claim_text must not be None")
    lowered = str(claim_text).lower()
    return tuple(phrase for phrase in _ADVANTAGE_TRIGGER_PHRASES if phrase in lowered)


def probe_advantage_language(
    claim_text: str,
    protocol_id: str | None = None,
    *,
    unknown_policy: Literal["raise", "refuse"] = "raise",
) -> AdvantageLanguageProbeResult:
    """Probe a claim under advantage-language governance.

    Default (no protocol): any advantage trigger phrase is refused. With a known
    protocol: ``no_advantage_default`` and ``research_observation`` allow only
    non-trigger claims (or research wording without marketing triggers);
    ``decisive_gated`` allows trigger phrases only when a protocol id is bound
    but still does not invent green advantage — allowed means “may enter the
    decisive evidence path”, not “advantage is proven”; ``refuse_advantage_language``
    always refuses.

    Parameters
    ----------
    claim_text
        Free-text claim.
    protocol_id
        Optional protocol identity.
    unknown_policy
        ``raise`` (default) rejects unknown protocol ids; ``refuse`` maps them
        to the ungoverned refuse row.

    Returns
    -------
    AdvantageLanguageProbeResult
        Deterministic language-gate decision.

    Raises
    ------
    ValueError
        If ``claim_text`` is not a string-like value that can be stripped when
        empty rules apply, if protocol id is blank when provided, or unknown
        under ``unknown_policy='raise'``.
    TypeError
        If ``claim_text`` is ``None``.

    """
    if claim_text is None:
        raise TypeError("claim_text must not be None")
    text = str(claim_text)
    triggers = find_advantage_language_triggers(text)

    if protocol_id is None or (isinstance(protocol_id, str) and not protocol_id.strip()):
        if not triggers:
            return AdvantageLanguageProbeResult(
                claim_text=text,
                allowed=True,
                language_status="no_advantage_default",
                protocol_id="protocol:default.no_advantage",
                triggers=(),
                reason=(
                    "no advantage-language triggers; default no-advantage posture allows "
                    "the claim as research/neutral wording"
                ),
            )
        refuse = get_advantage_protocol("protocol:ungoverned.advantage_language")
        return AdvantageLanguageProbeResult(
            claim_text=text,
            allowed=False,
            language_status="refuse_advantage_language",
            protocol_id=refuse.protocol_id,
            triggers=triggers,
            reason=refuse.reason,
        )

    key = str(protocol_id).strip()
    record = _PROTOCOL_BY_ID.get(key)
    if record is None:
        if unknown_policy == "raise":
            raise ValueError(
                f"unknown advantage protocol_id {key!r}; refuse invent-green advantage"
            )
        if unknown_policy != "refuse":
            raise ValueError(
                f"unknown_policy must be 'raise' or 'refuse' (got {unknown_policy!r})"
            )
        refuse = get_advantage_protocol("protocol:ungoverned.advantage_language")
        return AdvantageLanguageProbeResult(
            claim_text=text,
            allowed=False,
            language_status="refuse_advantage_language",
            protocol_id=key,
            triggers=triggers,
            reason=(
                f"unknown protocol_id {key!r}; mapped to ungoverned refuse path — {refuse.reason}"
            ),
        )

    if record.language_status == "refuse_advantage_language":
        return AdvantageLanguageProbeResult(
            claim_text=text,
            allowed=False,
            language_status="refuse_advantage_language",
            protocol_id=key,
            triggers=triggers,
            reason=record.reason,
        )

    if record.language_status == "no_advantage_default":
        if triggers:
            return AdvantageLanguageProbeResult(
                claim_text=text,
                allowed=False,
                language_status="refuse_advantage_language",
                protocol_id=key,
                triggers=triggers,
                reason=(
                    "no_advantage_default protocol forbids advantage-language triggers; "
                    "issue a research_observation or decisive_gated protocol id instead"
                ),
            )
        return AdvantageLanguageProbeResult(
            claim_text=text,
            allowed=True,
            language_status="no_advantage_default",
            protocol_id=key,
            triggers=(),
            reason="bound no_advantage_default protocol allows neutral wording only",
        )

    if record.language_status == "research_observation":
        if triggers:
            return AdvantageLanguageProbeResult(
                claim_text=text,
                allowed=False,
                language_status="research_observation",
                protocol_id=key,
                triggers=triggers,
                reason=(
                    "research_observation protocol forbids marketing advantage triggers; "
                    f"{record.reason}"
                ),
            )
        return AdvantageLanguageProbeResult(
            claim_text=text,
            allowed=True,
            language_status="research_observation",
            protocol_id=key,
            triggers=(),
            reason=(
                "research_observation protocol allows non-marketing research wording; "
                f"{record.reason}"
            ),
        )

    # decisive_gated: may enter evidence path; still not invent-green advantage.
    return AdvantageLanguageProbeResult(
        claim_text=text,
        allowed=True,
        language_status="decisive_gated",
        protocol_id=key,
        triggers=triggers,
        reason=(
            "decisive_gated protocol may enter the evidence path; this is not a "
            f"proven advantage claim — {record.reason}"
        ),
    )


def assert_advantage_language_registry_integrity(
    payload: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assert the registry payload contains zero blank entries.

    Parameters
    ----------
    payload
        Optional payload from :func:`build_advantage_language_registry`.

    Returns
    -------
    dict[str, object]
        Validated payload.

    Raises
    ------
    ValueError
        If blank entries or count drift are detected.

    """
    registry = dict(payload) if payload is not None else build_advantage_language_registry()
    protocols = registry.get("protocols")
    if not isinstance(protocols, list) or not protocols:
        raise ValueError("advantage language registry must contain a non-empty protocols list")
    blank = 0
    for index, row in enumerate(protocols):
        if not isinstance(row, Mapping):
            raise ValueError(f"protocol row {index} must be a mapping")
        protocol_id = row.get("protocol_id")
        status = row.get("language_status")
        if not protocol_id:
            blank += 1
            continue
        if status not in {
            "no_advantage_default",
            "research_observation",
            "decisive_gated",
            "refuse_advantage_language",
        }:
            blank += 1
            continue
        if status != "no_advantage_default" and not row.get("reason"):
            raise ValueError(f"protocol {protocol_id!r} is non-default without reason")
    if blank:
        raise ValueError(
            f"advantage language registry has {blank} blank or invalid entries; refuse green"
        )
    blank_entry_count = registry.get("blank_entry_count", -1)
    if not isinstance(blank_entry_count, int) or blank_entry_count != 0:
        raise ValueError("blank_entry_count must be 0")
    protocol_count = registry.get("protocol_count", -1)
    if not isinstance(protocol_count, int) or protocol_count != len(protocols):
        raise ValueError("protocol_count does not match protocols list length")
    return registry


__all__ = [
    "ADVANTAGE_LANGUAGE_CLAIM_BOUNDARY",
    "ADVANTAGE_LANGUAGE_PROTOCOL_SCHEMA",
    "AdvantageLanguageProbeResult",
    "AdvantageProtocolRecord",
    "NoAdvantageCertificate",
    "ProtocolLanguageStatus",
    "assert_advantage_language_registry_integrity",
    "build_advantage_language_registry",
    "find_advantage_language_triggers",
    "get_advantage_protocol",
    "issue_no_advantage_certificate",
    "iter_advantage_protocols",
    "list_advantage_protocol_ids",
    "probe_advantage_language",
]
