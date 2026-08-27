# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for advantage language protocol
"""Real-surface tests for ``scpn_quantum_control.advantage_language_protocol``."""

from __future__ import annotations

import pytest

import scpn_quantum_control.advantage_language_protocol as advantage_language_protocol
from scpn_quantum_control.advantage_language_protocol import (
    ADVANTAGE_LANGUAGE_CLAIM_BOUNDARY,
    ADVANTAGE_LANGUAGE_PROTOCOL_SCHEMA,
    AdvantageLanguageProbeResult,
    AdvantageProtocolRecord,
    NoAdvantageCertificate,
    assert_advantage_language_registry_integrity,
    build_advantage_language_registry,
    find_advantage_language_triggers,
    get_advantage_protocol,
    issue_no_advantage_certificate,
    iter_advantage_protocols,
    list_advantage_protocol_ids,
    probe_advantage_language,
)


def test_list_ids_stable_and_include_default() -> None:
    """Expose unique stable ids including the default no-advantage row."""
    ids = list_advantage_protocol_ids()
    assert ids
    assert len(ids) == len(set(ids))
    assert "protocol:default.no_advantage" in ids
    assert ids == list_advantage_protocol_ids()


def test_get_default_and_decisive_protocols() -> None:
    """Resolve default, decisive, and bounded research protocols."""
    default = get_advantage_protocol("protocol:default.no_advantage")
    assert default.language_status == "no_advantage_default"
    assert not default.reason
    assert default.claim_boundary == ADVANTAGE_LANGUAGE_CLAIM_BOUNDARY

    decisive = get_advantage_protocol("protocol:decisive.kuramoto_xy")
    assert decisive.language_status == "decisive_gated"
    assert decisive.reason
    assert any("decisive_advantage_protocol" in m for m in decisive.evidence_modules)

    entanglement_sync = get_advantage_protocol("protocol:entanglement.initial_state_observation")
    assert entanglement_sync.language_status == "research_observation"
    assert "entanglement-specific" in entanglement_sync.reason
    assert any(
        "entanglement_enhanced_sync" in module for module in entanglement_sync.evidence_modules
    )


def test_get_rejects_blank_and_unknown() -> None:
    """Reject blank and unknown protocol identifiers fail closed."""
    with pytest.raises(ValueError, match="non-empty"):
        get_advantage_protocol("  ")
    with pytest.raises(ValueError, match="unknown advantage protocol_id"):
        get_advantage_protocol("protocol:invent.green")


def test_iter_filters_by_status() -> None:
    """Return the full catalogue or deterministic status subsets."""
    all_rows = iter_advantage_protocols()
    assert len(all_rows) == len(list_advantage_protocol_ids())
    research = iter_advantage_protocols(language_status="research_observation")
    assert research
    assert all(row.language_status == "research_observation" for row in research)


def test_build_registry_zero_blanks() -> None:
    """Build a schema-tagged registry with complete status counts."""
    registry = build_advantage_language_registry()
    assert registry["schema"] == ADVANTAGE_LANGUAGE_PROTOCOL_SCHEMA
    assert registry["blank_entry_count"] == 0
    assert registry["protocol_count"] == len(registry["protocols"])  # type: ignore[arg-type]
    validated = assert_advantage_language_registry_integrity(registry)
    assert validated["blank_entry_count"] == 0
    counts = registry["status_counts"]
    assert isinstance(counts, dict)
    assert sum(int(v) for v in counts.values()) == int(registry["protocol_count"])


def test_issue_no_advantage_certificate_default() -> None:
    """Issue deterministic default and protocol-bound certificates."""
    cert = issue_no_advantage_certificate(context="docs homepage")
    assert isinstance(cert, NoAdvantageCertificate)
    assert cert.language_status == "no_advantage_default"
    assert "No quantum-advantage marketing claim" in cert.statement
    assert "no_advantage" in cert.certificate_id
    payload = cert.to_dict()
    assert payload["claim_boundary"] == ADVANTAGE_LANGUAGE_CLAIM_BOUNDARY

    entanglement_sync = issue_no_advantage_certificate(
        context="entanglement-sync initial-state study",
        protocol_id="protocol:entanglement.initial_state_observation",
    )
    assert entanglement_sync.protocol_id == "protocol:entanglement.initial_state_observation"
    assert entanglement_sync.language_status == "no_advantage_default"


def test_issue_certificate_rejects_blank_and_refuse_protocol() -> None:
    """Reject blank contexts and refuse-only protocol bindings."""
    with pytest.raises(ValueError, match="context"):
        issue_no_advantage_certificate(context="  ")
    with pytest.raises(ValueError, match="refuse protocol"):
        issue_no_advantage_certificate(protocol_id="protocol:ungoverned.advantage_language")


def test_find_triggers_and_probe_ungoverned_advantage() -> None:
    """Detect marketing triggers and refuse ungoverned advantage claims."""
    triggers = find_advantage_language_triggers(
        "This shows Quantum Advantage over classical solvers."
    )
    assert "quantum advantage" in triggers

    refused = probe_advantage_language("We claim quantum advantage without a protocol.")
    assert refused.allowed is False
    assert refused.language_status == "refuse_advantage_language"
    assert refused.triggers
    assert "ungoverned" in refused.protocol_id or "refuse" in refused.reason.lower()


def test_probe_neutral_default_allowed() -> None:
    """Allow neutral wording under the default no-advantage posture."""
    result = probe_advantage_language("Local statevector parity under claim_boundary.")
    assert result.allowed is True
    assert result.language_status == "no_advantage_default"
    assert not result.triggers


def test_probe_bound_protocols() -> None:
    """Apply default, research, decisive, and refuse protocol policies."""
    research_ok = probe_advantage_language(
        "S2 scaling matrix research observation only.",
        protocol_id="protocol:s2.scaling_matrix",
    )
    assert research_ok.allowed is True
    assert research_ok.language_status == "research_observation"

    research_block = probe_advantage_language(
        "S2 proves quantum advantage.",
        protocol_id="protocol:s2.scaling_matrix",
    )
    assert research_block.allowed is False

    default_block = probe_advantage_language(
        "beats classical at every size",
        protocol_id="protocol:default.no_advantage",
    )
    assert default_block.allowed is False

    default_ok = probe_advantage_language(
        "Neutral reporting of residuals.",
        protocol_id="protocol:default.no_advantage",
    )
    assert default_ok.allowed is True

    decisive = probe_advantage_language(
        "Entering decisive Kuramoto-XY protocol for quantum advantage question.",
        protocol_id="protocol:decisive.kuramoto_xy",
    )
    assert decisive.allowed is True
    assert decisive.language_status == "decisive_gated"
    assert "not a proven advantage" in decisive.reason.lower() or "evidence path" in (
        decisive.reason.lower()
    )

    refuse_row = probe_advantage_language(
        "anything",
        protocol_id="protocol:ungoverned.advantage_language",
    )
    assert refuse_row.allowed is False


def test_probe_unknown_protocol_policies() -> None:
    """Raise or refuse unknown protocols according to explicit policy."""
    with pytest.raises(ValueError, match="unknown advantage protocol_id"):
        probe_advantage_language("x", protocol_id="protocol:missing")
    refused = probe_advantage_language(
        "quantum advantage",
        protocol_id="protocol:missing",
        unknown_policy="refuse",
    )
    assert refused.allowed is False
    assert refused.language_status == "refuse_advantage_language"
    with pytest.raises(ValueError, match="unknown_policy"):
        probe_advantage_language(
            "x",
            protocol_id="protocol:missing",
            unknown_policy="invent",  # type: ignore[arg-type]
        )


def test_probe_none_claim_and_empty_protocol_id() -> None:
    """Reject absent claims and treat blank protocol ids as unbound."""
    with pytest.raises(TypeError, match="claim_text"):
        probe_advantage_language(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="claim_text"):
        find_advantage_language_triggers(None)  # type: ignore[arg-type]
    # empty protocol_id treated as unbound
    result = probe_advantage_language("neutral text", protocol_id="  ")
    assert result.allowed is True
    assert result.language_status == "no_advantage_default"


def test_record_and_certificate_validation() -> None:
    """Enforce catalogue-record and certificate construction invariants."""
    with pytest.raises(ValueError, match="protocol_id"):
        AdvantageProtocolRecord(
            protocol_id="",
            language_status="no_advantage_default",
            summary="s",
            evidence_modules=("m",),
        )
    with pytest.raises(ValueError, match="unknown language_status"):
        AdvantageProtocolRecord(
            protocol_id="p",
            language_status="green",  # type: ignore[arg-type]
            summary="s",
            evidence_modules=("m",),
        )
    with pytest.raises(ValueError, match="summary"):
        AdvantageProtocolRecord(
            protocol_id="p",
            language_status="no_advantage_default",
            summary="  ",
            evidence_modules=("m",),
        )
    with pytest.raises(ValueError, match="evidence_modules"):
        AdvantageProtocolRecord(
            protocol_id="p",
            language_status="no_advantage_default",
            summary="s",
            evidence_modules=("",),
        )
    with pytest.raises(ValueError, match="must not carry"):
        AdvantageProtocolRecord(
            protocol_id="p",
            language_status="no_advantage_default",
            summary="s",
            evidence_modules=("m",),
            reason="should not be here",
        )
    with pytest.raises(ValueError, match="require a non-empty reason"):
        AdvantageProtocolRecord(
            protocol_id="p",
            language_status="decisive_gated",
            summary="s",
            evidence_modules=("m",),
            reason="",
        )
    with pytest.raises(ValueError, match="certificate_id"):
        NoAdvantageCertificate(
            certificate_id="",
            language_status="no_advantage_default",
            statement="s",
        )
    with pytest.raises(ValueError, match="language_status must be"):
        NoAdvantageCertificate(
            certificate_id="c",
            language_status="research_observation",
            statement="s",
        )
    with pytest.raises(ValueError, match="statement"):
        NoAdvantageCertificate(
            certificate_id="c",
            language_status="no_advantage_default",
            statement="  ",
        )
    with pytest.raises(ValueError, match="protocol_id must be non-empty"):
        NoAdvantageCertificate(
            certificate_id="c",
            language_status="no_advantage_default",
            statement="s",
            protocol_id="",
        )


def test_probe_result_validation() -> None:
    """Enforce probe reason, refusal, and trigger invariants."""
    with pytest.raises(ValueError, match="claim_text"):
        AdvantageLanguageProbeResult(
            claim_text=None,  # type: ignore[arg-type]
            allowed=True,
            language_status="no_advantage_default",
            protocol_id="p",
            triggers=(),
            reason="r",
        )
    with pytest.raises(ValueError, match="reason"):
        AdvantageLanguageProbeResult(
            claim_text="x",
            allowed=True,
            language_status="no_advantage_default",
            protocol_id="p",
            triggers=(),
            reason="",
        )
    with pytest.raises(ValueError, match="must not be allowed"):
        AdvantageLanguageProbeResult(
            claim_text="x",
            allowed=True,
            language_status="refuse_advantage_language",
            protocol_id="p",
            triggers=(),
            reason="r",
        )
    with pytest.raises(ValueError, match="triggers"):
        AdvantageLanguageProbeResult(
            claim_text="x",
            allowed=False,
            language_status="refuse_advantage_language",
            protocol_id="p",
            triggers=("  ",),
            reason="r",
        )


def test_assert_integrity_rejects_invalid_payloads() -> None:
    """Reject malformed rows, blanks, missing reasons, and count drift."""
    with pytest.raises(ValueError, match="non-empty protocols"):
        assert_advantage_language_registry_integrity({"protocols": []})
    with pytest.raises(ValueError, match="blank"):
        assert_advantage_language_registry_integrity(
            {
                "protocols": [{"protocol_id": "", "language_status": "no_advantage_default"}],
                "blank_entry_count": 0,
                "protocol_count": 1,
            }
        )
    with pytest.raises(ValueError, match="without reason"):
        assert_advantage_language_registry_integrity(
            {
                "protocols": [
                    {
                        "protocol_id": "p",
                        "language_status": "decisive_gated",
                        "reason": "",
                    }
                ],
                "blank_entry_count": 0,
                "protocol_count": 1,
            }
        )
    good = get_advantage_protocol("protocol:default.no_advantage").to_dict()
    with pytest.raises(ValueError, match="blank_entry_count"):
        assert_advantage_language_registry_integrity(
            {"protocols": [good], "blank_entry_count": 1, "protocol_count": 1}
        )
    with pytest.raises(ValueError, match="protocol_count"):
        assert_advantage_language_registry_integrity(
            {"protocols": [good], "blank_entry_count": 0, "protocol_count": 99}
        )
    with pytest.raises(ValueError, match="mapping"):
        assert_advantage_language_registry_integrity(
            {
                "protocols": ["not-a-mapping"],
                "blank_entry_count": 0,
                "protocol_count": 1,
            }
        )
    with pytest.raises(ValueError, match="blank"):
        assert_advantage_language_registry_integrity(
            {
                "protocols": [
                    {
                        "protocol_id": "p",
                        "language_status": "not-a-status",
                        "reason": "r",
                    }
                ],
                "blank_entry_count": 0,
                "protocol_count": 1,
            }
        )


def test_catalogue_map_rejects_duplicates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail closed when canonical protocol identifiers are duplicated."""
    row = get_advantage_protocol("protocol:default.no_advantage")
    monkeypatch.setattr(
        advantage_language_protocol,
        "_CANONICAL_PROTOCOLS",
        (row, row),
    )
    with pytest.raises(RuntimeError, match="duplicate protocol_id"):
        advantage_language_protocol._catalogue_map()


def test_record_to_dict_and_probe_to_dict() -> None:
    """Serialise protocol records and probe decisions to JSON-ready maps."""
    row = get_advantage_protocol("protocol:neural_operator.structural_surrogate")
    payload = row.to_dict()
    assert payload["protocol_id"] == row.protocol_id
    assert isinstance(payload["evidence_modules"], list)
    probe = probe_advantage_language("neutral")
    assert probe.to_dict()["allowed"] is True
