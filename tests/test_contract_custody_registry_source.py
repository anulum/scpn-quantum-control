# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — registry evidence source tests
"""Own registry evidence extraction and isolation from the shared registry."""

from scpn_quantum_control.program_ad_registry import primitive_contract_for
from tools.contract_custody_registry_source import PROBE_IDENTITY, registry_evidence_source


def test_contract_snapshot_matches_actual_public_registry_contract() -> None:
    """Retain actual identity, rule metadata and unsupported lowering boundary."""
    source = registry_evidence_source()
    contract = primitive_contract_for(PROBE_IDENTITY)
    snapshot = source["contract"]
    assert snapshot["identity"] == contract.identity.key
    assert snapshot["derivative_rule"] == contract.derivative_rule.name
    assert snapshot["lowering_metadata"] == dict(contract.lowering_metadata)
    assert snapshot["has_lowering_rule"] is False
    assert contract.lowering_rule is None
    assert "executable lowering blocked" in snapshot["lowering_metadata"]["mlir"]
    assert source["report"]["supported"] is True
    assert source["inputs"]["probe_identity"] == PROBE_IDENTITY


def test_mutating_returned_evidence_cannot_modify_registry_metadata() -> None:
    """The capture owns mutable JSON copies; the source snapshot stays intact."""
    source = registry_evidence_source()
    original = dict(primitive_contract_for(PROBE_IDENTITY).lowering_metadata)
    source["contract"]["lowering_metadata"]["mlir"] = "tampered"
    source["report"]["rows"].clear()
    assert dict(primitive_contract_for(PROBE_IDENTITY).lowering_metadata) == original
    fresh = registry_evidence_source()
    assert fresh["contract"]["lowering_metadata"] == original
    assert fresh["report"]["rows"]


def test_empty_registry_capture_does_not_unregister_default_contracts() -> None:
    """Unsupported source evidence must not mutate the shared positive registry."""
    before = registry_evidence_source()
    empty = registry_evidence_source(empty=True)
    assert empty["inputs"]["registry"] == "empty"
    assert empty["contract"] is None
    assert empty["report"]["supported"] is False
    assert empty["report"]["covered_primitives"] == 0
    assert len(empty["report"]["blocked_identities"]) == empty["report"]["total_primitives"]
    assert registry_evidence_source() == before
