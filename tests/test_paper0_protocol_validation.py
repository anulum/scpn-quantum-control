# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Protocol validation tests
"""Tests for Paper 0 Protocol source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.protocol_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    ProtocolConfig,
    classify_protocol_component,
    protocol_labels,
    validate_protocol_fixture,
)


def test_protocol_fixture_preserves_source_boundary() -> None:
    result = validate_protocol_fixture()
    assert result.source_ledger_span == ("P0R05191", "P0R05201")
    assert result.source_record_count == 11
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R05202"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert result.problem_metadata["protocol_state"] == "source_protocol_only_no_experiment"
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05191"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05201"


def test_protocol_classification_and_labels_are_explicit() -> None:
    for component in ("protocol", "falsification_condition"):
        assert classify_protocol_component(component) == f"{component}_source_boundary"
    labels = protocol_labels()
    assert labels["section"] == "Protocol"
    assert labels["next_boundary"] == "P0R05202"


def test_protocol_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 11"):
        ProtocolConfig(expected_source_record_count=10)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        ProtocolConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05202"):
        ProtocolConfig(next_source_boundary="P0R05201")
    with pytest.raises(ValueError, match="unknown protocol component"):
        classify_protocol_component("empirical_validation_claim")
