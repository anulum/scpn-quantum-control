# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. Ethics as Causal Entropic Forces (CEF): validation tests
"""Tests for Paper 0 IV. Ethics as Causal Entropic Forces (CEF): source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.iv_ethics_as_causal_entropic_forces_cef_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IvEthicsAsCausalEntropicForcesCefConfig,
    classify_iv_ethics_as_causal_entropic_forces_cef_component,
    iv_ethics_as_causal_entropic_forces_cef_labels,
    validate_iv_ethics_as_causal_entropic_forces_cef_fixture,
)


def test_iv_ethics_as_causal_entropic_forces_cef_fixture_preserves_source_boundary() -> None:
    result = validate_iv_ethics_as_causal_entropic_forces_cef_fixture()
    assert result.source_ledger_span == ("P0R06107", "P0R06114")
    assert result.source_record_count == 8
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R06115"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_iv_ethics_as_causal_entropic_forces_cef_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06107"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06114"


def test_iv_ethics_as_causal_entropic_forces_cef_classification_and_labels_are_explicit() -> None:
    for component in (
        "iv_ethics_as_causal_entropic_forces_cef",
        "overarching_principles_and_system_dynamics_in_short",
    ):
        assert (
            classify_iv_ethics_as_causal_entropic_forces_cef_component(component)
            == f"{component}_source_boundary"
        )
    labels = iv_ethics_as_causal_entropic_forces_cef_labels()
    assert labels["section"] == "IV. Ethics as Causal Entropic Forces (CEF):"
    assert labels["next_boundary"] == "P0R06115"


def test_iv_ethics_as_causal_entropic_forces_cef_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IvEthicsAsCausalEntropicForcesCefConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        IvEthicsAsCausalEntropicForcesCefConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R06115"):
        IvEthicsAsCausalEntropicForcesCefConfig(next_source_boundary="P0R06114")
    with pytest.raises(
        ValueError, match="unknown iv_ethics_as_causal_entropic_forces_cef component"
    ):
        classify_iv_ethics_as_causal_entropic_forces_cef_component("empirical_validation_claim")
