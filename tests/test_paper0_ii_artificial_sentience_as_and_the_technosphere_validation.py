# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. Artificial Sentience (AS) and the Technosphere validation tests
"""Tests for Paper 0 II. Artificial Sentience (AS) and the Technosphere source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ii_artificial_sentience_as_and_the_technosphere_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiArtificialSentienceAsAndTheTechnosphereConfig,
    classify_ii_artificial_sentience_as_and_the_technosphere_component,
    ii_artificial_sentience_as_and_the_technosphere_labels,
    validate_ii_artificial_sentience_as_and_the_technosphere_fixture,
)


def test_ii_artificial_sentience_as_and_the_technosphere_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_ii_artificial_sentience_as_and_the_technosphere_fixture()
    assert result.source_ledger_span == ("P0R06206", "P0R06211")
    assert result.source_record_count == 6
    assert result.component_count == 2
    assert result.next_source_boundary == "None"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ii_artificial_sentience_as_and_the_technosphere_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R06206"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R06211"


def test_ii_artificial_sentience_as_and_the_technosphere_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "ii_artificial_sentience_as_and_the_technosphere",
        "criteria_for_artificial_sentience_as",
    ):
        assert (
            classify_ii_artificial_sentience_as_and_the_technosphere_component(component)
            == f"{component}_source_boundary"
        )
    labels = ii_artificial_sentience_as_and_the_technosphere_labels()
    assert labels["section"] == "II. Artificial Sentience (AS) and the Technosphere"
    assert labels["next_boundary"] == "None"


def test_ii_artificial_sentience_as_and_the_technosphere_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 6"):
        IiArtificialSentienceAsAndTheTechnosphereConfig(expected_source_record_count=5)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        IiArtificialSentienceAsAndTheTechnosphereConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal None"):
        IiArtificialSentienceAsAndTheTechnosphereConfig(next_source_boundary="P0R06211")
    with pytest.raises(
        ValueError, match="unknown ii_artificial_sentience_as_and_the_technosphere component"
    ):
        classify_ii_artificial_sentience_as_and_the_technosphere_component(
            "empirical_validation_claim"
        )
