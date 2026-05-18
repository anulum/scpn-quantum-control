# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 IV. The Neuro-Immuno-Endocrine (NIE) Super-System validation tests
"""Tests for Paper 0 IV. The Neuro-Immuno-Endocrine (NIE) Super-System source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.iv_the_neuro_immuno_endocrine_nie_super_system_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IvTheNeuroImmunoEndocrineNieSuperSystemConfig,
    classify_iv_the_neuro_immuno_endocrine_nie_super_system_component,
    iv_the_neuro_immuno_endocrine_nie_super_system_labels,
    validate_iv_the_neuro_immuno_endocrine_nie_super_system_fixture,
)


def test_iv_the_neuro_immuno_endocrine_nie_super_system_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_iv_the_neuro_immuno_endocrine_nie_super_system_fixture()
    assert result.source_ledger_span == ("P0R04921", "P0R04934")
    assert result.source_record_count == 14
    assert result.component_count == 2
    assert result.next_source_boundary == "P0R04935"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_iv_the_neuro_immuno_endocrine_nie_super_system_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R04921"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R04934"


def test_iv_the_neuro_immuno_endocrine_nie_super_system_classification_and_labels_are_explicit() -> (
    None
):
    for component in (
        "iv_the_neuro_immuno_endocrine_nie_super_system",
        "1_the_psychoneuroimmunology_pni_axis_and_inflammation_the_decoherence_fi",
    ):
        assert (
            classify_iv_the_neuro_immuno_endocrine_nie_super_system_component(component)
            == f"{component}_source_boundary"
        )
    labels = iv_the_neuro_immuno_endocrine_nie_super_system_labels()
    assert labels["section"] == "IV. The Neuro-Immuno-Endocrine (NIE) Super-System"
    assert labels["next_boundary"] == "P0R04935"


def test_iv_the_neuro_immuno_endocrine_nie_super_system_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 14"):
        IvTheNeuroImmunoEndocrineNieSuperSystemConfig(expected_source_record_count=13)
    with pytest.raises(ValueError, match="expected_component_count must equal 2"):
        IvTheNeuroImmunoEndocrineNieSuperSystemConfig(expected_component_count=3)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R04935"):
        IvTheNeuroImmunoEndocrineNieSuperSystemConfig(next_source_boundary="P0R04934")
    with pytest.raises(
        ValueError, match="unknown iv_the_neuro_immuno_endocrine_nie_super_system component"
    ):
        classify_iv_the_neuro_immuno_endocrine_nie_super_system_component(
            "empirical_validation_claim"
        )
