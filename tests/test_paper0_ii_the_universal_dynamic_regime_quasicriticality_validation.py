# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 II. The Universal Dynamic Regime: Quasicriticality validation tests
"""Tests for Paper 0 II. The Universal Dynamic Regime: Quasicriticality source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.ii_the_universal_dynamic_regime_quasicriticality_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    IiTheUniversalDynamicRegimeQuasicriticalityConfig,
    classify_ii_the_universal_dynamic_regime_quasicriticality_component,
    ii_the_universal_dynamic_regime_quasicriticality_labels,
    validate_ii_the_universal_dynamic_regime_quasicriticality_fixture,
)


def test_ii_the_universal_dynamic_regime_quasicriticality_fixture_preserves_source_boundary() -> (
    None
):
    result = validate_ii_the_universal_dynamic_regime_quasicriticality_fixture()
    assert result.source_ledger_span == ("P0R02513", "P0R02520")
    assert result.source_record_count == 8
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R02521"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_ii_the_universal_dynamic_regime_quasicriticality_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R02513"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R02520"


def test_ii_the_universal_dynamic_regime_quasicriticality_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("ii_the_universal_dynamic_regime_quasicriticality",):
        assert (
            classify_ii_the_universal_dynamic_regime_quasicriticality_component(component)
            == f"{component}_source_boundary"
        )
    labels = ii_the_universal_dynamic_regime_quasicriticality_labels()
    assert labels["section"] == "II. The Universal Dynamic Regime: Quasicriticality"
    assert labels["next_boundary"] == "P0R02521"


def test_ii_the_universal_dynamic_regime_quasicriticality_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 8"):
        IiTheUniversalDynamicRegimeQuasicriticalityConfig(expected_source_record_count=7)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        IiTheUniversalDynamicRegimeQuasicriticalityConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R02521"):
        IiTheUniversalDynamicRegimeQuasicriticalityConfig(next_source_boundary="P0R02520")
    with pytest.raises(
        ValueError, match="unknown ii_the_universal_dynamic_regime_quasicriticality component"
    ):
        classify_ii_the_universal_dynamic_regime_quasicriticality_component(
            "empirical_validation_claim"
        )
