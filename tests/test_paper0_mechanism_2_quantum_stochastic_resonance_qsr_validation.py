# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Mechanism 2: Quantum Stochastic Resonance (QSR) validation tests
"""Tests for Paper 0 Mechanism 2: Quantum Stochastic Resonance (QSR) source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.mechanism_2_quantum_stochastic_resonance_qsr_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    Mechanism2QuantumStochasticResonanceQsrConfig,
    classify_mechanism_2_quantum_stochastic_resonance_qsr_component,
    mechanism_2_quantum_stochastic_resonance_qsr_labels,
    validate_mechanism_2_quantum_stochastic_resonance_qsr_fixture,
)


def test_mechanism_2_quantum_stochastic_resonance_qsr_fixture_preserves_source_boundary() -> None:
    result = validate_mechanism_2_quantum_stochastic_resonance_qsr_fixture()
    assert result.source_ledger_span == ("P0R03343", "P0R03359")
    assert result.source_record_count == 17
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R03360"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_mechanism_2_quantum_stochastic_resonance_qsr_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R03343"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R03359"


def test_mechanism_2_quantum_stochastic_resonance_qsr_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("mechanism_2_quantum_stochastic_resonance_qsr",):
        assert (
            classify_mechanism_2_quantum_stochastic_resonance_qsr_component(component)
            == f"{component}_source_boundary"
        )
    labels = mechanism_2_quantum_stochastic_resonance_qsr_labels()
    assert labels["section"] == "Mechanism 2: Quantum Stochastic Resonance (QSR)"
    assert labels["next_boundary"] == "P0R03360"


def test_mechanism_2_quantum_stochastic_resonance_qsr_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 17"):
        Mechanism2QuantumStochasticResonanceQsrConfig(expected_source_record_count=16)
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        Mechanism2QuantumStochasticResonanceQsrConfig(expected_component_count=2)
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R03360"):
        Mechanism2QuantumStochasticResonanceQsrConfig(next_source_boundary="P0R03359")
    with pytest.raises(
        ValueError, match="unknown mechanism_2_quantum_stochastic_resonance_qsr component"
    ):
        classify_mechanism_2_quantum_stochastic_resonance_qsr_component(
            "empirical_validation_claim"
        )
