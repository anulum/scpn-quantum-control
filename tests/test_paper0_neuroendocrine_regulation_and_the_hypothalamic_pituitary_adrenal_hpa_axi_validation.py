# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Neuroendocrine Regulation and the Hypothalamic-Pituitary-Adrenal (HPA) Axis validation tests
"""Tests for Paper 0 Neuroendocrine Regulation and the Hypothalamic-Pituitary-Adrenal (HPA) Axis source fixture."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_validation import (
    CLAIM_BOUNDARY,
    HARDWARE_STATUS,
    NeuroendocrineRegulationAndTheHypothalamicPituitaryAdrenalHpaAxiConfig,
    classify_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_component,
    neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_labels,
    validate_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_fixture,
)


def test_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_fixture_preserves_source_boundary() -> (
    None
):
    result = (
        validate_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_fixture()
    )
    assert result.source_ledger_span == ("P0R05445", "P0R05454")
    assert result.source_record_count == 10
    assert result.component_count == 1
    assert result.next_source_boundary == "P0R05455"
    assert result.hardware_status == HARDWARE_STATUS
    assert result.claim_boundary == CLAIM_BOUNDARY
    assert (
        result.problem_metadata["protocol_state"]
        == "source_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_only_no_experiment"
    )
    assert tuple(result.problem_metadata["source_ledger_ids"])[0] == "P0R05445"
    assert tuple(result.problem_metadata["source_ledger_ids"])[-1] == "P0R05454"


def test_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_classification_and_labels_are_explicit() -> (
    None
):
    for component in ("neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi",):
        assert (
            classify_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_component(
                component
            )
            == f"{component}_source_boundary"
        )
    labels = neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_labels()
    assert (
        labels["section"]
        == "Neuroendocrine Regulation and the Hypothalamic-Pituitary-Adrenal (HPA) Axis"
    )
    assert labels["next_boundary"] == "P0R05455"


def test_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_rejects_invalid_configuration() -> (
    None
):
    with pytest.raises(ValueError, match="expected_source_record_count must equal 10"):
        NeuroendocrineRegulationAndTheHypothalamicPituitaryAdrenalHpaAxiConfig(
            expected_source_record_count=9
        )
    with pytest.raises(ValueError, match="expected_component_count must equal 1"):
        NeuroendocrineRegulationAndTheHypothalamicPituitaryAdrenalHpaAxiConfig(
            expected_component_count=2
        )
    with pytest.raises(ValueError, match="next_source_boundary must equal P0R05455"):
        NeuroendocrineRegulationAndTheHypothalamicPituitaryAdrenalHpaAxiConfig(
            next_source_boundary="P0R05454"
        )
    with pytest.raises(
        ValueError,
        match="unknown neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi component",
    ):
        classify_neuroendocrine_regulation_and_the_hypothalamic_pituitary_adrenal_hpa_axi_component(
            "empirical_validation_claim"
        )
