# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom III SEC-Ntilde equivalence validation tests
"""Tests for Paper 0 Axiom III SEC/Ntilde equivalence validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiom_iii_sec_ntilde_equivalence_validation import (
    AxiomIIISECNtildeEquivalenceConfig,
    axiom_iii_sec_ntilde_equivalence_labels,
    classify_sec_ntilde_equivalence_component,
    validate_axiom_iii_sec_ntilde_equivalence_fixture,
)


def test_sec_ntilde_equivalence_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 7"):
        AxiomIIISECNtildeEquivalenceConfig(expected_source_record_count=6)

    with pytest.raises(ValueError, match="expected_blank_terminal_record_count must equal 1"):
        AxiomIIISECNtildeEquivalenceConfig(expected_blank_terminal_record_count=0)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00818"):
        AxiomIIISECNtildeEquivalenceConfig(next_source_boundary="P0R00817")


def test_sec_ntilde_equivalence_classifiers_are_source_bounded() -> None:
    assert (
        classify_sec_ntilde_equivalence_component("equivalence_heading")
        == "sec_ntilde_unity_equivalence_subsection"
    )
    assert (
        classify_sec_ntilde_equivalence_component("macroscopic_realisation")
        == "sec_as_macroscopic_realisation_of_ntilde_unity"
    )
    assert (
        classify_sec_ntilde_equivalence_component("quasicritical_efficiency")
        == "ntilde_unity_as_quasicritical_efficiency_target"
    )
    assert (
        classify_sec_ntilde_equivalence_component("causal_imperative_architecture")
        == "causal_imperative_and_15_layer_locking_architecture"
    )

    with pytest.raises(ValueError, match="unknown SEC-Ntilde-equivalence component"):
        classify_sec_ntilde_equivalence_component("tripartite_ontology")


def test_sec_ntilde_equivalence_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_axiom_iii_sec_ntilde_equivalence_fixture()

    assert result.source_ledger_span == ("P0R00811", "P0R00817")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 7
    assert result.equivalence_claim_count == 2
    assert result.architecture_target_count == 2
    assert result.efficiency_claim_count == 2
    assert result.blank_terminal_record_count == 1
    assert result.next_source_boundary == "P0R00818"
    assert result.null_controls == {
        "sec_ntilde_equivalence_is_source_claim_not_empirical_evidence": 1.0,
        "truncated_p0r00816_requires_source_audit_before_completion_claim": 1.0,
        "blank_p0r00817_is_preserved_not_silently_omitted": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == (
        "source_axiom_iii_sec_ntilde_equivalence_only_no_experiment"
    )
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00811"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00817"


def test_sec_ntilde_equivalence_labels_name_tripartite_boundary() -> None:
    labels = axiom_iii_sec_ntilde_equivalence_labels()

    assert labels["section"] == "Equivalence of SEC and the tilde_N_t = 1 State"
    assert labels["equivalence"] == "SEC is macroscopic physical realisation of tilde_N_t = 1"
    assert labels["source_integrity"] == "P0R00816 truncated; P0R00817 blank"
    assert labels["next_boundary"] == "1.4 Tripartite Ontology: The Substance of Information"
