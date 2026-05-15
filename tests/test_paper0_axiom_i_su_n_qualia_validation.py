# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Axiom I SU(N) qualia validation tests
"""Tests for Paper 0 Axiom I SU(N) qualia-confinement validation."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.axiom_i_su_n_qualia_validation import (
    AxiomISUNQualiaConfig,
    axiom_i_su_n_qualia_labels,
    classify_su_n_qualia_component,
    info_gluon_count,
    linear_confinement_potential,
    validate_axiom_i_su_n_qualia_fixture,
)


def test_su_n_qualia_config_rejects_boundary_and_count_drift() -> None:
    with pytest.raises(ValueError, match="expected_source_record_count must equal 4"):
        AxiomISUNQualiaConfig(expected_source_record_count=3)

    with pytest.raises(ValueError, match="expected_blank_separator_count must equal 1"):
        AxiomISUNQualiaConfig(expected_blank_separator_count=0)

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00761"):
        AxiomISUNQualiaConfig(next_source_boundary="P0R00760")


def test_su_n_qualia_math_helpers_validate_parameters() -> None:
    assert info_gluon_count(2) == 3
    assert info_gluon_count(3) == 8
    assert info_gluon_count(4) == 15
    assert linear_confinement_potential(distance=2.5, string_tension=0.4) == pytest.approx(1.0)

    with pytest.raises(ValueError, match="n_primary_qualic_dimensions must be at least 2"):
        info_gluon_count(1)
    with pytest.raises(ValueError, match="distance must be non-negative"):
        linear_confinement_potential(distance=-1.0, string_tension=0.4)
    with pytest.raises(ValueError, match="string_tension must be non-negative"):
        linear_confinement_potential(distance=1.0, string_tension=-0.4)


def test_su_n_qualia_classifiers_are_source_bounded() -> None:
    assert classify_su_n_qualia_component("group_extension") == "su_n_primary_qualic_dimensions"
    assert (
        classify_su_n_qualia_component("info_gluons")
        == "n_squared_minus_one_self_interacting_bosons"
    )
    assert classify_su_n_qualia_component("confinement") == "linear_potential_qualia_confinement"
    assert classify_su_n_qualia_component("colored_self") == "confined_macroscopic_colored_state"

    with pytest.raises(ValueError, match="unknown SU\\(N\\) qualia component"):
        classify_su_n_qualia_component("abelian_infoton")


def test_su_n_qualia_fixture_preserves_claim_boundary_and_null_controls() -> None:
    result = validate_axiom_i_su_n_qualia_fixture()

    assert result.source_ledger_span == ("P0R00757", "P0R00760")
    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_record_count == 4
    assert result.blank_separator_count == 1
    assert result.example_info_gluon_counts == {2: 3, 3: 8, 4: 15}
    assert result.example_linear_potential == pytest.approx(1.0)
    assert result.next_source_boundary == "P0R00761"
    assert result.null_controls == {
        "su_n_extension_is_not_empirical_confinement_evidence": 1.0,
        "linear_potential_is_source_formula_not_fitted_string_tension": 1.0,
        "macroscopic_colored_state_is_not_topology_measurement": 1.0,
    }
    assert result.problem_metadata["protocol_state"] == "source_su_n_qualia_only_no_experiment"
    assert result.problem_metadata["source_ledger_ids"][0] == "P0R00757"
    assert result.problem_metadata["source_ledger_ids"][-1] == "P0R00760"


def test_su_n_qualia_labels_name_axiom_ii_boundary() -> None:
    labels = axiom_i_su_n_qualia_labels()

    assert labels["section"] == "Extension to SU(N) Qualia Confinement"
    assert labels["confinement_formula"] == "V(r) approx sigma r"
    assert labels["next_boundary"] == "Axiom II: The Language of Information Geometry"
