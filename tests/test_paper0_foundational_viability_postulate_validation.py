# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 foundational viability postulate fixtures
"""Tests for Paper 0 foundational viability and Psi-field postulate fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.foundational_viability_postulate_validation import (
    FoundationalViabilityPostulateConfig,
    classify_physical_postulate_component,
    classify_viability_pillar,
    psi_field_quantum_numbers,
    validate_foundational_viability_postulate_fixture,
)


def test_viability_pillar_classifier_preserves_three_pillars() -> None:
    assert classify_viability_pillar("ontological_postulate") == "psi_field_primitive_ontology"
    assert classify_viability_pillar("derived_interactions") == "u1_fim_interaction_derivation"
    assert (
        classify_viability_pillar("multiscale_architecture")
        == "hierarchy_rg_bidirectional_causality"
    )

    with pytest.raises(ValueError, match="unknown foundational viability pillar"):
        classify_viability_pillar("extra_pillar")


def test_physical_postulate_classifier_preserves_field_theory_components() -> None:
    assert classify_physical_postulate_component("complex_scalar") == "standard_qft_scalar_field"
    assert classify_physical_postulate_component("spin") == "spin_0_bosonic_quanta"
    assert classify_physical_postulate_component("phase_symmetry") == "global_u1_phase_symmetry"
    assert (
        classify_physical_postulate_component("fim_coupling") == "informational_geometry_coupling"
    )

    with pytest.raises(ValueError, match="unknown physical postulate component"):
        classify_physical_postulate_component("untracked_force")


def test_psi_field_quantum_numbers_preserve_source_formalisation() -> None:
    numbers = psi_field_quantum_numbers()

    assert numbers["spin"] == "0"
    assert numbers["statistics"] == "bosonic"
    assert numbers["symmetry"] == "global U(1) phase"
    assert numbers["decomposition"] == "Psi = |Psi| e^{i theta}"


def test_foundational_viability_postulate_fixture_preserves_scope_counts_and_boundaries() -> None:
    result = validate_foundational_viability_postulate_fixture()

    assert result.hardware_status == "source_methodology_no_experiment"
    assert result.source_ledger_span == ("P0R00464", "P0R00505")
    assert result.pillar_count == 3
    assert result.physics_postulate_count == 4
    assert result.next_source_boundary == "P0R00506"
    assert result.coupling_equation == "H_int = -lambda * Psi_s * sigma"
    assert result.null_controls["internal_consistency_is_not_empirical_validation"] == 1.0
    assert result.null_controls["gauge_derivation_requires_later_section_boundary"] == 1.0

    with pytest.raises(ValueError, match="next_source_boundary must equal P0R00506"):
        FoundationalViabilityPostulateConfig(next_source_boundary="P0R00505")
