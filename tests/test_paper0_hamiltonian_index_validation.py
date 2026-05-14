# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 Hamiltonian index fixture tests
"""Tests for Paper 0 Appendix C Hamiltonian/operator index fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.hamiltonian_index_validation import (
    HamiltonianIndexConfig,
    classify_operator_layer,
    operator_catalogue,
    validate_hamiltonian_index_fixture,
    validate_operator_locations,
)


def test_operator_catalogue_preserves_source_symbols_and_locations() -> None:
    catalogue = operator_catalogue()

    assert tuple(entry.symbol for entry in catalogue) == (
        "L_Anulum",
        "H_MT",
        "H_PQT",
        "H_iso",
        "H_NI",
        "H_syn",
        "H_RP",
        "R_Psi",
        "O_sem",
    )
    assert classify_operator_layer("H_MT") == "microscopic_layer1"
    assert classify_operator_layer("R_Psi") == "informational_operator"
    assert validate_operator_locations(catalogue) is True


def test_hamiltonian_index_guards_reject_unknown_or_unlocated_entries() -> None:
    with pytest.raises(ValueError, match="operator symbol is not in Appendix C catalogue"):
        classify_operator_layer("H_missing")
    with pytest.raises(ValueError, match="expected_operator_count must be at least 1"):
        HamiltonianIndexConfig(expected_operator_count=0)

    catalogue = list(operator_catalogue())
    broken = catalogue.copy()
    broken[0] = broken[0].__class__(
        symbol=broken[0].symbol,
        label=broken[0].label,
        layer_group=broken[0].layer_group,
        equation=broken[0].equation,
        source_ledger_id=broken[0].source_ledger_id,
        location="",
    )
    with pytest.raises(ValueError, match="operator location must be non-empty"):
        validate_operator_locations(tuple(broken))


def test_hamiltonian_index_fixture_is_a_bounded_index_not_validation() -> None:
    result = validate_hamiltonian_index_fixture()

    assert result.hardware_status == "operator_index_no_execution"
    assert result.source_ledger_span == ("P0R06878", "P0R06915")
    assert result.operator_count == 9
    assert result.expected_operator_count == 9
    assert result.location_coverage_valid is True
    assert result.layer_groups == (
        "fundamental_meta_universal",
        "microscopic_layer1",
        "mesoscopic_layers2_4",
        "macroscopic_layers6_8",
        "informational_operator",
    )
    assert result.null_controls["unknown_operator_rejection_label"] == 1.0
    assert result.null_controls["missing_location_rejection_label"] == 1.0
    assert result.null_controls["unsupported_executed_validation_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
