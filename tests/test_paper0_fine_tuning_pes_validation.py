# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 fine-tuning PES fixture tests
"""Tests for Paper 0 fine-tuning PES simulator fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.fine_tuning_pes_validation import (
    FineTuningPESConfig,
    pes_selection_probability,
    protocol_catalogue_completeness,
    validate_fine_tuning_pes_fixture,
)


def test_pes_selection_probability_matches_source_proportionality() -> None:
    probability = pes_selection_probability(
        observed_sec=0.81,
        candidate_sec_values=(0.27, 0.54, 0.81),
    )

    assert probability == pytest.approx(1.0)


def test_pes_selection_probability_rejects_invalid_sec_values() -> None:
    with pytest.raises(ValueError, match="candidate SEC values must be finite and positive"):
        pes_selection_probability(observed_sec=0.5, candidate_sec_values=(0.0, 0.5))
    with pytest.raises(ValueError, match="observed SEC must be finite and positive"):
        pes_selection_probability(observed_sec=-0.1, candidate_sec_values=(0.2, 0.5))
    with pytest.raises(ValueError, match="candidate SEC values must not be empty"):
        pes_selection_probability(observed_sec=0.5, candidate_sec_values=())
    with pytest.raises(ValueError, match="protocol_threshold must be finite and positive"):
        FineTuningPESConfig(protocol_threshold=0.0)


def test_protocol_catalogue_completeness_requires_all_source_protocols() -> None:
    complete = protocol_catalogue_completeness(
        pta=True,
        awva=True,
        qrng_tsvf=True,
        cef_rg=True,
    )
    partial = protocol_catalogue_completeness(
        pta=True,
        awva=False,
        qrng_tsvf=True,
        cef_rg=False,
    )

    assert complete == pytest.approx(1.0)
    assert partial == pytest.approx(0.5)


def test_fine_tuning_pes_fixture_preserves_protocol_boundaries() -> None:
    result = validate_fine_tuning_pes_fixture()

    assert result.spec_keys == (
        "fine_tuning_pes.selection_formula_boundary",
        "fine_tuning_pes.advanced_protocol_catalogue",
        "fine_tuning_pes.protocol_falsification_boundaries",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06378", "P0R06381")
    assert result.selection_probability > result.config_thresholds["selection_threshold"]
    assert result.protocol_completeness == pytest.approx(1.0)
    assert result.null_controls["missing_protocol_rejection_label"] == 1.0
    assert result.null_controls["unsupported_empirical_evidence_rejection_label"] == 1.0
    assert result.null_controls["nonpositive_sec_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
