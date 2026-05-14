# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 RAG QEC stack fixture tests
"""Tests for Paper 0 RAG Layer 1 QEC stack fixtures."""

from __future__ import annotations

import pytest

from scpn_quantum_control.paper0.rag_qec_stack_validation import (
    RAGQECStackConfig,
    coherence_time_fs,
    error_threshold_source_formula,
    protection_factor,
    qec_hamiltonian_total,
    validate_rag_qec_stack_fixture,
)


def test_qec_hamiltonian_total_preserves_source_sum() -> None:
    total = qec_hamiltonian_total(
        microtubule_lattice=-1.2,
        stabilisers=-0.4,
        syndrome=-0.2,
    )

    assert total == pytest.approx(-1.8)


def test_gap_coherence_and_protection_factor_match_source_values() -> None:
    cfg = RAGQECStackConfig(delta_e_ev=1.64, physiological_kbt_ev=0.026)

    assert coherence_time_fs(delta_e_ev=cfg.delta_e_ev) == pytest.approx(0.401, rel=0.02)
    assert coherence_time_fs(delta_e_ev=cfg.physiological_kbt_ev) == pytest.approx(25.31, rel=0.02)
    assert protection_factor(
        protected_time_fs=400.0,
        unprotected_time_fs=25.0,
    ) == pytest.approx(16.0)


def test_error_threshold_formula_flags_source_approximation_mismatch() -> None:
    result = error_threshold_source_formula(delta_e_ev=1.64, physiological_kbt_ev=0.026)

    assert result.formula_value > 0.999
    assert result.source_approximation == pytest.approx(1e-14)
    assert result.source_consistency_warning is True


def test_rag_qec_stack_fixture_preserves_boundaries_and_controls() -> None:
    with pytest.raises(ValueError, match="delta_e_ev must be finite and positive"):
        RAGQECStackConfig(delta_e_ev=0.0)
    with pytest.raises(ValueError, match="Hamiltonian components must be finite"):
        qec_hamiltonian_total(microtubule_lattice=float("nan"), stabilisers=-0.4, syndrome=-0.2)

    result = validate_rag_qec_stack_fixture()

    assert result.spec_keys == (
        "rag_qec_stack.insert_framing",
        "rag_qec_stack.layer1_qec_hamiltonian",
        "rag_qec_stack.gap_coherence_protection",
        "rag_qec_stack.programmability_and_observable",
    )
    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.source_ledger_span == ("P0R06530", "P0R06559")
    assert result.gap_ratio > 60.0
    assert result.protection_factor == pytest.approx(16.0)
    assert result.error_threshold_source_warning is True
    assert result.null_controls["non_positive_gap_rejection_label"] == 1.0
    assert result.null_controls["threshold_approximation_warning_label"] == 1.0
    assert result.null_controls["unsupported_spectroscopy_evidence_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
