# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 grand-synthesis validation tests
"""Executable fixture tests for Paper 0 Grand Synthesis NTHS phase records."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.grand_synthesis_validation import (
    GrandSynthesisConfig,
    compute_phase_metrics,
    engagement_surprise_coupling,
    free_energy_minimising_coupling,
    validate_grand_synthesis_fixture,
    validate_nths_phase_test_fixture,
)


def test_nths_policy_couplings_are_finite_symmetric_and_distinct() -> None:
    config = GrandSynthesisConfig(agent_count=6)

    coherent = free_energy_minimising_coupling(config)
    engagement = engagement_surprise_coupling(config)

    assert coherent.shape == (6, 6)
    assert engagement.shape == (6, 6)
    assert np.allclose(coherent, coherent.T)
    assert np.allclose(engagement, engagement.T)
    assert np.allclose(np.diag(coherent), 0.0)
    assert np.allclose(np.diag(engagement), 0.0)
    assert np.all(np.isfinite(coherent))
    assert np.all(np.isfinite(engagement))
    assert not np.allclose(coherent, engagement)


def test_nths_phase_metrics_separate_coherence_and_engagement_regimes() -> None:
    config = GrandSynthesisConfig(agent_count=6)

    coherent = compute_phase_metrics(free_energy_minimising_coupling(config), config)
    engagement = compute_phase_metrics(engagement_surprise_coupling(config), config)
    result = validate_nths_phase_test_fixture(config)

    assert coherent.frustration_index < engagement.frustration_index
    assert coherent.consensus_order > engagement.consensus_order
    assert coherent.sec_proxy > engagement.sec_proxy
    assert result.engagement_spin_glass_label is True
    assert result.coherence_consensus_label is True
    assert result.sec_delta > 0.0
    assert "not empirical evidence" in result.claim_boundary


def test_grand_synthesis_fixture_rejects_missing_or_invalid_controls() -> None:
    with pytest.raises(ValueError, match="at least four agents"):
        GrandSynthesisConfig(agent_count=3)

    with pytest.raises(ValueError, match="finite and positive"):
        GrandSynthesisConfig(coherence_gain=0.0)

    with pytest.raises(ValueError, match="cluster labels"):
        GrandSynthesisConfig(agent_count=6, cluster_labels=(0, 0, 1))

    with pytest.raises(ValueError, match="at least two clusters"):
        GrandSynthesisConfig(agent_count=6, cluster_labels=(0, 0, 0, 0, 0, 0))


def test_grand_synthesis_default_fixture_wires_claims_mechanism_and_phase_test() -> None:
    result = validate_grand_synthesis_fixture()

    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.spec_keys == (
        "grand_synthesis.anulum_claim_boundary",
        "grand_synthesis.architecture_mechanism_map",
        "grand_synthesis.nths_phase_test",
        "grand_synthesis.figure_caption_boundary",
    )
    assert result.nths_phase.engagement_spin_glass_label is True
    assert result.nths_phase.coherence_consensus_label is True
    assert result.nths_phase.null_controls[
        "missing_policy_regime_rejection_label"
    ] == pytest.approx(1.0)
    assert result.nths_phase.null_controls[
        "missing_adaptive_coupling_rejection_label"
    ] == pytest.approx(1.0)
    assert "not empirical evidence" in result.claim_boundary
