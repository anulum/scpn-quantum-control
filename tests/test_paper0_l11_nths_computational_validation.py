# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 L11 NTHS computational fixture tests
"""Tests for Paper 0 L11 NTHS computational experiment fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.l11_nths_computational_validation import (
    L11NTHSComputationalConfig,
    anova_divergence_decision,
    expected_free_energy_score,
    noosphere_hamiltonian,
    order_parameter_summary,
    validate_l11_nths_computational_fixture,
)


def test_noosphere_hamiltonian_and_order_parameters_match_source_forms() -> None:
    spins = np.array([1, 1, -1, -1], dtype=np.int8)
    coupling = np.array(
        [
            [0.0, 0.8, -0.4, -0.3],
            [0.8, 0.0, -0.2, -0.5],
            [-0.4, -0.2, 0.0, 0.7],
            [-0.3, -0.5, 0.7, 0.0],
        ],
        dtype=np.float64,
    )

    energy = noosphere_hamiltonian(spins=spins, coupling_matrix=coupling)
    summary = order_parameter_summary(np.vstack([spins, spins, -spins]))

    assert energy == pytest.approx(-2.9)
    assert abs(summary.magnetization) <= 1.0
    assert 0.0 <= summary.edwards_anderson_q <= 1.0


def test_expected_free_energy_score_is_lower_for_better_policy() -> None:
    exploit = expected_free_energy_score(
        posterior_entropy=0.2,
        expected_surprise=0.1,
        preference_alignment=0.8,
    )
    polarizing = expected_free_energy_score(
        posterior_entropy=0.7,
        expected_surprise=0.8,
        preference_alignment=0.1,
    )

    assert exploit < polarizing


def test_l11_nths_falsification_and_fixture_boundary() -> None:
    assert anova_divergence_decision(p_value=0.0005, cohen_d=2.4) is True
    assert anova_divergence_decision(p_value=0.01, cohen_d=2.4) is False
    assert anova_divergence_decision(p_value=0.0005, cohen_d=1.0) is False

    with pytest.raises(ValueError, match="agent_count must be at least 1"):
        L11NTHSComputationalConfig(agent_count=0)
    with pytest.raises(ValueError, match="coupling_matrix must be square"):
        noosphere_hamiltonian(
            spins=np.array([1, -1], dtype=np.int8),
            coupling_matrix=np.ones((2, 3), dtype=np.float64),
        )
    with pytest.raises(ValueError, match="replicas must contain only -1 or \\+1"):
        order_parameter_summary(np.array([[0, 1]], dtype=np.int8))

    result = validate_l11_nths_computational_fixture()

    assert result.spec_keys == (
        "l11_nths_computational.block_framing",
        "l11_nths_computational.agent_architecture",
        "l11_nths_computational.environment_spin_glass",
        "l11_nths_computational.ai_objective_conditions",
        "l11_nths_computational.simulation_protocol",
        "l11_nths_computational.order_parameters",
        "l11_nths_computational.predicted_outcomes",
        "l11_nths_computational.statistics_falsification_extensions",
    )
    assert result.hardware_status == "computational_protocol_no_external_execution"
    assert result.source_ledger_span == ("P0R06730", "P0R06814")
    assert result.agent_count == 1000
    assert result.initial_topology == "barabasi_albert_m3"
    assert result.control_magnetization_abs > result.experimental_magnetization_abs
    assert result.experimental_edwards_anderson_q > 0.0
    assert result.experimental_noosphere_energy > result.control_noosphere_energy
    assert result.significant_divergence is True
    assert result.null_controls["invalid_agent_count_rejection_label"] == 1.0
    assert result.null_controls["invalid_spin_rejection_label"] == 1.0
    assert result.null_controls["unsupported_external_execution_claim_rejection_label"] == 1.0
    assert "not empirical evidence" in result.claim_boundary
