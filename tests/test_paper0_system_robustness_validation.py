# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 system-robustness validation tests
"""Executable fixture tests for Paper 0 system-robustness records."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.paper0.system_robustness_validation import (
    SystemRobustnessConfig,
    critical_recovery_time,
    largest_component_fraction,
    ms_qec_success_probability,
    validate_cascading_failure_percolation_fixture,
    validate_critical_slowing_recovery_fixture,
    validate_decoherence_attack_ms_qec_boundary_fixture,
    validate_system_robustness_fixture,
)


def test_cascading_failure_percolation_detects_fragmentation() -> None:
    config = SystemRobustnessConfig(
        coupling_matrix=np.array(
            [
                [0.0, 0.9, 0.8, 0.1],
                [0.9, 0.0, 0.7, 0.1],
                [0.8, 0.7, 0.0, 0.1],
                [0.1, 0.1, 0.1, 0.0],
            ],
            dtype=np.float64,
        ),
        percolation_threshold=0.5,
    )

    intact = largest_component_fraction(config.coupling_matrix, threshold=0.5)
    attacked = largest_component_fraction(
        config.coupling_matrix,
        threshold=0.5,
        removed_nodes=(0,),
    )
    result = validate_cascading_failure_percolation_fixture(config)

    assert intact == pytest.approx(0.75)
    assert attacked == pytest.approx(0.5)
    assert result.largest_component_loss > 0.0
    assert result.null_controls["empty_graph_fragmentation_label"] == pytest.approx(1.0)
    assert "not operational security evidence" in result.claim_boundary


def test_critical_slowing_recovery_time_diverges_near_sigma_one() -> None:
    config = SystemRobustnessConfig(sigma=1.04, reference_sigma=1.4)

    near = critical_recovery_time(config.sigma)
    far = critical_recovery_time(config.reference_sigma)
    result = validate_critical_slowing_recovery_fixture(config)

    assert near > far
    assert result.recovery_time_ratio > 1.0
    assert result.null_controls["far_from_transition_ratio"] < 1.0
    assert result.null_controls["critical_point_rejection_label"] == pytest.approx(1.0)


def test_decoherence_attack_boundary_tracks_ms_qec_success() -> None:
    config = SystemRobustnessConfig(
        decoherence_exposure=0.6,
        ms_qec_redundancy=4,
        qec_correction_strength=0.72,
    )

    protected = ms_qec_success_probability(
        exposure=config.decoherence_exposure,
        redundancy=config.ms_qec_redundancy,
        correction_strength=config.qec_correction_strength,
    )
    unprotected = ms_qec_success_probability(
        exposure=config.decoherence_exposure,
        redundancy=1,
        correction_strength=0.0,
    )
    result = validate_decoherence_attack_ms_qec_boundary_fixture(config)

    assert protected > unprotected
    assert 0.0 < result.ms_qec_success_probability <= 1.0
    assert result.failure_probability < result.unprotected_failure_probability
    assert result.null_controls["zero_redundancy_rejection_label"] == pytest.approx(1.0)
    assert result.null_controls["out_of_range_correction_rejection_label"] == pytest.approx(1.0)


def test_system_robustness_fixture_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="symmetric"):
        SystemRobustnessConfig(coupling_matrix=np.array([[0.0, 1.0], [0.2, 0.0]]))

    with pytest.raises(ValueError, match="positive"):
        SystemRobustnessConfig(sigma=0.0)

    with pytest.raises(ValueError, match="unit interval"):
        SystemRobustnessConfig(qec_correction_strength=1.2)

    with pytest.raises(ValueError, match="redundancy"):
        SystemRobustnessConfig(ms_qec_redundancy=0)


def test_system_robustness_default_fixture_wires_all_boundaries() -> None:
    result = validate_system_robustness_fixture()

    assert result.hardware_status == "simulator_only_no_provider_submission"
    assert result.spec_keys == (
        "applied.system_robustness.cascading_failure_percolation",
        "applied.system_robustness.critical_slowing_recovery",
        "applied.system_robustness.decoherence_attack_ms_qec_boundary",
    )
    assert result.cascade.largest_component_loss >= 0.0
    assert result.critical_slowing.recovery_time_ratio > 1.0
    assert result.decoherence.failure_probability < 1.0
    assert "not operational security evidence" in result.claim_boundary
