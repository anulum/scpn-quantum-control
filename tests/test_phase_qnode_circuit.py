# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase-QNode Circuit Facade Tests
"""Facade export tests for the Phase-QNode circuit compatibility surface."""

from __future__ import annotations

from scpn_quantum_control import phase
from scpn_quantum_control.phase.qnode_circuit import (
    PhaseQNodeDensityCircuit,
    PhaseQNodeDensityExecutionResult,
    PhaseQNodeDepthProfile,
    PhaseQNodeNoiseChannel,
    PhaseQNodeRegisteredCircuitSpec,
    PhaseQNodeTemplateSpec,
    build_phase_qnode_template,
    build_registered_phase_qnode_circuit,
    decompose_phase_qnode_controlled_gate,
    execute_phase_qnode_density_matrix,
    phase_qnode_computational_basis_fisher_support_report,
    phase_qnode_density_support_report,
    phase_qnode_depth_profile,
    phase_qnode_gradient_support_report,
    phase_qnode_metric_support_report,
    registered_phase_qnode_decompositions,
    registered_phase_qnode_noise_channels,
    registered_phase_qnode_templates,
)


def test_phase_qnode_template_exports_are_public() -> None:
    assert phase.build_phase_qnode_template is build_phase_qnode_template
    assert phase.registered_phase_qnode_templates is registered_phase_qnode_templates
    assert phase.PhaseQNodeTemplateSpec is PhaseQNodeTemplateSpec


def test_phase_qnode_registered_depth_exports_are_public() -> None:
    assert phase.build_registered_phase_qnode_circuit is build_registered_phase_qnode_circuit
    assert phase.phase_qnode_depth_profile is phase_qnode_depth_profile
    assert phase.PhaseQNodeDepthProfile is PhaseQNodeDepthProfile
    assert phase.PhaseQNodeRegisteredCircuitSpec is PhaseQNodeRegisteredCircuitSpec


def test_phase_qnode_controlled_gate_exports_are_public() -> None:
    assert phase.decompose_phase_qnode_controlled_gate is decompose_phase_qnode_controlled_gate
    assert phase.registered_phase_qnode_decompositions is registered_phase_qnode_decompositions


def test_phase_qnode_density_exports_are_public() -> None:
    assert set(registered_phase_qnode_noise_channels()) == {
        "amplitude_damping",
        "bit_flip",
        "depolarizing",
        "phase_flip",
    }
    assert phase.PhaseQNodeDensityCircuit is PhaseQNodeDensityCircuit
    assert phase.PhaseQNodeDensityExecutionResult is PhaseQNodeDensityExecutionResult
    assert phase.PhaseQNodeNoiseChannel is PhaseQNodeNoiseChannel
    assert phase.execute_phase_qnode_density_matrix is execute_phase_qnode_density_matrix
    assert phase.phase_qnode_density_support_report is phase_qnode_density_support_report
    assert phase.phase_qnode_gradient_support_report is phase_qnode_gradient_support_report
    assert phase.phase_qnode_metric_support_report is phase_qnode_metric_support_report
    assert (
        phase.phase_qnode_computational_basis_fisher_support_report
        is phase_qnode_computational_basis_fisher_support_report
    )
    assert phase.registered_phase_qnode_noise_channels is registered_phase_qnode_noise_channels
