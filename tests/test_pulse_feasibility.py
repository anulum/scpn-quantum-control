# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S3 pulse feasibility probes
"""Tests for no-submit S3 pulse feasibility probes."""

from __future__ import annotations

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.hardware.pulse_feasibility import (
    PulseProviderSnapshot,
    assess_pulse_provider_feasibility,
    pulse_snapshot_from_metadata,
)
from scpn_quantum_control.phase.pulse_shaping import build_trotter_pulse_schedule


def _schedule():
    return build_trotter_pulse_schedule(4, build_knm_paper27(4), t_step=0.2)


def test_pulse_provider_ready_when_metadata_satisfies_schedule() -> None:
    decision = assess_pulse_provider_feasibility(
        PulseProviderSnapshot(
            provider="pulse",
            backend_name="ready",
            n_qubits=8,
            supports_pulse_control=True,
            supports_native_xy=False,
            min_time_step=0.0001,
            max_pulse_duration=1.0,
            max_pulses=16,
            supported_features=("pulse_control",),
        ),
        _schedule(),
    )

    assert decision.status == "ready"
    assert decision.hardware_submission is False
    assert decision.schedule.pulse_count == 6


def test_pulse_provider_blocks_missing_execution_support() -> None:
    decision = assess_pulse_provider_feasibility(
        PulseProviderSnapshot(
            provider="gate_only",
            backend_name="blocked",
            n_qubits=4,
            supports_pulse_control=False,
            supports_native_xy=False,
            supported_features=("gate_model",),
        ),
        _schedule(),
    )

    assert decision.status == "blocked"
    assert "neither pulse control nor native XY" in decision.reasons[0]


def test_pulse_snapshot_from_metadata_parses_provider_neutral_record() -> None:
    snapshot = pulse_snapshot_from_metadata(
        {
            "provider": "analog",
            "backend_name": "review",
            "n_qubits": 16,
            "supports_pulse_control": False,
            "supports_native_xy": True,
            "supported_features": ["native_xy"],
        }
    )

    assert snapshot.provider == "analog"
    assert snapshot.supported_features == ("native_xy",)
