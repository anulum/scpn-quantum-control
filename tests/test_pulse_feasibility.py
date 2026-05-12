# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S3 pulse feasibility probes
"""Tests for no-submit S3 pulse feasibility probes."""

from __future__ import annotations

import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27
from scpn_quantum_control.hardware.pulse_feasibility import (
    PulseProviderSnapshot,
    assess_pulse_provider_feasibility,
    assess_pulse_provider_fleet,
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


def test_pulse_snapshot_from_metadata_rejects_non_text_feature_entries() -> None:
    with pytest.raises(ValueError, match="string sequences"):
        pulse_snapshot_from_metadata(
            {
                "provider": "analog",
                "backend_name": "bad_features",
                "n_qubits": 4,
                "supports_pulse_control": True,
                "supports_native_xy": False,
                "supported_features": ["pulse_control", object()],
            }
        )


def test_pulse_provider_fleet_keeps_native_xy_limits_for_manual_review() -> None:
    decisions = assess_pulse_provider_fleet(
        (
            PulseProviderSnapshot(
                provider="analog",
                backend_name="too_short_native_xy",
                n_qubits=8,
                supports_pulse_control=False,
                supports_native_xy=True,
                max_pulse_duration=0.01,
                supported_features=("native_xy",),
            ),
            PulseProviderSnapshot(
                provider="gate",
                backend_name="metadata_light",
                n_qubits=8,
                supports_pulse_control=True,
                supports_native_xy=False,
            ),
        ),
        _schedule(),
    )

    assert [decision.status for decision in decisions] == ["manual_review", "unknown"]
    assert "max_pulse_duration" in decisions[0].reasons[0]
    assert decisions[0].hardware_submission is False
    assert decisions[1].reasons == ("provider did not declare supported_features",)


def test_pulse_provider_blocks_declared_pulse_backend_over_schedule_limits() -> None:
    decision = assess_pulse_provider_feasibility(
        PulseProviderSnapshot(
            provider="pulse",
            backend_name="tight_limits",
            n_qubits=8,
            supports_pulse_control=True,
            supports_native_xy=False,
            min_time_step=1.0,
            max_pulses=1,
            supported_features=("pulse_control",),
        ),
        _schedule(),
    )
    data = decision.to_dict()

    assert decision.status == "blocked"
    assert any("max_pulses" in reason for reason in decision.reasons)
    assert any("min_time_step" in reason for reason in decision.reasons)
    assert data["hardware_submission"] is False
    assert data["schedule"]["pulse_count"] == 6


def test_pulse_provider_blocks_insufficient_qubits_with_declared_execution_support() -> None:
    decision = assess_pulse_provider_feasibility(
        PulseProviderSnapshot(
            provider="pulse",
            backend_name="too_few_qubits",
            n_qubits=2,
            supports_pulse_control=True,
            supports_native_xy=False,
            supported_features=("pulse_control",),
        ),
        _schedule(),
    )

    assert decision.status == "blocked"
    assert decision.reasons == ("provider has 2 qubits but schedule requires 4",)


@pytest.mark.parametrize(
    ("metadata", "message"),
    (
        (
            {
                "provider": "",
                "backend_name": "x",
                "n_qubits": 4,
                "supports_pulse_control": True,
                "supports_native_xy": False,
            },
            "provider",
        ),
        (
            {
                "provider": "pulse",
                "backend_name": "x",
                "n_qubits": 0,
                "supports_pulse_control": True,
                "supports_native_xy": False,
            },
            "n_qubits",
        ),
        (
            {
                "provider": "pulse",
                "backend_name": "x",
                "n_qubits": 4,
                "supports_pulse_control": "yes",
                "supports_native_xy": False,
            },
            "supports_pulse_control",
        ),
        (
            {
                "provider": "pulse",
                "backend_name": "x",
                "n_qubits": 4,
                "supports_pulse_control": True,
                "supports_native_xy": False,
                "min_time_step": 0.0,
            },
            "min_time_step",
        ),
        (
            {
                "provider": "pulse",
                "backend_name": "x",
                "n_qubits": 4,
                "supports_pulse_control": True,
                "supports_native_xy": False,
                "max_pulse_duration": 0.0,
            },
            "max_pulse_duration",
        ),
        (
            {
                "provider": "pulse",
                "backend_name": "x",
                "n_qubits": 4,
                "supports_pulse_control": True,
                "supports_native_xy": False,
                "max_pulses": 0,
            },
            "max_pulses",
        ),
    ),
)
def test_pulse_snapshot_from_metadata_rejects_invalid_boundary_fields(
    metadata: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        pulse_snapshot_from_metadata(metadata)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"provider": ""}, "provider"),
        ({"backend_name": ""}, "backend_name"),
        ({"n_qubits": 0}, "n_qubits"),
        ({"min_time_step": 0.0}, "min_time_step"),
        ({"max_pulse_duration": 0.0}, "max_pulse_duration"),
        ({"max_pulses": 0}, "max_pulses"),
    ),
)
def test_pulse_provider_snapshot_rejects_invalid_boundaries(
    kwargs: dict[str, object],
    message: str,
) -> None:
    params = {
        "provider": "pulse",
        "backend_name": "target",
        "n_qubits": 4,
        "supports_pulse_control": True,
        "supports_native_xy": False,
    } | kwargs

    with pytest.raises(ValueError, match=message):
        PulseProviderSnapshot(**params)


def test_pulse_snapshot_from_metadata_accepts_single_feature_text_and_ignores_private_blob() -> (
    None
):
    snapshot = pulse_snapshot_from_metadata(
        {
            "provider": "pulse",
            "backend_name": "single_feature",
            "n_qubits": 4,
            "supports_pulse_control": True,
            "supports_native_xy": False,
            "supported_features": "pulse_control",
            "metadata": "provider private blob",
        }
    )

    assert snapshot.supported_features == ("pulse_control",)
    assert snapshot.metadata == {}


@pytest.mark.parametrize("supported_features", (7, ["pulse_control", ""]))
def test_pulse_snapshot_from_metadata_rejects_malformed_feature_sequences(
    supported_features: object,
) -> None:
    with pytest.raises(ValueError):
        pulse_snapshot_from_metadata(
            {
                "provider": "pulse",
                "backend_name": "bad_sequence",
                "n_qubits": 4,
                "supports_pulse_control": True,
                "supports_native_xy": False,
                "supported_features": supported_features,
            }
        )
