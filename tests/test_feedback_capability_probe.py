# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for feedback capability probes
"""Tests for no-submit S1 capability probes."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.control.realtime_feedback import RealtimeSyncFeedbackController
from scpn_quantum_control.hardware.feedback_capability_probe import (
    BackendCapabilitySnapshot,
    assess_feedback_backend_capability,
    assess_feedback_backend_fleet,
    required_s1_dynamic_features,
)
from scpn_quantum_control.hardware.feedback_submission import (
    build_s1_feedback_submission_package,
)


def _package():
    controller = RealtimeSyncFeedbackController(
        np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64),
        np.array([0.1, 0.3], dtype=np.float64),
    )
    return build_s1_feedback_submission_package(controller, n_rounds=1)


def test_required_s1_dynamic_features_follow_package_circuit() -> None:
    features = required_s1_dynamic_features(_package())

    assert features == (
        "cross_shot_batches",
        "mid_circuit_measurement",
        "conditional_control",
        "conditional_reset",
    )


def test_feedback_backend_capability_ready_when_metadata_satisfies_package() -> None:
    package = _package()
    snapshot = BackendCapabilitySnapshot(
        provider="ibm",
        backend_name="ibm_example",
        n_qubits=8,
        supported_features=required_s1_dynamic_features(package),
        max_shots=4096,
        max_circuits=8,
    )

    decision = assess_feedback_backend_capability(snapshot, package)

    assert decision.status == "ready"
    assert decision.missing_features == ()
    assert decision.to_dict()["backend_name"] == "ibm_example"


def test_feedback_backend_capability_blocks_missing_features_and_budget_limits() -> None:
    package = _package()
    snapshot = BackendCapabilitySnapshot(
        provider="gate",
        backend_name="limited",
        n_qubits=2,
        supported_features=("cross_shot_batches",),
        max_shots=128,
        max_circuits=1,
    )

    decision = assess_feedback_backend_capability(snapshot, package)

    assert decision.status == "blocked"
    assert "mid_circuit_measurement" in decision.missing_features
    assert any("max_shots" in reason for reason in decision.reasons)
    assert any("max_circuits" in reason for reason in decision.reasons)


def test_feedback_backend_capability_unknown_without_declared_features() -> None:
    decision = assess_feedback_backend_capability(
        BackendCapabilitySnapshot(provider="unknown", backend_name="metadata-light", n_qubits=8),
        _package(),
    )

    assert decision.status == "unknown"
    assert "backend did not declare supported_features" in decision.reasons


def test_feedback_backend_fleet_assesses_all_snapshots() -> None:
    package = _package()
    decisions = assess_feedback_backend_fleet(
        (
            BackendCapabilitySnapshot(
                provider="sim",
                backend_name="ready",
                n_qubits=8,
                supported_features=required_s1_dynamic_features(package),
            ),
            BackendCapabilitySnapshot(provider="sim", backend_name="unknown", n_qubits=8),
        ),
        package,
    )

    assert [decision.status for decision in decisions] == ["ready", "unknown"]


def test_feedback_backend_capability_blocks_budget_even_when_features_match() -> None:
    package = _package()
    snapshot = BackendCapabilitySnapshot(
        provider="ibm",
        backend_name="budget_limited",
        n_qubits=8,
        supported_features=required_s1_dynamic_features(package),
        max_shots=package.budget.shots_per_circuit - 1,
        max_circuits=package.budget.circuits,
    )

    decision = assess_feedback_backend_capability(snapshot, package)

    assert decision.status == "blocked"
    assert decision.missing_features == ()
    assert decision.required_features == required_s1_dynamic_features(package)
    assert decision.to_dict()["max_shots"] == package.budget.shots_per_circuit - 1
    assert any("max_shots" in reason for reason in decision.reasons)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"provider": ""}, "provider"),
        ({"backend_name": ""}, "backend_name"),
        ({"n_qubits": 0}, "n_qubits"),
        ({"max_shots": 0}, "max_shots"),
        ({"max_circuits": 0}, "max_circuits"),
    ),
)
def test_backend_capability_snapshot_rejects_invalid_metadata_boundaries(
    kwargs: dict[str, object],
    message: str,
) -> None:
    params = {"provider": "ibm", "backend_name": "target", "n_qubits": 4} | kwargs

    with pytest.raises(ValueError, match=message):
        BackendCapabilitySnapshot(**params)
