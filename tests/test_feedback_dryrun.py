# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for feedback dry-runs
"""Tests for no-submit S1 provider dry-run payloads."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.control.realtime_feedback import RealtimeSyncFeedbackController
from scpn_quantum_control.hardware.feedback_dryrun import (
    FeedbackDryRunPayload,
    build_s1_feedback_dry_run_bundle,
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


def test_s1_feedback_dry_run_bundle_contains_no_submit_provider_payloads() -> None:
    bundle = build_s1_feedback_dry_run_bundle(_package())

    providers = [payload.provider for payload in bundle]

    assert providers == ["ibm_runtime", "openqasm3_gate", "analog_native_review"]
    assert all(payload.submission_enabled is False for payload in bundle)
    assert bundle[0].payload["required_backend_features"] == [
        "mid_circuit_measurement",
        "conditional_reset",
        "conditional_rotation",
        "dynamic_circuit_control_flow",
    ]
    assert bundle[1].payload["program_requirements"]["conditionals"] is True
    assert "separate native-feedback dossier" in bundle[2].warnings[1]


def test_feedback_dry_run_payload_rejects_submission_enabled() -> None:
    with pytest.raises(ValueError, match="must not enable submission"):
        FeedbackDryRunPayload(
            provider="ibm_runtime",
            submission_enabled=True,
            payload={},
            warnings=(),
        )
