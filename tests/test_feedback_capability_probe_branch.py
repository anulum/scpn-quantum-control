# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the feedback capability probe
"""Conditional-reset feature branch test for the dynamic-feature probe."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from scpn_quantum_control.hardware.feedback_capability_probe import (
    required_s1_dynamic_features,
)
from scpn_quantum_control.hardware.feedback_submission import FeedbackSubmissionPackage


def test_required_features_includes_conditional_reset() -> None:
    """A payload requiring conditional reset lists that dynamic feature."""
    package = cast(
        FeedbackSubmissionPackage,
        SimpleNamespace(
            circuit=SimpleNamespace(
                has_mid_circuit_measurement=False,
                has_conditional_control=False,
                has_conditional_reset=True,
            )
        ),
    )
    features: tuple[str, ...] = required_s1_dynamic_features(package)
    assert "conditional_reset" in features
