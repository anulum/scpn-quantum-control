# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the feedback dry-run payload serialiser
"""Serialisation test for the feedback dry-run payload."""

from __future__ import annotations

from scpn_quantum_control.hardware.feedback_dryrun import FeedbackDryRunPayload


def test_dry_run_payload_serialises() -> None:
    """A dry-run payload serialises to a dictionary."""
    payload = FeedbackDryRunPayload(
        provider="ibm_runtime",
        submission_enabled=False,
        payload={"shots": 1024},
        warnings=(),
    )
    serialised = payload.to_dict()
    assert serialised["submission_enabled"] is False
    assert serialised["payload"] == {"shots": 1024}
