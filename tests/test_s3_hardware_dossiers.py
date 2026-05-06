# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S3 hardware dossiers
"""Tests for no-submit S3 hardware-job dossier export."""

from __future__ import annotations

from scripts.export_s3_hardware_dossiers import (
    PULSE_PATH,
    _ansatz_dossier,
    _best_ansatz_row,
    _load_json,
    _pulse_dossier,
    _ready_pulse_targets,
)


def test_ansatz_dossier_is_no_submit_and_has_claim_boundary() -> None:
    dossier = _ansatz_dossier(
        _best_ansatz_row(
            _load_json(PULSE_PATH.with_name("s3_ansatz_observable_validation_2026-05-06.json"))
        )
    )

    data = dossier.to_dict()
    assert data["qpu_budget"]["hardware_submission"] is False
    assert "does not authorise submission" in data["claim_boundary"]


def test_pulse_dossier_uses_ready_targets() -> None:
    targets = _ready_pulse_targets(_load_json(PULSE_PATH))
    dossier = _pulse_dossier(targets)

    data = dossier.to_dict()
    assert data["qpu_budget"]["hardware_submission"] is False
    assert data["platform_fit"]
