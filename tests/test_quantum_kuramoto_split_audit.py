# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — quantum kuramoto split audit tests
# scpn-quantum-control -- S6 split audit tests
"""Tests for the S6 quantum-kuramoto split audit."""

from __future__ import annotations

from scripts.audit_quantum_kuramoto_split import build_split_audit


def test_split_audit_has_rows_and_status_counts() -> None:
    payload = build_split_audit()

    assert payload["schema"] == "s6_quantum_kuramoto_split_audit_v1"
    assert payload["rows"]
    statuses = payload["statuses"]
    assert isinstance(statuses, dict)
    assert statuses["reusable"] > 0
    assert statuses["scpn_specific"] > 0


def test_split_audit_keeps_package_publish_blocked() -> None:
    payload = build_split_audit()
    boundary = payload["acceptance_boundary"]

    assert isinstance(boundary, dict)
    assert boundary["safe_to_publish_package_now"] is False
    assert "no package skeleton" in str(boundary["reason"])


def test_split_audit_classifies_core_and_scpn_specific_modules() -> None:
    payload = build_split_audit()
    rows = {row["module"]: row for row in payload["rows"]}

    assert rows["scpn_quantum_control.phase.xy_kuramoto"]["status"] == "reusable"
    assert rows["scpn_quantum_control.bridge.ssgf_adapter"]["status"] == "scpn_specific"
