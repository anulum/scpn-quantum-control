# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the sync order parameter observable
"""Contract tests for the synchronisation proxy observable."""

from __future__ import annotations

from scpn_quantum_control.analysis.sync_order_parameter import SyncOrderParameter


def test_sync_order_zero_for_empty_counts() -> None:
    """Empty or absent counts yield zero-valued proxy fields."""
    assert SyncOrderParameter()(counts=None) == {
        "sync_order": 0.0,
        "sync_order_z_magnetisation": 0.0,
        "is_xy_kuramoto_order_parameter": 0.0,
    }


def test_sync_order_exposes_z_magnetisation_alias_without_r_claim() -> None:
    """The legacy key is an alias for the Z-basis proxy, not true Kuramoto R."""
    result = SyncOrderParameter()(counts={"000": 75, "111": 25})

    assert result["sync_order"] == result["sync_order_z_magnetisation"]
    assert result["sync_order_z_magnetisation"] == 0.5
    assert result["is_xy_kuramoto_order_parameter"] == 0.0
