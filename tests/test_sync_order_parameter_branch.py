# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for the sync order parameter observable
"""Empty-counts branch test for the synchronisation order-parameter observable."""

from __future__ import annotations

from scpn_quantum_control.analysis.sync_order_parameter import SyncOrderParameter


def test_sync_order_zero_for_empty_counts() -> None:
    """Empty or absent counts yield a zero synchronisation order parameter."""
    assert SyncOrderParameter()(counts=None) == {"sync_order": 0.0}
