# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch test for symmetry-sector replay counts
"""Guard test for the symmetry-sector replay count normaliser."""

from __future__ import annotations

import pytest

from scpn_quantum_control.mitigation.symmetry_sector_replay import _normalise_counts


def test_normalise_counts_rejects_empty() -> None:
    """An empty raw-count mapping is rejected."""
    with pytest.raises(ValueError, match="raw counts must not be empty"):
        _normalise_counts({}, n_qubits=2)
