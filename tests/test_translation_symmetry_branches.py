# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for translation symmetry analysis
"""Guard and branch tests for the translation-symmetry momentum solver.

Covers the empty-sector budget shortcut, the integer-momentum guard and the
empty momentum-sector result branch.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis import translation_symmetry as ts
from scpn_quantum_control.analysis.translation_symmetry import (
    _require_momentum_sector_budget,
    eigh_with_translation,
)

_K = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
_OMEGA = np.array([1.0, 1.0], dtype=np.float64)


def test_budget_guard_skips_empty_sector() -> None:
    """A sub-unit sector dimension skips the workspace budget check without raising."""
    assert _require_momentum_sector_budget(0, max_dense_gib=None) is None


def test_eigh_rejects_non_integer_momentum() -> None:
    """A non-integer (boolean) momentum is rejected."""
    with pytest.raises(ValueError, match="momentum must be an integer"):
        eigh_with_translation(_K, _OMEGA, momentum=True)


def test_eigh_returns_empty_for_unrepresented_momentum(monkeypatch: pytest.MonkeyPatch) -> None:
    """A momentum with no representative states yields an empty spectrum."""
    monkeypatch.setattr(ts, "momentum_sectors", lambda _n: {})
    result = eigh_with_translation(_K, _OMEGA, momentum=0)
    assert result["dim"] == 0
    assert result["eigvals"].size == 0
