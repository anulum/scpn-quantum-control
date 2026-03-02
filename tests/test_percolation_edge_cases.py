"""Cover edge cases in percolation.py: lines 122, 198, 238."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.crypto.percolation import (
    best_entanglement_path,
    key_rate_per_channel,
    robustness_targeted_removal,
)


def test_entanglement_rate_clamped_entropy():
    """Concurrence near zero → rate=0 (line 122: e < _ENTROPY_CLAMP_EPS)."""
    conc = np.array([[0.0, 1e-20], [1e-20, 0.0]])
    rates = key_rate_per_channel(conc)
    assert np.all(rates == 0.0)


def test_robustness_targeted_full_disconnect():
    """All edges removed (line 198: loop exhausts all edges)."""
    K = np.array([[0.0, 0.1], [0.1, 0.0]])
    result = robustness_targeted_removal(K)
    assert result["edges_to_disconnect"] >= 1
    assert 0 < result["fraction"] <= 1.0


def test_best_path_boundary_validation():
    """Source/target out of range raises ValueError (line 238)."""
    K = np.array([[0.0, 0.5], [0.5, 0.0]])
    with pytest.raises(ValueError, match="out of range"):
        best_entanglement_path(K, source=0, target=5)
