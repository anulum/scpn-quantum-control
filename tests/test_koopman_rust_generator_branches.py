# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the Rust-preferred Koopman generator
"""Branch tests for the Rust-preferred Koopman generator and its fallback.

Covers the shape-mismatch and non-finite guards on a native engine result and
the NumPy fallback taken when no native engine is available.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.analysis import koopman

_K = np.array([[0.0, 0.4], [0.4, 0.0]], dtype=np.float64)
_OMEGA = np.array([0.1, -0.1], dtype=np.float64)


class _FakeEngine:
    """Stand-in native engine exposing a configurable ``koopman_generator``."""

    def __init__(self, result: NDArray[np.float64]) -> None:
        self._result = result

    def koopman_generator(
        self,
        _k: NDArray[np.float64],
        _omega: NDArray[np.float64],
        _theta: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return self._result


def test_rust_generator_rejects_wrong_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """A native generator with the wrong shape is rejected."""
    engine = _FakeEngine(np.zeros((3, 3), dtype=np.float64))
    monkeypatch.setattr(koopman, "optional_rust_engine", lambda: engine)
    with pytest.raises(ValueError, match="expected"):
        koopman.build_koopman_generator_rust(_K, _OMEGA)


def test_rust_generator_rejects_non_finite(monkeypatch: pytest.MonkeyPatch) -> None:
    """A native generator with non-finite entries is rejected."""
    bad = np.zeros((4, 4), dtype=np.float64)
    bad[0, 0] = np.nan
    engine = _FakeEngine(bad)
    monkeypatch.setattr(koopman, "optional_rust_engine", lambda: engine)
    with pytest.raises(ValueError, match="non-finite"):
        koopman.build_koopman_generator_rust(_K, _OMEGA)


def test_rust_generator_falls_back_to_numpy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without a native engine the NumPy generator is returned."""
    monkeypatch.setattr(koopman, "optional_rust_engine", lambda: None)
    generator, labels = koopman.build_koopman_generator_rust(_K, _OMEGA)
    expected, expected_labels = koopman.build_koopman_generator(_K, _OMEGA)
    assert generator.shape == (4, 4)
    assert labels == expected_labels
    np.testing.assert_allclose(generator, expected)
