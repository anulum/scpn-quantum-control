# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Validation tests for the EEG PLV→VQE mapping
"""Validation-branch tests for the EEG classification mapping.

Covers the PLV-matrix, natural-frequency, reps, threshold, and statevector
validators' rejection paths, plus the eeg_plv_to_vqe guards against a VQE solver
returning a non-numeric energy or a non-boolean convergence flag.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import scpn_quantum_control.applications.eeg_classification as eeg
from scpn_quantum_control.applications.eeg_classification import (
    _validated_natural_frequencies,
    _validated_plv_matrix,
    _validated_reps,
    _validated_statevector,
    _validated_threshold,
    eeg_plv_to_vqe,
)


class TestPlvValidation:
    """The PLV matrix validator rejects malformed coupling matrices."""

    def test_rejects_empty_matrix(self) -> None:
        """A zero-channel matrix is rejected."""
        with pytest.raises(ValueError, match="at least one channel"):
            _validated_plv_matrix(np.zeros((0, 0), dtype=np.float64))

    def test_rejects_non_finite(self) -> None:
        """Non-finite PLV entries are rejected."""
        with pytest.raises(ValueError, match="finite"):
            _validated_plv_matrix(np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64))

    def test_rejects_non_zero_diagonal(self) -> None:
        """A non-zero diagonal is rejected."""
        with pytest.raises(ValueError, match="diagonal must be zero"):
            _validated_plv_matrix(np.array([[0.5, 0.1], [0.1, 0.0]], dtype=np.float64))


class TestScalarAndFrequencyValidation:
    """Frequency, reps, threshold, and statevector validators."""

    def test_rejects_non_finite_frequencies(self) -> None:
        """Non-finite natural frequencies are rejected."""
        with pytest.raises(ValueError, match="finite"):
            _validated_natural_frequencies(np.array([np.inf, 1.0], dtype=np.float64), 2)

    @pytest.mark.parametrize("reps", [True, 1.5])
    def test_rejects_non_integer_reps(self, reps: Any) -> None:
        """Booleans and non-integers are rejected as repetition counts."""
        with pytest.raises(ValueError, match="positive integer"):
            _validated_reps(reps)

    def test_rejects_out_of_range_threshold(self) -> None:
        """A threshold outside [0, 1] is rejected."""
        with pytest.raises(ValueError, match="threshold must be in"):
            _validated_threshold(1.5)

    def test_rejects_non_1d_statevector(self) -> None:
        """A non-1-D statevector is rejected."""
        with pytest.raises(ValueError, match="1-D statevector"):
            _validated_statevector(np.zeros((2, 2), dtype=np.complex128), "state")

    def test_rejects_non_finite_statevector(self) -> None:
        """A statevector with non-finite amplitudes is rejected."""
        with pytest.raises(ValueError, match="finite amplitudes"):
            _validated_statevector(np.array([complex(1.0, np.inf)], dtype=np.complex128), "state")


class _FakeVQE:
    """Minimal PhaseVQE stand-in whose solve result is caller-controlled."""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result
        self.ansatz: Any = None
        self.n_params = 0

    def solve(self, *, maxiter: int, seed: int) -> dict[str, Any]:
        """Return the preconfigured solver payload."""
        return self._result


def _patch_vqe(monkeypatch: pytest.MonkeyPatch, result: dict[str, Any]) -> None:
    """Replace PhaseVQE with a stand-in returning a fixed solve payload."""
    monkeypatch.setattr(eeg, "PhaseVQE", lambda *a, **k: _FakeVQE(result))


def test_rejects_non_numeric_energy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A solver energy that is not a real number is rejected."""
    _patch_vqe(monkeypatch, {"vqe_energy": "not-a-number", "converged": True})
    plv = np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64)
    freqs = np.array([10.0, 11.0], dtype=np.float64)
    with pytest.raises(TypeError, match="non-numeric energy"):
        eeg_plv_to_vqe(plv, freqs, reps=1)


def test_rejects_non_boolean_convergence(monkeypatch: pytest.MonkeyPatch) -> None:
    """A solver convergence flag that is not boolean is rejected."""
    _patch_vqe(monkeypatch, {"vqe_energy": 1.0, "converged": "yes"})
    plv = np.array([[0.0, 0.2], [0.2, 0.0]], dtype=np.float64)
    freqs = np.array([10.0, 11.0], dtype=np.float64)
    with pytest.raises(TypeError, match="non-boolean convergence"):
        eeg_plv_to_vqe(plv, freqs, reps=1)
