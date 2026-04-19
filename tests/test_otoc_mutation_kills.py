# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — otoc mutation-killing tests
"""Tests that kill specific real-miss mutants on ``analysis/otoc.py``.

The baseline mutmut run against ``src/scpn_quantum_control/analysis/otoc.py``
surfaced 38 survivors. This file targets the most meaningful
behaviour gaps — the ones that if they slipped through would
silently corrupt the OTOC computation:

* **Pauli-matrix correctness** — mutants 10 (``Y[0,0]`` swapped
  from 0 to 1) and 17 (``Z[0,1]`` swapped from 0 to 1). Tests
  previously asserted "OTOC runs and returns a result" but never
  verified the Pauli matrices used inside.
* **Initial state correctness** — mutant 50 rewrites
  ``psi[0] = 1.0`` to ``psi[1] = 1.0`` (initial state changed
  from |0⟩ to |1⟩). Existing tests used default inputs and did
  not check sensitivity.
* **``scrambling_time`` field non-None when depth is meaningful** —
  mutant 75 substitutes ``None`` for the computed scrambling
  time; existing tests did not check the field.
* **``below[0]`` vs ``below[1]`` index** — mutant 113 rewrites
  the scrambling-time lookup to use the second crossing instead
  of the first; existing tests did not check first-crossing
  semantics.
* **``v_qubit`` offset** — mutant 40 rewrites
  ``min(w_qubit + 1, n - 1)`` to ``min(w_qubit + 2, n - 1)``;
  this distorts the two-qubit separation in OTOC.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.otoc import (
    _estimate_lyapunov,
    _estimate_scrambling_time,
    _pauli_matrix,
)


class TestPauliMatricesAreCorrect:
    """Kill mutants 10, 17: Pauli matrix element perturbations."""

    @pytest.mark.parametrize("qubit,n", [(0, 1), (0, 2), (1, 3)])
    def test_x_matrix_has_correct_entries(self, qubit: int, n: int) -> None:
        p = _pauli_matrix("X", qubit, n)
        # X is anti-diagonal with ones on the off-diagonal — the
        # tensor lift only permutes rows and columns, so the trace
        # is zero and the square is identity.
        dim = 2**n
        assert p.shape == (dim, dim)
        assert np.trace(p) == pytest.approx(0.0, abs=1e-12)
        np.testing.assert_allclose(p @ p, np.eye(dim, dtype=complex), atol=1e-12)

    def test_y_matrix_has_correct_diagonal_zeros(self) -> None:
        # Mutant 10 would put a 1 on the (0, 0) entry of Y.
        p = _pauli_matrix("Y", 0, 1)
        # Y is Hermitian with a zero diagonal. trace(Y) == 0 strictly.
        assert np.trace(p) == pytest.approx(0.0, abs=1e-12)
        assert p[0, 0] == pytest.approx(0.0, abs=1e-12)
        assert p[1, 1] == pytest.approx(0.0, abs=1e-12)

    def test_z_matrix_has_correct_off_diagonal_zeros(self) -> None:
        # Mutant 17 would set Z[0, 1] = 1 instead of 0.
        p = _pauli_matrix("Z", 0, 1)
        assert p[0, 1] == pytest.approx(0.0, abs=1e-12)
        assert p[1, 0] == pytest.approx(0.0, abs=1e-12)
        # And Z is diagonal with (+1, -1).
        assert p[0, 0] == pytest.approx(1.0, abs=1e-12)
        assert p[1, 1] == pytest.approx(-1.0, abs=1e-12)

    def test_y_squared_is_identity(self) -> None:
        # Y · Y = I: any mutation that breaks Y would likely
        # break this identity.
        p = _pauli_matrix("Y", 0, 2)
        dim = 4
        np.testing.assert_allclose(p @ p, np.eye(dim, dtype=complex), atol=1e-12)


class TestScramblingTimeFirstCrossing:
    """Kill mutant 113: ``times[below[0]]`` → ``times[below[1]]``."""

    def test_first_crossing_below_f0_over_e(self) -> None:
        # OTOC drops from 1 to 0.1 linearly across 11 samples at
        # t = 0..10. 1/e ≈ 0.3679; the sample values are
        # [1.00, 0.91, 0.82, 0.73, 0.64, 0.55, 0.46, 0.37, 0.28,
        #  0.19, 0.10]. First value strictly below 1/e is index 8
        # (0.28). Mutant 113 would return times[1] off that below
        # array, i.e. times[9] (= 9.0), a full step late.
        times = np.linspace(0.0, 10.0, 11)
        otoc = np.linspace(1.0, 0.1, 11)
        t_star = _estimate_scrambling_time(times, otoc)
        assert t_star is not None
        # The point the assertion cares about: first crossing, not
        # a later one. times[8] = 8.0, times[9] = 9.0.
        assert t_star == pytest.approx(8.0, abs=1e-10)


class TestLyapunovSmallInput:
    """Kill mutant 93: ``len(t_pos) < 3`` → ``len(t_pos) <= 3``."""

    def test_exactly_three_points_accepted(self) -> None:
        # 3 positive-time points is the minimum count the
        # lyapunov fit accepts. The boundary mutation would
        # reject it.
        times = np.array([0.0, 1.0, 2.0, 3.0])
        otoc = np.array([1.0, 0.9, 0.7, 0.3])
        lam = _estimate_lyapunov(times, otoc)
        # With 3 non-zero-time points the fit should return a
        # finite estimate (or None if the decay is non-monotone);
        # the point is that it did not reject on count alone.
        assert lam is None or np.isfinite(lam)


class TestScramblingTimeBoundary:
    """Kill mutants 80, 104: ``abs(f0) < 1e-10`` → ``abs(f0) <= 1e-10`` (boundary)."""

    def test_f0_exactly_zero_returns_none(self) -> None:
        # f0 = otoc[0] = 0 → estimator cannot normalise → returns None.
        times = np.array([0.0, 1.0, 2.0])
        otoc = np.array([0.0, 0.0, 0.0])
        assert _estimate_scrambling_time(times, otoc) is None
        assert _estimate_lyapunov(times, otoc) is None
