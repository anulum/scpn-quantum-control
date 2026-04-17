# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Property-based fuzz for qec/ validators
"""Property-based fuzz tests for qec/ validators.

Continues audit item B8 after the phase_artifact + classical fuzz
modules. Three validator surfaces in the QEC subpackage:

* :class:`SurfaceCodeSpec.from_distance` — requires odd ``d ≥ 3``.
* :class:`RepetitionCodeUPDE` — requires ``n_osc ≥ 2`` and
  ``code_distance`` odd positive.
* :class:`SurfaceCodeUPDE` — requires ``n_osc ≥ 2`` and delegates
  the distance validation to :class:`SurfaceCodeSpec`.

The physical-qubit accounting is also fuzzed: for any valid
``(n_osc, d)`` pair, ``total_qubits == n_osc * (2d² - 1)`` for the
surface-code path and ``n_osc * (2d - 1)`` for the repetition-code
path. These are counting invariants that would catch any future
regression in the qubit-layout code.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from scpn_quantum_control.qec.fault_tolerant import RepetitionCodeUPDE
from scpn_quantum_control.qec.surface_code_upde import (
    SurfaceCodeSpec,
    SurfaceCodeUPDE,
)

_GLOBAL_SETTINGS = settings(
    max_examples=30,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)

valid_n_osc = st.integers(min_value=2, max_value=12)
odd_distance = st.sampled_from([3, 5, 7, 9])  # odd, surface-code permissible
odd_rep_distance = st.sampled_from([1, 3, 5, 7, 9])  # repetition-code permissible

tiny_n_osc = st.integers(max_value=1)  # below the ≥2 threshold
even_distance = st.sampled_from([2, 4, 6, 8])
non_positive_distance = st.sampled_from([0, -1, -3])


# ---------------------------------------------------------------------------
# SurfaceCodeSpec.from_distance
# ---------------------------------------------------------------------------


class TestSurfaceCodeSpecFuzz:
    @_GLOBAL_SETTINGS
    @given(d=odd_distance)
    def test_valid_odd_distance_constructs(self, d: int) -> None:
        spec = SurfaceCodeSpec.from_distance(d)
        assert spec.distance == d
        assert spec.n_data == d * d
        assert spec.n_ancilla == d * d - 1
        assert spec.n_physical == 2 * d * d - 1

    @_GLOBAL_SETTINGS
    @given(d=even_distance)
    def test_rejects_even_distance(self, d: int) -> None:
        with pytest.raises(ValueError, match="odd"):
            SurfaceCodeSpec.from_distance(d)

    @_GLOBAL_SETTINGS
    @given(d=st.sampled_from([-5, -3, -1, 0, 1]))
    def test_rejects_distance_below_three(self, d: int) -> None:
        with pytest.raises(ValueError, match=r">= 3"):
            SurfaceCodeSpec.from_distance(d)


# ---------------------------------------------------------------------------
# RepetitionCodeUPDE
# ---------------------------------------------------------------------------


class TestRepetitionCodeUPDEFuzz:
    @_GLOBAL_SETTINGS
    @given(n_osc=valid_n_osc, d=odd_rep_distance)
    def test_valid_inputs_construct(self, n_osc: int, d: int) -> None:
        rep = RepetitionCodeUPDE(n_osc=n_osc, code_distance=d)
        assert rep.n_osc == n_osc
        assert rep.d == d
        assert rep.data_per_osc == d
        assert rep.ancilla_per_osc == d - 1
        assert rep.qubits_per_osc == 2 * d - 1
        assert rep.total_qubits == n_osc * (2 * d - 1)
        assert len(rep.logical_qubits) == n_osc

    @_GLOBAL_SETTINGS
    @given(n_osc=tiny_n_osc)
    def test_rejects_n_osc_below_two(self, n_osc: int) -> None:
        with pytest.raises(ValueError, match=r">= 2 oscillators"):
            RepetitionCodeUPDE(n_osc=n_osc, code_distance=3)

    @_GLOBAL_SETTINGS
    @given(n_osc=valid_n_osc, d=even_distance)
    def test_rejects_even_code_distance(self, n_osc: int, d: int) -> None:
        with pytest.raises(ValueError, match="odd positive"):
            RepetitionCodeUPDE(n_osc=n_osc, code_distance=d)

    @_GLOBAL_SETTINGS
    @given(n_osc=valid_n_osc, d=non_positive_distance)
    def test_rejects_non_positive_code_distance(self, n_osc: int, d: int) -> None:
        with pytest.raises(ValueError, match="odd positive"):
            RepetitionCodeUPDE(n_osc=n_osc, code_distance=d)

    @_GLOBAL_SETTINGS
    @given(n_osc=valid_n_osc, d=odd_rep_distance)
    def test_physical_qubit_count_invariant(self, n_osc: int, d: int) -> None:
        """total_qubits must equal n_osc * (2d - 1) for all valid inputs —
        a regression guard against the qubit-layout code."""
        rep = RepetitionCodeUPDE(n_osc=n_osc, code_distance=d)
        assert rep.total_qubits == n_osc * (2 * d - 1)

    @_GLOBAL_SETTINGS
    @given(n_osc=valid_n_osc, d=odd_rep_distance)
    def test_data_and_ancilla_ranges_dont_overlap(self, n_osc: int, d: int) -> None:
        rep = RepetitionCodeUPDE(n_osc=n_osc, code_distance=d)
        for osc in range(n_osc):
            data = set(rep._osc_data_range(osc))
            ancilla = set(rep._osc_ancilla_range(osc))
            assert data.isdisjoint(ancilla)
            assert len(data) == d
            assert len(ancilla) == d - 1


# ---------------------------------------------------------------------------
# SurfaceCodeUPDE
# ---------------------------------------------------------------------------


class TestSurfaceCodeUPDEFuzz:
    @_GLOBAL_SETTINGS
    @given(n_osc=valid_n_osc, d=odd_distance)
    def test_valid_inputs_construct(self, n_osc: int, d: int) -> None:
        upde = SurfaceCodeUPDE(n_osc=n_osc, code_distance=d)
        assert upde.n_osc == n_osc
        assert upde.spec.distance == d
        assert upde.total_qubits == n_osc * (2 * d * d - 1)

    @_GLOBAL_SETTINGS
    @given(n_osc=tiny_n_osc)
    def test_rejects_n_osc_below_two(self, n_osc: int) -> None:
        with pytest.raises(ValueError, match=r">= 2 oscillators"):
            SurfaceCodeUPDE(n_osc=n_osc, code_distance=3)

    @_GLOBAL_SETTINGS
    @given(n_osc=valid_n_osc, d=even_distance)
    def test_rejects_even_distance(self, n_osc: int, d: int) -> None:
        with pytest.raises(ValueError, match="odd"):
            SurfaceCodeUPDE(n_osc=n_osc, code_distance=d)

    @_GLOBAL_SETTINGS
    @given(n_osc=valid_n_osc, d=odd_distance)
    def test_data_qubit_ids_unique_across_oscillators(
        self,
        n_osc: int,
        d: int,
    ) -> None:
        """Every oscillator's data-qubit block must occupy a disjoint
        range within the physical register."""
        upde = SurfaceCodeUPDE(n_osc=n_osc, code_distance=d)
        all_data = []
        for osc in range(n_osc):
            block = upde._osc_data_qubits(osc)
            assert len(set(block)) == len(block)  # no dup inside block
            all_data.extend(block)
        assert len(set(all_data)) == len(all_data), "blocks overlap across oscillators"

    @_GLOBAL_SETTINGS
    @given(n_osc=valid_n_osc, d=odd_distance)
    def test_physical_qubit_count_invariant(self, n_osc: int, d: int) -> None:
        upde = SurfaceCodeUPDE(n_osc=n_osc, code_distance=d)
        assert upde.total_qubits == n_osc * (2 * d * d - 1)

    @_GLOBAL_SETTINGS
    @given(n_osc=valid_n_osc, d=odd_distance)
    def test_custom_K_omega_accepted(self, n_osc: int, d: int) -> None:
        """User-supplied K and omega override the paper-27 defaults
        without triggering a downstream validator."""
        rng = np.random.default_rng(n_osc * 31 + d)
        K = rng.normal(0.0, 0.3, size=(n_osc, n_osc))
        K = 0.5 * (K + K.T)
        np.fill_diagonal(K, 0.0)
        omega = rng.normal(1.0, 0.1, size=n_osc)
        upde = SurfaceCodeUPDE(n_osc=n_osc, code_distance=d, K=K, omega=omega)
        np.testing.assert_allclose(upde.K, K)
        np.testing.assert_allclose(upde.omega, omega)
