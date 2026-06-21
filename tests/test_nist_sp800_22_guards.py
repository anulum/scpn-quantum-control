# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Guard and fallback tests for the NIST SP 800-22 suite
"""Input-guard, helper-edge and fail-closed contract tests for the NIST suite.

These exercise the validation guards of the SP 800-22 statistical tests, the
helper edge branches that the canonical tests do not reach, the Berlekamp–Massey
native-engine fallback, and the defensive minimum-length contracts that the
``as_bits`` floor normally precludes (verified by patching the floor away).
"""

from __future__ import annotations

import sys
import types
from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.entropy import nist_sp800_22 as nist


def test_as_bits_rejects_two_dimensional() -> None:
    """A non-1-D sequence is rejected."""
    with pytest.raises(ValueError, match="one-dimensional"):
        nist.as_bits(np.zeros((2, 2), dtype=np.int8))


def test_as_bits_rejects_too_few_bits() -> None:
    """A sequence below the requested minimum length is rejected."""
    with pytest.raises(ValueError, match="at least 4 bits"):
        nist.as_bits([0, 1], minimum=4)


def test_block_frequency_rejects_non_positive_block() -> None:
    """A non-positive block size is rejected."""
    with pytest.raises(ValueError, match="block_size must be a positive integer"):
        nist.block_frequency_test([0, 1, 0, 1], block_size=0)


def test_block_frequency_rejects_oversized_block() -> None:
    """A block larger than the whole sequence is rejected."""
    with pytest.raises(ValueError, match="block_size larger than the sequence length"):
        nist.block_frequency_test([0, 1, 0, 1], block_size=8)


def test_runs_test_reports_precondition_failure() -> None:
    """An imbalanced sequence trips the frequency precondition of the runs test."""
    result = nist.runs_test([1] * 100)
    assert result.passed is False
    assert result.p_value == 0.0
    assert result.details["precondition_failed"] == 1.0


def test_serial_test_rejects_block_size_below_two() -> None:
    """A serial-test block size below two is rejected."""
    with pytest.raises(ValueError, match="block_size must be an integer >= 2"):
        nist.serial_test([0, 1, 1, 0, 1, 0], block_size=1)


def test_serial_test_block_size_two_exercises_zero_order_psi() -> None:
    """A block size of two drives the zero-order ψ² helper branch."""
    result = nist.serial_test([0, 1] * 64, block_size=2)
    assert result.name == "serial"
    assert result.p_values is not None


def test_phi_m_zero_order_is_zero() -> None:
    """The zero-order φ helper returns zero."""
    bits = np.array([0, 1, 0, 1], dtype=np.int8)
    assert nist._phi_m(bits, 0) == 0.0


def test_approximate_entropy_rejects_non_positive_block() -> None:
    """A non-positive approximate-entropy block size is rejected."""
    with pytest.raises(ValueError, match="block_size must be a positive integer"):
        nist.approximate_entropy_test([0, 1, 0, 1], block_size=0)


def test_cusum_p_value_zero_excursion_is_unity() -> None:
    """A zero maximal excursion yields a unit cusum P-value."""
    assert nist._cusum_p_value(0, 100) == 1.0


def test_cumulative_sums_rejects_unknown_mode() -> None:
    """An unknown cusum mode is rejected."""
    with pytest.raises(ValueError, match="mode must be 'forward' or 'backward'"):
        nist.cumulative_sums_test([0, 1, 0, 1], mode="sideways")


def test_gf2_rank_breaks_when_rows_exhausted() -> None:
    """A wide full-row-rank matrix stops once every row is a pivot."""
    matrix = np.array([[1, 0, 1, 1], [0, 1, 0, 1]], dtype=np.int8)
    assert nist._gf2_rank(matrix) == 2


def test_non_overlapping_template_rejects_empty_template() -> None:
    """An empty template is rejected."""
    with pytest.raises(ValueError, match="template must be non-empty"):
        nist.non_overlapping_template_test([0, 1, 0, 1], template=())


def test_non_overlapping_template_rejects_short_blocks() -> None:
    """A block length not exceeding the template length is rejected."""
    with pytest.raises(ValueError, match="block length must exceed the template length"):
        nist.non_overlapping_template_test([0, 1] * 8, n_blocks=8)


def test_berlekamp_massey_falls_back_on_engine_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising native Berlekamp–Massey export falls back to the Python kernel."""

    def _boom(_bits: NDArray[np.int8]) -> int:
        raise ValueError("engine refused")

    stub = types.ModuleType("scpn_quantum_engine")
    stub.nist_berlekamp_massey = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "scpn_quantum_engine", stub)

    bits = np.array([1, 0, 0, 1, 0, 1, 1, 0], dtype=np.int8)
    assert nist.berlekamp_massey(bits) == nist._berlekamp_massey_python(bits)


def _short_bits_factory(size: int) -> Any:
    """Return an ``as_bits`` replacement that ignores the minimum-length floor."""

    def _as_bits(_sequence: Sequence[int] | NDArray[np.integer], *, minimum: int = 1) -> Any:
        return np.zeros(size, dtype=np.int8)

    return _as_bits


def test_longest_run_defensive_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the length floor removed, too-short input fails the longest-run lookup."""
    monkeypatch.setattr(nist, "as_bits", _short_bits_factory(64))
    with pytest.raises(ValueError, match="sequence too short for the longest-run test"):
        nist.longest_run_of_ones_test([0, 1] * 32)


def test_binary_matrix_rank_defensive_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the length floor removed, too-short input fails the one-matrix guard."""
    monkeypatch.setattr(nist, "as_bits", _short_bits_factory(16))
    with pytest.raises(ValueError, match="sequence too short for one matrix"):
        nist.binary_matrix_rank_test([0, 1] * 8)


def test_overlapping_template_defensive_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the length floor removed, too-short input fails the one-block guard."""
    monkeypatch.setattr(nist, "as_bits", _short_bits_factory(16))
    with pytest.raises(ValueError, match="sequence too short for one block"):
        nist.overlapping_template_test([0, 1] * 8)


def test_maurers_defensive_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the length floor removed, too-short input fails Maurer's block guard."""
    monkeypatch.setattr(nist, "as_bits", _short_bits_factory(6))
    with pytest.raises(ValueError, match="sequence too short for Maurer"):
        nist.maurers_universal_test([0, 1, 0, 1, 1, 0])


def test_linear_complexity_defensive_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """With the length floor removed, too-short input fails the one-block guard."""
    monkeypatch.setattr(nist, "as_bits", _short_bits_factory(16))
    with pytest.raises(ValueError, match="sequence too short for one block"):
        nist.linear_complexity_test([0, 1] * 8)


def test_random_excursions_defensive_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cycle-free walk fails the random-excursions cycle guard."""
    walk = np.zeros(4, dtype=np.int64)
    monkeypatch.setattr(nist, "_excursion_cycles", lambda _bits: (walk, 0))
    with pytest.raises(
        ValueError, match="no cycles found; sequence unsuitable for random-excursions"
    ):
        nist.random_excursions_test([0, 1] * 500)


def test_random_excursions_variant_defensive_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cycle-free walk fails the variant cycle guard."""
    walk = np.zeros(4, dtype=np.int64)
    monkeypatch.setattr(nist, "_excursion_cycles", lambda _bits: (walk, 0))
    with pytest.raises(ValueError, match="no cycles found; sequence unsuitable for the variant"):
        nist.random_excursions_variant_test([0, 1] * 500)
