# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Random-state helper contract tests
"""``rust_random_state`` draws locally and returns the bytes it always has.

The helper used to call :func:`numpy.random.seed`, which reseeds and then
consumes the caller's global generator, so a call reached out and replaced an
unrelated stream the caller had seeded itself.

The repair keeps a legacy :class:`numpy.random.RandomState` rather than moving
to a modern :class:`numpy.random.Generator`, and that choice is load-bearing: a
``RandomState`` reproduces the exact bytes the global path produced, so no state
this helper has ever returned changes. A test below pins that byte equality
against the historical algorithm written out in full, so a later switch to
``default_rng`` fails loudly instead of silently changing every value.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from oscillatools.accel import (
    DEFAULT_RANDOM_STATE_MAX_GIB,
    RANDOM_STATE_BYTES_PER_AMPLITUDE,
    rust_random_state,
)


def _historical_state(n_qubits: int, seed: int) -> NDArray[np.complex128]:
    """Reproduce the pre-repair algorithm, restoring the global generator.

    Written out rather than imported, because the point is to compare against
    what the helper used to do, not against what it does now.

    Parameters
    ----------
    n_qubits
        Qubit count.
    seed
        Seed for the global generator.

    Returns
    -------
    numpy.ndarray
        The vector the old implementation returned for these arguments.
    """
    saved = np.random.get_state()
    try:
        np.random.seed(seed)
        drawn = np.random.randn(2**n_qubits) + 1j * np.random.randn(2**n_qubits)
        return np.asarray(drawn / np.linalg.norm(drawn), dtype=np.complex128)
    finally:
        np.random.set_state(saved)


class TestGlobalGeneratorIsUntouched:
    """The recorded defect: the helper reached into the caller's stream."""

    def test_an_unrelated_seeded_stream_is_preserved(self) -> None:
        """The card's case: a stream seeded to 7 must survive the call."""
        np.random.seed(7)
        expected = np.random.rand()

        np.random.seed(7)
        rust_random_state(2, seed=42)
        actual = np.random.rand()

        assert actual == expected

    def test_the_global_generator_is_not_consumed(self) -> None:
        """Two draws around a call must be consecutive in the same stream."""
        np.random.seed(11)
        reference = [np.random.rand(), np.random.rand()]

        np.random.seed(11)
        first = np.random.rand()
        rust_random_state(3, seed=5)
        second = np.random.rand()

        assert [first, second] == reference

    def test_the_global_state_object_is_unchanged(self) -> None:
        """Not even the internal position of the global generator moves."""
        np.random.seed(3)
        # The legacy tuple form is what MT19937 exposes; its stub is a union,
        # so the local name carries the type rather than a suppression.
        before: Any = np.random.get_state(legacy=True)

        rust_random_state(4, seed=99)
        after: Any = np.random.get_state(legacy=True)

        assert before[0] == after[0]
        np.testing.assert_array_equal(before[1], after[1])
        assert tuple(before[2:]) == tuple(after[2:])


class TestByteCompatibility:
    """The compatibility choice, pinned so it cannot be dropped silently."""

    @pytest.mark.parametrize(("n_qubits", "seed"), [(0, 42), (1, 0), (3, 123), (3, 124), (12, 49)])
    def test_output_matches_the_historical_algorithm(self, n_qubits: int, seed: int) -> None:
        """A modern Generator would change every one of these values.

        Parameters
        ----------
        n_qubits
            Qubit count.
        seed
            Seed under test.
        """
        np.testing.assert_array_equal(
            rust_random_state(n_qubits, seed=seed), _historical_state(n_qubits, seed)
        )

    def test_the_two_generator_families_disagree(self) -> None:
        """The fact the compatibility choice rests on, asserted not assumed.

        The same seed drives the legacy and modern generators to different
        bytes, so switching families would change every recorded state.
        """
        saved = np.random.get_state()
        try:
            np.random.seed(123)
            legacy_draw = np.random.randn(8)
        finally:
            np.random.set_state(saved)
        modern_draw = np.random.default_rng(123).standard_normal(8)

        assert not np.allclose(legacy_draw, modern_draw)


class TestOutputContract:
    """Deterministic, normalised, and sensitive to the seed."""

    @pytest.mark.parametrize("n_qubits", [0, 1, 2, 3, 6])
    def test_shape_dtype_and_norm(self, n_qubits: int) -> None:
        """Small sizes, including the degenerate single-amplitude case.

        Parameters
        ----------
        n_qubits
            Qubit count under test.
        """
        state = rust_random_state(n_qubits, seed=17)

        assert state.shape == (2**n_qubits,)
        assert state.dtype == np.complex128
        assert np.linalg.norm(state) == pytest.approx(1.0)

    def test_the_same_seed_gives_the_same_state(self) -> None:
        """Determinism does not depend on call order any more."""
        first = rust_random_state(4, seed=8)
        rust_random_state(4, seed=999)
        second = rust_random_state(4, seed=8)

        np.testing.assert_array_equal(first, second)

    def test_different_seeds_give_different_states(self) -> None:
        """The seed still selects the stream."""
        assert not np.allclose(rust_random_state(3, seed=1), rust_random_state(3, seed=2))


class TestArgumentDomain:
    """Invalid sizes and budgets are named rather than left to NumPy."""

    @pytest.mark.parametrize("n_qubits", [-1, 1.5, True, "3", None])
    def test_rejects_an_inadmissible_qubit_count(self, n_qubits: Any) -> None:
        """A bool is an int in Python, and is rejected on purpose.

        Parameters
        ----------
        n_qubits
            Inadmissible qubit count.
        """
        with pytest.raises(ValueError, match="n_qubits must be a non-negative integer"):
            rust_random_state(n_qubits)

    @pytest.mark.parametrize("seed", [-1, 2.5, True, None])
    def test_rejects_an_inadmissible_seed(self, seed: Any) -> None:
        """A seed selects a stream, so it must be a real non-negative integer.

        Parameters
        ----------
        seed
            Inadmissible seed.
        """
        with pytest.raises(ValueError, match="seed must be a non-negative integer"):
            rust_random_state(2, seed=seed)

    @pytest.mark.parametrize("max_gib", [0.0, -1.0, float("nan"), float("inf")])
    def test_rejects_an_inadmissible_budget(self, max_gib: float) -> None:
        """A non-positive or non-finite budget admits nothing meaningful.

        Parameters
        ----------
        max_gib
            Inadmissible budget.
        """
        with pytest.raises(ValueError, match="max_gib must be positive and finite"):
            rust_random_state(2, max_gib=max_gib)


class TestMemoryAdmission:
    """A large request refuses before it draws anything."""

    def test_a_large_request_refuses(self) -> None:
        """Forty qubits would need tens of terabytes at peak."""
        with pytest.raises(MemoryError, match="above the .* GiB budget"):
            rust_random_state(40)

    def test_the_refusal_precedes_the_draw(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Nothing is allocated, so this never approaches real memory.

        Parameters
        ----------
        monkeypatch
            Used to install a generator that fails if it is constructed.
        """
        calls = []

        def _spy(*args: Any, **kwargs: Any) -> Any:
            calls.append(args)
            raise AssertionError("the generator was constructed despite the budget")

        monkeypatch.setattr(np.random, "RandomState", _spy)

        with pytest.raises(MemoryError):
            rust_random_state(40)

        assert calls == []

    def test_the_budget_is_the_peak_not_the_result(self) -> None:
        """The peak holds two real draws and a complex temporary as well."""
        n_qubits = 10
        result_bytes = (2**n_qubits) * np.dtype(np.complex128).itemsize
        peak_bytes = (2**n_qubits) * RANDOM_STATE_BYTES_PER_AMPLITUDE

        assert peak_bytes == 3 * result_bytes
        # A budget that fits the result but not the peak must still refuse.
        with pytest.raises(MemoryError):
            rust_random_state(n_qubits, max_gib=(result_bytes + 8) / 1024**3)

    def test_an_explicit_budget_admits_what_fits(self) -> None:
        """The budget is a limit, not a fixed size."""
        state = rust_random_state(4, max_gib=DEFAULT_RANDOM_STATE_MAX_GIB)

        assert state.shape == (16,)


class TestDegenerateDraw:
    """The normalisation guard, which a real draw cannot reach."""

    def test_a_zero_draw_is_refused_rather_than_divided_by(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the guard this returned a vector of NaNs.

        A Gaussian draw is never exactly zero, so the guard is unreachable from
        a real generator and is exercised through a substitute instead. It is
        kept because the alternative to refusing is dividing by zero and
        returning something that is not a state.

        Parameters
        ----------
        monkeypatch
            Used to install a generator that draws only zeros.
        """

        class _ZeroGenerator:
            def __init__(self, seed: int) -> None:
                """Accept the seed and ignore it.

                Parameters
                ----------
                seed
                    Ignored.
                """
                self.seed = seed

            def randn(self, size: int) -> NDArray[np.float64]:
                """Return zeros of the requested size.

                Parameters
                ----------
                size
                    Number of amplitudes.

                Returns
                -------
                numpy.ndarray
                    An all-zero draw.
                """
                return np.zeros(size)

        monkeypatch.setattr(np.random, "RandomState", _ZeroGenerator)

        with pytest.raises(ValueError, match="could not be normalised"):
            rust_random_state(2, seed=1)
