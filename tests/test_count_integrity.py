# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for strict shot-count coercion helpers
"""Tests for the strict shot-count and bitstring coercion utilities.

Exercises the accept and fail-closed paths of the integer coercion, binary and
fixed-width bitstring normalisers, provider job-id validation, and shot
conservation, including the strict rejection of booleans, non-integral
numerics, padded or non-binary keys, control characters, and shot mismatches.
"""

from __future__ import annotations

import pytest

from scpn_quantum_control.hardware._count_integrity import (
    strict_binary_bitstring_key,
    strict_fixed_width_bitstring_key,
    strict_integer_value,
    strict_non_negative_count,
    strict_provider_job_id,
    strict_shot_conservation,
)


class TestStrictInteger:
    """Integer coercion accepts genuine integers and rejects ambiguity."""

    def test_accepts_integer(self) -> None:
        """A plain integer is returned unchanged."""
        assert strict_non_negative_count(5) == 5

    def test_accepts_integral_string(self) -> None:
        """A string naming an exact integer is accepted."""
        assert strict_integer_value("7") == 7

    def test_rejects_bool(self) -> None:
        """Booleans are not accepted as integer counts."""
        with pytest.raises(ValueError, match="must be an integer"):
            strict_non_negative_count(True)

    def test_rejects_negative(self) -> None:
        """A negative value fails the non-negative contract."""
        with pytest.raises(ValueError, match="non-negative"):
            strict_non_negative_count(-1)

    def test_rejects_empty_string(self) -> None:
        """An empty/whitespace string is not an integer."""
        with pytest.raises(ValueError, match="must be an integer"):
            strict_integer_value("   ")

    def test_rejects_non_numeric_string(self) -> None:
        """A non-numeric string is rejected."""
        with pytest.raises(ValueError, match="must be an integer"):
            strict_integer_value("abc")

    def test_rejects_non_integral_decimal_string(self) -> None:
        """A fractional numeric string is rejected rather than truncated."""
        with pytest.raises(ValueError, match="must be an integer"):
            strict_integer_value("1.5")

    def test_rejects_unsupported_type(self) -> None:
        """A non-int, non-string value is rejected."""
        with pytest.raises(ValueError, match="must be an integer"):
            strict_integer_value(None)


class TestBinaryBitstringKey:
    """Binary bitstring normalisation from strings and bit sequences."""

    def test_accepts_binary_string(self) -> None:
        """A clean binary string passes through unchanged."""
        assert strict_binary_bitstring_key("0101") == "0101"

    def test_accepts_bit_sequence(self) -> None:
        """A sequence of 0/1 bits is joined into a bitstring."""
        assert strict_binary_bitstring_key([1, 0, 1]) == "101"

    def test_rejects_empty_string(self) -> None:
        """An empty string is not a valid key."""
        with pytest.raises(ValueError, match="non-empty bitstring"):
            strict_binary_bitstring_key("")

    def test_rejects_whitespace_padding(self) -> None:
        """Surrounding whitespace is rejected, not stripped silently."""
        with pytest.raises(ValueError, match="whitespace padding"):
            strict_binary_bitstring_key(" 01")

    def test_rejects_non_binary_string(self) -> None:
        """Non-binary digits are rejected."""
        with pytest.raises(ValueError, match="only binary digits"):
            strict_binary_bitstring_key("012")

    def test_rejects_empty_sequence(self) -> None:
        """An empty sequence is not a valid key."""
        with pytest.raises(ValueError, match="non-empty bitstring"):
            strict_binary_bitstring_key([])

    def test_rejects_non_binary_sequence(self) -> None:
        """A sequence containing a non-bit integer is rejected."""
        with pytest.raises(ValueError, match="only binary digits"):
            strict_binary_bitstring_key([1, 2])

    def test_rejects_unsupported_type(self) -> None:
        """A non-string, non-sequence value is rejected."""
        with pytest.raises(ValueError, match="bitstring or sequence"):
            strict_binary_bitstring_key(42)


class TestFixedWidthBitstringKey:
    """Fixed-width bitstring normalisation."""

    def test_zero_pads_to_width(self) -> None:
        """A short key is left-zero-padded to the requested width."""
        assert strict_fixed_width_bitstring_key("11", width=4) == "0011"

    def test_rejects_non_positive_width(self) -> None:
        """The target width must be positive."""
        with pytest.raises(ValueError, match="width must be positive"):
            strict_fixed_width_bitstring_key("1", width=0)

    def test_rejects_overlong_key(self) -> None:
        """A key wider than the target width is rejected."""
        with pytest.raises(ValueError, match="exactly 2 bits wide"):
            strict_fixed_width_bitstring_key("101", width=2)


class TestProviderJobId:
    """Provider job-id canonicalisation."""

    def test_accepts_trimmed_id(self) -> None:
        """A surrounding-space id is trimmed and returned."""
        assert strict_provider_job_id("  job-123  ") == "job-123"

    def test_rejects_empty(self) -> None:
        """An empty id is rejected."""
        with pytest.raises(ValueError, match="must be non-empty"):
            strict_provider_job_id("   ")

    def test_rejects_control_whitespace(self) -> None:
        """Tabs and other control whitespace are rejected."""
        with pytest.raises(ValueError, match="control whitespace"):
            strict_provider_job_id("job\t123")

    def test_rejects_control_character(self) -> None:
        """Non-whitespace control characters are rejected."""
        with pytest.raises(ValueError, match="control characters"):
            strict_provider_job_id("job\x01")

    def test_rejects_object_repr(self) -> None:
        """An object-repr placeholder is not a stable job id."""
        with pytest.raises(ValueError, match="object representation"):
            strict_provider_job_id("<Job object at 0x7f00>")


class TestShotConservation:
    """Decoded counts must match the declared provider shots exactly."""

    def test_accepts_matching_shots(self) -> None:
        """Counts that sum to the expected shots are accepted."""
        assert strict_shot_conservation({"0": 6, "1": 4}, expected_shots=10) == 10

    def test_rejects_non_positive_expectation(self) -> None:
        """The shot expectation must be positive."""
        with pytest.raises(ValueError, match="expectation must be positive"):
            strict_shot_conservation({"0": 1}, expected_shots=0)

    def test_rejects_shot_mismatch(self) -> None:
        """A count sum that disagrees with the expectation fails closed."""
        with pytest.raises(ValueError, match="mismatch: expected 10, observed 9"):
            strict_shot_conservation({"0": 5, "1": 4}, expected_shots=10)
