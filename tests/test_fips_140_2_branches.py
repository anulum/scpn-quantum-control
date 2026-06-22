# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the FIPS 140-2 RNG health tests
"""Guard and pass-path tests for the FIPS 140-2 power-up health tests."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.entropy.fips_140_2 import (
    FIPS_SAMPLE_BITS,
    enforce_fips_140_2,
    fips_140_2_tests,
)


def test_fips_tests_reject_non_binary_bits() -> None:
    """A full-length sequence with a non-binary value is rejected."""
    bits = np.zeros(FIPS_SAMPLE_BITS, dtype=np.int8)
    bits[0] = 2
    with pytest.raises(ValueError, match="bit sequence must contain only 0 and 1"):
        fips_140_2_tests(bits)


def test_enforce_returns_report_for_passing_stream() -> None:
    """A balanced random stream passes the power-up tests and returns a report."""
    bits = np.random.default_rng(0).integers(0, 2, FIPS_SAMPLE_BITS).astype(np.int8)
    report = enforce_fips_140_2(bits)
    assert report.monobit_pass is True
