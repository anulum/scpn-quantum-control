# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the Aer quantum entropy source contract
"""Contract tests for the Aer-backed quantum entropy source.

Covers construction validation (unknown kind, non-positive register, the
statevector phase-estimation register cap), the kind/bits-per-shot properties,
and the sample_bits input contract (negative count rejection, empty request)
without depending on a wide simulator run.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.entropy.quantum_source import AerQuantumEntropySource


class TestConstructionValidation:
    """The source validates its configuration before building any circuit."""

    def test_rejects_unknown_kind(self) -> None:
        """An unsupported source kind is rejected."""
        with pytest.raises(ValueError, match="unknown quantum source kind"):
            AerQuantumEntropySource(kind="bogus")  # type: ignore[arg-type]

    def test_rejects_non_positive_register(self) -> None:
        """A non-positive register size is rejected."""
        with pytest.raises(ValueError, match="register_qubits must be a positive integer"):
            AerQuantumEntropySource(register_qubits=0)

    def test_rejects_oversized_statevector_register(self) -> None:
        """The non-Clifford phase-estimation register is capped for statevector."""
        with pytest.raises(ValueError, match="capped at 12"):
            AerQuantumEntropySource(kind="phase_estimation", register_qubits=13)


class TestProperties:
    """Configured metadata is exposed for the streaming harness."""

    def test_kind_and_bits_per_shot(self) -> None:
        """kind echoes the configured source; bits_per_shot equals the register width."""
        source = AerQuantumEntropySource(kind="xy_measurement", register_qubits=4, seed=7)
        assert source.kind == "xy_measurement"
        assert source.bits_per_shot == 4


class TestSampleBitsContract:
    """sample_bits validates its request before touching the simulator."""

    def test_rejects_negative_bit_count(self) -> None:
        """A negative bit request is rejected."""
        source = AerQuantumEntropySource(register_qubits=4, seed=1)
        with pytest.raises(ValueError, match="n_bits must be a non-negative integer"):
            source.sample_bits(-1)

    def test_zero_bits_returns_empty_int8_array(self) -> None:
        """A zero-bit request returns an empty int8 array without simulating."""
        source = AerQuantumEntropySource(register_qubits=4, seed=1)
        result = source.sample_bits(0)
        assert result.dtype == np.int8
        assert result.size == 0
