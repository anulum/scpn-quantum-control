# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
"""Tests for Z₂ parity symmetry verification error mitigation."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.mitigation.symmetry_verification import (
    SymmetryVerificationResult,
    bitstring_parity,
    initial_state_parity,
    parity_postselect,
    parity_verified_expectation,
    parity_verified_R,
    symmetry_expand,
)


class TestBitstringParity:
    def test_all_zeros_is_even(self):
        assert bitstring_parity("0000") == 0

    def test_single_one_is_odd(self):
        assert bitstring_parity("0001") == 1
        assert bitstring_parity("1000") == 1

    def test_two_ones_is_even(self):
        assert bitstring_parity("0011") == 0
        assert bitstring_parity("1010") == 0
        assert bitstring_parity("1001") == 0

    def test_three_ones_is_odd(self):
        assert bitstring_parity("0111") == 1
        assert bitstring_parity("1110") == 1

    def test_all_ones_even_n(self):
        assert bitstring_parity("1111") == 0

    def test_all_ones_odd_n(self):
        assert bitstring_parity("111") == 1

    def test_spaces_ignored(self):
        assert bitstring_parity("00 11") == 0
        assert bitstring_parity("0 0 0 1") == 1


class TestInitialStateParity:
    def test_zero_frequencies_even(self):
        omega = np.array([0.0, 0.0, 0.0, 0.0])
        assert initial_state_parity(omega) == 0

    def test_small_frequencies_even(self):
        omega = np.array([0.1, 0.2, 0.05, 0.15])
        assert initial_state_parity(omega) == 0

    def test_pi_frequencies(self):
        # Ry(pi)|0> = |1>, so 4 qubits all at pi -> 4 ones -> even parity
        omega = np.array([np.pi, np.pi, np.pi, np.pi])
        assert initial_state_parity(omega) == 0

    def test_single_pi_frequency(self):
        # One qubit flipped -> odd parity
        omega = np.array([np.pi, 0.0, 0.0, 0.0])
        assert initial_state_parity(omega) == 1

    def test_three_pi_frequencies(self):
        # 3 qubits flipped -> odd parity
        omega = np.array([np.pi, np.pi, np.pi, 0.0])
        assert initial_state_parity(omega) == 1


class TestParityPostselect:
    def test_all_even_parity(self):
        counts = {"0000": 500, "0011": 300, "1100": 200}
        result = parity_postselect(counts, expected_parity=0)
        assert result.verified_counts == counts
        assert result.rejected_counts == {}
        assert result.rejection_rate == 0.0
        assert result.verified_shots == 1000

    def test_mixed_parity(self):
        counts = {"0000": 400, "0001": 100, "0011": 300, "0111": 200}
        result = parity_postselect(counts, expected_parity=0)
        assert result.verified_counts == {"0000": 400, "0011": 300}
        assert result.rejected_counts == {"0001": 100, "0111": 200}
        assert result.verified_shots == 700
        assert result.rejected_shots == 300
        assert abs(result.rejection_rate - 0.3) < 1e-10

    def test_odd_parity_selection(self):
        counts = {"0000": 400, "0001": 600}
        result = parity_postselect(counts, expected_parity=1)
        assert result.verified_counts == {"0001": 600}
        assert result.rejected_counts == {"0000": 400}
        assert result.rejection_rate == 0.4

    def test_empty_counts(self):
        result = parity_postselect({}, expected_parity=0)
        assert result.verified_shots == 0
        assert result.rejected_shots == 0
        assert result.rejection_rate == 0.0

    def test_all_rejected(self):
        counts = {"0001": 500, "0111": 500}
        result = parity_postselect(counts, expected_parity=0)
        assert result.verified_shots == 0
        assert result.rejected_shots == 1000
        assert result.rejection_rate == 1.0

    def test_returns_correct_type(self):
        counts = {"00": 100}
        result = parity_postselect(counts, expected_parity=0)
        assert isinstance(result, SymmetryVerificationResult)


class TestSymmetryExpand:
    def test_correct_parity_unchanged(self):
        counts = {"0000": 500, "0011": 300}
        expanded = symmetry_expand(counts, expected_parity=0)
        assert expanded == {"0000": 500, "0011": 300}

    def test_wrong_parity_flipped(self):
        counts = {"0001": 100}  # odd parity -> flip LSB -> "0000"
        expanded = symmetry_expand(counts, expected_parity=0)
        assert expanded == {"0000": 100}

    def test_mixed_accumulates(self):
        counts = {"0000": 400, "0001": 100}
        expanded = symmetry_expand(counts, expected_parity=0)
        # "0001" (odd) -> flip LSB -> "0000", added to existing 400
        assert expanded == {"0000": 500}

    def test_preserves_total_shots(self):
        counts = {"0000": 300, "0001": 200, "0011": 150, "0111": 350}
        expanded = symmetry_expand(counts, expected_parity=0)
        total_in = sum(counts.values())
        total_out = sum(expanded.values())
        assert total_out == total_in

    def test_spaces_stripped(self):
        counts = {"00 00": 100}
        expanded = symmetry_expand(counts, expected_parity=0)
        assert "0000" in expanded


class TestParityVerifiedExpectation:
    def test_pure_zero_state(self):
        counts = {"0000": 1000}
        exp, std, rej = parity_verified_expectation(counts, 4, expected_parity=0)
        np.testing.assert_array_almost_equal(exp, [1.0, 1.0, 1.0, 1.0])
        assert rej == 0.0

    def test_pure_one_state(self):
        counts = {"1111": 1000}
        exp, std, rej = parity_verified_expectation(counts, 4, expected_parity=0)
        np.testing.assert_array_almost_equal(exp, [-1.0, -1.0, -1.0, -1.0])
        assert rej == 0.0

    def test_removes_errors(self):
        # Ground truth: all |0000>, but 30% flipped to |0001> by noise
        counts = {"0000": 700, "0001": 300}
        exp_raw = np.zeros(4)
        for bs, c in counts.items():
            for q in range(4):
                bit = int(bs[-(q + 1)])
                exp_raw[q] += (1 - 2 * bit) * c
        exp_raw /= 1000

        exp_ver, _, rej = parity_verified_expectation(counts, 4, expected_parity=0)
        # Verified should recover perfect |0000> expectations
        np.testing.assert_array_almost_equal(exp_ver, [1.0, 1.0, 1.0, 1.0])
        assert abs(rej - 0.3) < 1e-10
        # Raw qubit 0 was degraded by the 300 errors
        assert exp_raw[0] < 1.0

    def test_all_rejected_returns_zeros(self):
        counts = {"0001": 1000}
        exp, std, rej = parity_verified_expectation(counts, 4, expected_parity=0)
        np.testing.assert_array_almost_equal(exp, [0.0, 0.0, 0.0, 0.0])
        assert rej == 1.0


class TestParityVerifiedR:
    def test_perfect_sync_state(self):
        # All qubits measure +1 in X and 0 in Y -> R = 1.0
        z_counts = {"0000": 1000}
        x_counts = {"0000": 1000}  # all +X eigenstate
        y_counts = {"0000": 1000}
        result = parity_verified_R(z_counts, x_counts, y_counts, 4, expected_parity=0)
        assert "R_raw" in result
        assert "R_verified" in result
        assert "z_rejection_rate" in result
        assert "improvement" in result

    def test_noisy_improves(self):
        # Simulate: 80% correct (even parity), 20% single-bit errors (odd parity)
        z_counts = {"0000": 800, "0001": 200}
        x_counts = {"0000": 800, "0001": 200}
        y_counts = {"0000": 800, "0001": 200}
        result = parity_verified_R(z_counts, x_counts, y_counts, 4, expected_parity=0)
        # Z rejection should remove the 20% errors
        assert result["z_rejection_rate"] == pytest.approx(0.2)

    def test_two_qubit_system(self):
        z_counts = {"00": 500, "11": 500}
        x_counts = {"00": 500, "11": 500}
        y_counts = {"00": 500, "11": 500}
        result = parity_verified_R(z_counts, x_counts, y_counts, 2, expected_parity=0)
        assert result["z_rejection_rate"] == 0.0


class TestIntegrationWithExperiments:
    """Verify the module integrates with existing experiment helpers."""

    def test_expectation_matches_format(self):
        from scpn_quantum_control.hardware.experiments import _expectation_per_qubit

        counts = {"0000": 700, "0011": 300}
        # Raw
        exp_raw, _ = _expectation_per_qubit(counts, 4)
        # Verified (all even parity, so identical)
        exp_ver, _, rej = parity_verified_expectation(counts, 4, expected_parity=0)
        np.testing.assert_array_almost_equal(exp_raw, exp_ver)
        assert rej == 0.0

    def test_parity_conservation_proof(self):
        """The XY Hamiltonian conserves parity within each sector.

        Start from |00⟩ (definite even parity), evolve under XX+YY,
        verify all output amplitudes remain in the even sector.
        Ry initialization mixes parity sectors, so we test from |00⟩.
        """
        try:
            from qiskit import QuantumCircuit
            from qiskit.quantum_info import Statevector
        except ImportError:
            pytest.skip("Qiskit not available")

        # Start from |00⟩ (definite even parity)
        qc = QuantumCircuit(2)
        # XX+YY interaction conserves parity
        qc.rxx(0.4, 0, 1)
        qc.ryy(0.4, 0, 1)

        sv = Statevector.from_instruction(qc)
        probs = sv.probabilities_dict()

        # |00⟩ has even parity; XY evolution preserves it
        for bs, p in probs.items():
            if p > 1e-10:
                assert bitstring_parity(bs) == 0, (
                    f"Bitstring {bs} has wrong parity (p={p:.4f})"
                )
