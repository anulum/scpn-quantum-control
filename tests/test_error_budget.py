# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Error Budget
"""Tests for surface code error budget."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.qec.error_budget import (
    ErrorBudget,
    compare_error_budgets,
    compute_error_budget,
    logical_error_rate,
    minimum_code_distance,
)


class TestLogicalErrorRate:
    """Exercise the surface-code logical-error ansatz."""

    def test_below_threshold_decreases_with_distance(self) -> None:
        """Logical error decreases as below-threshold distance increases."""
        p3 = logical_error_rate(3, 0.003)
        p5 = logical_error_rate(5, 0.003)
        p7 = logical_error_rate(7, 0.003)
        assert p3 > p5 > p7

    def test_above_threshold_returns_one(self) -> None:
        """Above-threshold physical error saturates the logical rate."""
        p = logical_error_rate(5, 0.02)
        assert p == 1.0

    def test_at_threshold_returns_one(self) -> None:
        """At p_phys = p_th, surface code provides no correction."""
        p = logical_error_rate(3, 0.01)
        assert p == 1.0

    def test_just_below_threshold(self) -> None:
        """A physical rate just below threshold remains correctable."""
        p = logical_error_rate(3, 0.0099)
        assert p < 1.0

    def test_zero_physical_rate(self) -> None:
        """A zero physical error rate yields zero logical error."""
        p = logical_error_rate(5, 0.0)
        assert p == 0.0

    def test_exponential_suppression(self) -> None:
        """Each distance increase by 2 should suppress by (p/p_th) factor."""
        p_phys = 0.001
        p5 = logical_error_rate(5, p_phys)
        p7 = logical_error_rate(7, p_phys)
        ratio = p5 / p7
        expected_ratio = (p_phys / 0.01) ** (-1)  # one more power of ratio
        assert ratio == pytest.approx(expected_ratio, rel=0.01)


class TestMinimumCodeDistance:
    """Exercise minimum surface-code distance selection."""

    def test_returns_odd(self) -> None:
        """The selected code distance is odd."""
        d = minimum_code_distance(1e-6, 0.003)
        assert d % 2 == 1

    def test_at_least_3(self) -> None:
        """The selected code distance respects the minimum of three."""
        d = minimum_code_distance(1.0, 0.003)
        assert d >= 3

    def test_lower_target_needs_higher_distance(self) -> None:
        """A tighter logical target cannot require a lower distance."""
        d_easy = minimum_code_distance(1e-3, 0.003)
        d_hard = minimum_code_distance(1e-9, 0.003)
        assert d_hard >= d_easy

    def test_lower_physical_rate_needs_lower_distance(self) -> None:
        """Cleaner hardware cannot require a higher distance."""
        d_noisy = minimum_code_distance(1e-6, 0.005)
        d_clean = minimum_code_distance(1e-6, 0.001)
        assert d_clean <= d_noisy


class TestComputeErrorBudget:
    """Exercise complete Kuramoto-XY error-budget construction."""

    def test_returns_error_budget(self) -> None:
        """The estimator returns its public result record."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        budget = compute_error_budget(K, omega)
        assert isinstance(budget, ErrorBudget)

    def test_n_oscillators(self) -> None:
        """The result records the coupling-matrix oscillator count."""
        K = build_knm_paper27(L=8)
        omega = OMEGA_N_16[:8]
        budget = compute_error_budget(K, omega)
        assert budget.n_oscillators == 8

    def test_physical_qubits_formula(self) -> None:
        """Physical-qubit totals follow the surface-code formula."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        budget = compute_error_budget(K, omega)
        d = budget.code_distance
        expected = 2 * d * d - 1
        assert budget.physical_qubits_per_osc == expected
        assert budget.total_physical_qubits == 4 * expected

    def test_code_distance_odd(self) -> None:
        """The computed error budget uses a valid odd code distance."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        budget = compute_error_budget(K, omega)
        assert budget.code_distance >= 3
        assert budget.code_distance % 2 == 1

    def test_trotter_error_within_budget(self) -> None:
        """The selected Trotter step count satisfies its allocation."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        budget = compute_error_budget(K, omega, target_total_error=0.01)
        assert budget.trotter_error <= 0.01 / 3.0 + 1e-10

    def test_lower_physical_rate_fewer_qubits(self) -> None:
        """Cleaner hardware cannot increase the qubit estimate."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        b_noisy = compute_error_budget(K, omega, p_physical=0.005)
        b_clean = compute_error_budget(K, omega, p_physical=0.001)
        assert b_clean.total_physical_qubits <= b_noisy.total_physical_qubits

    def test_scpn_16_osc_resource_estimate(self) -> None:
        """Record the actual resource estimate for 16-oscillator SCPN."""
        K = build_knm_paper27(L=16)
        omega = OMEGA_N_16
        budget = compute_error_budget(K, omega, target_total_error=0.01)
        print("\n  16-osc error budget:")
        print(f"  Code distance: d={budget.code_distance}")
        print(f"  Physical qubits/osc: {budget.physical_qubits_per_osc}")
        print(f"  Total physical qubits: {budget.total_physical_qubits}")
        print(f"  Trotter steps: {budget.n_trotter_steps}")
        print(f"  QEC rounds: {budget.qec_rounds_total}")
        print(f"  Trotter error: {budget.trotter_error:.6f}")
        print(f"  Gate error/step: {budget.gate_error_per_step:.6f}")
        print(f"  Logical error/round: {budget.logical_error_rate:.2e}")
        print(f"  Total error: {budget.total_error:.6f}")
        assert isinstance(budget.total_physical_qubits, int)


class TestCompareErrorBudgets:
    """Exercise comparison across physical-error-rate scenarios."""

    def test_returns_list(self) -> None:
        """The default comparison returns three representative budgets."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        budgets = compare_error_budgets(K, omega)
        assert len(budgets) == 3

    def test_better_hardware_fewer_qubits(self) -> None:
        """The cleanest default hardware does not require more qubits."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        budgets = compare_error_budgets(K, omega)
        # budgets[0] = 0.3%, budgets[2] = 0.01%
        assert budgets[2].total_physical_qubits <= budgets[0].total_physical_qubits

    def test_explicit_physical_rates_preserve_order(self) -> None:
        """Explicit physical rates produce one ordered budget each."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        rates = np.array([0.002, 0.0005])

        budgets = compare_error_budgets(K, omega, rates)

        assert len(budgets) == len(rates)
        assert budgets[1].total_physical_qubits <= budgets[0].total_physical_qubits
