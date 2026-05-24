# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Backend selector contract tests
"""Contract tests for backend recommendation and automatic solver dispatch."""

from __future__ import annotations

import numpy as np
import pytest


def _system(n: int = 4):
    """Standard heterogeneous Kuramoto-XY system."""
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


def _zero_coupling(n: int = 4):
    """Decoupled system — K=0, eigenstates are product states."""
    K = np.zeros((n, n))
    omega = np.linspace(0.8, 1.2, n)
    return n, K, omega


class TestBackendSelector:
    """Tests for recommend_backend and auto_solve."""

    @pytest.mark.parametrize(
        "n,expected",
        [
            (2, "exact_diag"),
            (4, "exact_diag"),
            (8, "exact_diag"),
            (14, "exact_diag"),
        ],
    )
    def test_small_systems_select_ed(self, n, expected):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(n)
        assert rec["backend"] == expected
        assert rec["feasible"]
        assert rec["memory_mb"] > 0

    def test_medium_system_selects_sector(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(16, ram_gb=32.0)
        assert rec["backend"] in ("u1_sector_ed", "sector_ed", "statevector")
        assert rec["feasible"]

    def test_large_system_selects_mps(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(32, has_quimb=True)
        assert rec["backend"] == "mps_dmrg"

    def test_huge_system_selects_hardware(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(100, ram_gb=32.0, has_quimb=False)
        assert rec["backend"] == "hardware"

    def test_open_system_selects_lindblad(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(4, want_open_system=True)
        assert rec["backend"] == "lindblad_scipy"

    def test_open_system_large_selects_mcwf(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(14, want_open_system=True)
        assert rec["backend"] in ("mcwf", "lindblad_scipy")

    def test_recommendation_output_keys(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        rec = recommend_backend(4)
        assert {"backend", "reason", "memory_mb", "feasible"} <= set(rec.keys())
        assert isinstance(rec["backend"], str)
        assert isinstance(rec["reason"], str)
        assert isinstance(rec["memory_mb"], (int, float))
        assert isinstance(rec["feasible"], bool)

    def test_memory_increases_with_n(self):
        from scpn_quantum_control.phase.backend_selector import recommend_backend

        mem_4 = recommend_backend(4)["memory_mb"]
        mem_8 = recommend_backend(8)["memory_mb"]
        mem_12 = recommend_backend(12)["memory_mb"]
        assert mem_4 < mem_8 < mem_12, "Memory should increase with system size"

    @pytest.mark.parametrize("n", [4, 6, 8])
    def test_auto_solve_produces_ground_energy(self, n):
        from scpn_quantum_control.phase.backend_selector import auto_solve

        _, K, omega = _system(n)
        result = auto_solve(K, omega)
        assert "backend_used" in result
        assert "result" in result
        assert result["result"]["ground_energy"] < 0
        assert np.isfinite(result["result"]["ground_energy"])

    def test_auto_solve_zero_coupling(self):
        """Decoupled system: ground energy = -sum(|omega|)."""
        from scpn_quantum_control.phase.backend_selector import auto_solve

        _, K, omega = _zero_coupling(4)
        result = auto_solve(K, omega)
        E = result["result"]["ground_energy"]
        E_expected = -np.sum(np.abs(omega))
        np.testing.assert_allclose(E, E_expected, atol=1e-8)

    def test_auto_solve_matches_recommend(self):
        from scpn_quantum_control.phase.backend_selector import (
            auto_solve,
            recommend_backend,
        )

        _, K, omega = _system(6)
        rec = recommend_backend(6)
        result = auto_solve(K, omega)
        assert result["backend_used"] == rec["backend"]
