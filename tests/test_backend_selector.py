# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Automatic Backend Selection
"""Tests for auto-select simulation backend.

Covers:
    - recommend_backend all branches: exact_diag, u1_sector_ed, sector_ed,
      mps_dmrg, gpu_statevector, statevector, hardware, lindblad_scipy,
      tjm_mps, open-system fallback
    - auto_solve for exact_diag, u1_sector_ed, sector_ed, lindblad, fallback
    - Memory estimates
    - Edge cases: very large n, tiny RAM
"""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.phase.backend_selector import auto_solve, recommend_backend


def _system(n: int = 4):
    K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
    np.fill_diagonal(K, 0.0)
    omega = np.linspace(0.8, 1.2, n)
    return K, omega


class TestRecommendBackend:
    def test_small_n_exact_diag(self):
        rec = recommend_backend(4)
        assert rec["backend"] == "exact_diag"
        assert rec["feasible"] is True

    def test_n14_exact_diag(self):
        rec = recommend_backend(14)
        assert rec["backend"] == "exact_diag"

    def test_n16_u1_sector(self):
        rec = recommend_backend(16)
        assert rec["backend"] == "u1_sector_ed"

    def test_n16_z2_sector_when_u1_too_big(self):
        """With tiny RAM, u1 won't fit, fall to z2."""
        rec = recommend_backend(16, ram_gb=0.001)
        # With 1 MB RAM, sector_ed won't fit either
        assert rec["backend"] in ("sector_ed", "statevector", "hardware")

    def test_n18_u1_sector(self):
        """n=18 u1 largest sector C(18,9)=48620 → ~37 GB → fits in 128 GB."""
        rec = recommend_backend(18, ram_gb=128)
        assert rec["backend"] == "u1_sector_ed"

    def test_mps_dmrg(self):
        rec = recommend_backend(30, has_quimb=True)
        assert rec["backend"] == "mps_dmrg"

    def test_gpu_statevector(self):
        rec = recommend_backend(25, has_gpu=True, ram_gb=0.001)
        assert rec["backend"] == "gpu_statevector"

    def test_cpu_statevector(self):
        """n=20 u1 sector too large → falls to statevector."""
        rec = recommend_backend(20, ram_gb=128)
        assert rec["backend"] == "statevector"

    def test_hardware_fallback(self):
        rec = recommend_backend(40, ram_gb=0.001)
        assert rec["backend"] == "hardware"

    def test_open_system_lindblad(self):
        rec = recommend_backend(4, want_open_system=True)
        assert rec["backend"] == "lindblad_scipy"

    def test_open_system_tjm(self):
        rec = recommend_backend(20, want_open_system=True, has_quimb=True)
        assert rec["backend"] == "tjm_mps"

    def test_open_system_fallback(self):
        rec = recommend_backend(20, want_open_system=True, has_quimb=False)
        assert rec["backend"] == "lindblad_scipy"
        assert "may be slow" in rec["reason"]

    def test_memory_mb_positive(self):
        rec = recommend_backend(10)
        assert rec["memory_mb"] > 0

    def test_output_keys(self):
        rec = recommend_backend(4)
        assert "backend" in rec
        assert "reason" in rec
        assert "memory_mb" in rec
        assert "feasible" in rec

    def test_statevector_path(self):
        """n=22 without quimb/gpu but enough RAM → statevector."""
        rec = recommend_backend(22, ram_gb=128, has_quimb=False, has_gpu=False)
        assert rec["backend"] == "statevector"


class TestAutoSolve:
    def test_exact_diag(self):
        K, omega = _system(4)
        result = auto_solve(K, omega)
        assert result["backend_used"] == "exact_diag"
        assert "eigvals" in result["result"]

    def test_u1_sector(self):
        """n=16 with RAM forcing u1 path, but use n=8 to avoid OOM."""
        from unittest.mock import patch

        K, omega = _system(8)
        with patch(
            "scpn_quantum_control.phase.backend_selector.recommend_backend",
            return_value={
                "backend": "u1_sector_ed",
                "reason": "test",
                "memory_mb": 100,
                "feasible": True,
            },
        ):
            result = auto_solve(K, omega)
            assert result["backend_used"] == "u1_sector_ed"
            assert "ground_energy" in result["result"]

    def test_lindblad_open(self):
        K, omega = _system(4)
        result = auto_solve(K, omega, want_open_system=True, gamma_amp=0.05, t_max=0.3, dt=0.1)
        assert result["backend_used"] == "lindblad_scipy"
        assert "R" in result["result"]

    def test_sector_ed(self):
        """Force sector_ed path via mock."""
        from unittest.mock import patch

        K, omega = _system(6)
        with patch(
            "scpn_quantum_control.phase.backend_selector.recommend_backend",
            return_value={
                "backend": "sector_ed",
                "reason": "test",
                "memory_mb": 10,
                "feasible": True,
            },
        ):
            result = auto_solve(K, omega)
            assert result["backend_used"] == "sector_ed"

    def test_result_has_recommendation(self):
        K, omega = _system(4)
        result = auto_solve(K, omega)
        assert "recommendation" in result
        assert "backend" in result["recommendation"]

    def test_mps_dmrg_path(self):
        """n=20 with quimb → DMRG path."""
        from unittest.mock import MagicMock, patch

        K, omega = _system(20)
        mock_dmrg = MagicMock(return_value={"energy": -1.0, "bond_dim": 64})
        with (
            patch(
                "scpn_quantum_control.phase.backend_selector.recommend_backend",
                return_value={
                    "backend": "mps_dmrg",
                    "reason": "test",
                    "memory_mb": 100,
                    "feasible": True,
                },
            ),
            patch(
                "scpn_quantum_control.phase.mps_evolution.dmrg_ground_state",
                mock_dmrg,
            ),
        ):
            result = auto_solve(K, omega)
            assert result["backend_used"] == "mps_dmrg"

    def test_fallback_statevector(self):
        """Force statevector fallback via mock."""
        from unittest.mock import patch

        K, omega = _system(4)
        with patch(
            "scpn_quantum_control.phase.backend_selector.recommend_backend",
            return_value={
                "backend": "statevector",
                "reason": "test",
                "memory_mb": 1,
                "feasible": True,
            },
        ):
            result = auto_solve(K, omega, t_max=0.1, dt=0.1)
            assert result["backend_used"] == "statevector"

    def test_quimb_import_exception(self):
        """Cover except branch when mps_evolution import fails."""
        from unittest.mock import patch

        K, omega = _system(4)
        with patch.dict(
            "sys.modules",
            {"scpn_quantum_control.phase.mps_evolution": None},
        ):
            result = auto_solve(K, omega)
            assert result["backend_used"] == "exact_diag"
