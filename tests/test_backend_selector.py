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
import pytest

from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.phase import backend_selector as backend_selector_module
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

    def test_n15_z2_sector_when_u1_disabled(self):
        rec = recommend_backend(15, ram_gb=32, allow_u1_sector=False)
        assert rec["backend"] == "sector_ed"
        assert rec["feasible"] is True

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
        assert "AsyncHardwareRunner" in rec["note"]

    def test_open_system_lindblad(self):
        rec = recommend_backend(4, want_open_system=True)
        assert rec["backend"] == "lindblad_scipy"

    def test_open_system_tjm(self):
        rec = recommend_backend(20, want_open_system=True, has_quimb=True)
        assert rec["backend"] == "tjm_mps"
        assert rec["feasible"] is False
        assert "not yet implemented" in rec["note"]

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

    def test_lindblad_open_receives_dense_budget(self):
        from unittest.mock import patch

        K, omega = _system(4)

        class FakeLindbladSolver:
            def __init__(self, n, K_arg, omega_arg, *, gamma_amp, gamma_deph, max_dense_gib):
                assert n == 4
                assert K_arg is K
                assert omega_arg is omega
                assert gamma_amp == 0.05
                assert gamma_deph == 0.0
                assert max_dense_gib == 0.25

            def run(self, *, t_max, dt, max_dense_gib):
                assert t_max == 0.3
                assert dt == 0.1
                assert max_dense_gib == 0.25
                return {"R": np.array([0.0, 0.1])}

        with patch(
            "scpn_quantum_control.phase.lindblad.LindbladKuramotoSolver",
            FakeLindbladSolver,
        ):
            result = auto_solve(
                K,
                omega,
                want_open_system=True,
                gamma_amp=0.05,
                t_max=0.3,
                dt=0.1,
                max_dense_gib=0.25,
            )

        assert result["backend_used"] == "lindblad_scipy"

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

    def test_auto_solve_forwards_u1_policy(self):
        from unittest.mock import patch

        K, omega = _system(4)
        captured = {}

        def fake_recommend(n, **kwargs):
            captured.update(kwargs)
            return {
                "backend": "exact_diag",
                "reason": "test",
                "memory_mb": 1,
                "feasible": True,
            }

        with patch(
            "scpn_quantum_control.phase.backend_selector.recommend_backend",
            fake_recommend,
        ):
            result = auto_solve(K, omega, allow_u1_sector=False)

        assert result["backend_used"] == "exact_diag"
        assert captured["allow_u1_sector"] is False

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

    def test_hardware_recommendation_does_not_submit_qpu_job(self):
        """The selector recommends hardware; auto_solve fails closed locally."""
        from unittest.mock import patch

        K, omega = _system(4)
        with (
            patch(
                "scpn_quantum_control.phase.backend_selector.recommend_backend",
                return_value={
                    "backend": "hardware",
                    "reason": "test",
                    "memory_mb": 0,
                    "feasible": True,
                    "note": "Recommendation only; submit with AsyncHardwareRunner.",
                },
            ),
            pytest.raises(RuntimeError, match="will not substitute a statevector proxy"),
        ):
            auto_solve(K, omega, t_max=0.1, dt=0.1)

    def test_tjm_mps_recommendation_does_not_fall_through_to_statevector(self):
        """Open-system MPS recommendations must fail closed until executable."""
        from unittest.mock import patch

        K, omega = _system(4)
        with (
            patch(
                "scpn_quantum_control.phase.backend_selector.recommend_backend",
                return_value={
                    "backend": "tjm_mps",
                    "reason": "test",
                    "memory_mb": 1,
                    "feasible": True,
                    "note": "not executable",
                },
            ),
            pytest.raises(RuntimeError, match="tjm_mps"),
        ):
            auto_solve(K, omega, want_open_system=True, t_max=0.1, dt=0.1)

    def test_exact_diag_propagates_dense_budget_before_builder(self, monkeypatch):
        from unittest.mock import patch

        K, omega = _system(4)

        def fail_dense(*args, **kwargs):
            raise AssertionError("dense builder must not run after budget rejection")

        monkeypatch.setattr(backend_selector_module, "knm_to_dense_matrix", fail_dense)
        with (
            patch(
                "scpn_quantum_control.phase.backend_selector.recommend_backend",
                return_value={
                    "backend": "exact_diag",
                    "reason": "test",
                    "memory_mb": 1,
                    "feasible": True,
                },
            ),
            pytest.raises(DenseAllocationError, match="auto_solve exact diagonalisation"),
        ):
            auto_solve(K, omega, max_dense_gib=1e-12)

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
