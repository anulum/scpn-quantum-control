# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Mps Evolution
"""Tests for MPS/DMRG backend (quimb)."""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pytest

quimb = pytest.importorskip("quimb")

from scpn_quantum_control.phase.mps_evolution import (
    dmrg_ground_state,
    is_quimb_available,
    tebd_evolution,
)


class TestQuimbAvailable:
    def test_quimb_installed(self):
        assert is_quimb_available()


class TestDMRG:
    def setup_method(self):
        self.n = 4
        self.K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(self.n), range(self.n))))
        self.omega = np.linspace(0.8, 1.2, self.n)

    def test_dmrg_returns_energy(self):
        result = dmrg_ground_state(self.K, self.omega, bond_dim=16, max_sweeps=5)
        assert "energy" in result
        assert isinstance(result["energy"], float)

    def test_dmrg_energy_below_zero(self):
        result = dmrg_ground_state(self.K, self.omega, bond_dim=16, max_sweeps=10)
        assert result["energy"] < 0, "Ground state energy should be negative"

    def test_dmrg_energy_reasonable(self):
        # MPS uses NN-only coupling (long-range dropped), so won't match full ED exactly.
        # Just verify energy is negative and finite.
        result = dmrg_ground_state(self.K, self.omega, bond_dim=32, max_sweeps=20)
        assert result["energy"] < 0, "DMRG ground energy should be negative"
        assert result["energy"] > -20, "DMRG energy should be finite"

    def test_dmrg_bond_dims(self):
        result = dmrg_ground_state(self.K, self.omega, bond_dim=16, max_sweeps=5)
        assert len(result["bond_dims"]) == self.n - 1
        assert all(d >= 1 for d in result["bond_dims"])

    def test_dmrg_output_keys(self):
        result = dmrg_ground_state(self.K, self.omega, bond_dim=8, max_sweeps=3)
        assert set(result.keys()) == {"energy", "mps", "converged", "bond_dims", "n_oscillators"}


class TestTEBD:
    def setup_method(self):
        self.n = 4
        self.K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(self.n), range(self.n))))
        self.omega = np.linspace(0.8, 1.2, self.n)

    def test_tebd_returns_R(self):
        result = tebd_evolution(self.K, self.omega, t_max=0.2, dt=0.05, bond_dim=16)
        assert "R" in result
        assert len(result["R"]) == 5  # 0.0, 0.05, 0.10, 0.15, 0.20

    def test_tebd_R_bounded(self):
        result = tebd_evolution(self.K, self.omega, t_max=0.5, dt=0.1, bond_dim=16)
        assert all(0 <= r <= 1.01 for r in result["R"])

    def test_tebd_output_keys(self):
        result = tebd_evolution(self.K, self.omega, t_max=0.1, dt=0.05, bond_dim=8)
        assert set(result.keys()) == {"times", "R", "bond_dims_final", "mps_final"}

    def test_tebd_bond_dims(self):
        result = tebd_evolution(self.K, self.omega, t_max=0.2, dt=0.05, bond_dim=16)
        assert len(result["bond_dims_final"]) == self.n - 1


# ---------------------------------------------------------------------------
# MPS physics: bond dimension and entanglement
# ---------------------------------------------------------------------------


class TestMPSPhysics:
    def test_higher_bond_dim_lower_energy(self):
        """More bond dim → better variational energy (more entanglement captured)."""
        n = 4
        K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
        omega = np.linspace(0.8, 1.2, n)
        r8 = dmrg_ground_state(K, omega, bond_dim=8, max_sweeps=10)
        r32 = dmrg_ground_state(K, omega, bond_dim=32, max_sweeps=10)
        assert r32["energy"] <= r8["energy"] + 0.1  # generous tolerance

    def test_tebd_times_monotonic(self):
        n = 4
        K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
        omega = np.linspace(0.8, 1.2, n)
        result = tebd_evolution(K, omega, t_max=0.2, dt=0.05, bond_dim=16)
        assert np.all(np.diff(result["times"]) > 0)


# ---------------------------------------------------------------------------
# Pipeline: Knm → DMRG → ground energy → wired
# ---------------------------------------------------------------------------


class TestMPSImportErrors:
    """Cover ImportError paths when quimb unavailable (lines 45, 97, 151)."""

    def test_import_guard_without_quimb(self):
        """A host without quimb gets explicit unavailable status/errors."""
        import scpn_quantum_control.phase.mps_evolution as mps_mod

        class BlockQuimbImport:
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "quimb" or fullname.startswith("quimb."):
                    raise ImportError("blocked quimb")
                return None

        blocker = BlockQuimbImport()
        saved_modules = {
            name: module
            for name, module in sys.modules.items()
            if name == "quimb" or name.startswith("quimb.")
        }
        for name in saved_modules:
            sys.modules.pop(name, None)
        sys.meta_path.insert(0, blocker)
        try:
            reloaded = importlib.reload(mps_mod)
            assert reloaded.is_quimb_available() is False
            with pytest.raises(ImportError, match="quimb not installed"):
                reloaded.dmrg_ground_state(np.eye(2), np.ones(2))
        finally:
            sys.meta_path.remove(blocker)
            sys.modules.update(saved_modules)
            importlib.reload(mps_mod)

    def test_build_mpo_raises_without_quimb(self):
        import scpn_quantum_control.phase.mps_evolution as mps_mod

        orig = mps_mod._QUIMB_AVAILABLE
        try:
            mps_mod._QUIMB_AVAILABLE = False
            with pytest.raises(ImportError, match="quimb not installed"):
                mps_mod._build_mpo_hamiltonian(np.eye(2), np.ones(2))
        finally:
            mps_mod._QUIMB_AVAILABLE = orig

    def test_dmrg_raises_without_quimb(self):
        import scpn_quantum_control.phase.mps_evolution as mps_mod

        orig = mps_mod._QUIMB_AVAILABLE
        try:
            mps_mod._QUIMB_AVAILABLE = False
            with pytest.raises(ImportError, match="quimb not installed"):
                mps_mod.dmrg_ground_state(np.eye(2), np.ones(2))
        finally:
            mps_mod._QUIMB_AVAILABLE = orig

    def test_tebd_raises_without_quimb(self):
        import scpn_quantum_control.phase.mps_evolution as mps_mod

        orig = mps_mod._QUIMB_AVAILABLE
        try:
            mps_mod._QUIMB_AVAILABLE = False
            with pytest.raises(ImportError, match="quimb not installed"):
                mps_mod.tebd_evolution(np.eye(2), np.ones(2))
        finally:
            mps_mod._QUIMB_AVAILABLE = orig


class TestMPSZeroCoupling:
    """Cover 'continue' paths when K[i, i+1] ≈ 0 (lines 58, 162)."""

    def test_dmrg_sparse_coupling(self):
        """K with zero nearest-neighbour entry → continue in _build_mpo_hamiltonian."""
        n = 4
        K = np.zeros((n, n))
        # Only K[0,1] non-zero, K[1,2] = K[2,3] = 0
        K[0, 1] = K[1, 0] = 0.5
        omega = np.ones(n)
        result = dmrg_ground_state(K, omega, bond_dim=8, max_sweeps=3)
        assert "energy" in result

    def test_tebd_with_zero_omega(self):
        """TEBD with zero omega — only coupling terms, no Z field."""
        n = 4
        K = 0.45 * np.exp(-0.3 * np.abs(np.subtract.outer(range(n), range(n))))
        np.fill_diagonal(K, 0.0)
        omega = np.zeros(n)
        result = tebd_evolution(K, omega, t_max=0.1, dt=0.05, bond_dim=8)
        assert "R" in result
        assert len(result["R"]) > 0

    def test_tebd_skips_zero_nearest_neighbour_coupling(self):
        """A missing NN edge is allowed and still yields bounded dynamics."""
        n = 4
        K = np.zeros((n, n))
        K[0, 1] = K[1, 0] = 0.4
        K[2, 3] = K[3, 2] = 0.2
        omega = np.linspace(0.8, 1.2, n)
        result = tebd_evolution(K, omega, t_max=0.1, dt=0.05, bond_dim=8)
        assert len(result["R"]) == 3
        assert np.all((result["R"] >= 0.0) & (result["R"] <= 1.01))


class TestMPSPipeline:
    def test_pipeline_knm_to_dmrg(self):
        """Full pipeline: Knm → DMRG ground state → energy.
        Verifies MPS backend is wired and produces physical energies.
        """
        import time

        from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        t0 = time.perf_counter()
        result = dmrg_ground_state(K, omega, bond_dim=16, max_sweeps=10)
        dt = (time.perf_counter() - t0) * 1000

        assert result["energy"] < 0
        assert result["n_oscillators"] == 4

        print(f"\n  PIPELINE Knm→DMRG (4q, χ=16): {dt:.1f} ms")
        print(f"  E_0 = {result['energy']:.4f}, bond_dims = {result['bond_dims']}")
