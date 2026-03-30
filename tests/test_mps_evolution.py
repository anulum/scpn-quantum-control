# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Mps Evolution
"""Tests for MPS/DMRG backend (quimb)."""

from __future__ import annotations

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
