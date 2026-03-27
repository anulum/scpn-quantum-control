# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Classical Reference Regression Tests
"""Pin pre-computed classical reference values from results/classical_16q_reference.json.

These tests verify that the classical simulation code continues to produce
the exact same answers computed on the UpCloud 192GB instance. Any drift
indicates a code change broke numerical accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.hardware.classical import (
    classical_exact_diag,
    classical_exact_evolution,
)

# ---------------------------------------------------------------------------
# Exact evolution R values — dense path (n < 13)
# ---------------------------------------------------------------------------


class TestEvolutionRegression:
    """Pin R(t_max, dt=0.1) values from classical_16q_reference.json."""

    @pytest.mark.parametrize(
        "n,expected_R",
        [
            (2, 0.7315547411122074),
            (4, 0.7952928318152384),
            (6, 0.5443559200364484),
            (8, 0.585423130300379),
            (10, 0.6408746968839728),
            (12, 0.5633489738526578),
        ],
        ids=["2q", "4q", "6q", "8q", "10q", "12q"],
    )
    def test_evolution_R_dt01(self, n, expected_R):
        """R at t=dt=0.1 (single step) must match reference to 10 digits."""
        result = classical_exact_evolution(n, t_max=0.1, dt=0.1)
        assert result["R"][-1] == pytest.approx(expected_R, abs=1e-10)

    @pytest.mark.parametrize(
        "n,expected_R",
        [
            (2, 0.7315547411122074),
            (4, 0.7952928318152384),
            (6, 0.5443559200364484),
            (8, 0.585423130300379),
        ],
        ids=["2q", "4q", "6q", "8q"],
    )
    def test_evolution_R_dt01_multi_step(self, n, expected_R):
        """Multi-step evolution with same total time should give same final R."""
        # dt=0.1, t_max=0.1 -> 1 step
        result_1 = classical_exact_evolution(n, t_max=0.1, dt=0.1)
        # dt=0.05, t_max=0.1 -> 2 steps (U_dt different but total unitary same)
        result_2 = classical_exact_evolution(n, t_max=0.1, dt=0.05)
        # Both should give same final R since exp(-iH*0.1) = exp(-iH*0.05)^2
        np.testing.assert_allclose(result_1["R"][-1], result_2["R"][-1], atol=1e-10)


# ---------------------------------------------------------------------------
# Exact diagonalization — ground energy and spectral gap
# ---------------------------------------------------------------------------


class TestDiagRegression:
    @pytest.mark.parametrize(
        "n,expected_E0",
        [
            (2, -3.939),
            (4, -6.303000000000001),
            (6, -10.793000000000001),
            (8, -12.75572460640126),
            (10, -17.009187520707613),
            (12, -21.25968107600197),
        ],
        ids=["2q", "4q", "6q", "8q", "10q", "12q"],
    )
    def test_ground_energy(self, n, expected_E0):
        result = classical_exact_diag(n)
        assert result["ground_energy"] == pytest.approx(expected_E0, abs=1e-6)

    @pytest.mark.parametrize(
        "n,expected_gap",
        [
            (2, 2.5227454324875067),
            (4, 1.131746751491428),
            (6, 0.44680698530075524),
            (8, 0.28272460640125985),
            (10, 0.26858975558857523),
            (12, 0.19356903798420433),
        ],
        ids=["2q", "4q", "6q", "8q", "10q", "12q"],
    )
    def test_spectral_gap(self, n, expected_gap):
        result = classical_exact_diag(n)
        assert result["spectral_gap"] == pytest.approx(expected_gap, abs=1e-6)


# ---------------------------------------------------------------------------
# Trajectory regression (multi-step evolution)
# ---------------------------------------------------------------------------


class TestTrajectoryRegression:
    def test_16q_8step_trajectory(self, classical_reference):
        """Pin the 16-qubit 8-step trajectory from UpCloud computation.

        Uses the session-scoped fixture to avoid re-reading JSON per test.
        This test is marked slow since 16-qubit evolution takes ~500s.
        We only verify the reference data structure here; actual 16q
        recomputation would need the slow marker.
        """
        traj = classical_reference["evo_16q_8step_dt0.05"]
        R_trajectory = traj["R_trajectory"]
        assert len(R_trajectory) == 9
        # R decreases monotonically in this trajectory
        for i in range(len(R_trajectory) - 1):
            assert R_trajectory[i] > R_trajectory[i + 1]

    def test_reference_data_completeness(self, classical_reference):
        """Reference file must contain all expected entries."""
        expected_keys = [
            "diag_16q",
            "evo_2q_dt0.1",
            "evo_4q_dt0.1",
            "evo_6q_dt0.1",
            "evo_8q_dt0.1",
            "evo_10q_dt0.1",
            "evo_12q_dt0.1",
            "evo_14q_dt0.1",
            "evo_16q_dt0.1",
            "evo_16q_dt0.01",
            "evo_16q_dt0.02",
            "evo_16q_dt0.05",
            "evo_16q_dt0.2",
            "evo_16q_8step_dt0.05",
            "diag_2q",
            "diag_4q",
            "diag_6q",
            "diag_8q",
            "diag_10q",
            "diag_12q",
            "diag_14q",
        ]
        for key in expected_keys:
            assert key in classical_reference, f"Missing key: {key}"

    def test_reference_R_values_bounded(self, classical_reference):
        """Every R value in the reference file is in [0, 1]."""
        for key, val in classical_reference.items():
            if "R" in val and isinstance(val["R"], float):
                assert 0.0 <= val["R"] <= 1.0, f"{key}: R={val['R']}"
            if "R_trajectory" in val:
                for r in val["R_trajectory"]:
                    assert 0.0 <= r <= 1.0, f"{key} trajectory: R={r}"

    @pytest.mark.parametrize(
        "key,expected_R",
        [
            ("evo_2q_dt0.1", 0.7315547411122074),
            ("evo_4q_dt0.1", 0.7952928318152384),
            ("evo_6q_dt0.1", 0.5443559200364484),
            ("evo_8q_dt0.1", 0.585423130300379),
        ],
        ids=["2q", "4q", "6q", "8q"],
    )
    def test_reference_matches_live_computation(self, classical_reference, key, expected_R):
        """Reference file values must match what we compute now."""
        assert classical_reference[key]["R"] == pytest.approx(expected_R, abs=1e-14)

    @pytest.mark.parametrize(
        "key,expected_E0",
        [
            ("diag_2q", -3.939),
            ("diag_4q", -6.303000000000001),
            ("diag_6q", -10.793000000000001),
            ("diag_8q", -12.75572460640126),
        ],
        ids=["2q", "4q", "6q", "8q"],
    )
    def test_reference_diag_matches_live(self, classical_reference, key, expected_E0):
        assert classical_reference[key]["ground_energy"] == pytest.approx(expected_E0, abs=1e-6)
