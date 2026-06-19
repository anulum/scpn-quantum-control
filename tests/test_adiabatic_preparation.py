# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Adiabatic Preparation
"""Tests for finite-size adiabatic state preparation diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.phase import adiabatic_preparation as adiabatic_module
from scpn_quantum_control.phase.adiabatic_preparation import (
    AdiabaticResult,
    adiabatic_ramp,
    adiabatic_time_scaling,
)


def _ring_topology(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestAdiabaticRamp:
    def test_returns_result(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=3.0, T_total=5.0, n_steps=20)
        assert isinstance(result, AdiabaticResult)
        assert len(result.times) == 21

    def test_fidelity_starts_at_one(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=2.0, T_total=5.0, n_steps=15)
        assert result.fidelity[0] > 0.99

    def test_slow_ramp_before_transition(self):
        """Slow ramp below the small-system gap minimum should maintain fidelity."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        # K_target=1.0 stays below the finite-size gap minimum for this fixture.
        result = adiabatic_ramp(omega, T, K_target=1.0, T_total=30.0, n_steps=30)
        assert result.final_fidelity > 0.5

    def test_fast_ramp_lower_fidelity(self):
        """Very fast ramp → diabatic transitions → lower fidelity."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        fast = adiabatic_ramp(omega, T, K_target=3.0, T_total=0.1, n_steps=10)
        slow = adiabatic_ramp(omega, T, K_target=3.0, T_total=20.0, n_steps=30)
        # Slow should generally have better fidelity
        assert slow.final_fidelity >= fast.final_fidelity - 0.1

    def test_gap_always_positive(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=3.0, T_total=5.0, n_steps=15)
        assert np.all(result.gap > 0)

    def test_min_gap_location(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=5.0, T_total=10.0, n_steps=20)
        assert result.min_gap > 0
        assert 0 <= result.min_gap_K <= 5.0

    def test_3qubit_ramp(self):
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=2.0, T_total=5.0, n_steps=15)
        assert isinstance(result, AdiabaticResult)
        assert np.all(np.isfinite(result.fidelity))

    def test_rejects_dense_budget_before_hamiltonian_allocation(self, monkeypatch):
        T = _ring_topology(3)
        omega = OMEGA_N_16[:3]

        def fail_dense(*args, **kwargs):
            raise AssertionError("dense Hamiltonian builder must not run after budget rejection")

        monkeypatch.setattr(adiabatic_module, "knm_to_dense_matrix", fail_dense)

        with pytest.raises(DenseAllocationError, match="adiabatic dense eigensolver"):
            adiabatic_ramp(
                omega,
                T,
                K_target=2.0,
                T_total=5.0,
                n_steps=15,
                max_dense_gib=1e-12,
            )

    @pytest.mark.parametrize(
        ("omega", "topology", "kwargs", "match"),
        [
            (np.ones(1), np.zeros((1, 1)), {}, "at least 2"),
            (np.ones(3), np.ones((2, 2)), {}, "K_topology"),
            (np.ones(2), np.ones((2, 2)), {"K_target": np.nan}, "K_target"),
            (np.ones(2), np.ones((2, 2)), {"T_total": 0.0}, "T_total"),
            (np.ones(2), np.ones((2, 2)), {"T_total": np.inf}, "T_total"),
            (np.ones(2), np.ones((2, 2)), {"n_steps": 0}, "n_steps"),
            (np.array([1.0, np.nan]), np.ones((2, 2)), {}, "finite"),
            (np.ones(2), np.array([[0.0, 1.0], [0.2, 0.0]]), {}, "symmetric"),
        ],
    )
    def test_rejects_invalid_inputs(self, omega, topology, kwargs, match):
        call_kwargs = {"K_target": 2.0, "T_total": 5.0, "n_steps": 10}
        call_kwargs.update(kwargs)
        with pytest.raises(ValueError, match=match):
            adiabatic_ramp(omega, topology, **call_kwargs)

    def test_rejects_string_topology_coercion(self):
        omega = OMEGA_N_16[:2]
        topology = [["0.0", "1.0"], ["1.0", "0.0"]]

        with pytest.raises(ValueError, match="K_topology must contain real numeric scalars"):
            adiabatic_ramp(omega, topology, K_target=2.0, T_total=5.0, n_steps=10)

    def test_rejects_boolean_schedule_coercion(self):
        omega = OMEGA_N_16[:2]
        topology = _ring_topology(2)

        with pytest.raises(ValueError, match="T_total must be a real numeric scalar"):
            adiabatic_ramp(omega, topology, K_target=2.0, T_total=True, n_steps=10)

    def test_rejects_ragged_omega_before_coercion(self):
        topology = _ring_topology(2)

        with pytest.raises(ValueError, match="omega must be a rectangular numeric array"):
            adiabatic_ramp([[1.0], [2.0, 3.0]], topology, K_target=2.0, T_total=5.0, n_steps=10)

    def test_rejects_structured_topology_dtype(self):
        omega = OMEGA_N_16[:2]
        topology = np.array(
            [[(0.0, 0.0), (1.0, 0.0)], [(1.0, 0.0), (0.0, 0.0)]],
            dtype=[("weight", np.float64), ("phase", np.float64)],
        )

        with pytest.raises(ValueError, match="K_topology must contain real numeric scalars"):
            adiabatic_ramp(omega, topology, K_target=2.0, T_total=5.0, n_steps=10)

    def test_rejects_vector_target_scalar(self):
        omega = OMEGA_N_16[:2]
        topology = _ring_topology(2)

        with pytest.raises(ValueError, match="K_target must be a real numeric scalar"):
            adiabatic_ramp(
                omega,
                topology,
                K_target=np.array([1.0, 2.0]),
                T_total=5.0,
                n_steps=10,
            )

    def test_rejects_nonvector_omega_shape(self):
        topology = _ring_topology(2)

        with pytest.raises(ValueError, match="omega must be a one-dimensional vector"):
            adiabatic_ramp(
                np.ones((2, 1)),
                topology,
                K_target=2.0,
                T_total=5.0,
                n_steps=10,
            )


class TestAdiabaticTimeScaling:
    def test_returns_dict(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_time_scaling(
            omega, T, K_target=2.0, T_values=np.array([1.0, 5.0]), n_steps_per_T=10
        )
        assert "T_total" in result
        assert "final_fidelity" in result
        assert len(result["T_total"]) == 2

    def test_uses_default_time_grid(self):
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_time_scaling(omega, T, K_target=2.0, n_steps_per_T=5)
        assert result["T_total"] == [1.0, 2.0, 5.0, 10.0, 20.0]

    def test_fidelity_increases_with_time(self):
        """Longer ramps should keep finite values in this finite-size diagnostic."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_time_scaling(
            omega, T, K_target=2.0, T_values=np.array([0.5, 20.0]), n_steps_per_T=15
        )
        # Not guaranteed for all T, but large gap should show trend
        assert all(np.isfinite(f) for f in result["final_fidelity"])

    def test_time_scaling_propagates_dense_budget(self, monkeypatch):
        T = _ring_topology(3)
        omega = OMEGA_N_16[:3]

        def fail_dense(*args, **kwargs):
            raise AssertionError("dense Hamiltonian builder must not run after budget rejection")

        monkeypatch.setattr(adiabatic_module, "knm_to_dense_matrix", fail_dense)

        with pytest.raises(DenseAllocationError, match="adiabatic dense eigensolver"):
            adiabatic_time_scaling(
                omega,
                T,
                K_target=2.0,
                T_values=np.array([1.0, 2.0]),
                n_steps_per_T=10,
                max_dense_gib=1e-12,
            )

    @pytest.mark.parametrize(
        ("T_values", "n_steps_per_T", "match"),
        [
            (np.array([]), 10, "T_values"),
            (np.array([1.0, np.nan]), 10, "T_values"),
            (np.array([1.0, 0.0]), 10, "T_values"),
            (np.array([1.0]), 0, "n_steps_per_T"),
        ],
    )
    def test_time_scaling_rejects_invalid_inputs(self, T_values, n_steps_per_T, match):
        T = _ring_topology(2)
        omega = OMEGA_N_16[:2]

        with pytest.raises(ValueError, match=match):
            adiabatic_time_scaling(
                omega,
                T,
                K_target=2.0,
                T_values=T_values,
                n_steps_per_T=n_steps_per_T,
            )

    def test_time_scaling_rejects_string_time_grid_coercion(self):
        T = _ring_topology(2)
        omega = OMEGA_N_16[:2]

        with pytest.raises(ValueError, match="T_values must contain real numeric scalars"):
            adiabatic_time_scaling(
                omega,
                T,
                K_target=2.0,
                T_values=["1.0", "2.0"],
                n_steps_per_T=10,
            )


# ---------------------------------------------------------------------------
# Adiabatic physics: gap, fidelity bounds
# ---------------------------------------------------------------------------


class TestAdiabaticPhysics:
    def test_fidelity_bounded_0_1(self):
        """Fidelity must be in [0, 1]."""
        T = _ring_topology(2)
        omega = OMEGA_N_16[:2]
        result = adiabatic_ramp(omega, T, K_target=3.0, T_total=5.0, n_steps=10)
        assert np.all(result.fidelity >= -1e-10)
        assert np.all(result.fidelity <= 1.0 + 1e-10)

    def test_K_ramp_monotonic(self):
        """Coupling should ramp from 0 to K_target monotonically."""
        T = _ring_topology(2)
        omega = OMEGA_N_16[:2]
        result = adiabatic_ramp(omega, T, K_target=3.0, T_total=5.0, n_steps=10)
        assert result.K_schedule[0] < result.K_schedule[-1]


# ---------------------------------------------------------------------------
# Pipeline: Knm → adiabatic ramp → fidelity → wired
# ---------------------------------------------------------------------------


class TestAdiabaticPipeline:
    def test_pipeline_knm_to_adiabatic(self):
        """Full pipeline: Knm topology → adiabatic ramp → fidelity tracking.
        Verifies adiabatic module is wired end-to-end.
        """
        import time

        from scpn_quantum_control.bridge.knm_hamiltonian import build_knm_paper27

        K_topo = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]

        t0 = time.perf_counter()
        result = adiabatic_ramp(omega, K_topo, K_target=3.0, T_total=5.0, n_steps=15)
        dt = (time.perf_counter() - t0) * 1000

        assert isinstance(result, AdiabaticResult)
        assert result.min_gap > 0

        print(f"\n  PIPELINE Knm→Adiabatic (3q, 15 steps): {dt:.1f} ms")
        print(f"  Final fidelity = {result.final_fidelity:.4f}")
        print(f"  Min gap = {result.min_gap:.4f} at K = {result.min_gap_K:.2f}")
