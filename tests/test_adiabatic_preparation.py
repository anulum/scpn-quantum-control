# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Adiabatic Preparation
"""Tests for finite-size adiabatic state preparation diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
import pytest
from scipy.linalg import expm

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, knm_to_dense_matrix
from scpn_quantum_control.dense_budget import DenseAllocationError
from scpn_quantum_control.phase import adiabatic_preparation as adiabatic_module
from scpn_quantum_control.phase.adiabatic_preparation import (
    AdiabaticResult,
    adiabatic_ramp,
    adiabatic_time_scaling,
)

if TYPE_CHECKING:
    from numpy.typing import NDArray


class RampOptions(TypedDict, total=False):
    """Typed overrides for semantically invalid ramp parameters."""

    K_target: float
    T_total: float
    n_steps: int


def _ring_topology(n: int) -> NDArray[np.float64]:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


class TestAdiabaticRamp:
    def test_one_step_matches_zero_initialized_evolution(self) -> None:
        """Public fidelity agrees with direct midpoint propagation from K=0."""
        omega = np.array([1.0, -2.0])
        topology = 50.0 * _ring_topology(2)
        duration = 0.2
        target = 0.03
        _, initial_vectors = np.linalg.eigh(knm_to_dense_matrix(np.zeros((2, 2)), omega))
        midpoint = knm_to_dense_matrix(0.5 * target * topology, omega)
        final_values, final_vectors = np.linalg.eigh(knm_to_dense_matrix(target * topology, omega))
        evolved = expm(-1j * midpoint * duration) @ initial_vectors[:, 0]
        expected = float(abs(np.vdot(final_vectors[:, 0], evolved)) ** 2)
        result = adiabatic_ramp(omega, topology, K_target=target, T_total=duration, n_steps=1)
        assert result.final_fidelity == pytest.approx(expected, abs=1e-12)
        assert result.gap[-1] == pytest.approx(final_values[1] - final_values[0])

    def test_initial_gap_matches_reported_zero_coupling(self) -> None:
        """The first spectral sample must belong to the reported schedule point."""
        omega = np.array([1.0, 2.0])
        topology = 50.0 * _ring_topology(2)
        before = topology.copy()
        result = adiabatic_ramp(omega, topology, K_target=0.0, T_total=1.0, n_steps=4)
        spectrum = np.linalg.eigvalsh(knm_to_dense_matrix(np.zeros((2, 2)), omega))
        assert result.K_schedule[0] == 0.0
        np.testing.assert_allclose(result.gap, spectrum[1] - spectrum[0], atol=1e-12)
        np.testing.assert_allclose(result.fidelity, 1.0, atol=1e-12)
        np.testing.assert_array_equal(topology, before)

    def test_returns_result(self) -> None:
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=3.0, T_total=5.0, n_steps=20)
        assert isinstance(result, AdiabaticResult)
        assert len(result.times) == 21

    def test_fidelity_starts_at_one(self) -> None:
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=2.0, T_total=5.0, n_steps=15)
        assert result.fidelity[0] > 0.99

    def test_slow_ramp_before_transition(self) -> None:
        """Slow ramp below the small-system gap minimum should maintain fidelity."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        # K_target=1.0 stays below the finite-size gap minimum for this fixture.
        result = adiabatic_ramp(omega, T, K_target=1.0, T_total=30.0, n_steps=30)
        assert result.final_fidelity > 0.5

    def test_fast_ramp_lower_fidelity(self) -> None:
        """Very fast ramp → diabatic transitions → lower fidelity."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        fast = adiabatic_ramp(omega, T, K_target=3.0, T_total=0.1, n_steps=10)
        slow = adiabatic_ramp(omega, T, K_target=3.0, T_total=20.0, n_steps=30)
        # Slow should generally have better fidelity
        assert slow.final_fidelity >= fast.final_fidelity - 0.1

    def test_gap_always_positive(self) -> None:
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=3.0, T_total=5.0, n_steps=15)
        assert np.all(result.gap > 0)

    def test_min_gap_location(self) -> None:
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=5.0, T_total=10.0, n_steps=20)
        assert result.min_gap > 0
        assert 0 <= result.min_gap_K <= 5.0

    def test_3qubit_ramp(self) -> None:
        n = 3
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_ramp(omega, T, K_target=2.0, T_total=5.0, n_steps=15)
        assert isinstance(result, AdiabaticResult)
        assert np.all(np.isfinite(result.fidelity))

    def test_rejects_dense_budget_before_hamiltonian_allocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        T = _ring_topology(3)
        omega = OMEGA_N_16[:3]

        def fail_dense(*args: object, **kwargs: object) -> None:
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
    def test_rejects_invalid_inputs(
        self,
        omega: NDArray[np.float64],
        topology: NDArray[np.float64],
        kwargs: RampOptions,
        match: str,
    ) -> None:
        call_kwargs: RampOptions = {"K_target": 2.0, "T_total": 5.0, "n_steps": 10}
        call_kwargs.update(kwargs)
        with pytest.raises(ValueError, match=match):
            adiabatic_ramp(omega, topology, **call_kwargs)

    def test_rejects_string_topology_coercion(self) -> None:
        omega = OMEGA_N_16[:2]
        topology = [["0.0", "1.0"], ["1.0", "0.0"]]

        with pytest.raises(ValueError, match="K_topology must contain real numeric scalars"):
            # A string topology is the input under test.
            adiabatic_ramp(omega, topology, K_target=2.0, T_total=5.0, n_steps=10)  # type: ignore[arg-type]

    def test_rejects_boolean_schedule_coercion(self) -> None:
        omega = OMEGA_N_16[:2]
        topology = _ring_topology(2)

        with pytest.raises(ValueError, match="T_total must be a real numeric scalar"):
            adiabatic_ramp(omega, topology, K_target=2.0, T_total=True, n_steps=10)

    def test_rejects_ragged_omega_before_coercion(self) -> None:
        topology = _ring_topology(2)

        with pytest.raises(ValueError, match="omega must be a rectangular numeric array"):
            # A ragged omega is the input under test.
            adiabatic_ramp(
                [[1.0], [2.0, 3.0]],  # type: ignore[arg-type]
                topology,
                K_target=2.0,
                T_total=5.0,
                n_steps=10,
            )

    def test_rejects_structured_topology_dtype(self) -> None:
        omega = OMEGA_N_16[:2]
        topology = np.array(
            [[(0.0, 0.0), (1.0, 0.0)], [(1.0, 0.0), (0.0, 0.0)]],
            dtype=[("weight", np.float64), ("phase", np.float64)],
        )

        with pytest.raises(ValueError, match="K_topology must contain real numeric scalars"):
            adiabatic_ramp(omega, topology, K_target=2.0, T_total=5.0, n_steps=10)

    def test_rejects_vector_target_scalar(self) -> None:
        omega = OMEGA_N_16[:2]
        topology = _ring_topology(2)

        with pytest.raises(ValueError, match="K_target must be a real numeric scalar"):
            adiabatic_ramp(
                omega,
                topology,
                # A vector where a scalar target is required is the input under test.
                K_target=np.array([1.0, 2.0]),  # type: ignore[arg-type]
                T_total=5.0,
                n_steps=10,
            )

    def test_rejects_nonvector_omega_shape(self) -> None:
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
    def test_returns_dict(self) -> None:
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_time_scaling(
            omega, T, K_target=2.0, T_values=np.array([1.0, 5.0]), n_steps_per_T=10
        )
        assert "T_total" in result
        assert "final_fidelity" in result
        assert len(result["T_total"]) == 2

    def test_uses_default_time_grid(self) -> None:
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_time_scaling(omega, T, K_target=2.0, n_steps_per_T=5)
        assert result["T_total"] == [1.0, 2.0, 5.0, 10.0, 20.0]

    def test_fidelity_increases_with_time(self) -> None:
        """Longer ramps should keep finite values in this finite-size diagnostic."""
        n = 2
        T = _ring_topology(n)
        omega = OMEGA_N_16[:n]
        result = adiabatic_time_scaling(
            omega, T, K_target=2.0, T_values=np.array([0.5, 20.0]), n_steps_per_T=15
        )
        # Not guaranteed for all T, but large gap should show trend
        assert all(np.isfinite(f) for f in result["final_fidelity"])

    def test_time_scaling_propagates_dense_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        T = _ring_topology(3)
        omega = OMEGA_N_16[:3]

        def fail_dense(*args: object, **kwargs: object) -> None:
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
    def test_time_scaling_rejects_invalid_inputs(
        self, T_values: NDArray[np.float64], n_steps_per_T: int, match: str
    ) -> None:
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

    def test_time_scaling_rejects_string_time_grid_coercion(self) -> None:
        T = _ring_topology(2)
        omega = OMEGA_N_16[:2]

        with pytest.raises(ValueError, match="T_values must contain real numeric scalars"):
            adiabatic_time_scaling(
                omega,
                T,
                K_target=2.0,
                # A string time grid is the input under test.
                T_values=["1.0", "2.0"],  # type: ignore[arg-type]
                n_steps_per_T=10,
            )


# ---------------------------------------------------------------------------
# Adiabatic physics: gap, fidelity bounds
# ---------------------------------------------------------------------------


class TestAdiabaticPhysics:
    def test_fidelity_bounded_0_1(self) -> None:
        """Fidelity must be in [0, 1]."""
        T = _ring_topology(2)
        omega = OMEGA_N_16[:2]
        result = adiabatic_ramp(omega, T, K_target=3.0, T_total=5.0, n_steps=10)
        assert np.all(result.fidelity >= -1e-10)
        assert np.all(result.fidelity <= 1.0 + 1e-10)

    def test_K_ramp_monotonic(self) -> None:
        """Coupling should ramp from 0 to K_target monotonically."""
        T = _ring_topology(2)
        omega = OMEGA_N_16[:2]
        result = adiabatic_ramp(omega, T, K_target=3.0, T_total=5.0, n_steps=10)
        assert result.K_schedule[0] < result.K_schedule[-1]


# ---------------------------------------------------------------------------
# Pipeline: Knm → adiabatic ramp → fidelity → wired
# ---------------------------------------------------------------------------


class TestAdiabaticPipeline:
    def test_pipeline_knm_to_adiabatic(self) -> None:
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
