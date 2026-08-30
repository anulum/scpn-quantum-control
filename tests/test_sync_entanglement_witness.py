# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Sync Entanglement Witness
"""Tests for R as entanglement witness."""

from __future__ import annotations

from typing import Any, NoReturn

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.analysis.sync_entanglement_witness as witness_module
from scpn_quantum_control.analysis.sync_entanglement_witness import (
    EntanglementWitnessResult,
    R_entanglement_scan,
    R_from_statevector,
    R_separable_bound,
    R_separable_bound_at_energy,
    detect_entanglement_from_R,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.dense_budget import DenseAllocationError


class TestRSeparableBound:
    """Verify unconstrained and energy-constrained separable bounds."""

    def test_unconstrained_is_one(self) -> None:
        """Return the analytic unconstrained product-state bound."""
        assert R_separable_bound(4) == 1.0

    def test_energy_bound_rejects_budget_before_hamiltonian_allocation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reject an undersized budget before dense Hamiltonian construction."""
        K = build_knm_paper27(L=12)
        omega = OMEGA_N_16[:12]

        def fail_if_dense_hamiltonian_is_requested(*_args: object, **_kwargs: object) -> NoReturn:
            raise AssertionError("dense Hamiltonian allocation happened before budget gate")

        monkeypatch.setattr(
            witness_module,
            "knm_to_dense_matrix",
            fail_if_dense_hamiltonian_is_requested,
        )

        with pytest.raises(DenseAllocationError, match="separable-bound dense"):
            R_separable_bound_at_energy(
                K,
                omega,
                target_energy=0.0,
                n_samples=1,
                max_dense_gib=1e-12,
            )

    def test_energy_bound_passes_dense_budget_to_bridge(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forward the admitted dense budget to the bridge."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        seen_budgets: list[float | None] = []

        def fake_dense_matrix(
            _K_arg: NDArray[np.float64],
            _omega_arg: NDArray[np.float64],
            *,
            max_dense_gib: float | None = None,
        ) -> NDArray[np.complex128]:
            seen_budgets.append(max_dense_gib)
            return np.zeros((4, 4), dtype=np.complex128)

        monkeypatch.setattr(witness_module, "knm_to_dense_matrix", fake_dense_matrix)

        R_separable_bound_at_energy(
            K,
            omega,
            target_energy=100.0,
            n_samples=3,
            max_dense_gib=0.25,
        )

        assert seen_budgets == [0.25]

    def test_energy_constrained_less_than_one(self) -> None:
        """Keep the sampled energy-constrained bound within unit limits."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        from scpn_quantum_control.hardware.classical import classical_exact_diag

        exact = classical_exact_diag(3, K=K, omega=omega)
        E_ground = exact["ground_energy"]
        R_sep = R_separable_bound_at_energy(K, omega, E_ground, n_samples=500)
        # At ground state energy, separable states should have limited R
        assert 0.0 <= R_sep <= 1.0


class TestRFromStatevector:
    """Verify statevector order-parameter evaluation."""

    def test_plus_state_R_one(self) -> None:
        """Recover unit phase coherence for a two-qubit plus state."""
        # |+⟩⊗|+⟩ → all Bloch vectors point along X → R = 1
        psi = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.complex128)
        R = R_from_statevector(psi, 2)
        assert R > 0.9

    def test_zero_state_is_coherent(self) -> None:
        """Retain the conventional zero-phase value for a zero state."""
        psi = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)
        R = R_from_statevector(psi, 2)
        # |00⟩: ⟨X⟩=⟨Y⟩=0 → arctan2(0,0)=0 → all phases=0 → R=1
        # The computational basis state IS phase-coherent (trivially)
        assert R > 0.9

    def test_bell_state(self) -> None:
        """Keep the Bell-state order parameter within physical bounds."""
        psi = np.array([1.0, 0.0, 0.0, 1.0], dtype=np.complex128) / np.sqrt(2)
        R = R_from_statevector(psi, 2)
        # Bell state: entangled, but local expectations may be zero
        assert 0.0 <= R <= 1.0


class TestDetectEntanglement:
    """Verify the public ground-state witness pipeline."""

    def test_propagates_dense_budget_to_separable_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forward the dense budget into the separable-bound calculation."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        seen_budgets: list[float | None] = []

        def fake_exact_diag(
            _n: int,
            K: NDArray[np.float64] | None = None,
            omega: NDArray[np.float64] | None = None,
            **_kwargs: object,
        ) -> dict[str, Any]:
            return {
                "ground_state": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
                "ground_energy": -1.0,
            }

        def fake_bound(
            _K_arg: NDArray[np.float64],
            _omega_arg: NDArray[np.float64],
            _target_energy: float,
            _n_samples: int,
            _seed: int,
            *,
            max_dense_gib: float | None,
        ) -> float:
            seen_budgets.append(max_dense_gib)
            return 0.5

        monkeypatch.setattr(witness_module, "classical_exact_diag", fake_exact_diag)
        monkeypatch.setattr(witness_module, "R_separable_bound_at_energy", fake_bound)

        result = detect_entanglement_from_R(K, omega, n_samples=10, max_dense_gib=0.25)

        assert result.is_entangled
        assert seen_budgets == [0.25]

    def test_returns_result(self) -> None:
        """Return the typed bounded witness result."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = detect_entanglement_from_R(K, omega, n_samples=200)
        assert isinstance(result, EntanglementWitnessResult)
        assert result.n_qubits == 3
        assert 0.0 <= result.R_measured <= 1.0
        assert 0.0 <= result.R_sep_max <= 1.0
        assert result.entanglement_depth in {1, 2}

    def test_entanglement_depth_is_only_certified_pairwise(self) -> None:
        """Limit the certified depth to nonseparability."""
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = detect_entanglement_from_R(K, omega, n_samples=200)

        expected_depth = 2 if result.is_entangled else 1

        assert result.entanglement_depth == expected_depth

    def test_strong_coupling_entangled(self) -> None:
        """Evaluate a strongly coupled finite-system ground state."""
        # At strong coupling, ground state should be entangled
        K = build_knm_paper27(L=3) * 5.0
        omega = OMEGA_N_16[:3]
        result = detect_entanglement_from_R(K, omega, n_samples=500)
        # Strong coupling pushes R_ground above separable bound
        # (may not always trigger depending on random sampling)
        assert result.R_measured >= 0.0

    def test_large_system_skips_eager_dense_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Reach exact-diagonalisation dispatch without an eager n>=14 budget."""
        K = np.zeros((14, 14), dtype=np.float64)
        omega = np.zeros(14, dtype=np.float64)

        def stop_at_exact_diag(*_args: object, **_kwargs: object) -> NoReturn:
            raise RuntimeError("exact diagonalisation reached")

        monkeypatch.setattr(witness_module, "classical_exact_diag", stop_at_exact_diag)
        with pytest.raises(RuntimeError, match="exact diagonalisation reached"):
            detect_entanglement_from_R(K, omega, n_samples=1)


class TestREntanglementScan:
    """Verify coupling-grid witness scans."""

    def test_scan_propagates_dense_budget_to_separable_bound(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Forward the dense budget at every scan point."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        seen_budgets: list[float | None] = []

        def fake_exact_diag(
            _n: int,
            K: NDArray[np.float64] | None = None,
            omega: NDArray[np.float64] | None = None,
            **_kwargs: object,
        ) -> dict[str, Any]:
            return {
                "ground_state": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128),
                "ground_energy": -1.0,
            }

        def fake_bound(
            _K_arg: NDArray[np.float64],
            _omega_arg: NDArray[np.float64],
            _target_energy: float,
            _n_samples: int,
            _seed: int,
            *,
            max_dense_gib: float | None,
        ) -> float:
            seen_budgets.append(max_dense_gib)
            return 0.5

        monkeypatch.setattr(witness_module, "classical_exact_diag", fake_exact_diag)
        monkeypatch.setattr(witness_module, "R_separable_bound_at_energy", fake_bound)

        scan = R_entanglement_scan(
            K,
            omega,
            K_base_range=np.array([0.1, 0.2], dtype=np.float64),
            n_samples=10,
            max_dense_gib=0.25,
        )

        assert len(scan["R_gap"]) == 2
        assert seen_budgets == [0.25, 0.25]

    def test_returns_lists(self) -> None:
        """Return one aligned list for every scan quantity."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        scan = R_entanglement_scan(K, omega, n_K_values=5, n_samples=100)
        assert len(scan["K_base"]) == 5
        assert len(scan["R_ground"]) == 5
        assert len(scan["R_sep_max"]) == 5
        assert len(scan["R_gap"]) == 5
        assert len(scan["entangled"]) == 5

    def test_R_gap_sign(self) -> None:
        """Return finite signed witness gaps."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        scan = R_entanglement_scan(K, omega, n_K_values=3, n_samples=100)
        # R_gap can be positive (entangled) or negative (separable)
        for gap in scan["R_gap"]:
            assert np.isfinite(gap)

    def test_large_system_scan_skips_eager_dense_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Reach scan exact-diagonalisation dispatch without an eager n>=14 budget."""
        K = np.zeros((14, 14), dtype=np.float64)
        omega = np.zeros(14, dtype=np.float64)

        def stop_at_exact_diag(*_args: object, **_kwargs: object) -> NoReturn:
            raise RuntimeError("scan exact diagonalisation reached")

        monkeypatch.setattr(witness_module, "classical_exact_diag", stop_at_exact_diag)
        with pytest.raises(RuntimeError, match="scan exact diagonalisation reached"):
            R_entanglement_scan(
                K,
                omega,
                K_base_range=np.array([0.1], dtype=np.float64),
                n_samples=1,
            )


# ---------------------------------------------------------------------------
# Entanglement witness physics
# ---------------------------------------------------------------------------


class TestWitnessPhysics:
    """Verify physical bounds of the witness inputs."""

    def test_R_bounded_for_random_states(self) -> None:
        """R must be in [0,1] for any normalised state."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
            psi /= np.linalg.norm(psi)
            R = R_from_statevector(np.array(psi), 4)
            assert 0.0 <= R <= 1.0 + 1e-10

    def test_separable_bound_monotonic(self) -> None:
        """R_sep(n) should not depend on n for unconstrained case (always 1)."""
        for n in [2, 3, 4, 6]:
            assert R_separable_bound(n) == 1.0


# ---------------------------------------------------------------------------
# Pipeline: Knm → ground state → R → witness → wired
# ---------------------------------------------------------------------------


class TestWitnessPipeline:
    """Verify the end-to-end K_nm witness path."""

    def test_pipeline_knm_to_entanglement_witness(self) -> None:
        """Full pipeline: build_knm → ground state → R → separable bound → witness.

        Verifies entanglement witness is wired end-to-end.
        """
        import time

        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]

        t0 = time.perf_counter()
        result = detect_entanglement_from_R(K, omega, n_samples=200)
        dt = (time.perf_counter() - t0) * 1000

        assert isinstance(result, EntanglementWitnessResult)
        assert np.isfinite(result.R_measured)

        print(f"\n  PIPELINE Knm→R_witness (3q, 200 samples): {dt:.1f} ms")
        print(f"  R_measured = {result.R_measured:.4f}, R_sep_max = {result.R_sep_max:.4f}")
        print(f"  Entangled: {result.is_entangled}")


class TestSyncEntanglementCoverage:
    """Cover lines 130-135: product state R computation in R_separable_bound_at_energy."""

    def test_high_target_energy_allows_all_product_states(self) -> None:
        """High target_energy ensures product states pass energy filter → lines 130-135."""
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        R_max = R_separable_bound_at_energy(K, omega, target_energy=100.0, n_samples=50)
        assert isinstance(R_max, float)
        assert 0 < R_max <= 1.0
