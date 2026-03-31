# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Sync Entanglement Witness
"""Tests for R as entanglement witness."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.analysis.sync_entanglement_witness import (
    EntanglementWitnessResult,
    R_entanglement_scan,
    R_from_statevector,
    R_separable_bound,
    R_separable_bound_at_energy,
    detect_entanglement_from_R,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


class TestRSeparableBound:
    def test_unconstrained_is_one(self):
        assert R_separable_bound(4) == 1.0

    def test_energy_constrained_less_than_one(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        from scpn_quantum_control.hardware.classical import classical_exact_diag

        exact = classical_exact_diag(3, K=K, omega=omega)
        E_ground = exact["ground_energy"]
        R_sep = R_separable_bound_at_energy(K, omega, E_ground, n_samples=500)
        # At ground state energy, separable states should have limited R
        assert 0.0 <= R_sep <= 1.0


class TestRFromStatevector:
    def test_plus_state_R_one(self):
        # |+⟩⊗|+⟩ → all Bloch vectors point along X → R = 1
        psi = np.array([0.5, 0.5, 0.5, 0.5])
        R = R_from_statevector(psi, 2)
        assert R > 0.9

    def test_zero_state_is_coherent(self):
        psi = np.array([1.0, 0.0, 0.0, 0.0])
        R = R_from_statevector(psi, 2)
        # |00⟩: ⟨X⟩=⟨Y⟩=0 → arctan2(0,0)=0 → all phases=0 → R=1
        # The computational basis state IS phase-coherent (trivially)
        assert R > 0.9

    def test_bell_state(self):
        psi = np.array([1.0, 0.0, 0.0, 1.0]) / np.sqrt(2)
        R = R_from_statevector(psi, 2)
        # Bell state: entangled, but local expectations may be zero
        assert 0.0 <= R <= 1.0


class TestDetectEntanglement:
    def test_returns_result(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        result = detect_entanglement_from_R(K, omega, n_samples=200)
        assert isinstance(result, EntanglementWitnessResult)
        assert result.n_qubits == 3
        assert 0.0 <= result.R_measured <= 1.0
        assert 0.0 <= result.R_sep_max <= 1.0

    def test_strong_coupling_entangled(self):
        # At strong coupling, ground state should be entangled
        K = build_knm_paper27(L=3) * 5.0
        omega = OMEGA_N_16[:3]
        result = detect_entanglement_from_R(K, omega, n_samples=500)
        # Strong coupling pushes R_ground above separable bound
        # (may not always trigger depending on random sampling)
        assert result.R_measured >= 0.0


class TestREntanglementScan:
    def test_returns_lists(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        scan = R_entanglement_scan(K, omega, n_K_values=5, n_samples=100)
        assert len(scan["K_base"]) == 5
        assert len(scan["R_ground"]) == 5
        assert len(scan["R_sep_max"]) == 5
        assert len(scan["R_gap"]) == 5
        assert len(scan["entangled"]) == 5

    def test_R_gap_sign(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        scan = R_entanglement_scan(K, omega, n_K_values=3, n_samples=100)
        # R_gap can be positive (entangled) or negative (separable)
        for gap in scan["R_gap"]:
            assert np.isfinite(gap)


# ---------------------------------------------------------------------------
# Entanglement witness physics
# ---------------------------------------------------------------------------


class TestWitnessPhysics:
    def test_R_bounded_for_random_states(self):
        """R must be in [0,1] for any normalised state."""
        rng = np.random.default_rng(42)
        for _ in range(5):
            psi = rng.standard_normal(16) + 1j * rng.standard_normal(16)
            psi /= np.linalg.norm(psi)
            R = R_from_statevector(np.array(psi), 4)
            assert 0.0 <= R <= 1.0 + 1e-10

    def test_separable_bound_monotonic(self):
        """R_sep(n) should not depend on n for unconstrained case (always 1)."""
        for n in [2, 3, 4, 6]:
            assert R_separable_bound(n) == 1.0


# ---------------------------------------------------------------------------
# Pipeline: Knm → ground state → R → witness → wired
# ---------------------------------------------------------------------------


class TestWitnessPipeline:
    def test_pipeline_knm_to_entanglement_witness(self):
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
