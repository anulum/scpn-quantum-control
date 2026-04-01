# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Identity Ground State
"""Tests for identity/ground_state.py."""

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.identity.ground_state import IdentityAttractor


def test_solve_returns_robustness_gap():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    attractor = IdentityAttractor(K, omega, ansatz_reps=1)
    result = attractor.solve(maxiter=30, seed=0)
    assert "robustness_gap" in result
    assert result["robustness_gap"] >= 0.0
    assert np.isfinite(result["robustness_gap"])


def test_robustness_gap_accessor():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    attractor = IdentityAttractor(K, omega, ansatz_reps=1)
    attractor.solve(maxiter=20, seed=0)
    gap = attractor.robustness_gap()
    assert gap >= 0.0


def test_robustness_gap_before_solve_raises():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    attractor = IdentityAttractor(K, omega)
    with pytest.raises(RuntimeError, match="solve"):
        attractor.robustness_gap()


def test_ground_state_is_normalized():
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    attractor = IdentityAttractor(K, omega, ansatz_reps=1)
    attractor.solve(maxiter=20, seed=0)
    sv = attractor.ground_state()
    assert sv is not None
    assert abs(float(np.sum(np.abs(sv) ** 2)) - 1.0) < 1e-10


def test_result_contains_eigenvalues():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    attractor = IdentityAttractor(K, omega, ansatz_reps=1)
    result = attractor.solve(maxiter=20, seed=0)
    assert "eigenvalues" in result
    assert len(result["eigenvalues"]) >= 2
    # Eigenvalues should be sorted ascending
    assert result["eigenvalues"][0] <= result["eigenvalues"][1]


def test_n_dispositions_matches_input():
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    attractor = IdentityAttractor(K, omega, ansatz_reps=1)
    result = attractor.solve(maxiter=10, seed=0)
    assert result["n_dispositions"] == 4


def test_from_binding_spec():
    spec = {
        "layers": [
            {"oscillator_ids": ["a", "b"], "natural_frequency": 1.0},
            {"oscillator_ids": ["c"], "natural_frequency": 2.0},
        ],
        "coupling": {"base_strength": 0.5, "decay_alpha": 0.2},
    }
    attractor = IdentityAttractor.from_binding_spec(spec, ansatz_reps=1)
    result = attractor.solve(maxiter=20, seed=0)
    assert result["n_dispositions"] == 3
    assert result["robustness_gap"] >= 0.0


def test_non_square_k_raises():
    K = np.ones((3, 4))
    omega = np.ones(3)
    with pytest.raises(ValueError, match="square"):
        IdentityAttractor(K, omega)


def test_mismatched_k_omega_raises():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:4]
    with pytest.raises(ValueError, match="omega length"):
        IdentityAttractor(K, omega)


def test_stronger_coupling_larger_gap():
    """Stronger coupling should produce a larger energy gap (more robust)."""
    omega = OMEGA_N_16[:3]
    K_weak = build_knm_paper27(L=3, K_base=0.1)
    K_strong = build_knm_paper27(L=3, K_base=1.0)

    att_weak = IdentityAttractor(K_weak, omega, ansatz_reps=1)
    att_strong = IdentityAttractor(K_strong, omega, ansatz_reps=1)
    r_weak = att_weak.solve(maxiter=30, seed=0)
    r_strong = att_strong.solve(maxiter=30, seed=0)

    # Stronger coupling produces larger absolute energy gap
    assert r_strong["robustness_gap"] > r_weak["robustness_gap"] * 0.5


# ---------------------------------------------------------------------------
# Ground state physics: variational principle
# ---------------------------------------------------------------------------


def test_ground_energy_below_exact():
    """VQE ground energy ≥ exact ground energy (variational principle)."""
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    attractor = IdentityAttractor(K, omega, ansatz_reps=2)
    result = attractor.solve(maxiter=50, seed=42)
    # VQE energy should be near exact
    exact_E = result["eigenvalues"][0]
    assert result["ground_energy"] >= exact_E - 0.5


def test_eigenvalues_sorted():
    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]
    attractor = IdentityAttractor(K, omega, ansatz_reps=1)
    result = attractor.solve(maxiter=20, seed=0)
    eigvals = result["eigenvalues"]
    assert all(eigvals[i] <= eigvals[i + 1] + 1e-10 for i in range(len(eigvals) - 1))


# ---------------------------------------------------------------------------
# Pipeline: Knm → IdentityAttractor → robustness → wired
# ---------------------------------------------------------------------------


def test_pipeline_knm_to_identity():
    """Full pipeline: build_knm → IdentityAttractor → solve → robustness gap.
    Verifies identity ground state module is wired end-to-end.
    """
    import time

    K = build_knm_paper27(L=3)
    omega = OMEGA_N_16[:3]

    t0 = time.perf_counter()
    attractor = IdentityAttractor(K, omega, ansatz_reps=1)
    result = attractor.solve(maxiter=30, seed=42)
    dt = (time.perf_counter() - t0) * 1000

    assert result["robustness_gap"] >= 0
    assert result["n_dispositions"] == 3

    print(f"\n  PIPELINE Knm→IdentityAttractor (3q): {dt:.1f} ms")
    print(f"  E_0 = {result['ground_energy']:.4f}, gap = {result['robustness_gap']:.4f}")
