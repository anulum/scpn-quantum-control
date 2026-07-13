# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Tcbo Observer
"""Tests for quantum TCBO observer."""

from __future__ import annotations

import time

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
from scpn_quantum_control.tcbo.quantum_observer import (
    TCBOResult,
    compute_tcbo_observables,
)


class TestComputeTCBOObservables:
    def test_returns_result(self) -> None:
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_tcbo_observables(K, omega)
        assert isinstance(result, TCBOResult)

    def test_p_h1_bounded(self) -> None:
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_tcbo_observables(K, omega)
        assert 0 <= result.p_h1 <= 1.0

    def test_n_qubits(self) -> None:
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_tcbo_observables(K, omega)
        assert result.n_qubits == 4

    def test_tee_type(self) -> None:
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_tcbo_observables(K, omega)
        assert isinstance(result.tee, float)

    def test_string_order_bounded(self) -> None:
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_tcbo_observables(K, omega)
        assert abs(result.string_order) <= 1.0 + 1e-10

    def test_betti_proxies(self) -> None:
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_tcbo_observables(K, omega)
        assert 0 <= result.betti_0_proxy <= 1.0
        assert result.betti_1_proxy == result.p_h1

    def test_6_oscillators(self) -> None:
        K = build_knm_paper27(L=6)
        omega = OMEGA_N_16[:6]
        result = compute_tcbo_observables(K, omega)
        assert result.n_qubits == 6

    def test_scpn_tcbo_measurement(self) -> None:
        """Record TCBO observables at SCPN defaults."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_tcbo_observables(K, omega)
        print("\n  TCBO (4 osc):")
        print(f"  p_h1 (vortex density): {result.p_h1:.4f}")
        print(f"  TEE: {result.tee:.6f}")
        print(f"  String order: {result.string_order:.6f}")
        print(f"  β_0 proxy: {result.betti_0_proxy:.4f}")
        assert isinstance(result.p_h1, float)


# ---------------------------------------------------------------------------
# Physical invariants — topological observables
# ---------------------------------------------------------------------------


class TestTCBOPhysics:
    def test_tee_finite(self) -> None:
        """The entropy inclusion-exclusion proxy must be finite."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result = compute_tcbo_observables(K, omega)
        assert np.isfinite(result.tee)

    def test_different_coupling_changes_string_order(self) -> None:
        """Strong coupling changes the Pauli-string diagnostic."""
        omega = OMEGA_N_16[:4]
        r1 = compute_tcbo_observables(build_knm_paper27(L=4, K_base=0.1), omega)
        r2 = compute_tcbo_observables(build_knm_paper27(L=4, K_base=2.0), omega)
        assert not np.isclose(r1.string_order, r2.string_order)

    def test_betti_proxies_are_not_artificially_complemented(self) -> None:
        """The independently defined Betti-labelled proxies need not sum to one."""
        K = build_knm_paper27(L=4, K_base=2.0)
        omega = OMEGA_N_16[:4]
        result = compute_tcbo_observables(K, omega)
        assert result.betti_1_proxy == result.p_h1
        assert not np.isclose(result.betti_0_proxy + result.betti_1_proxy, 1.0)


# ---------------------------------------------------------------------------
# Pipeline: Knm → TCBO observables → wired end-to-end
# ---------------------------------------------------------------------------


class TestTCBOPipeline:
    def test_pipeline_knm_to_tcbo(self) -> None:
        """Full pipeline: build_knm → TCBO → all observables wired."""
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]

        t0 = time.perf_counter()
        result = compute_tcbo_observables(K, omega)
        dt = (time.perf_counter() - t0) * 1000

        assert result.n_qubits == 4
        assert 0 <= result.p_h1 <= 1.0

        print(f"\n  PIPELINE Knm→TCBO (4q): {dt:.1f} ms")
        print(f"  TEE={result.tee:.6f}, string_order={result.string_order:.6f}")
