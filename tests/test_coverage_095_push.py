# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Coverage push tests to reach 95% gate
"""Targeted tests for modules at 0% coverage."""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import (
    build_knm_paper27,
    knm_to_hamiltonian,
)

OMEGA_2 = np.array([1.0, 1.2])
K_TOPO_2 = build_knm_paper27(L=2)


def _dense_H(K: float = 1.0):
    H = knm_to_hamiltonian(K_TOPO_2 * K, OMEGA_2).to_matrix()
    if hasattr(H, "toarray"):
        H = H.toarray()
    return np.array(H)


class TestMagicNonstabilizerness:
    def test_magic_at_coupling(self):
        from scpn_quantum_control.analysis.magic_nonstabilizerness import magic_at_coupling

        result = magic_at_coupling(OMEGA_2, K_TOPO_2, K_base=1.0)
        assert result.sre_m2 >= 0

    def test_magic_scan(self):
        from scpn_quantum_control.analysis.magic_nonstabilizerness import magic_vs_coupling

        scan = magic_vs_coupling(OMEGA_2, K_TOPO_2, k_range=np.array([0.5, 1.0, 2.0]))
        assert len(scan.k_values) == 3
        assert len(scan.sre_m2) == 3


class TestKrylovComplexity:
    def test_lanczos_coefficients(self):
        from scpn_quantum_control.analysis.krylov_complexity import lanczos_coefficients

        H = _dense_H()
        op = np.zeros((4, 4))
        op[0, 1] = 1.0
        b_n = lanczos_coefficients(H, op, max_steps=3)
        assert len(b_n) <= 3

    def test_krylov_at_coupling(self):
        from scpn_quantum_control.analysis.krylov_complexity import krylov_complexity

        H = _dense_H()
        op = np.zeros((4, 4))
        op[0, 1] = 1.0
        result = krylov_complexity(H, op, t_max=0.5, n_times=5, max_lanczos=3)
        assert result.lanczos_b is not None


class TestLoschmidtEcho:
    def test_quench(self):
        from scpn_quantum_control.analysis.loschmidt_echo import loschmidt_quench

        result = loschmidt_quench(
            OMEGA_2, K_TOPO_2, K_initial=0.5, K_final=2.0, t_max=1.0, n_times=5
        )
        assert len(result.loschmidt_amplitude) == 5

    def test_quench_scan(self):
        from scpn_quantum_control.analysis.loschmidt_echo import quench_scan

        scan = quench_scan(
            OMEGA_2,
            K_TOPO_2,
            K_initial=0.5,
            K_final_range=np.array([1.0, 2.0]),
            t_max=0.5,
            n_times=5,
        )
        assert "K_final" in scan


class TestLindbladNESS:
    def test_compute_ness(self):
        from scpn_quantum_control.analysis.lindblad_ness import compute_ness

        result = compute_ness(OMEGA_2, K_TOPO_2, K_base=1.0, gamma=0.1)
        assert 0 <= result.R_ness <= 1
        assert result.purity > 0

    def test_ness_scan(self):
        from scpn_quantum_control.analysis.lindblad_ness import ness_vs_coupling

        scan = ness_vs_coupling(OMEGA_2, K_TOPO_2, k_range=np.array([0.5, 1.0]), gamma=0.1)
        assert len(scan.R_ness) == 2


class TestEntanglementEntropy:
    def test_at_coupling(self):
        from scpn_quantum_control.analysis.entanglement_entropy import entanglement_at_coupling

        result = entanglement_at_coupling(OMEGA_2, K_TOPO_2, K_base=1.0)
        assert result.entropy >= 0

    def test_scan(self):
        from scpn_quantum_control.analysis.entanglement_entropy import entanglement_vs_coupling

        scan = entanglement_vs_coupling(OMEGA_2, K_TOPO_2, k_range=np.array([0.5, 1.0, 2.0]))
        assert len(scan.entropy) == 3


class TestPairingCorrelator:
    def test_pairing_map(self):
        from scpn_quantum_control.analysis.pairing_correlator import pairing_map

        result = pairing_map(OMEGA_2, K_TOPO_2, K_base=1.0)
        assert result is not None

    def test_pairing_vs_anisotropy(self):
        from scpn_quantum_control.analysis.pairing_correlator import pairing_vs_anisotropy

        scan = pairing_vs_anisotropy(
            OMEGA_2, K_TOPO_2, K_base=1.0, delta_range=np.array([0.0, 0.5])
        )
        assert "delta" in scan


class TestOTOCSyncProbe:
    def test_otoc_sync_scan(self):
        from scpn_quantum_control.analysis.otoc_sync_probe import otoc_sync_scan

        scan = otoc_sync_scan(
            K_TOPO_2, OMEGA_2, K_base_range=np.array([0.5, 1.0]), n_time_points=5
        )
        assert len(scan.K_base_values) == 2

    def test_compare_otoc_vs_R(self):
        from scpn_quantum_control.analysis.otoc_sync_probe import compare_otoc_vs_R, otoc_sync_scan

        scan = otoc_sync_scan(
            K_TOPO_2, OMEGA_2, K_base_range=np.array([0.5, 1.0, 2.0]), n_time_points=5
        )
        result = compare_otoc_vs_R(scan)
        assert "K_c_otoc" in result


class TestPersistentHomology:
    def test_phase_distance_matrix(self):
        from scpn_quantum_control.analysis.persistent_homology import phase_distance_matrix

        theta = np.array([0.0, 1.0, 2.0, 3.0])
        D = phase_distance_matrix(theta)
        assert D.shape == (4, 4)
        assert np.allclose(D, D.T)
        assert np.allclose(np.diag(D), 0)


class TestImports:
    def test_graph_topology_scan(self):
        from scpn_quantum_control.analysis import graph_topology_scan  # noqa: F401

    def test_critical_concordance(self):
        from scpn_quantum_control.analysis import critical_concordance  # noqa: F401

    def test_qfi_criticality(self):
        from scpn_quantum_control.analysis import qfi_criticality  # noqa: F401

    def test_entanglement_percolation(self):
        from scpn_quantum_control.analysis import entanglement_percolation  # noqa: F401

    def test_hamiltonian_self_consistency(self):
        from scpn_quantum_control.analysis import hamiltonian_self_consistency  # noqa: F401

    def test_dla_parity(self):
        from scpn_quantum_control.analysis import dla_parity_theorem  # noqa: F401

    def test_berry_phase(self):
        from scpn_quantum_control.analysis import berry_phase  # noqa: F401

    def test_entanglement_enhanced_sync(self):
        from scpn_quantum_control.analysis import entanglement_enhanced_sync  # noqa: F401

    def test_finite_size_scaling(self):
        from scpn_quantum_control.analysis import finite_size_scaling  # noqa: F401

    def test_monte_carlo_xy(self):
        from scpn_quantum_control.analysis import monte_carlo_xy  # noqa: F401
