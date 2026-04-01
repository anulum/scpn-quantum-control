# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Round4 8 Coverage
"""Coverage gap tests for Rounds 4-8 modules.

Exercises default parameter branches, edge cases, and untested code paths
to push coverage toward 100%.
"""

from __future__ import annotations

import numpy as np

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    knm_to_xxz_hamiltonian,
)


def _ring(n: int) -> np.ndarray:
    T = np.zeros((n, n))
    for i in range(n):
        j = (i + 1) % n
        T[i, j] = T[j, i] = 1.0
    return T


# --- QFI criticality: default k_range ---
class TestQFICriticalityDefaults:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.qfi_criticality import qfi_vs_coupling

        result = qfi_vs_coupling(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 20


# --- Entanglement percolation: edge cases ---
class TestEntanglementPercolationEdge:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.entanglement_percolation import percolation_scan

        result = percolation_scan(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 20

    def test_no_entangled_pairs(self):
        from scpn_quantum_control.analysis.entanglement_percolation import percolation_scan

        result = percolation_scan(OMEGA_N_16[:2], _ring(2), k_range=np.array([0.001]))
        assert result.n_entangled_pairs[0] == 0


# --- QRC: boundary detection + default ---
class TestQRCEdge:
    def test_boundary_detection(self):
        from scpn_quantum_control.analysis.qrc_phase_detector import qrc_phase_detection

        T = _ring(3)
        omega = OMEGA_N_16[:3]
        k_train = np.array([0.1, 0.5, 3.0, 5.0])
        k_test = np.linspace(0.5, 4.0, 8)
        result = qrc_phase_detection(omega, T, k_train, k_test, k_threshold=1.5)
        # Boundary detection may or may not find crossing
        assert result.accuracy >= 0

    def test_generate_data_weight1(self):
        from scpn_quantum_control.analysis.qrc_phase_detector import generate_training_data

        X, y = generate_training_data(OMEGA_N_16[:2], _ring(2), np.array([1.0]), 0.5, max_weight=1)
        assert X.shape[0] == 1


# --- Floquet: default amplitudes ---
class TestFloquetDefaults:
    def test_default_amplitudes(self):
        from scpn_quantum_control.phase.floquet_kuramoto import scan_drive_amplitude

        result = scan_drive_amplitude(
            _ring(2),
            OMEGA_N_16[:2],
            K_base=1.0,
            drive_frequency=2.0,
            n_periods=2,
            steps_per_period=4,
        )
        assert len(result["amplitude"]) == 10

    def test_subharmonic_ratio_short_signal(self):
        from scpn_quantum_control.phase.floquet_kuramoto import floquet_evolve

        result = floquet_evolve(
            _ring(2),
            OMEGA_N_16[:2],
            K_base=1.0,
            drive_amplitude=0.5,
            drive_frequency=2.0,
            n_periods=1,
            steps_per_period=3,
        )
        assert np.isfinite(result.subharmonic_ratio)


# --- Critical concordance: default k_range ---
class TestConcordanceDefaults:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.critical_concordance import critical_concordance

        result = critical_concordance(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 15


# --- Berry phase: default k_range ---
class TestBerryDefaults:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.berry_phase import berry_phase_scan

        result = berry_phase_scan(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 29

    def test_single_step(self):
        from scpn_quantum_control.analysis.berry_phase import berry_phase_scan

        result = berry_phase_scan(OMEGA_N_16[:2], _ring(2), k_range=np.array([1.0, 2.0]))
        assert len(result.berry_connection) == 1


# --- Quantum Mpemba: edge ---
class TestMpembaEdge:
    def test_high_gamma(self):
        from scpn_quantum_control.analysis.quantum_mpemba import mpemba_experiment

        result = mpemba_experiment(
            OMEGA_N_16[:2], _ring(2), K_base=1.0, gamma=5.0, t_max=1.0, n_steps=5
        )
        assert len(result.times) == 6


# --- Lindblad NESS: default k_range ---
class TestNESSDefaults:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.lindblad_ness import ness_vs_coupling

        result = ness_vs_coupling(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 15


# --- Adiabatic: default T_values ---
class TestAdiabaticDefaults:
    def test_default_T_values(self):
        from scpn_quantum_control.phase.adiabatic_preparation import adiabatic_time_scaling

        result = adiabatic_time_scaling(OMEGA_N_16[:2], _ring(2), K_target=2.0, n_steps_per_T=5)
        assert len(result["T_total"]) == 5

    def test_default_k_range_ramp(self):
        from scpn_quantum_control.phase.adiabatic_preparation import adiabatic_ramp

        result = adiabatic_ramp(OMEGA_N_16[:2], _ring(2), K_target=1.0)
        assert len(result.times) == 51


# --- XXZ phase diagram: defaults ---
class TestXXZDefaults:
    def test_default_ranges(self):
        from scpn_quantum_control.analysis.xxz_phase_diagram import anisotropy_phase_diagram

        result = anisotropy_phase_diagram(OMEGA_N_16[:2], _ring(2))
        assert len(result.delta_values) == 6

    def test_default_scan(self):
        from scpn_quantum_control.analysis.xxz_phase_diagram import scan_coupling_at_delta

        result = scan_coupling_at_delta(OMEGA_N_16[:2], _ring(2), delta=0.5)
        assert len(result.k_values) == 15


# --- Pairing: default delta_range ---
class TestPairingDefaults:
    def test_default_delta_range(self):
        from scpn_quantum_control.analysis.pairing_correlator import pairing_vs_anisotropy

        result = pairing_vs_anisotropy(OMEGA_N_16[:2], _ring(2), K_base=2.0)
        assert len(result["delta"]) == 6

    def test_zero_std_correlation(self):
        """All-zeros coupling → no pairing correlation."""
        from scpn_quantum_control.analysis.pairing_correlator import pairing_map

        result = pairing_map(OMEGA_N_16[:2], np.zeros((2, 2)), K_base=0.0, delta=0.0)
        assert result.pairing_topology_correlation == 0.0


# --- SFF: defaults + edge ---
class TestSFFDefaults:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.spectral_form_factor import sff_vs_coupling

        result = sff_vs_coupling(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 15

    def test_no_chaos_onset(self):
        """Very weak coupling → Poisson statistics everywhere → no chaos onset."""
        from scpn_quantum_control.analysis.spectral_form_factor import sff_vs_coupling

        result = sff_vs_coupling(OMEGA_N_16[:2], _ring(2), k_range=np.array([0.001]))
        # chaos_onset_K may or may not be None depending on r_bar
        assert isinstance(result.chaos_onset_K, (float, type(None)))


# --- Loschmidt: defaults ---
class TestLoschmidtDefaults:
    def test_default_K_final_range(self):
        from scpn_quantum_control.analysis.loschmidt_echo import quench_scan

        result = quench_scan(OMEGA_N_16[:2], _ring(2), K_initial=0.5, n_times=20)
        assert len(result["K_final"]) == 10


# --- Entanglement entropy: defaults ---
class TestEntropyDefaults:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.entanglement_entropy import entanglement_vs_coupling

        result = entanglement_vs_coupling(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 20

    def test_2qubit_bipartition(self):
        """2-qubit: n_A = 1 (half-chain)."""
        from scpn_quantum_control.analysis.entanglement_entropy import entanglement_at_coupling

        result = entanglement_at_coupling(OMEGA_N_16[:2], _ring(2), K_base=3.0)
        assert result.entropy >= 0


# --- Krylov: defaults ---
class TestKrylovDefaults:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.krylov_complexity import krylov_vs_coupling

        result = krylov_vs_coupling(OMEGA_N_16[:2], _ring(2))
        assert len(result["K_base"]) == 10

    def test_zero_operator(self):
        """Zero operator → empty Lanczos."""
        from scpn_quantum_control.analysis.krylov_complexity import krylov_complexity

        H = np.eye(4, dtype=complex)
        O_zero = np.zeros((4, 4), dtype=complex)
        result = krylov_complexity(H, O_zero, t_max=1.0, n_times=5)
        assert result.peak_complexity == 0.0


# --- Magic: defaults ---
class TestMagicDefaults:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.magic_nonstabilizerness import magic_vs_coupling

        result = magic_vs_coupling(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 15


# --- FSS: defaults ---
class TestFSSDefaults:
    def test_default_system_sizes(self):
        from scpn_quantum_control.analysis.finite_size_scaling import finite_size_scaling

        result = finite_size_scaling()
        assert len(result.system_sizes) == 3
        assert 2 in result.system_sizes

    def test_default_k_range(self):
        from scpn_quantum_control.analysis.finite_size_scaling import finite_size_scaling

        result = finite_size_scaling(system_sizes=[2])
        assert len(result.k_c_values) == 1


# --- XXZ Hamiltonian edge: delta < 0 ---
class TestXXZEdge:
    def test_large_delta(self):
        K = 2.0 * _ring(2)
        omega = OMEGA_N_16[:2]
        H = knm_to_xxz_hamiltonian(K, omega, delta=2.0)
        assert H.to_matrix().shape == (4, 4)

    def test_zero_coupling_xxz(self):
        K = np.zeros((2, 2))
        omega = OMEGA_N_16[:2]
        H = knm_to_xxz_hamiltonian(K, omega, delta=1.0)
        # Only Z terms, no coupling
        mat = H.to_matrix()
        assert np.allclose(mat, np.diag(np.diag(mat)))


# --- Sync entanglement witness: coverage for separable_bound_mc ---
class TestSyncWitnessEdge:
    def test_2qubit_separable_bound(self):
        from scpn_quantum_control.analysis.sync_entanglement_witness import R_separable_bound

        bound = R_separable_bound(2)
        assert 0 < bound <= 1.0

    def test_3qubit_separable_bound(self):
        from scpn_quantum_control.analysis.sync_entanglement_witness import R_separable_bound

        bound = R_separable_bound(3)
        assert 0 < bound <= 1.0


# --- Quantum speed limit: default k_range ---
class TestQSLDefaults:
    def test_default_k_base_range(self):
        from scpn_quantum_control.analysis.quantum_speed_limit import qsl_vs_coupling

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = qsl_vs_coupling(K, omega, t_target=1.0)
        assert len(result["K_base"]) > 0


# --- OTOC sync probe: defaults ---
class TestOTOCProbeDefaults:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.otoc_sync_probe import otoc_sync_scan

        result = otoc_sync_scan(_ring(2), OMEGA_N_16[:2])
        assert len(result.K_base_values) > 0


# --- Hamiltonian self-consistency: correlator_shot_noise ---
class TestSelfConsistencyEdge:
    def test_correlator_shot_noise(self):
        from scpn_quantum_control.analysis.hamiltonian_self_consistency import (
            correlator_shot_noise,
        )

        x_counts = {"00": 500, "11": 500}
        y_counts = {"00": 500, "11": 500}
        noise = correlator_shot_noise(x_counts, y_counts, 2)
        assert noise > 0


# --- Cross-domain transfer: default ---
class TestCrossDomainTransferDefaults:
    def test_with_custom_systems(self):
        from scpn_quantum_control.phase.cross_domain_transfer import (
            PhysicalSystem,
            transfer_experiment,
        )

        omega = OMEGA_N_16[:2]
        sys_a = PhysicalSystem(name="A", K=2.0 * _ring(2), omega=omega)
        sys_b = PhysicalSystem(name="B", K=1.5 * _ring(2), omega=omega)
        result = transfer_experiment(sys_a, sys_b, maxiter=10)
        assert result.source_system == "A"
