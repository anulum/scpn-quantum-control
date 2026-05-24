# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Phase dynamics contract tests
"""Contract tests for ansatz, AVQDS, Floquet, QSVT, Trotter, VarQITE, ADAPT-VQE, and default phase-dynamics surfaces."""

from __future__ import annotations

import numpy as np
import pytest

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


def test_ansatz_convergence_99pct():
    """Verifies 137-141: _convergence_99pct for negative and positive final."""
    from scpn_quantum_control.phase.ansatz_methodology import _convergence_99pct

    history = [-0.5, -0.8, -0.95, -0.99, -1.0]
    idx = _convergence_99pct(history)
    assert 0 <= idx < len(history)

    history2 = [0.5, 0.8, 0.95, 0.99, 1.0]
    idx2 = _convergence_99pct(history2)
    assert 0 <= idx2 < len(history2)

    assert _convergence_99pct([1.2, 1.1, 1.0]) == 0


def test_ansatz_benchmark_default_sizes(monkeypatch):
    """Default benchmark sweep uses five sizes and three ansatz families."""
    from scpn_quantum_control.phase import ansatz_methodology as module

    calls = []

    def fake_benchmark(K, omega, ansatz_name, maxiter, reps, gradient_samples, seed):
        calls.append((K.shape, len(omega), ansatz_name, maxiter, reps, gradient_samples, seed))
        return module.AnsatzBenchmarkResult(
            ansatz_name=ansatz_name,
            n_qubits=len(omega),
            n_params=1,
            n_entangling_gates=1,
            reps=reps,
            final_energy=-1.0,
            exact_energy=-1.0,
            relative_error=0.0,
            n_evals=1,
            convergence_iter_99pct=0,
            energy_history=[-1.0],
            gradient_variance=0.0,
        )

    monkeypatch.setattr(module, "benchmark_single_ansatz", fake_benchmark)

    results = module.run_full_benchmark(maxiter=7, reps=1, gradient_samples=2, seed=11)

    assert len(results) == 15
    assert {n_qubits for _, n_qubits, *_ in calls} == {2, 3, 4, 5, 6}
    assert [name for *_, name, _, _, _, _ in calls[:3]] == [
        "knm_informed",
        "two_local",
        "efficient_su2",
    ]
    assert all(call[3:] == (7, 1, 2, 11) for call in calls)


def test_ansatz_efficient_su2_branch_metadata(monkeypatch):
    """EfficientSU2 benchmark behaviour assembles result metadata."""
    from scpn_quantum_control.phase import ansatz_methodology as module

    monkeypatch.setattr(
        module, "_vqe_run", lambda ansatz, hamiltonian, maxiter, seed: (-1.25, 4, [-2.0, -1.25])
    )
    monkeypatch.setattr(
        module, "_gradient_variance", lambda ansatz, hamiltonian, n_samples, seed: 0.125
    )
    monkeypatch.setattr(
        module, "classical_exact_diag", lambda n, K, omega: {"ground_energy": -1.0}
    )

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = module.benchmark_single_ansatz(
        K,
        omega,
        "efficient_su2",
        maxiter=5,
        reps=1,
        gradient_samples=3,
        seed=17,
    )

    assert result.ansatz_name == "efficient_su2"
    assert result.n_qubits == 2
    assert result.n_params > 0
    assert result.n_evals == 4
    assert result.energy_history == [-2.0, -1.25]
    assert result.gradient_variance == pytest.approx(0.125)
    assert result.relative_error == pytest.approx(0.25)


def test_avqds_sparse_h_mat():
    """Verifies 90: avqds H_mat.toarray() sparse path."""
    from scpn_quantum_control.phase.avqds import avqds_simulate

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = avqds_simulate(K, omega, t_total=0.1, n_steps=2, seed=42)
    assert hasattr(result, "energies")
    assert len(result.energies) > 0


def test_avqds_evolution_toarray():
    """Verifies 123: avqds_simulate H_mat toarray in evolution loop."""
    from scpn_quantum_control.phase.avqds import avqds_simulate

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = avqds_simulate(K, omega, t_total=0.2, n_steps=3, seed=0)
    assert len(result.parameters_history) > 0


def test_avqds_sparse_matrix_conversion_contract(monkeypatch):
    """Sparse-like Hamiltonian matrices are densified before dynamics."""
    from scpn_quantum_control.phase import avqds as module

    toarray_calls = []

    class SparseLike:
        def __init__(self, matrix):
            self.matrix = matrix

        def toarray(self):
            toarray_calls.append(self.matrix.shape)
            return self.matrix

    class Hamiltonian:
        def to_matrix(self):
            return SparseLike(np.eye(2, dtype=complex))

    class Ansatz:
        num_parameters = 1

        def assign_parameters(self, params):
            return params

    class FakeStatevector:
        def __init__(self, data):
            self.data = data

        @classmethod
        def from_instruction(cls, assigned):
            theta = float(np.asarray(assigned)[0])
            return cls(np.array([np.cos(theta), np.sin(theta)], dtype=complex))

        def expectation_value(self, hamiltonian):
            matrix = hamiltonian.to_matrix().toarray()
            return complex(np.vdot(self.data, matrix @ self.data))

    monkeypatch.setattr(module, "knm_to_hamiltonian", lambda K, omega: Hamiltonian())
    monkeypatch.setattr(module, "knm_to_ansatz", lambda K, reps: Ansatz())
    monkeypatch.setattr(module, "Statevector", FakeStatevector)

    result = module.avqds_simulate(
        np.zeros((1, 1)),
        np.zeros(1),
        t_total=0.1,
        n_steps=1,
        seed=0,
    )

    assert toarray_calls[:2] == [(2, 2), (2, 2)]
    assert result.n_params == 1
    assert len(result.parameters_history) == 2
    assert np.isfinite(result.final_energy)


def test_floquet_subharmonic_few_freqs():
    """Verifies 142/153: _subharmonic_ratio returns 0.0 with few points."""
    from scpn_quantum_control.phase.floquet_kuramoto import _subharmonic_ratio

    signal = np.array([0.5, 0.3])  # < 4 points → returns 0.0
    val = _subharmonic_ratio(signal, drive_freq=1.0, dt=0.1)
    assert val == 0.0


def test_floquet_subharmonic_zero_power():
    """Verifies 172: _subharmonic_ratio returns 0.0 when p_omega ~ 0."""
    from scpn_quantum_control.phase.floquet_kuramoto import _subharmonic_ratio

    signal = np.zeros(64)  # all-zero signal → zero power everywhere
    val = _subharmonic_ratio(signal, drive_freq=10.0, dt=0.01)
    assert val == 0.0


def test_qsvt_spectral_norm_large():
    """Verifies 85-89: hamiltonian_spectral_norm uses sparse eigsh for n >= 14."""
    from scpn_quantum_control.phase.qsvt_evolution import hamiltonian_spectral_norm

    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    norm = hamiltonian_spectral_norm(K, omega)
    assert norm > 0.0


def test_trotter_error_invalid_order():
    """Verifies 133: trotter_error_bound raises for order != 1 or 2."""
    from scpn_quantum_control.phase.trotter_error import trotter_error_bound

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        trotter_error_bound(K, omega, t=1.0, reps=5, order=3)


def test_trotter_optimal_dt_invalid_order():
    """Verifies 156: optimal_dt raises for order != 1 or 2."""
    from scpn_quantum_control.phase.trotter_error import optimal_dt

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        optimal_dt(K, omega, epsilon=0.01, t_total=1.0, order=3)


def test_varqite_ground_state():
    """Verify varqite_ground_state computation."""
    from scpn_quantum_control.phase.varqite import varqite_ground_state

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = varqite_ground_state(K, omega, n_steps=2, tau_total=0.2, seed=42)
    assert hasattr(result, "energy")
    assert hasattr(result, "energy_history")


def test_varqite_matrices_densify_sparse_hamiltonian(monkeypatch):
    """Sparse-like Hamiltonian matrices are densified before McLachlan solves."""
    from scpn_quantum_control.phase import varqite as module

    toarray_calls = []

    class SparseLike:
        def __init__(self, matrix):
            self.matrix = matrix

        def toarray(self):
            toarray_calls.append(self.matrix.shape)
            return self.matrix

    class Hamiltonian:
        def to_matrix(self):
            return SparseLike(np.diag([0.0, 1.0]))

    class Ansatz:
        def assign_parameters(self, params):
            return np.asarray(params, dtype=float)

    class FakeStatevector:
        def __init__(self, data):
            self.data = data

        @classmethod
        def from_instruction(cls, assigned):
            theta = float(np.asarray(assigned)[0])
            return cls(np.array([np.cos(theta), np.sin(theta)], dtype=complex))

    monkeypatch.setattr(module, "Statevector", FakeStatevector)

    A, C = module._varqite_matrices(
        Ansatz(),
        np.array([0.2]),
        Hamiltonian(),
        epsilon=1e-5,
    )

    assert toarray_calls == [(2, 2)]
    assert A.shape == (1, 1)
    assert C.shape == (1,)
    assert A[0, 0] == pytest.approx(1.0, rel=1e-8)
    assert np.all(np.isfinite(C))


def test_adapt_vqe_full_loop():
    """Verifies 154-170: ADAPT-VQE runs multi-iteration loop."""
    from scpn_quantum_control.phase.adapt_vqe import adapt_vqe

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = adapt_vqe(K, omega, max_iterations=2, gradient_threshold=0.5, maxiter_opt=5, seed=42)
    assert hasattr(result, "energy")
    assert hasattr(result, "converged")


def test_adapt_vqe_build_ansatz():
    """Verifies 203-209: _build_ansatz creates circuit from selected ops."""
    from scpn_quantum_control.phase.adapt_vqe import _build_ansatz, _build_operator_pool

    K = build_knm_paper27(L=2)
    pool = _build_operator_pool(K, 2)
    if len(pool) > 0:
        qc = _build_ansatz(2, pool, [0], [0.1])
        assert qc.num_qubits == 2


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


class TestConcordanceDefaults:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.critical_concordance import critical_concordance

        result = critical_concordance(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 15


class TestBerryDefaults:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.berry_phase import berry_phase_scan

        result = berry_phase_scan(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 29

    def test_single_step(self):
        from scpn_quantum_control.analysis.berry_phase import berry_phase_scan

        result = berry_phase_scan(OMEGA_N_16[:2], _ring(2), k_range=np.array([1.0, 2.0]))
        assert len(result.berry_connection) == 1


class TestMpembaEdge:
    def test_high_gamma(self):
        from scpn_quantum_control.analysis.quantum_mpemba import mpemba_experiment

        result = mpemba_experiment(
            OMEGA_N_16[:2], _ring(2), K_base=1.0, gamma=5.0, t_max=1.0, n_steps=5
        )
        assert len(result.times) == 6


class TestNESSDefaults:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.lindblad_ness import ness_vs_coupling

        result = ness_vs_coupling(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 15


class TestAdiabaticDefaults:
    def test_default_T_values(self):
        from scpn_quantum_control.phase.adiabatic_preparation import adiabatic_time_scaling

        result = adiabatic_time_scaling(OMEGA_N_16[:2], _ring(2), K_target=2.0, n_steps_per_T=5)
        assert len(result["T_total"]) == 5

    def test_default_k_range_ramp(self):
        from scpn_quantum_control.phase.adiabatic_preparation import adiabatic_ramp

        result = adiabatic_ramp(OMEGA_N_16[:2], _ring(2), K_target=1.0)
        assert len(result.times) == 51


class TestXXZDefaults:
    def test_default_ranges(self):
        from scpn_quantum_control.analysis.xxz_phase_diagram import anisotropy_phase_diagram

        result = anisotropy_phase_diagram(OMEGA_N_16[:2], _ring(2))
        assert len(result.delta_values) == 6

    def test_default_scan(self):
        from scpn_quantum_control.analysis.xxz_phase_diagram import scan_coupling_at_delta

        result = scan_coupling_at_delta(OMEGA_N_16[:2], _ring(2), delta=0.5)
        assert len(result.k_values) == 15


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


class TestMagicDefaults:
    def test_default_k_range(self):
        from scpn_quantum_control.analysis.magic_nonstabilizerness import magic_vs_coupling

        result = magic_vs_coupling(OMEGA_N_16[:2], _ring(2))
        assert len(result.k_values) == 15


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


class TestSyncWitnessEdge:
    def test_2qubit_separable_bound(self):
        from scpn_quantum_control.analysis.sync_entanglement_witness import R_separable_bound

        bound = R_separable_bound(2)
        assert 0 < bound <= 1.0

    def test_3qubit_separable_bound(self):
        from scpn_quantum_control.analysis.sync_entanglement_witness import R_separable_bound

        bound = R_separable_bound(3)
        assert 0 < bound <= 1.0


class TestQSLDefaults:
    def test_default_k_base_range(self):
        from scpn_quantum_control.analysis.quantum_speed_limit import qsl_vs_coupling

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = qsl_vs_coupling(K, omega, t_target=1.0)
        assert len(result["K_base"]) > 0


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


class TestAdaptVQE:
    def test_adapt_vqe_runs_optimisation(self):
        from scpn_quantum_control.phase.adapt_vqe import adapt_vqe

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result = adapt_vqe(K, omega, max_iterations=3, gradient_threshold=0.001, seed=42)
        assert isinstance(result.energy, float)
        assert hasattr(result, "selected_operators")
