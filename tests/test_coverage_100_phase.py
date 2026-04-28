# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Coverage tests for phase/ module gaps
"""Tests targeting specific uncovered lines in the phase/ subpackage."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

# --- ansatz_methodology.py line 141: convergence index for negative final energy ---


def test_ansatz_convergence_99pct():
    """Cover line 137-141: _convergence_99pct for negative and positive final."""
    from scpn_quantum_control.phase.ansatz_methodology import _convergence_99pct

    history = [-0.5, -0.8, -0.95, -0.99, -1.0]
    idx = _convergence_99pct(history)
    assert 0 <= idx < len(history)

    history2 = [0.5, 0.8, 0.95, 0.99, 1.0]
    idx2 = _convergence_99pct(history2)
    assert 0 <= idx2 < len(history2)

    assert _convergence_99pct([1.2, 1.1, 1.0]) == 0


# --- ansatz_methodology.py line 205: system_sizes default ---


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
    """EfficientSU2 benchmark branch assembles result metadata."""
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


# --- avqds.py line 90: sparse H_mat.toarray() ---


def test_avqds_sparse_h_mat():
    """Cover line 90: avqds H_mat.toarray() sparse path."""
    from scpn_quantum_control.phase.avqds import avqds_simulate

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = avqds_simulate(K, omega, t_total=0.1, n_steps=2, seed=42)
    assert hasattr(result, "energies")
    assert len(result.energies) > 0


# --- avqds.py line 123: H_mat toarray for evolution ---


def test_avqds_evolution_toarray():
    """Cover line 123: avqds_simulate H_mat toarray in evolution loop."""
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


# --- floquet_kuramoto.py line 152-153: len(freqs) < 2 → returns 0.0 ---


def test_floquet_subharmonic_few_freqs():
    """Cover line 142/153: _subharmonic_ratio returns 0.0 with few points."""
    from scpn_quantum_control.phase.floquet_kuramoto import _subharmonic_ratio

    signal = np.array([0.5, 0.3])  # < 4 points → returns 0.0 at line 143
    val = _subharmonic_ratio(signal, drive_freq=1.0, dt=0.1)
    assert val == 0.0


# --- floquet_kuramoto.py line 171-172: p_omega near zero → returns 0.0 ---


def test_floquet_subharmonic_zero_power():
    """Cover line 172: _subharmonic_ratio returns 0.0 when p_omega ~ 0."""
    from scpn_quantum_control.phase.floquet_kuramoto import _subharmonic_ratio

    signal = np.zeros(64)  # all-zero signal → zero power everywhere
    val = _subharmonic_ratio(signal, drive_freq=10.0, dt=0.01)
    assert val == 0.0


# --- qsvt_evolution.py lines 85-89: large system sparse eigsh path ---


def test_qsvt_spectral_norm_large():
    """Cover lines 85-89: hamiltonian_spectral_norm uses sparse eigsh for n >= 14."""
    from scpn_quantum_control.phase.qsvt_evolution import hamiltonian_spectral_norm

    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    norm = hamiltonian_spectral_norm(K, omega)
    assert norm > 0.0


# --- trotter_error.py line 133: order != 1 and != 2 raises ValueError ---


def test_trotter_error_invalid_order():
    """Cover line 133: trotter_error_bound raises for order != 1 or 2."""
    from scpn_quantum_control.phase.trotter_error import trotter_error_bound

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        trotter_error_bound(K, omega, t=1.0, reps=5, order=3)


# --- trotter_error.py line 156: optimal_dt order != 1 or 2 raises ---


def test_trotter_optimal_dt_invalid_order():
    """Cover line 156: optimal_dt raises for order != 1 or 2."""
    from scpn_quantum_control.phase.trotter_error import optimal_dt

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    with pytest.raises(ValueError, match="order must be 1 or 2"):
        optimal_dt(K, omega, epsilon=0.01, t_total=1.0, order=3)


# --- varqite.py line 75: H_mat toarray path ---


def test_varqite_ground_state():
    """Cover varqite_ground_state computation."""
    from scpn_quantum_control.phase.varqite import varqite_ground_state

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = varqite_ground_state(K, omega, n_steps=2, tau_total=0.2, seed=42)
    assert hasattr(result, "energy")
    assert hasattr(result, "energy_history")


# --- adapt_vqe.py lines 154-170: ADAPT-VQE full optimization loop ---


def test_adapt_vqe_full_loop():
    """Cover lines 154-170: ADAPT-VQE runs multi-iteration loop."""
    from scpn_quantum_control.phase.adapt_vqe import adapt_vqe

    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    result = adapt_vqe(K, omega, max_iterations=2, gradient_threshold=0.5, maxiter_opt=5, seed=42)
    assert hasattr(result, "energy")
    assert hasattr(result, "converged")


# --- adapt_vqe.py lines 203-209: _build_ansatz with PauliEvolutionGate ---


def test_adapt_vqe_build_ansatz():
    """Cover lines 203-209: _build_ansatz creates circuit from selected ops."""
    from scpn_quantum_control.phase.adapt_vqe import _build_ansatz, _build_operator_pool

    K = build_knm_paper27(L=2)
    pool = _build_operator_pool(K, 2)
    if len(pool) > 0:
        qc = _build_ansatz(2, pool, [0], [0.1])
        assert qc.num_qubits == 2
