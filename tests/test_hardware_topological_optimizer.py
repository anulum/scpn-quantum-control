# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Project Configuration
"""Tests for Hardware-in-the-Loop Topological Feedback."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit

import scpn_quantum_control.control.hardware_topological_optimizer as hardware_topology
import scpn_quantum_control.control.topological_optimizer as topology
from scpn_quantum_control.analysis.quantum_persistent_homology import _RIPSER_AVAILABLE
from scpn_quantum_control.control.hardware_topological_optimizer import (
    HardwareTopologicalOptimizer,
)
from scpn_quantum_control.hardware.runner import HardwareRunner


class TestHardwareTopologicalOptimizer:
    def test_hardware_optimizer_step(self):
        """Verify one cycle of hardware-in-the-loop topological gradient descent."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")

        n = 2
        initial_K = np.array([[0.0, 0.1], [0.1, 0.0]])
        omega = np.array([5.0, 10.0])

        runner = HardwareRunner(use_simulator=True)
        runner.connect()

        opt = HardwareTopologicalOptimizer(
            runner=runner, n_qubits=n, initial_K=initial_K, omega=omega, learning_rate=0.5, dt=0.5
        )

        # We step it with a very small number of samples and shots to keep it fast
        # Note: with n=2, p_h1 is likely 0, so gradient will be 0, but we test the pipeline
        res = opt.step(n_samples=1)

        assert "K_updated" in res
        assert "p_h1_current" in res
        assert "gradient_norm" in res
        assert res["K_updated"].shape == (n, n)
        np.testing.assert_allclose(res["K_updated"], res["K_updated"].T)

    def test_hardware_optimizer_multi_step(self):
        """Multiple steps produce expected history length."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")
        n = 2
        initial_K = np.array([[0.0, 0.2], [0.2, 0.0]])
        omega = np.array([5.0, 10.0])
        runner = HardwareRunner(use_simulator=True)
        runner.connect()
        opt = HardwareTopologicalOptimizer(
            runner=runner, n_qubits=n, initial_K=initial_K, omega=omega, dt=0.5
        )
        history = opt.optimize(steps=2, n_samples=1)
        assert len(history) == 2
        for step in history:
            np.testing.assert_allclose(step["K_updated"], step["K_updated"].T, atol=1e-12)
            assert np.all(step["K_updated"] >= -1e-15)

    def test_hardware_optimizer_k_non_negative(self):
        """K entries must remain >= 0 with hardware path."""
        if not _RIPSER_AVAILABLE:
            pytest.skip("ripser not available")
        n = 2
        initial_K = np.array([[0.0, 0.05], [0.05, 0.0]])
        omega = np.array([5.0, 10.0])
        runner = HardwareRunner(use_simulator=True)
        runner.connect()
        opt = HardwareTopologicalOptimizer(
            runner=runner,
            n_qubits=n,
            initial_K=initial_K,
            omega=omega,
            learning_rate=1.0,
            dt=0.5,
        )
        res = opt.step(n_samples=1)
        assert np.all(res["K_updated"] >= -1e-15)

    def test_hardware_finite_difference_measures_candidate_couplings(self, monkeypatch):
        """Hardware finite differences must run circuits for K, K+delta, and K-delta."""
        monkeypatch.setattr(topology, "_RIPSER_AVAILABLE", True)
        observed_couplings: list[np.ndarray] = []
        initial_K = np.array([[0.0, 0.2], [0.2, 0.0]], dtype=float)
        delta = np.array([[0.0, 0.05], [0.05, 0.0]], dtype=float)

        def fake_fast_sparse_evolution(K, omega, *, t_total, n_steps):
            return {"final_state": np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)}

        def fake_build_evo_base(n, K, omega, *, t, trotter_reps):
            observed_couplings.append(np.array(K, dtype=float).copy())
            return QuantumCircuit(n)

        def fake_build_xyz_circuits(base_qc, n):
            return QuantumCircuit(n, n), QuantumCircuit(n, n), QuantumCircuit(n, n)

        class RecordingRunner:
            def run_sampler(self, circuits, *, shots, name):
                return [
                    SimpleNamespace(counts={"00": shots}),
                    SimpleNamespace(counts={"00": shots}),
                ]

        monkeypatch.setattr(topology, "fast_sparse_evolution", fake_fast_sparse_evolution)
        monkeypatch.setattr(hardware_topology, "_build_evo_base", fake_build_evo_base)
        monkeypatch.setattr(hardware_topology, "_build_xyz_circuits", fake_build_xyz_circuits)
        monkeypatch.setattr(topology.np.random, "normal", lambda *args, **kwargs: delta)
        monkeypatch.setattr(
            topology,
            "quantum_persistent_homology",
            lambda x_counts, y_counts, n, persistence_threshold: SimpleNamespace(p_h1=0.0),
        )

        opt = HardwareTopologicalOptimizer(
            runner=RecordingRunner(),
            n_qubits=2,
            initial_K=initial_K,
            omega=np.array([5.0, 10.0]),
        )

        opt.step(n_samples=1)

        np.testing.assert_allclose(observed_couplings[0], initial_K)
        assert any(np.allclose(K, initial_K + delta) for K in observed_couplings[1:])
        assert any(np.allclose(K, initial_K - delta) for K in observed_couplings[1:])
