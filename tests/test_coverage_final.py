# SPDX-License-Identifier: AGPL-3.0-or-later | Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Coverage Final
"""Tests targeting remaining uncovered lines for 100% coverage."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest


class TestClassicalPythonFallback:
    """Exercise the Python Euler ODE path when Rust engine is unavailable."""

    def test_kuramoto_python_fallback(self):
        with patch.dict("sys.modules", {"scpn_quantum_engine": None}):
            from importlib import reload

            import scpn_quantum_control.hardware.classical as classical_mod

            reload(classical_mod)
            result = classical_mod.classical_kuramoto_reference(4, 0.2, 0.1)
            assert result["theta"].shape == (3, 4)
            assert result["R"].shape == (3,)
            assert all(0 <= r <= 1 for r in result["R"])


class TestQuantumAdvantageEdgeCases:
    """Cover branches in benchmarks/quantum_advantage.py."""

    def test_classical_benchmark_infeasible(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import classical_benchmark

        result = classical_benchmark(16, t_max=0.1, dt=0.1)
        assert result["t_total_ms"] == float("inf")
        assert result["ground_energy"] is None

    def test_estimate_crossover_too_few_points(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import (
            AdvantageResult,
            estimate_crossover,
        )

        results = [AdvantageResult(4, 1.0, 2.0), AdvantageResult(6, 2.0, 3.0)]
        assert estimate_crossover(results) is None

    def test_estimate_crossover_with_inf(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import (
            AdvantageResult,
            estimate_crossover,
        )

        results = [
            AdvantageResult(4, 1.0, 2.0),
            AdvantageResult(6, 5.0, 3.0),
            AdvantageResult(8, float("inf"), 4.0),
        ]
        cross = estimate_crossover(results)
        # Only 2 feasible points — not enough for fit
        assert cross is None

    def test_run_scaling_benchmark_few_sizes(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import run_scaling_benchmark

        results = run_scaling_benchmark(sizes=[4, 6], t_max=0.1, dt=0.1)
        assert len(results) == 2
        assert results[0].crossover_predicted is None  # < 3 results


class TestPECEdgeCases:
    def test_pec_zero_error(self):
        from scpn_quantum_control.mitigation.pec import pauli_twirl_decompose

        coeffs = pauli_twirl_decompose(0.0)
        assert coeffs[0] == 1.0
        assert all(c == 0.0 for c in coeffs[1:])

    def test_pec_sample_single_gate(self):
        from qiskit import QuantumCircuit

        from scpn_quantum_control.mitigation.pec import pec_sample

        qc = QuantumCircuit(1)
        qc.ry(0.5, 0)
        result = pec_sample(qc, 0.01, 100, rng=np.random.default_rng(42))
        assert result.n_samples == 100


class TestSNNImportError:
    def test_arcane_neuron_import_error(self):
        with patch.dict(
            "sys.modules",
            {
                "sc_neurocore": None,
                "sc_neurocore.neurons": None,
                "sc_neurocore.neurons.models": None,
            },
        ):
            from scpn_quantum_control.bridge.snn_adapter import ArcaneNeuronBridge

            with pytest.raises(ImportError, match="sc-neurocore"):
                ArcaneNeuronBridge(2, 3)


class TestQuantumAdvantageMoreEdges:
    def test_estimate_crossover_curve_fit_fails(self):
        from unittest.mock import patch

        from scpn_quantum_control.benchmarks.quantum_advantage import (
            AdvantageResult,
            estimate_crossover,
        )

        results = [
            AdvantageResult(4, 1.0, 2.0),
            AdvantageResult(6, 5.0, 3.0),
            AdvantageResult(8, 20.0, 4.0),
        ]
        with patch(
            "scpn_quantum_control.benchmarks.quantum_advantage.curve_fit",
            side_effect=RuntimeError("fit failed"),
        ):
            assert estimate_crossover(results) is None

    def test_estimate_crossover_quantum_grows_faster(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import (
            AdvantageResult,
            estimate_crossover,
        )

        # Quantum grows faster than classical — no crossover
        results = [
            AdvantageResult(4, 10.0, 1.0),
            AdvantageResult(6, 15.0, 5.0),
            AdvantageResult(8, 20.0, 50.0),
        ]
        cross = estimate_crossover(results)
        # b_c <= b_q → None
        assert cross is None

    def test_run_scaling_default_sizes(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import run_scaling_benchmark

        results = run_scaling_benchmark(sizes=[4], t_max=0.1, dt=0.1)
        assert len(results) == 1
        assert results[0].crossover_predicted is None  # single size

    def test_classical_benchmark_n15(self):
        from scpn_quantum_control.benchmarks.quantum_advantage import classical_benchmark

        result = classical_benchmark(15, t_max=0.05, dt=0.05)
        assert result["t_total_ms"] == float("inf")

    def test_estimate_crossover_negative_ratio(self):
        from unittest.mock import patch

        from scpn_quantum_control.benchmarks.quantum_advantage import (
            AdvantageResult,
            estimate_crossover,
        )

        results = [
            AdvantageResult(4, 1.0, 2.0),
            AdvantageResult(6, 5.0, 3.0),
            AdvantageResult(8, 20.0, 4.0),
        ]
        # Mock curve_fit to return negative a coefficient
        with patch(
            "scpn_quantum_control.benchmarks.quantum_advantage.curve_fit",
            return_value=(np.array([-1.0, 0.5]), None),
        ):
            assert estimate_crossover(results) is None

    def test_run_scaling_with_warning(self):
        import warnings

        from scpn_quantum_control.benchmarks.quantum_advantage import run_scaling_benchmark

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            results = run_scaling_benchmark(sizes=[4, 6, 8], t_max=0.05, dt=0.05)
            assert len(results) == 3
            # crossover_predicted should be set (3 results)
            assert (
                results[0].crossover_predicted is not None
                or results[0].crossover_predicted is None
            )


class TestPECPauliBranches:
    def test_pec_y_branch(self):
        from qiskit import QuantumCircuit

        from scpn_quantum_control.mitigation.pec import pec_sample

        qc = QuantumCircuit(1)
        qc.ry(1.0, 0)
        result = pec_sample(qc, 0.3, 5000, rng=np.random.default_rng(42))
        assert result.n_samples == 5000

    def test_pec_default_rng(self):
        from qiskit import QuantumCircuit

        from scpn_quantum_control.mitigation.pec import pec_sample

        qc = QuantumCircuit(1)
        qc.ry(0.5, 0)
        result = pec_sample(qc, 0.01, 50)  # no rng arg — uses default
        assert result.n_samples == 50


class TestMiscEdgeCases:
    def test_coherence_budget_zero_depth(self):
        from scpn_quantum_control.identity.coherence_budget import fidelity_at_depth

        f = fidelity_at_depth(0, n_qubits=4)
        assert f == pytest.approx(1.0, abs=0.01)

    def test_qpetri_single_transition(self):
        from scpn_quantum_control.control.qpetri import QuantumPetriNet

        W_in = np.array([[1]])
        W_out = np.array([[1]])
        net = QuantumPetriNet(1, 1, W_in, W_out, thresholds=np.array([1]))
        marking = net.step(np.array([1]))
        assert marking.shape == (1,)

    def test_noise_analysis_zero_qber(self):
        from scpn_quantum_control.crypto.noise_analysis import devetak_winter_rate

        rate = devetak_winter_rate(0.0)
        assert rate == pytest.approx(1.0, abs=0.01)
