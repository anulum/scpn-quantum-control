# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Pipeline Wiring & Performance Verification
"""Verify every module is WIRED into the pipeline (not decorative).

For each component:
1. Import succeeds from top-level package
2. Core function/class is callable
3. End-to-end data flows through the component
4. Performance metrics recorded (wall time, output shape/type)

If any import fails or function returns garbage, the module is decorative → FAIL.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import scpn_quantum_control as sqc
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timed(fn, *args, **kwargs):
    """Run fn, return (result, wall_time_ms)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    dt = (time.perf_counter() - t0) * 1000
    return result, dt


def _report(name, dt_ms, extra=""):
    """Print pipeline performance line."""
    tag = f"  [{dt_ms:7.1f} ms]"
    print(f"\n  PIPELINE {name}: {tag} {extra}")


# ---------------------------------------------------------------------------
# 1. Top-level __init__.py exports — every symbol must be importable
# ---------------------------------------------------------------------------


class TestTopLevelExports:
    """Every __all__ symbol must be importable and non-None."""

    @pytest.mark.parametrize("name", sqc.__all__)
    def test_export_exists(self, name):
        obj = getattr(sqc, name, None)
        assert obj is not None, f"{name} is None — not wired"


# ---------------------------------------------------------------------------
# 2. Bridge layer — Knm → Hamiltonian → Ansatz pipeline
# ---------------------------------------------------------------------------


class TestBridgePipeline:
    @pytest.mark.parametrize("L", [2, 4, 8, 16])
    def test_knm_to_hamiltonian_pipeline(self, L):
        K, dt_k = _timed(build_knm_paper27, L=L)
        omega = OMEGA_N_16[:L]
        H, dt_h = _timed(sqc.knm_to_hamiltonian, K, omega)
        assert H.num_qubits == L
        _report(f"Knm→H (L={L})", dt_k + dt_h, f"qubits={L}")

    @pytest.mark.parametrize("L", [2, 4, 8])
    def test_knm_to_ansatz_pipeline(self, L):
        K = build_knm_paper27(L=L)
        qc, dt = _timed(sqc.knm_to_ansatz, K, reps=2)
        assert qc.num_qubits == L
        assert qc.num_parameters == L * 2 * 2
        _report(f"Knm→Ansatz (L={L})", dt, f"params={qc.num_parameters}")


# ---------------------------------------------------------------------------
# 3. Phase solvers — VQE, Trotter UPDE, Kuramoto
# ---------------------------------------------------------------------------


class TestPhaseSolverPipeline:
    def test_phase_vqe_pipeline(self):
        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        vqe = sqc.PhaseVQE(K, omega, ansatz_reps=1)
        result, dt = _timed(vqe.solve, maxiter=20, seed=42)
        assert np.isfinite(result["ground_energy"])
        assert result["ground_energy"] < 0
        _report("PhaseVQE (2q)", dt, f"E={result['ground_energy']:.4f}")

    def test_trotter_upde_pipeline(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        solver = sqc.QuantumUPDESolver(K=K, omega=omega)
        result, dt = _timed(solver.run, n_steps=5, dt=0.05)
        assert len(result["R"]) == 6
        _report("TrotterUPDE (3q, 5 steps)", dt, f"R_final={result['R'][-1]:.4f}")

    def test_kuramoto_solver_pipeline(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        solver = sqc.QuantumKuramotoSolver(4, K, omega)
        result, dt = _timed(solver.run, t_max=0.3, dt=0.1, trotter_per_step=5)
        assert len(result["R"]) > 0
        _report("KuramotoSolver (4q)", dt, f"R_final={result['R'][-1]:.4f}")


# ---------------------------------------------------------------------------
# 4. Hardware layer — runner, transpile, noise model
# ---------------------------------------------------------------------------


class TestHardwarePipeline:
    def test_hardware_runner_full_pipeline(self, tmp_path):
        from qiskit import QuantumCircuit

        runner = sqc.HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "results"))
        runner.connect()
        qc = QuantumCircuit(4)
        qc.h(0)
        for i in range(1, 4):
            qc.cx(0, i)
        qc.measure_all()

        isa, dt_t = _timed(runner.transpile, qc)
        stats = runner.circuit_stats(isa)
        results, dt_r = _timed(runner.run_sampler, qc, shots=1000, name="ghz")
        assert results[0].counts is not None
        total = sum(results[0].counts.values())
        assert total == 1000
        _report("HW sampler (4q GHZ)", dt_t + dt_r, f"depth={stats['depth']}")

    def test_noise_model_pipeline(self, tmp_path):
        from scpn_quantum_control.hardware.noise_model import heron_r2_noise_model

        nm, dt_n = _timed(heron_r2_noise_model)
        runner = sqc.HardwareRunner(
            use_simulator=True, noise_model=nm, results_dir=str(tmp_path / "results")
        )
        runner.connect()
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        results, dt_r = _timed(runner.run_sampler, qc, shots=2000, name="noisy")
        assert len(results[0].counts) > 0
        _report("Noisy sim (2q Bell)", dt_n + dt_r)


# ---------------------------------------------------------------------------
# 5. Mitigation layer — ZNE, PEC
# ---------------------------------------------------------------------------


class TestMitigationPipeline:
    def test_zne_pipeline(self):
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        solver = sqc.QuantumKuramotoSolver(3, K, omega)
        qc = solver.evolve(0.1, trotter_steps=2)

        from qiskit.quantum_info import Statevector

        def measure_R(circuit):
            sv = Statevector.from_instruction(circuit)
            R, _ = solver.measure_order_parameter(sv)
            return R

        R_values = []
        t0 = time.perf_counter()
        for scale in [1, 3, 5]:
            folded = sqc.gate_fold_circuit(qc, scale)
            R_values.append(measure_R(folded))
        zne_result = sqc.zne_extrapolate([1, 3, 5], R_values, order=1)
        dt = (time.perf_counter() - t0) * 1000

        assert np.isfinite(zne_result.zero_noise_estimate)
        _report("ZNE (3q Kuramoto)", dt, f"R_zne={zne_result.zero_noise_estimate:.4f}")

    def test_pec_pipeline(self):
        decomp, dt = _timed(sqc.pauli_twirl_decompose, 0.01, n_qubits=1)
        assert decomp.shape[0] > 0
        _report("PEC decompose (1q, p=0.01)", dt, f"shape={decomp.shape}")


# ---------------------------------------------------------------------------
# 6. QEC layer
# ---------------------------------------------------------------------------


class TestQECPipeline:
    def test_control_qec_pipeline(self):
        qec, dt_init = _timed(sqc.ControlQEC, distance=3)
        n_data = 2 * 3**2
        err_x = np.zeros(n_data, dtype=np.int8)
        err_z = np.zeros(n_data, dtype=np.int8)
        syn_z, syn_x = qec.get_syndrome(err_x, err_z)
        assert int(syn_z.sum()) == 0
        _report("ControlQEC (d=3)", dt_init)

    def test_surface_code_pipeline(self):
        upde, dt = _timed(sqc.SurfaceCodeUPDE, n_osc=4, code_distance=3)
        assert upde is not None
        assert upde.n_osc == 4
        _report("SurfaceCodeUPDE (4 osc, d=3)", dt)


# ---------------------------------------------------------------------------
# 7. QSNN layer — synapse, neuron, STDP, trainer
# ---------------------------------------------------------------------------


class TestQSNNPipeline:
    def test_synapse_pipeline(self):
        syn = sqc.QuantumSynapse(0.7)
        from qiskit import QuantumCircuit

        qc = QuantumCircuit(2)
        syn.apply(qc, 0, 1)
        assert qc.size() > 0
        _report("QuantumSynapse", 0, f"theta={syn.theta:.4f}")

    def test_lif_neuron_pipeline(self):
        neuron = sqc.QuantumLIFNeuron()
        spike, dt = _timed(neuron.step, 1.5)
        assert spike in (0, 1)
        qc = neuron.get_circuit()
        assert qc is not None
        _report("QuantumLIFNeuron step", dt, f"spike={spike}")

    def test_stdp_pipeline(self):
        stdp = sqc.QuantumSTDP()
        syn = sqc.QuantumSynapse(0.5)
        w_before = syn.weight
        stdp.update(syn, pre_measured=1, post_measured=1)
        _report("QuantumSTDP update", 0, f"w: {w_before:.3f} → {syn.weight:.3f}")


# ---------------------------------------------------------------------------
# 8. Identity layer — Arcane Sapience binding
# ---------------------------------------------------------------------------


class TestIdentityPipeline:
    def test_identity_attractor_pipeline(self):
        attractor, dt = _timed(sqc.build_identity_attractor)
        assert attractor is not None
        _report("IdentityAttractor (default spec)", dt)

    def test_identity_key_pipeline(self):
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        fp, dt = _timed(sqc.identity_fingerprint, K, omega, maxiter=10)
        assert isinstance(fp, dict)
        assert "commitment" in fp
        _report("identity_fingerprint (4q)", dt, f"commitment={fp['commitment'][:16]}...")


# ---------------------------------------------------------------------------
# 9. Crypto layer — key hierarchy, QKD
# ---------------------------------------------------------------------------


class TestCryptoPipeline:
    def test_key_hierarchy_pipeline(self):
        from scpn_quantum_control.crypto.hierarchical_keys import (
            key_hierarchy,
            verify_key_chain,
        )

        K = build_knm_paper27(L=4)
        phases = OMEGA_N_16[:4]
        R = 0.8
        h, dt = _timed(key_hierarchy, K, phases, R, nonce=b"bench")
        assert len(h["master"]) == 32
        ok = verify_key_chain(h["master"], h["layers"], K, phases, R, nonce=b"bench")
        assert ok
        _report("key_hierarchy (4 layers)", dt)

    def test_qkd_protocol_pipeline(self):
        from scpn_quantum_control.crypto.entanglement_qkd import scpn_qkd_protocol

        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        result, dt = _timed(scpn_qkd_protocol, K, omega, [0, 1], [2, 3], seed=42)
        assert 0 <= result["qber"] <= 1
        assert result["ground_energy"] < 0
        _report("SCPN-QKD (4q)", dt, f"QBER={result['qber']:.4f}")


# ---------------------------------------------------------------------------
# 10. Analysis layer — FSS, H1 persistence, OTOC, XXZ
# ---------------------------------------------------------------------------


class TestAnalysisPipeline:
    def test_finite_size_scaling_pipeline(self):
        from scpn_quantum_control.analysis.finite_size_scaling import finite_size_scaling

        result, dt = _timed(
            finite_size_scaling, system_sizes=[2, 3], k_range=np.linspace(0.5, 4.0, 6)
        )
        assert len(result.k_c_values) == 2
        _report("FSS (L=2,3)", dt, f"K_c={result.k_c_values}")

    def test_h1_persistence_pipeline(self):
        from scpn_quantum_control.analysis.h1_persistence import scan_h1_persistence

        omega = OMEGA_N_16[:3]
        result, dt = _timed(scan_h1_persistence, omega, n_points=6)
        assert result.k_critical > 0
        _report("H1 persistence (3 osc)", dt, f"K_c={result.k_critical:.4f}")

    def test_otoc_pipeline(self):
        from scpn_quantum_control.analysis.otoc_sync_probe import otoc_sync_scan

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        result, dt = _timed(otoc_sync_scan, K, omega, n_K_values=3, n_time_points=5, t_max=0.5)
        assert result.n_qubits == 2
        _report("OTOC scan (2q)", dt, f"n_K={len(result.K_base_values)}")

    def test_xxz_phase_diagram_pipeline(self):
        from scpn_quantum_control.analysis.xxz_phase_diagram import anisotropy_phase_diagram

        T = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        omega = OMEGA_N_16[:3]
        result, dt = _timed(
            anisotropy_phase_diagram,
            omega,
            T,
            delta_range=np.array([0.0, 1.0]),
            k_range=np.array([1.0, 3.0]),
        )
        assert len(result.k_c_values) == 2
        _report("XXZ phase diagram (3q)", dt)


# ---------------------------------------------------------------------------
# 11. SSGF bridge
# ---------------------------------------------------------------------------


class TestSSGFPipeline:
    def test_ssgf_full_loop(self):
        class _NS:
            def __init__(self):
                self.W = build_knm_paper27(L=4).copy()
                np.fill_diagonal(self.W, 0.0)
                self.theta = np.random.default_rng(42).uniform(0, 2 * np.pi, 4)

        class _Engine:
            def __init__(self):
                self.ns = _NS()

        engine = _Engine()
        loop = sqc.SSGFQuantumLoop(engine, dt=0.1, trotter_reps=1)
        result, dt = _timed(loop.quantum_step)
        assert len(result["theta"]) == 4
        assert 0 <= result["R_global"] <= 1.0
        _report("SSGF quantum loop (4 osc)", dt, f"R={result['R_global']:.4f}")


# ---------------------------------------------------------------------------
# 12. Orchestrator adapter
# ---------------------------------------------------------------------------


class TestOrchestratorPipeline:
    def test_full_adapter_roundtrip(self):
        payload = {
            "layers": [
                {"R": 0.8, "psi": 0.5, "locks": {"0_1": {"plv": 0.9, "lag": 0.1}}},
                {"R": 0.6, "psi": 1.2, "locks": {}},
            ],
            "cross_alignment": [[1.0, 0.5], [0.5, 1.0]],
            "stability": -0.3,
            "regime": "NOMINAL",
        }
        adapter = sqc.PhaseOrchestratorAdapter
        artifact, dt_in = _timed(adapter.from_orchestrator_state, payload)
        telemetry, dt_out = _timed(adapter.to_scpn_control_telemetry, artifact)
        assert telemetry["regime"] == "NOMINAL"
        _report("Orchestrator roundtrip", dt_in + dt_out)


# ---------------------------------------------------------------------------
# 13. Cutting runner — large-scale partitioned simulation
# ---------------------------------------------------------------------------


class TestCuttingPipeline:
    def test_cutting_24_oscillators(self):
        from scpn_quantum_control.hardware.cutting_runner import run_cutting_simulation

        result, dt = _timed(run_cutting_simulation, n_oscillators=24, reps=1, max_partition_size=8)
        assert result.n_partitions == 3
        assert 0 <= result.combined_r_global <= 1.0
        _report(
            "Cutting sim (24 osc, 3 partitions)",
            dt,
            f"R={result.combined_r_global:.4f}, E={result.total_energy_estimate:.4f}",
        )


# ---------------------------------------------------------------------------
# 14. Control — QAOA-MPC, VQLS, Quantum Petri Net
# ---------------------------------------------------------------------------


class TestControlPipeline:
    def test_vqls_pipeline(self):
        vqls = sqc.VQLS_GradShafranov(n_qubits=2)
        result, dt = _timed(vqls.solve, reps=1, maxiter=5, seed=42)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))
        _report("VQLS Grad-Shafranov (2q)", dt)

    def test_qaoa_mpc_importable(self):
        assert sqc.QAOA_MPC is not None
        _report("QAOA_MPC", 0, "importable ✓")

    def test_quantum_petri_net_importable(self):
        assert sqc.QuantumPetriNet is not None
        _report("QuantumPetriNet", 0, "importable ✓")


# ---------------------------------------------------------------------------
# 15. Benchmarks — quantum advantage scaling
# ---------------------------------------------------------------------------


class TestBenchmarkPipeline:
    def test_scaling_benchmark(self):
        result, dt = _timed(sqc.run_scaling_benchmark, sizes=[2, 3])
        assert len(result) > 0
        _report("Scaling benchmark (L=2,3)", dt, f"n_results={len(result)}")
