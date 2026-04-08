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


# ---------------------------------------------------------------------------
# 16. MS-QEC Multi-Scale — hierarchical concatenated QEC
# ---------------------------------------------------------------------------


class TestMSQECPipeline:
    def test_build_multiscale_qec(self):
        from scpn_quantum_control.qec.multiscale_qec import build_multiscale_qec

        K = build_knm_paper27()
        result, dt = _timed(build_multiscale_qec, K, p_physical=0.001)
        assert result.concatenation_depth == 5
        assert result.total_physical_qubits > 0
        assert dt < 100, f"build_multiscale_qec must complete in <100ms, took {dt:.1f}ms"
        _report(
            "MS-QEC build",
            dt,
            f"levels={result.concatenation_depth}, qubits={result.total_physical_qubits}",
        )

    def test_concatenated_logical_rate(self):
        from scpn_quantum_control.qec.multiscale_qec import concatenated_logical_rate

        result, dt = _timed(concatenated_logical_rate, 0.001, [5, 5, 5, 5, 5])
        assert len(result) == 5
        assert all(r < 1.0 for r in result)
        assert dt < 1, f"concatenated_logical_rate must complete in <1ms, took {dt:.1f}ms"
        _report("Concatenated rates (5 levels)", dt, f"p_L_final={result[-1]:.2e}")

    def test_syndrome_flow(self):
        from scpn_quantum_control.qec.multiscale_qec import build_multiscale_qec
        from scpn_quantum_control.qec.syndrome_flow import syndrome_flow_analysis

        K = build_knm_paper27()
        ms = build_multiscale_qec(K, p_physical=0.001, distances=[3, 3, 3, 3, 3])
        flows, dt = _timed(syndrome_flow_analysis, K, ms)
        assert len(flows) == 4
        assert dt < 10, f"syndrome_flow_analysis must complete in <10ms, took {dt:.1f}ms"
        _report(
            "Syndrome flow (4 edges)",
            dt,
            f"max_weight={max(f.syndrome_weight for f in flows):.4f}",
        )


# ---------------------------------------------------------------------------
# 17. FEP — Free Energy Principle
# ---------------------------------------------------------------------------


class TestFEPPipeline:
    def test_variational_free_energy(self):
        from scpn_quantum_control.fep.variational_free_energy import variational_free_energy

        K = build_knm_paper27()
        n = K.shape[0]
        mu = np.zeros(n)
        sigma = 0.1 * np.eye(n)
        x = np.random.default_rng(42).standard_normal(n) * 0.1
        result, dt = _timed(variational_free_energy, mu, sigma, x, K)
        assert isinstance(result.free_energy, float)
        assert dt < 5, f"variational_free_energy must complete in <5ms, took {dt:.1f}ms"
        _report("Variational free energy (n=16)", dt, f"F={result.free_energy:.4f}")

    def test_predictive_coding_step(self):
        from scpn_quantum_control.fep.predictive_coding import predictive_coding_step

        K = build_knm_paper27(L=4)
        x = np.array([0.5, 0.3, -0.2, 0.1])
        beliefs = np.zeros(4)
        result, dt = _timed(predictive_coding_step, x, beliefs, K, learning_rate=0.001)
        assert isinstance(result.free_energy, float)
        assert dt < 5, f"predictive_coding_step must complete in <5ms, took {dt:.1f}ms"
        _report(
            "PC step (n=4)",
            dt,
            f"F={result.free_energy:.4f}, error_norm={result.total_error_norm:.4f}",
        )

    def test_free_energy_gradient(self):
        from scpn_quantum_control.fep.variational_free_energy import free_energy_gradient

        K = build_knm_paper27()
        n = K.shape[0]
        mu = np.zeros(n)
        sigma = 0.1 * np.eye(n)
        x = np.random.default_rng(42).standard_normal(n) * 0.1
        grad, dt = _timed(free_energy_gradient, mu, sigma, x, K)
        assert grad.shape == (n,)
        assert dt < 2, f"free_energy_gradient must complete in <2ms, took {dt:.1f}ms"
        _report("FE gradient (n=16, Rust)", dt, f"||grad||={np.linalg.norm(grad):.4f}")


# ---------------------------------------------------------------------------
# 18. Ψ-field — Lattice Gauge Simulator
# ---------------------------------------------------------------------------


class TestPsiFieldPipeline:
    def test_scpn_to_lattice(self):
        from scpn_quantum_control.psi_field.scpn_mapping import scpn_to_lattice

        lattice, dt = _timed(scpn_to_lattice, beta=2.0, seed=42)
        assert lattice.n_layers == 16
        assert lattice.gauge.n_edges == 120
        assert dt < 50, f"scpn_to_lattice must complete in <50ms, took {dt:.1f}ms"
        _report(
            "SCPN→lattice (16 layers)",
            dt,
            f"edges={lattice.gauge.n_edges}, plaq={len(lattice.gauge.plaquettes)}",
        )

    def test_hmc_update(self):
        from scpn_quantum_control.psi_field.lattice import U1LatticGauge, hmc_update

        K = build_knm_paper27(L=4)
        g = U1LatticGauge(K, beta=2.0, seed=42)
        (accepted, dH), dt = _timed(hmc_update, g, n_leapfrog=10, step_size=0.02)
        assert isinstance(accepted, bool)
        assert dt < 10, f"HMC step must complete in <10ms, took {dt:.1f}ms"
        _report("HMC step (n=4, 10 leapfrog)", dt, f"accepted={accepted}, dH={dH:.4f}")

    def test_topological_charge(self):
        from scpn_quantum_control.psi_field.observables import topological_charge
        from scpn_quantum_control.psi_field.scpn_mapping import scpn_to_lattice

        lattice = scpn_to_lattice(beta=2.0, seed=42)
        q, dt = _timed(topological_charge, lattice.gauge)
        assert isinstance(q, float)
        assert dt < 5, f"topological_charge must complete in <5ms, took {dt:.1f}ms"
        _report("Topological charge (16 layers, Rust)", dt, f"Q={q:.4f}")

    def test_gauge_covariant_kinetic(self):
        from scpn_quantum_control.psi_field.infoton import gauge_covariant_kinetic
        from scpn_quantum_control.psi_field.scpn_mapping import scpn_to_lattice

        lattice = scpn_to_lattice(beta=2.0, seed=42)
        T, dt = _timed(gauge_covariant_kinetic, lattice.infoton, lattice.gauge)
        assert T >= 0.0
        assert dt < 5, f"gauge_covariant_kinetic must complete in <5ms, took {dt:.1f}ms"
        _report("Gauge kinetic (16 layers)", dt, f"T={T:.4f}")


# ---------------------------------------------------------------------------
# 19. GUESS — Symmetry Decay ZNE
# ---------------------------------------------------------------------------


class TestGUESSPipeline:
    def test_learn_decay(self):
        from scpn_quantum_control.mitigation.symmetry_decay import learn_symmetry_decay

        model, dt = _timed(learn_symmetry_decay, 4.0, [3.8, 3.5, 3.0, 2.5, 2.0], [1, 3, 5, 7, 9])
        assert model.alpha > 0.0
        assert dt < 2, f"learn_symmetry_decay must complete in <2ms, took {dt:.1f}ms"
        _report("GUESS learn (5 scales, Rust)", dt, f"alpha={model.alpha:.4f}")

    def test_extrapolate(self):
        from scpn_quantum_control.mitigation.symmetry_decay import (
            guess_extrapolate,
            learn_symmetry_decay,
        )

        model = learn_symmetry_decay(4.0, [3.8, 3.0], [1, 3])
        result, dt = _timed(guess_extrapolate, 0.5, 3.8, model)
        assert isinstance(result.mitigated_value, float)
        assert dt < 1, f"guess_extrapolate must complete in <1ms, took {dt:.1f}ms"
        _report("GUESS extrapolate", dt, f"correction={result.correction_factor:.4f}")


# ---------------------------------------------------------------------------
# 20. DynQ — Topology-Agnostic Qubit Mapper
# ---------------------------------------------------------------------------


class TestDynQPipeline:
    def test_community_detection(self):
        import numpy as np

        from scpn_quantum_control.hardware.qubit_mapper import (
            build_calibration_graph,
            detect_execution_regions,
        )

        rng = np.random.default_rng(42)
        errors = {}
        for i in range(156):
            for j in [i + 1, i + 2]:
                if j < 156:
                    errors[(i, j)] = rng.uniform(0.001, 0.02)
        G = build_calibration_graph(errors)
        regions, dt = _timed(detect_execution_regions, G, 3, 1.0, 42)
        assert len(regions) > 0
        assert dt < 50, f"156-qubit detection must complete in <50ms, took {dt:.1f}ms"
        _report("DynQ detection (156 qubits)", dt, f"n_regions={len(regions)}")

    def test_full_pipeline(self):
        import numpy as np

        from scpn_quantum_control.hardware.qubit_mapper import dynq_initial_layout

        rng = np.random.default_rng(42)
        errors = {}
        for i in range(20):
            for j in [i + 1, i + 2]:
                if j < 20:
                    errors[(i, j)] = rng.uniform(0.001, 0.02)
        result, dt = _timed(dynq_initial_layout, errors, 5, None, 1.0, 3, 42)
        assert result is not None
        assert len(result.initial_layout) == 5
        assert dt < 20, f"DynQ pipeline must complete in <20ms, took {dt:.1f}ms"
        _report(
            "DynQ full pipeline (20 qubits)", dt, f"region_size={result.selected_region.n_qubits}"
        )
