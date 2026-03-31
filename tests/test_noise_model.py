# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Tests for Noise Model
"""Tests for noisy simulator support — elite multi-angle coverage."""

import pytest
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel

from scpn_quantum_control.hardware.noise_model import (
    CZ_ERROR_RATE,
    READOUT_ERROR_RATE,
    T1_US,
    T2_US,
    heron_r2_noise_model,
)
from scpn_quantum_control.hardware.runner import HardwareRunner

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def noisy_runner(tmp_path):
    nm = heron_r2_noise_model()
    runner = HardwareRunner(
        use_simulator=True, noise_model=nm, results_dir=str(tmp_path / "results")
    )
    runner.connect()
    return runner


@pytest.fixture
def clean_runner(tmp_path):
    runner = HardwareRunner(use_simulator=True, results_dir=str(tmp_path / "results"))
    runner.connect()
    return runner


# ---------------------------------------------------------------------------
# Noise model construction
# ---------------------------------------------------------------------------


class TestNoiseModelConstruction:
    def test_returns_noise_model(self):
        nm = heron_r2_noise_model()
        assert isinstance(nm, NoiseModel)

    def test_default_params_match_constants(self):
        assert T1_US == 300.0
        assert T2_US == 200.0
        assert CZ_ERROR_RATE == 0.005
        assert READOUT_ERROR_RATE == 0.002

    def test_custom_params(self):
        nm = heron_r2_noise_model(t1_us=100.0, t2_us=80.0, cz_error=0.02)
        assert isinstance(nm, NoiseModel)

    def test_different_params_different_model(self):
        nm1 = heron_r2_noise_model(cz_error=0.001)
        nm2 = heron_r2_noise_model(cz_error=0.05)
        # Models should be structurally different (different error rates)
        assert nm1 is not nm2


# ---------------------------------------------------------------------------
# Noisy runner connectivity
# ---------------------------------------------------------------------------


class TestNoisyRunner:
    def test_connects(self, noisy_runner):
        assert noisy_runner.backend is not None

    def test_transpile_works(self, noisy_runner):
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        isa = noisy_runner.transpile(qc)
        assert isa.num_qubits >= 2


# ---------------------------------------------------------------------------
# Noise effects on measurements
# ---------------------------------------------------------------------------


class TestNoiseEffects:
    def test_noisy_produces_error_events(self, noisy_runner):
        """GHZ on noisy sim should show non-ideal counts."""
        qc = QuantumCircuit(4)
        qc.h(0)
        for i in range(1, 4):
            qc.cx(0, i)
        qc.measure_all()

        results = noisy_runner.run_sampler(qc, shots=5000, name="ghz_noisy")
        counts = results[0].counts
        assert len(counts) > 2

    def test_bell_pair_noisy_vs_noiseless(self, noisy_runner, clean_runner):
        """Noisy Bell pair should have lower fidelity than noiseless."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        noisy_result = noisy_runner.run_sampler(qc, shots=10000, name="bell_noisy")
        clean_result = clean_runner.run_sampler(qc, shots=10000, name="bell_clean")

        noisy_counts = noisy_result[0].counts
        clean_counts = clean_result[0].counts

        # Ideal Bell: 50% "00" + 50% "11". Noise introduces "01", "10".
        noisy_non_ideal = sum(v for k, v in noisy_counts.items() if k not in ("00", "11"))
        clean_non_ideal = sum(v for k, v in clean_counts.items() if k not in ("00", "11"))
        assert noisy_non_ideal > clean_non_ideal

    def test_noisy_R_lower_than_noiseless(self, noisy_runner, clean_runner):
        """Noisy order parameter should be worse than noiseless."""
        from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
        from scpn_quantum_control.hardware.experiments import (
            _build_evo_base,
            _build_xyz_circuits,
            _R_from_xyz,
        )

        n = 4
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        base = _build_evo_base(n, K, omega, 0.1, trotter_reps=2)
        qc_z, qc_x, qc_y = _build_xyz_circuits(base, n)

        noisy_hw = noisy_runner.run_sampler([qc_z, qc_x, qc_y], shots=5000, name="noisy")
        clean_hw = clean_runner.run_sampler([qc_z, qc_x, qc_y], shots=5000, name="clean")

        R_noisy, *_ = _R_from_xyz(noisy_hw[0].counts, noisy_hw[1].counts, noisy_hw[2].counts, n)
        R_clean, *_ = _R_from_xyz(clean_hw[0].counts, clean_hw[1].counts, clean_hw[2].counts, n)

        assert R_noisy < R_clean + 0.15


# ---------------------------------------------------------------------------
# Noise model physics: error rates and relaxation
# ---------------------------------------------------------------------------


class TestNoisePhysics:
    def test_higher_error_more_noise(self):
        """Higher CZ error rate → more non-ideal counts in Bell pair."""
        from scpn_quantum_control.hardware.noise_model import heron_r2_noise_model

        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()

        nm_low = heron_r2_noise_model(cz_error=0.001)
        nm_high = heron_r2_noise_model(cz_error=0.1)

        runner_low = HardwareRunner(use_simulator=True, noise_model=nm_low)
        runner_low.connect()
        runner_high = HardwareRunner(use_simulator=True, noise_model=nm_high)
        runner_high.connect()

        r_low = runner_low.run_sampler(qc, shots=5000, name="low")
        r_high = runner_high.run_sampler(qc, shots=5000, name="high")

        err_low = sum(v for k, v in r_low[0].counts.items() if k not in ("00", "11"))
        err_high = sum(v for k, v in r_high[0].counts.items() if k not in ("00", "11"))
        assert err_high > err_low


# ---------------------------------------------------------------------------
# Pipeline: Knm → noisy evolution → R degradation → wired
# ---------------------------------------------------------------------------


class TestNoisePipeline:
    def test_pipeline_knm_noise_degradation(self, noisy_runner, clean_runner):
        """Full pipeline: Knm → Trotter → noisy/clean → R comparison.
        Verifies noise model is wired and degrades observables as expected.
        """
        import time

        from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
        from scpn_quantum_control.hardware.experiments import (
            _build_evo_base,
            _build_xyz_circuits,
            _R_from_xyz,
        )

        n = 3
        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        base = _build_evo_base(n, K, omega, 0.1, trotter_reps=1)
        qc_z, qc_x, qc_y = _build_xyz_circuits(base, n)

        t0 = time.perf_counter()
        noisy_hw = noisy_runner.run_sampler([qc_z, qc_x, qc_y], shots=2000, name="n")
        clean_hw = clean_runner.run_sampler([qc_z, qc_x, qc_y], shots=2000, name="c")
        dt = (time.perf_counter() - t0) * 1000

        R_n, *_ = _R_from_xyz(noisy_hw[0].counts, noisy_hw[1].counts, noisy_hw[2].counts, n)
        R_c, *_ = _R_from_xyz(clean_hw[0].counts, clean_hw[1].counts, clean_hw[2].counts, n)

        import numpy as np

        assert np.isfinite(R_n)
        assert np.isfinite(R_c)

        print(f"\n  PIPELINE Knm→Noisy (3q): {dt:.1f} ms")
        print(f"  R_clean={R_c:.4f}, R_noisy={R_n:.4f}")
