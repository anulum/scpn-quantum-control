"""Tests for noisy simulator support."""

import pytest

from scpn_quantum_control.hardware.noise_model import heron_r2_noise_model
from scpn_quantum_control.hardware.runner import HardwareRunner


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


def test_noise_model_returns_model():
    nm = heron_r2_noise_model()
    from qiskit_aer.noise import NoiseModel

    assert isinstance(nm, NoiseModel)


def test_noisy_runner_connects(noisy_runner):
    assert noisy_runner.backend is not None


def test_noisy_produces_error_events(noisy_runner):
    """A GHZ circuit on a noisy sim should show non-ideal counts."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    qc.h(0)
    for i in range(1, 4):
        qc.cx(0, i)
    qc.measure_all()

    results = noisy_runner.run_sampler(qc, shots=5000, name="ghz_noisy")
    counts = results[0].counts
    # Noiseless GHZ gives only "0000" and "1111". Noisy should have extras.
    assert len(counts) > 2


def test_noisy_R_lower_than_noiseless(noisy_runner, clean_runner):
    """Noisy order parameter should be worse than noiseless for same circuit."""
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

    R_noisy, _, _, _ = _R_from_xyz(noisy_hw[0].counts, noisy_hw[1].counts, noisy_hw[2].counts, n)
    R_clean, _, _, _ = _R_from_xyz(clean_hw[0].counts, clean_hw[1].counts, clean_hw[2].counts, n)

    # Noisy R should be lower (or at most equal within statistical noise)
    assert R_noisy < R_clean + 0.15


def test_custom_noise_params():
    """Custom T1/T2/error rates should produce a valid model."""
    nm = heron_r2_noise_model(t1_us=100.0, t2_us=80.0, cz_error=0.02)
    from qiskit_aer.noise import NoiseModel

    assert isinstance(nm, NoiseModel)
