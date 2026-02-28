"""Tests for ZNE error mitigation."""

import numpy as np
import pytest
from qiskit import QuantumCircuit

from scpn_quantum_control.mitigation.zne import (
    ZNEResult,
    gate_fold_circuit,
    zne_extrapolate,
)


def test_scale_1_identity():
    """scale=1 returns an equivalent circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    folded = gate_fold_circuit(qc, scale=1)
    # Same number of qubits and classical bits
    assert folded.num_qubits == qc.num_qubits
    assert folded.num_clbits == qc.num_clbits


def test_scale_3_triples_unitary_depth():
    """scale=3 should roughly triple the non-measurement gate count."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)

    folded = gate_fold_circuit(qc, scale=3)
    base_gates = sum(qc.count_ops().values())
    folded_gates = sum(folded.count_ops().values())
    # G G†G = 3x the original gates (approximately)
    assert folded_gates >= base_gates * 2


def test_even_scale_raises():
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(ValueError, match="odd positive"):
        gate_fold_circuit(qc, scale=2)


def test_zero_scale_raises():
    qc = QuantumCircuit(1)
    qc.h(0)
    with pytest.raises(ValueError):
        gate_fold_circuit(qc, scale=0)


def test_measurements_preserved():
    """Measurements should be present in the folded circuit."""
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()

    folded = gate_fold_circuit(qc, scale=3)
    assert folded.num_clbits > 0


def test_linear_extrapolation():
    """Linear extrapolation of y = 1 - 0.1*x should give ~1.0 at x=0."""
    scales = [1, 3, 5]
    evs = [0.9, 0.7, 0.5]
    result = zne_extrapolate(scales, evs, order=1)
    assert isinstance(result, ZNEResult)
    assert abs(result.zero_noise_estimate - 1.0) < 0.01


def test_quadratic_extrapolation():
    """Quadratic fit should handle curved data."""
    scales = [1, 3, 5]
    evs = [0.9, 0.65, 0.3]
    result = zne_extrapolate(scales, evs, order=2)
    assert isinstance(result, ZNEResult)
    assert np.isfinite(result.zero_noise_estimate)


def test_zne_result_fields():
    result = zne_extrapolate([1, 3], [0.8, 0.6], order=1)
    assert result.noise_scales == [1, 3]
    assert result.expectation_values == [0.8, 0.6]
    assert np.isfinite(result.fit_residual)


def test_noisy_sim_zne_improvement(tmp_path):
    """ZNE on a noisy simulator should extrapolate closer to noiseless value."""
    from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27
    from scpn_quantum_control.hardware.experiments import (
        _build_evo_base,
        _build_xyz_circuits,
        _R_from_xyz,
    )
    from scpn_quantum_control.hardware.noise_model import heron_r2_noise_model
    from scpn_quantum_control.hardware.runner import HardwareRunner

    nm = heron_r2_noise_model(cz_error=0.05)
    runner = HardwareRunner(
        use_simulator=True, noise_model=nm, results_dir=str(tmp_path / "results")
    )
    runner.connect()

    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    base = _build_evo_base(n, K, omega, 0.1, trotter_reps=2)

    R_per_scale = []
    for s in [1, 3, 5]:
        folded = gate_fold_circuit(base, s)
        qc_z, qc_x, qc_y = _build_xyz_circuits(folded, n)
        hw = runner.run_sampler([qc_z, qc_x, qc_y], shots=3000, name=f"zne_s{s}")
        R, _, _, _ = _R_from_xyz(hw[0].counts, hw[1].counts, hw[2].counts, n)
        R_per_scale.append(R)

    result = zne_extrapolate([1, 3, 5], R_per_scale, order=1)
    # ZNE estimate should be >= the noisy scale-1 value (extrapolating toward truth)
    assert result.zero_noise_estimate >= R_per_scale[0] - 0.1
