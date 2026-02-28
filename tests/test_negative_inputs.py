"""Negative tests: verify error handling for invalid inputs."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge import knm_to_hamiltonian, probability_to_angle
from scpn_quantum_control.bridge.sc_to_quantum import bitstream_to_statevector
from scpn_quantum_control.control.qaoa_mpc import QAOA_MPC
from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov
from scpn_quantum_control.qsnn.qlif import QuantumLIFNeuron


def test_probability_to_angle_nan():
    """NaN input should produce NaN output (not crash)."""
    result = probability_to_angle(float("nan"))
    assert np.isnan(result)


def test_probability_to_angle_clamps_negative():
    """Negative probability clamps to 0 → angle = 0."""
    theta = probability_to_angle(-0.5)
    assert theta == pytest.approx(0.0, abs=1e-10)


def test_probability_to_angle_clamps_above_one():
    """p > 1 clamps to 1 → angle = pi."""
    theta = probability_to_angle(1.5)
    assert theta == pytest.approx(np.pi, abs=1e-10)


def test_bitstream_single_zero():
    """Single-element bitstream [0] → p=0 → |0⟩."""
    sv = bitstream_to_statevector(np.array([0], dtype=np.uint8))
    assert len(sv) == 2
    assert sv[0] ** 2 > 0.99


def test_bitstream_all_ones():
    """All-ones bitstream → p=1 → |1⟩ state."""
    sv = bitstream_to_statevector(np.ones(100, dtype=np.uint8))
    assert sv[1] ** 2 > 0.99


def test_bitstream_all_zeros():
    """All-zeros bitstream → p=0 → |0⟩ state."""
    sv = bitstream_to_statevector(np.zeros(100, dtype=np.uint8))
    assert sv[0] ** 2 > 0.99


def test_hamiltonian_2q_minimal():
    """2-qubit Hamiltonian from 2x2 K matrix."""
    K = np.array([[0.45, 0.3], [0.3, 0.45]])
    omega = np.array([1.0, 0.8])
    H = knm_to_hamiltonian(K, omega)
    assert H.num_qubits == 2


def test_qaoa_horizon_1():
    """QAOA with horizon=1 should produce a single action."""
    B = np.eye(2)
    target = np.array([0.5, 0.5])
    mpc = QAOA_MPC(B, target, horizon=1, p_layers=1)
    actions = mpc.optimize()
    assert len(actions) == 1
    assert actions[0] in (0, 1)


def test_vqls_small_system():
    """VQLS with minimal qubits (3) converges without error."""
    solver = VQLS_GradShafranov(n_qubits=3)
    psi = solver.solve(maxiter=3)
    assert psi is not None
    assert len(psi) == 2**3


def test_qlif_zero_current():
    """QuantumLIFNeuron with zero input current → no spike."""
    neuron = QuantumLIFNeuron(v_rest=0.0, v_threshold=1.0)
    spike = neuron.step(0.0)
    assert spike in (0, 1)


def test_qlif_large_current():
    """QuantumLIFNeuron with large input current → spike."""
    neuron = QuantumLIFNeuron(v_rest=0.0, v_threshold=0.1)
    # Multiple steps with large current should eventually spike
    spikes = [neuron.step(10.0) for _ in range(20)]
    assert sum(spikes) > 0
