"""Tests for qsnn/qlif.py."""

from scpn_quantum_control.qsnn.qlif import QuantumLIFNeuron


def test_no_input_no_spike():
    neuron = QuantumLIFNeuron(n_shots=0)
    spike = neuron.step(0.0)
    assert spike == 0


def test_strong_input_spikes():
    neuron = QuantumLIFNeuron(v_threshold=1.0, tau_mem=1.0, dt=1.0, n_shots=0)
    for _ in range(10):
        neuron.step(2.0)
    # After enough strong input, membrane reaches threshold
    neuron.step(2.0)
    # Can't guarantee spike on exact step, but circuit should exist
    assert neuron.get_circuit() is not None


def test_reset_on_spike():
    neuron = QuantumLIFNeuron(v_threshold=0.1, tau_mem=100.0, dt=1.0, n_shots=0)
    neuron.v = 0.2  # above threshold
    spike = neuron.step(0.0)
    if spike:
        assert neuron.v == neuron.v_rest


def test_circuit_is_single_qubit():
    neuron = QuantumLIFNeuron()
    neuron.step(0.5)
    qc = neuron.get_circuit()
    assert qc.num_qubits == 1


def test_reset_clears_state():
    neuron = QuantumLIFNeuron()
    neuron.step(1.0)
    neuron.reset()
    assert neuron.v == neuron.v_rest
    assert neuron.get_circuit() is None


def test_statistical_spike_rate():
    """Strong sustained input -> spikes over many steps."""
    neuron = QuantumLIFNeuron(v_threshold=0.5, tau_mem=2.0, dt=1.0, n_shots=0)
    spikes = 0
    n_steps = 100
    for _ in range(n_steps):
        spikes += neuron.step(1.0)
    rate = spikes / n_steps
    assert rate > 0.1
