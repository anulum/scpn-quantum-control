"""Unit tests for experiments.py helper functions."""

import numpy as np

from scpn_quantum_control.hardware.experiments import (
    _build_evo_base,
    _build_xyz_circuits,
    _expectation_per_qubit,
    _R_from_xyz,
)


def test_expectation_per_qubit_all_zeros():
    """All |0> outcomes should give <Z> = +1 for each qubit."""
    counts = {"00": 1000}
    exp = _expectation_per_qubit(counts, 2)
    np.testing.assert_allclose(exp, [1.0, 1.0])


def test_expectation_per_qubit_all_ones():
    """All |1> outcomes should give <Z> = -1 for each qubit."""
    counts = {"11": 1000}
    exp = _expectation_per_qubit(counts, 2)
    np.testing.assert_allclose(exp, [-1.0, -1.0])


def test_expectation_per_qubit_mixed():
    """50/50 split should give <Z> ~ 0."""
    counts = {"00": 500, "11": 500}
    exp = _expectation_per_qubit(counts, 2)
    np.testing.assert_allclose(exp, [0.0, 0.0])


def test_expectation_per_qubit_single_qubit_asymmetric():
    """For one qubit: P(0)=0.7, P(1)=0.3 => <Z> = 0.4."""
    counts = {"0": 700, "1": 300}
    exp = _expectation_per_qubit(counts, 1)
    np.testing.assert_allclose(exp, [0.4])


def test_R_from_xyz_perfect_x_alignment():
    """All qubits measured as |+> in X-basis should give R ~ 1."""
    n = 4
    z_counts = {"0000": 500, "1111": 500}
    x_counts = {"0000": 1000}  # all <X> = +1
    y_counts = {"0000": 500, "1111": 500}  # <Y> ~ 0
    R, ex, ey, ez = _R_from_xyz(z_counts, x_counts, y_counts, n)
    assert abs(ex[0] - 1.0) < 0.01
    assert R > 0.5


def test_R_from_xyz_random_counts():
    """Random counts should give finite R in [0, 1]."""
    n = 3
    rng = np.random.default_rng(42)
    z_counts = {f"{i:03b}": int(v) for i, v in enumerate(rng.integers(1, 100, 8))}
    x_counts = {f"{i:03b}": int(v) for i, v in enumerate(rng.integers(1, 100, 8))}
    y_counts = {f"{i:03b}": int(v) for i, v in enumerate(rng.integers(1, 100, 8))}
    R, ex, ey, ez = _R_from_xyz(z_counts, x_counts, y_counts, n)
    assert 0.0 <= R <= 1.0 + 0.01
    assert len(ex) == n


def test_build_evo_base_circuit_structure():
    """Evolution base circuit should have n qubits and contain Ry + PauliEvolution."""
    from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27

    n = 4
    K = build_knm_paper27(L=n)
    omega = OMEGA_N_16[:n]
    qc = _build_evo_base(n, K, omega, t=0.1, trotter_reps=2)
    assert qc.num_qubits == 4
    assert qc.num_clbits == 0  # no measurement


def test_build_xyz_circuits_measurement_count():
    """XYZ circuits should each have n classical bits."""
    from qiskit import QuantumCircuit

    base = QuantumCircuit(3)
    base.h(0)
    qc_z, qc_x, qc_y = _build_xyz_circuits(base, 3)
    for qc in [qc_z, qc_x, qc_y]:
        assert qc.num_clbits == 3


def test_build_xyz_circuits_z_has_no_extra_gates():
    """Z-basis circuit should only add measurements, no basis rotations."""
    from qiskit import QuantumCircuit

    base = QuantumCircuit(2)
    base.h(0)
    qc_z, _, _ = _build_xyz_circuits(base, 2)
    ops = qc_z.count_ops()
    assert "h" in ops  # the original H gate
    assert "sdg" not in ops  # no Y-basis rotation


def test_build_xyz_circuits_y_has_sdg():
    """Y-basis circuit should include Sdg gates."""
    from qiskit import QuantumCircuit

    base = QuantumCircuit(2)
    base.h(0)
    _, _, qc_y = _build_xyz_circuits(base, 2)
    ops = qc_y.count_ops()
    assert "sdg" in ops


def test_all_experiments_registry_complete():
    """ALL_EXPERIMENTS must contain every *_experiment function in the module."""
    import inspect

    from scpn_quantum_control.hardware import experiments as mod
    from scpn_quantum_control.hardware.experiments import ALL_EXPERIMENTS

    defined = {
        name
        for name, obj in inspect.getmembers(mod, inspect.isfunction)
        if name.endswith("_experiment") and not name.startswith("_")
    }
    registered = set(ALL_EXPERIMENTS.values())
    registered_names = {f.__name__ for f in registered}
    missing = defined - registered_names
    assert not missing, f"Unregistered experiments: {missing}"
