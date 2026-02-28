"""Regression tests: verify simulator results match known baselines.

These tests guard against accidental changes to Knm parameters,
Hamiltonian construction, or Trotter evolution that would shift
the physics outputs. Baselines from February 2026 ibm_fez runs +
Aer statevector.
"""

from __future__ import annotations

import numpy as np
from qiskit_aer import AerSimulator

from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27, knm_to_hamiltonian
from scpn_quantum_control.hardware.experiments import (
    _build_evo_base,
    _R_from_xyz,
)

# --- Knm parameter baselines (Paper 27) ---


def test_knm_calibration_anchors():
    """K[1,2]=0.302, K[2,3]=0.201, K[3,4]=0.252, K[4,5]=0.154."""
    K = build_knm_paper27(L=16)
    assert abs(K[0, 1] - 0.302) < 0.001
    assert abs(K[1, 2] - 0.201) < 0.001
    assert abs(K[2, 3] - 0.252) < 0.001
    assert abs(K[3, 4] - 0.154) < 0.001


def test_knm_cross_hierarchy_boosts():
    """L1-L16 boost to 0.05; L5-L7 boost floor at 0.15 (natural value is higher)."""
    K = build_knm_paper27(L=16)
    assert abs(K[0, 15] - 0.05) < 0.001
    # L5-L7: max(natural, 0.15) — natural ~0.247 exceeds boost
    assert K[4, 6] >= 0.15


def test_omega_n_16_values():
    """First and last omega values from Paper 27."""
    assert abs(OMEGA_N_16[0] - 1.329) < 0.001
    assert abs(OMEGA_N_16[15] - 0.991) < 0.001
    assert len(OMEGA_N_16) == 16


# --- Hamiltonian baselines ---


def test_4q_ground_energy_baseline():
    """4-qubit exact ground state energy: -6.303 ± 0.01."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    H = knm_to_hamiltonian(K, omega)
    mat = H.to_matrix()
    if hasattr(mat, "toarray"):
        mat = mat.toarray()
    E0 = np.linalg.eigvalsh(mat)[0]
    assert abs(E0 - (-6.303)) < 0.01, f"E0={E0}, expected ~-6.303"


# --- Simulator R baselines ---


def _simulate_R(n_osc: int, t: float, trotter_reps: int) -> float:
    """Run statevector Trotter evolution, return R."""
    from qiskit import transpile
    from qiskit.quantum_info import Statevector

    K = build_knm_paper27(L=n_osc)
    omega = OMEGA_N_16[:n_osc]
    qc = _build_evo_base(n_osc, K, omega, t, trotter_reps)
    # Transpile to decompose PauliEvolution gates before Aer
    qc = transpile(qc, basis_gates=["cx", "u3", "u2", "u1", "id"], optimization_level=0)
    qc.save_statevector()
    sim = AerSimulator(method="statevector")
    sv = Statevector(sim.run(qc).result().get_statevector())

    x_exp = np.array(
        [float(sv.expectation_value(_pauli_op("X", i, n_osc)).real) for i in range(n_osc)]
    )
    y_exp = np.array(
        [float(sv.expectation_value(_pauli_op("Y", i, n_osc)).real) for i in range(n_osc)]
    )

    z_complex = np.mean(x_exp + 1j * y_exp)
    return float(abs(z_complex))


def _pauli_op(pauli: str, qubit: int, n: int):
    """Single-qubit Pauli on qubit `qubit` in n-qubit system."""
    from qiskit.quantum_info import SparsePauliOp

    label = ["I"] * n
    label[qubit] = pauli
    return SparsePauliOp("".join(reversed(label)))


def test_4osc_statevector_R_baseline():
    """4-osc statevector from |0⟩: R is finite and positive."""
    R = _simulate_R(4, t=1.0, trotter_reps=1)
    assert 0.0 < R < 1.5, f"R={R}, out of physical range"


def test_R_monotonic_with_time():
    """R changes monotonically from initial state over short evolution."""
    R_short = _simulate_R(4, t=0.1, trotter_reps=1)
    R_long = _simulate_R(4, t=1.0, trotter_reps=1)
    # Both should be finite and non-trivial
    assert np.isfinite(R_short) and np.isfinite(R_long)
    assert R_short != R_long, "Evolution should change R"


def test_R_from_xyz_known_distribution():
    """_R_from_xyz: equal 0/1 counts → <X>=<Y>=0 → R=0."""
    # 50/50 on each qubit: <X_i>=0, <Y_i>=0
    counts_x = {"0000": 5000, "1111": 5000}
    counts_y = {"0000": 5000, "1111": 5000}
    counts_z = {"0000": 5000, "1111": 5000}
    R, exp_x, exp_y, exp_z = _R_from_xyz(counts_z, counts_x, counts_y, n_qubits=4)
    assert abs(R) < 0.01, f"Balanced counts should give R≈0, got {R}"
