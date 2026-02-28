"""Circuit depth regression: guard against unintended depth changes."""

from __future__ import annotations

from qiskit import transpile

from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27, knm_to_ansatz
from scpn_quantum_control.hardware.experiments import _build_evo_base


def _transpiled_depth(qc) -> int:
    """Transpile to CX+U3 basis and return depth."""
    t = transpile(qc, basis_gates=["cx", "u3", "u2", "u1", "id"], optimization_level=0)
    return t.depth()


def test_4osc_1rep_trotter_depth():
    """4-qubit, 1 Trotter rep: depth should not exceed 100."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    qc = _build_evo_base(4, K, omega, t=1.0, trotter_reps=1)
    d = _transpiled_depth(qc)
    assert d < 100, f"4q 1-rep depth={d}, expected <100"


def test_8osc_1rep_trotter_depth():
    """8-qubit, 1 Trotter rep: depth should not exceed 300."""
    K = build_knm_paper27(L=8)
    omega = OMEGA_N_16[:8]
    qc = _build_evo_base(8, K, omega, t=1.0, trotter_reps=1)
    d = _transpiled_depth(qc)
    assert d < 300, f"8q 1-rep depth={d}, expected <300"


def test_16q_1rep_trotter_depth():
    """16-qubit, 1 Trotter rep: depth should not exceed 1000."""
    K = build_knm_paper27(L=16)
    omega = OMEGA_N_16
    qc = _build_evo_base(16, K, omega, t=0.1, trotter_reps=1)
    d = _transpiled_depth(qc)
    assert d < 1000, f"16q 1-rep depth={d}, expected <1000"


def test_depth_scales_with_reps():
    """More Trotter reps → proportionally deeper circuit."""
    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]
    qc1 = _build_evo_base(4, K, omega, t=1.0, trotter_reps=1)
    qc2 = _build_evo_base(4, K, omega, t=1.0, trotter_reps=3)
    d1 = _transpiled_depth(qc1)
    d3 = _transpiled_depth(qc2)
    assert d3 > d1, f"3-rep depth ({d3}) should exceed 1-rep depth ({d1})"
    assert d3 < 4 * d1, f"3-rep depth ({d3}) should be < 4x 1-rep depth ({d1})"


def test_ansatz_depth_scales_with_reps():
    """Ansatz depth grows with reps."""
    K = build_knm_paper27(L=4)
    qc1 = knm_to_ansatz(K, reps=1)
    qc2 = knm_to_ansatz(K, reps=3)
    d1 = _transpiled_depth(qc1)
    d3 = _transpiled_depth(qc2)
    assert d3 > d1
