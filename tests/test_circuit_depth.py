# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Circuit Depth
"""Circuit depth regression: guard against unintended depth changes."""

from __future__ import annotations

from qiskit import transpile

from scpn_quantum_control.bridge import OMEGA_N_16, build_knm_paper27, knm_to_ansatz
from scpn_quantum_control.hardware.experiments import _build_evo_base


def _transpiled_depth(qc) -> int:
    """Transpile to CX+U3 basis and return depth."""
    t = transpile(qc, basis_gates=["cx", "u3", "u2", "u1", "id"], optimization_level=0)
    return t.depth()


def test_4osc_1rep_trotter_depth(knm_4q):
    """4-qubit, 1 Trotter rep: depth should not exceed 100."""
    K, omega = knm_4q
    qc = _build_evo_base(4, K, omega, t=1.0, trotter_reps=1)
    d = _transpiled_depth(qc)
    assert d < 100, f"4q 1-rep depth={d}, expected <100"


def test_8osc_1rep_trotter_depth(knm_8q):
    """8-qubit, 1 Trotter rep: depth should not exceed 300."""
    K, omega = knm_8q
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


def test_depth_scales_with_reps(knm_4q):
    """More Trotter reps → proportionally deeper circuit."""
    K, omega = knm_4q
    qc1 = _build_evo_base(4, K, omega, t=1.0, trotter_reps=1)
    qc2 = _build_evo_base(4, K, omega, t=1.0, trotter_reps=3)
    d1 = _transpiled_depth(qc1)
    d3 = _transpiled_depth(qc2)
    assert d3 > d1, f"3-rep depth ({d3}) should exceed 1-rep depth ({d1})"
    assert d3 < 4 * d1, f"3-rep depth ({d3}) should be < 4x 1-rep depth ({d1})"


def test_ansatz_depth_scales_with_reps(knm_4q):
    """Ansatz depth grows with reps."""
    K, _ = knm_4q
    qc1 = knm_to_ansatz(K, reps=1)
    qc2 = knm_to_ansatz(K, reps=3)
    d1 = _transpiled_depth(qc1)
    d3 = _transpiled_depth(qc2)
    assert d3 > d1


def test_2osc_1rep_trotter_depth():
    """2-qubit, 1 Trotter rep: depth should be very shallow."""
    K = build_knm_paper27(L=2)
    omega = OMEGA_N_16[:2]
    qc = _build_evo_base(2, K, omega, t=1.0, trotter_reps=1)
    d = _transpiled_depth(qc)
    assert d < 50, f"2q 1-rep depth={d}, expected <50"


def test_depth_increases_with_system_size():
    """Larger systems → deeper circuits for same Trotter parameters."""
    depths = {}
    for L in [2, 4, 8]:
        K = build_knm_paper27(L=L)
        omega = OMEGA_N_16[:L]
        qc = _build_evo_base(L, K, omega, t=0.5, trotter_reps=1)
        depths[L] = _transpiled_depth(qc)
    assert depths[2] < depths[4] < depths[8]


def test_circuit_gate_count_4q(knm_4q):
    """4-qubit circuit should have reasonable gate counts."""
    from qiskit import transpile

    K, omega = knm_4q
    qc = _build_evo_base(4, K, omega, t=1.0, trotter_reps=1)
    t_qc = transpile(qc, basis_gates=["cx", "u3", "u2", "u1", "id"], optimization_level=0)
    ops = t_qc.count_ops()
    total = sum(ops.values())
    assert total > 0
    assert total < 500


def test_depth_finite_for_all_sizes():
    """Depth should be finite and positive for all standard sizes."""
    for L in [2, 3, 4, 6, 8]:
        K = build_knm_paper27(L=L)
        omega = OMEGA_N_16[:L]
        qc = _build_evo_base(L, K, omega, t=0.1, trotter_reps=1)
        d = _transpiled_depth(qc)
        assert 0 < d < 2000, f"L={L}: depth={d}"


def test_ansatz_qubit_count_matches_K(knm_4q):
    """Ansatz qubit count must match coupling matrix dimension."""
    K, _ = knm_4q
    qc = knm_to_ansatz(K, reps=2)
    assert qc.num_qubits == K.shape[0]


# ---------------------------------------------------------------------------
# Pipeline: Knm → circuit → depth regression → wired
# ---------------------------------------------------------------------------


def test_pipeline_depth_regression():
    """Full pipeline: build_knm → Trotter circuit → transpile → depth metrics.
    Verifies circuit depth module is wired and produces actionable data.
    """
    import time

    K = build_knm_paper27(L=4)
    omega = OMEGA_N_16[:4]

    t0 = time.perf_counter()
    qc = _build_evo_base(4, K, omega, t=0.5, trotter_reps=3)
    d = _transpiled_depth(qc)
    dt = (time.perf_counter() - t0) * 1000

    assert d > 0
    assert d < 500

    print(f"\n  PIPELINE Knm→Trotter→Depth (4q, 3 reps): {dt:.1f} ms")
    print(f"  Transpiled depth = {d}")
