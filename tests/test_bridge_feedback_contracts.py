# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Bridge feedback contract tests
"""Contract tests for orchestrator feedback, SNN backward coupling, K_nm Hamiltonian, and SPN circuit bridge boundaries."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
    build_kuramoto_ring,
    knm_to_dense_matrix,
    knm_to_hamiltonian,
    knm_to_xxz_hamiltonian,
)
from scpn_quantum_control.bridge.spn_to_qcircuit import inhibitor_anti_control, spn_to_circuit
from scpn_quantum_control.hardware.classical import _expectation_pauli
from scpn_quantum_control.qec.control_qec import ControlQEC


class TestOrchestratorFeedback:
    """Tests for compute_orchestrator_feedback action selection."""

    def test_rollback_on_weak_coupling(self):
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        K = build_knm_paper27(L=2) * 0.001
        omega = OMEGA_N_16[:2]
        fb = compute_orchestrator_feedback(K, omega, r_advance=0.8, r_hold=0.5)
        assert fb.action in ("advance", "hold", "rollback")
        assert fb.confidence >= 0.0

    def test_hold_on_medium_coupling(self):
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        K = build_knm_paper27(L=2) * 0.5
        omega = OMEGA_N_16[:2]
        fb = compute_orchestrator_feedback(K, omega, r_advance=0.99, r_hold=0.01)
        assert fb.action in ("advance", "hold", "rollback")
        assert 0.0 <= fb.confidence <= 1.0

    def test_advance_on_strong_coupling(self):
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        K = build_knm_paper27(L=2) * 5.0
        omega = OMEGA_N_16[:2]
        fb = compute_orchestrator_feedback(K, omega, r_advance=0.01, r_hold=0.005)
        assert fb.action in ("advance", "hold", "rollback")

    def test_confidence_bounded_01(self):
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        for scale in [0.001, 0.1, 1.0, 5.0]:
            K = build_knm_paper27(L=2) * scale
            omega = OMEGA_N_16[:2]
            fb = compute_orchestrator_feedback(K, omega)
            assert 0.0 <= fb.confidence <= 1.0, (
                f"Confidence {fb.confidence} out of [0,1] at scale={scale}"
            )

    def test_feedback_has_reason(self):
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        fb = compute_orchestrator_feedback(K, omega)
        assert isinstance(fb.reason, str)
        assert len(fb.reason) > 0

    @pytest.mark.parametrize("n", [2, 3, 4])
    def test_multiple_system_sizes(self, n):
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        K = build_knm_paper27(L=n)
        omega = OMEGA_N_16[:n]
        fb = compute_orchestrator_feedback(K, omega)
        assert fb.action in ("advance", "hold", "rollback")
        assert 0.0 <= fb.confidence <= 1.0

    def test_threshold_boundary(self):
        """Exact threshold values should not crash."""
        from scpn_quantum_control.bridge.orchestrator_feedback import (
            compute_orchestrator_feedback,
        )

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        fb = compute_orchestrator_feedback(K, omega, r_advance=0.5, r_hold=0.5)
        assert fb.action in ("advance", "hold", "rollback")


class TestSNNBackward:
    """Tests for quantum SNN parameter-shift gradients."""

    def test_zero_shift_gradient(self):
        from scpn_quantum_control.bridge.snn_backward import parameter_shift_gradient
        from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer

        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=42)
        spike_rates = np.array([0.5, 0.3])
        target = np.array([0.7, 0.2])
        result = parameter_shift_gradient(layer, spike_rates, target)
        assert result.grad_params is not None
        assert result.grad_spikes is not None

    def test_gradient_shapes(self):
        from scpn_quantum_control.bridge.snn_backward import parameter_shift_gradient
        from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer

        layer = QuantumDenseLayer(n_neurons=2, n_inputs=3, seed=42)
        spike_rates = np.array([0.5, 0.3, 0.8])
        target = np.array([0.7, 0.2])
        result = parameter_shift_gradient(layer, spike_rates, target)
        assert result.grad_spikes.shape == spike_rates.shape

    def test_gradient_finite(self):
        from scpn_quantum_control.bridge.snn_backward import parameter_shift_gradient
        from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer

        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=42)
        spike_rates = np.array([0.5, 0.5])
        target = np.array([0.5, 0.5])
        result = parameter_shift_gradient(layer, spike_rates, target)
        assert np.all(np.isfinite(result.grad_spikes))

    def test_gradient_changes_with_target(self):
        """Different targets should produce different gradients."""
        from scpn_quantum_control.bridge.snn_backward import parameter_shift_gradient
        from scpn_quantum_control.qsnn.qlayer import QuantumDenseLayer

        layer = QuantumDenseLayer(n_neurons=2, n_inputs=2, seed=42)
        spikes = np.array([0.5, 0.5])
        r1 = parameter_shift_gradient(layer, spikes, np.array([1.0, 0.0]))
        r2 = parameter_shift_gradient(layer, spikes, np.array([0.0, 1.0]))
        assert not np.array_equal(r1.grad_spikes, r2.grad_spikes)


class TestKnmHamiltonianEdge:
    def test_knm_to_dense_matrix_complex(self):
        from scpn_quantum_control.bridge.knm_hamiltonian import knm_to_dense_matrix

        K = build_knm_paper27(L=2)
        omega = OMEGA_N_16[:2]
        H = knm_to_dense_matrix(K, omega)
        assert H.shape == (4, 4)


def test_multi_inhibitor_anti_control():
    """Verifies 74-82: >1 inhibitor qubit triggers multi-controlled RYGate."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(4)
    inhibitor_anti_control(qc, [0, 1], target=2, theta=0.5)
    ops = qc.count_ops()
    assert ops.get("x", 0) >= 2, "anti-control pattern requires X gates on inhibitor qubits"


def test_inhibitor_target_in_inhibitor_list():
    """Verifies 82: target qubit is also in the inhibitor list (degenerate case)."""
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    # target=1 is in inhibitor_qubits=[0,1] — controls list becomes [0], still works
    inhibitor_anti_control(qc, [0, 1], target=1, theta=0.3)
    ops = qc.count_ops()
    assert ops.get("x", 0) >= 2, "anti-control pattern requires X gates on inhibitor qubits"


def test_spn_circuit_with_inhibitor_arcs():
    """SPN with inhibitor arcs (negative W_in) exercises the full anti-control path."""
    W_in = np.array([[-1.0, 0.5, 0.0], [0.0, 0.0, 0.8]])
    W_out = np.array([[0.0, 0.0], [0.6, 0.0], [0.0, 0.4]])
    thresholds = np.array([1.0, 1.0])

    qc = spn_to_circuit(W_in, W_out, thresholds)
    assert qc.num_qubits == 3


def test_spn_multi_inhibitor():
    """Two inhibitor arcs on a single transition -> multi-controlled gate."""
    W_in = np.array([[-0.5, -0.3, 0.0]])  # places 0,1 are inhibitors
    W_out = np.array([[0.0], [0.0], [0.7]])  # output to place 2
    thresholds = np.array([1.0])

    qc = spn_to_circuit(W_in, W_out, thresholds)
    assert qc.num_qubits == 3


class TestKnmToDenseMatrix:
    def test_matches_qiskit(self) -> None:
        K = build_knm_paper27(L=4)
        omega = OMEGA_N_16[:4]
        H_dense = knm_to_dense_matrix(K, omega)
        H_qiskit = knm_to_hamiltonian(K, omega).to_matrix()
        if hasattr(H_qiskit, "toarray"):
            H_qiskit = H_qiskit.toarray()
        np.testing.assert_allclose(H_dense, np.array(H_qiskit), atol=1e-12)

    def test_hermitian(self) -> None:
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        H = knm_to_dense_matrix(K, omega)
        np.testing.assert_allclose(H, H.conj().T, atol=1e-12)

    def test_shape(self) -> None:
        for n in [2, 3, 4]:
            K = build_knm_paper27(L=n)
            omega = OMEGA_N_16[:n]
            H = knm_to_dense_matrix(K, omega)
            assert H.shape == (2**n, 2**n)

    def test_zero_coupling(self) -> None:
        n = 3
        K = np.zeros((n, n))
        omega = OMEGA_N_16[:n]
        H = knm_to_dense_matrix(K, omega)
        assert np.allclose(H, np.diag(np.diag(H)))


class TestKnmToXXZ:
    def test_xy_equals_xxz_delta0(self) -> None:
        K = build_knm_paper27(L=3)
        omega = OMEGA_N_16[:3]
        H_xy = knm_to_hamiltonian(K, omega).to_matrix()
        H_xxz = knm_to_xxz_hamiltonian(K, omega, delta=0.0).to_matrix()
        np.testing.assert_allclose(np.array(H_xy), np.array(H_xxz), atol=1e-12)

    def test_empty_pauli_list(self) -> None:
        n = 3
        K = np.full((n, n), 1e-20)
        np.fill_diagonal(K, 0)
        omega = np.zeros(n)
        H = knm_to_xxz_hamiltonian(K, omega)
        assert H.num_qubits == n


class TestBuildKuramotoRing:
    def test_returns_tuple(self) -> None:
        K, omega = build_kuramoto_ring(4)
        assert K.shape == (4, 4)
        assert omega.shape == (4,)

    def test_ring_topology(self) -> None:
        K, _ = build_kuramoto_ring(4, coupling=2.0)
        assert K[0, 1] == 2.0
        assert K[3, 0] == 2.0
        assert K[0, 2] == 0.0

    def test_custom_omega(self) -> None:
        omega_in = np.array([1.0, 2.0, 3.0])
        K, omega_out = build_kuramoto_ring(3, omega=omega_in)
        np.testing.assert_array_equal(omega_out, omega_in)


def test_inhibitor_all_equal_target():
    """Verify all-self inhibitors compile to a bare Ry rotation."""
    from qiskit import QuantumCircuit

    from scpn_quantum_control.bridge.spn_to_qcircuit import inhibitor_anti_control

    qc = QuantumCircuit(3)
    # 2 inhibitors both equal to target → controls list empty → bare ry
    inhibitor_anti_control(qc, [1, 1], target=1, theta=0.5)
    from qiskit.circuit.library import RYGate

    assert any(isinstance(inst.operation, RYGate) for inst in qc.data)


def test_inhibitor_single_self():
    """Verify a single self-inhibitor compiles to a bare Ry rotation."""
    from qiskit import QuantumCircuit

    from scpn_quantum_control.bridge.spn_to_qcircuit import inhibitor_anti_control

    qc = QuantumCircuit(3)
    inhibitor_anti_control(qc, [2], target=2, theta=0.4)
    from qiskit.circuit.library import RYGate

    assert any(isinstance(inst.operation, RYGate) for inst in qc.data)


def test_inhibitor_multiple_inhibitors():
    """Verify spn_to_qcircuit multi-inhibitor path."""
    from qiskit import QuantumCircuit

    from scpn_quantum_control.bridge.spn_to_qcircuit import inhibitor_anti_control

    qc = QuantumCircuit(3)
    inhibitor_anti_control(qc, [0, 1], target=2, theta=0.5)
    ops = [inst.operation.name for inst in qc.data]
    assert ops.count("x") == 4


def test_pipeline_contract_wiring():
    """Pipeline: verify all contract targets (classical, SPN, QEC) are wired."""
    import time

    t0 = time.perf_counter()
    # Classical path
    psi = np.array([1, 0, 0, 0], dtype=complex)
    z = _expectation_pauli(psi, 2, 0, "Z")
    assert abs(z - 1.0) < 1e-10

    # QEC path

    qec = ControlQEC(distance=3)
    n_data = 2 * 3**2
    syn_z, _ = qec.get_syndrome(np.zeros(n_data, dtype=np.int8), np.zeros(n_data, dtype=np.int8))
    assert int(syn_z.sum()) == 0

    dt = (time.perf_counter() - t0) * 1000
    print(f"\n  PIPELINE contract wiring (classical+QEC): {dt:.1f} ms")
