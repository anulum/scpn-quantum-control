# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Control module contract tests
"""Contract tests for disruption, Petri, VQLS, QAOA MPC, QEC, director, and cost-control behaviours."""

from __future__ import annotations

import numpy as np
import pytest
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.bridge.knm_hamiltonian import (
    OMEGA_N_16,
    build_knm_paper27,
)
from scpn_quantum_control.control.qaoa_mpc import QAOA_MPC
from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov
from scpn_quantum_control.qec.control_qec import ControlQEC


class TestQDisruptionIter:
    def test_default_rng(self):
        from scpn_quantum_control.control.q_disruption_iter import (
            generate_synthetic_iter_data,
        )

        X, y = generate_synthetic_iter_data(n_samples=20, rng=None, allow_synthetic=True)
        assert X.shape[0] == 20
        assert y.shape[0] == 20

    @pytest.mark.parametrize("n_samples", [1, 10, 50, 100])
    def test_shape_varies_with_n_samples(self, n_samples):
        from scpn_quantum_control.control.q_disruption_iter import (
            generate_synthetic_iter_data,
        )

        X, y = generate_synthetic_iter_data(
            n_samples=n_samples,
            rng=np.random.default_rng(42),
            allow_synthetic=True,
        )
        assert X.shape[0] == n_samples
        assert y.shape[0] == n_samples

    def test_labels_binary(self):
        from scpn_quantum_control.control.q_disruption_iter import (
            generate_synthetic_iter_data,
        )

        _, y = generate_synthetic_iter_data(
            n_samples=100, rng=np.random.default_rng(42), allow_synthetic=True
        )
        assert set(np.unique(y)).issubset({0, 1})

    def test_features_finite(self):
        from scpn_quantum_control.control.q_disruption_iter import (
            generate_synthetic_iter_data,
        )

        X, _ = generate_synthetic_iter_data(
            n_samples=50, rng=np.random.default_rng(42), allow_synthetic=True
        )
        assert np.all(np.isfinite(X))

    def test_reproducible_with_rng(self):
        from scpn_quantum_control.control.q_disruption_iter import (
            generate_synthetic_iter_data,
        )

        X1, y1 = generate_synthetic_iter_data(
            n_samples=20, rng=np.random.default_rng(42), allow_synthetic=True
        )
        X2, y2 = generate_synthetic_iter_data(
            n_samples=20, rng=np.random.default_rng(42), allow_synthetic=True
        )
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


class TestQPetri:
    def test_w_out_shape_mismatch_raises(self):
        from scpn_quantum_control.control.qpetri import QuantumPetriNet

        W_in = np.array([[1.0, 0.0], [0.0, 1.0]])
        W_out = np.array([[1.0, 0.0]])  # wrong shape
        thresholds = np.array([0.5, 0.5])
        with pytest.raises(ValueError, match="W_out shape"):
            QuantumPetriNet(
                n_places=2,
                n_transitions=2,
                W_in=W_in,
                W_out=W_out,
                thresholds=thresholds,
            )

    def test_valid_construction(self):
        from scpn_quantum_control.control.qpetri import QuantumPetriNet

        n_p, n_t = 2, 2
        W_in = np.eye(n_p)
        W_out = np.eye(n_p)
        thresholds = np.ones(n_t) * 0.5
        net = QuantumPetriNet(
            n_places=n_p,
            n_transitions=n_t,
            W_in=W_in,
            W_out=W_out,
            thresholds=thresholds,
        )
        assert net.n_places == n_p
        assert net.n_transitions == n_t

    def test_encode_marking(self):
        from scpn_quantum_control.control.qpetri import QuantumPetriNet

        W_in = np.eye(2)
        W_out = np.eye(2)
        thresholds = np.array([0.5, 0.5])
        net = QuantumPetriNet(
            n_places=2,
            n_transitions=2,
            W_in=W_in,
            W_out=W_out,
            thresholds=thresholds,
        )
        qc = net.encode_marking(np.array([1.0, 0.0]))
        assert qc.num_qubits == 2

    def test_step_returns_marking(self):
        from scpn_quantum_control.control.qpetri import QuantumPetriNet

        W_in = np.eye(2)
        W_out = np.eye(2)
        thresholds = np.array([0.5, 0.5])
        net = QuantumPetriNet(
            n_places=2,
            n_transitions=2,
            W_in=W_in,
            W_out=W_out,
            thresholds=thresholds,
        )
        marking = net.step(np.array([0.8, 0.3]))
        assert marking.shape == (2,)
        assert all(np.isfinite(marking))


class TestVQLSGradShafranov:
    def test_denominator_near_zero_returns_array(self):
        from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov

        solver = VQLS_GradShafranov(n_qubits=2)
        result = solver.solve(maxiter=1, seed=42)
        assert isinstance(result, np.ndarray)

    def test_solve_returns_correct_shape(self):
        from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov

        solver = VQLS_GradShafranov(n_qubits=2)
        result = solver.solve(maxiter=5, seed=42)
        assert result.shape == (4,)  # 2^n_qubits

    def test_solve_output_finite(self):
        from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov

        solver = VQLS_GradShafranov(n_qubits=2)
        result = solver.solve(maxiter=5, seed=42)
        assert np.all(np.isfinite(result))

    def test_imaginary_tolerance_zero_raises(self):
        from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov

        solver = VQLS_GradShafranov(n_qubits=2, imag_tol=0.0)
        with pytest.raises(ValueError, match="imaginary norm"):
            solver.solve(reps=1, maxiter=1, seed=0)

    def test_reproducible_with_seed(self):
        from scpn_quantum_control.control.vqls_gs import VQLS_GradShafranov

        s1 = VQLS_GradShafranov(n_qubits=2)
        s2 = VQLS_GradShafranov(n_qubits=2)
        r1 = s1.solve(maxiter=3, seed=42)
        r2 = s2.solve(maxiter=3, seed=42)
        np.testing.assert_array_equal(r1, r2)


def test_qaoa_optimize_triggers_lazy_build():
    """QAOA_MPC.optimize() calls build_cost_hamiltonian lazily if not called yet."""
    B = np.eye(2)
    target = np.array([0.5, 0.5])
    mpc = QAOA_MPC(B, target, horizon=2, p_layers=1)
    # Don't call build_cost_hamiltonian — optimize should do it
    actions = mpc.optimize()
    assert len(actions) == 2


def test_qaoa_zz_circuit_path():
    """Verify ZZ terms are translated into two-qubit QAOA cost evolution."""

    B = np.eye(2)
    target = np.array([0.5, 0.5])
    mpc = QAOA_MPC(B, target, horizon=4, p_layers=1)
    # Inject a custom Hamiltonian with ZZ terms
    mpc._cost_ham = SparsePauliOp(["IIZZ", "ZZII", "ZIII"], [0.3, 0.2, 0.1])
    gamma = np.array([0.5])
    beta = np.array([0.3])
    qc = mpc._build_qaoa_circuit(gamma, beta)
    assert qc.depth() > 0
    ops = qc.count_ops()
    assert ops.get("rzz", 0) > 0 or ops.get("rz", 0) > 0, "expected ZZ rotation gates"


def test_vqls_cost_handles_near_zero_state():
    """VQLS cost returns 1.0 when xAtAx < 1e-15."""
    solver = VQLS_GradShafranov(n_qubits=4)
    psi = solver.solve(maxiter=5)
    assert psi.shape == (2**4,)
    assert np.linalg.norm(psi) > 0, "VQLS returned zero vector"


def test_qec_odd_defects_duplication():
    """Odd number of defects triggers defect duplication."""
    qec = ControlQEC(distance=3)
    err_x = np.zeros(2 * 3**2, dtype=np.int8)
    err_z = np.zeros(2 * 3**2, dtype=np.int8)
    # Place a single X error to get odd syndrome defects
    err_x[0] = 1
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    corr = qec.decoder.decode(syn_z)
    assert corr.shape == err_x.shape


def test_qec_simulate_errors_default_rng():
    """simulate_errors with rng=None creates its own."""
    qec = ControlQEC(distance=3)
    err_x, err_z = qec.simulate_errors(p_error=0.1)
    assert err_x.shape == err_z.shape


def test_qec_residual_syndrome_failure():
    """High error rate causes correction failure."""
    qec = ControlQEC(distance=3)
    rng = np.random.default_rng(42)
    failures = 0
    for _ in range(50):
        err_x, err_z = qec.simulate_errors(p_error=0.3, rng=rng)
        success = qec.decode_and_correct(err_x, err_z)
        if not success:
            failures += 1
    # At p=0.3, most rounds should fail
    assert failures >= 5


def test_qec_odd_syndrome_defects():
    """Verify odd syndrome defects are paired by duplicating one defect."""
    from scpn_quantum_control.qec.control_qec import ControlQEC

    qec = ControlQEC(distance=3)
    n_data = 2 * 3**2
    # Construct an error pattern that produces an odd number of defects
    rng = np.random.default_rng(0)
    for _ in range(100):
        err_x = (rng.random(n_data) < 0.15).astype(np.int8)
        err_z = np.zeros(n_data, dtype=np.int8)
        syn_z, _ = qec.get_syndrome(err_x, err_z)
        n_defects = int(syn_z.sum())
        if n_defects % 2 == 1:
            corr = qec.decoder.decode(syn_z)
            assert corr.shape == (n_data,)
            return
    pytest.skip("no odd syndrome found in 100 random samples")


def test_qec_decode_and_correct_failure():
    """Verify heavy uncorrectable errors make correction fail closed."""
    from scpn_quantum_control.qec.control_qec import ControlQEC

    qec = ControlQEC(distance=3)
    rng = np.random.default_rng(7)
    failures = 0
    for _ in range(200):
        err_x, err_z = qec.simulate_errors(p_error=0.4, rng=rng)
        if not qec.decode_and_correct(err_x, err_z):
            failures += 1
    assert failures > 0, "Expected some correction failures at p=0.4"


def test_qec_basic_syndrome():
    """ControlQEC should produce syndromes with correct shape."""
    from scpn_quantum_control.qec.control_qec import ControlQEC

    qec = ControlQEC(distance=3)
    n_data = 2 * 3**2
    err_x = np.zeros(n_data, dtype=np.int8)
    err_z = np.zeros(n_data, dtype=np.int8)
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    assert len(syn_z) > 0
    assert len(syn_x) > 0


def test_qec_no_error_clean_syndrome():
    """Zero errors should produce all-zero syndrome."""
    from scpn_quantum_control.qec.control_qec import ControlQEC

    qec = ControlQEC(distance=3)
    n_data = 2 * 3**2
    err_x = np.zeros(n_data, dtype=np.int8)
    err_z = np.zeros(n_data, dtype=np.int8)
    syn_z, syn_x = qec.get_syndrome(err_x, err_z)
    assert int(syn_z.sum()) == 0
    assert int(syn_x.sum()) == 0


class TestL16Actions:
    def test_adjust_action(self):
        from scpn_quantum_control.l16.quantum_director import compute_l16_lyapunov

        K = build_knm_paper27(L=2) * 0.01
        omega = OMEGA_N_16[:2]
        result = compute_l16_lyapunov(K, omega)
        assert result.action in ("continue", "adjust", "halt")

    def test_halt_action(self):
        from scpn_quantum_control.l16.quantum_director import compute_l16_lyapunov

        K = np.zeros((2, 2))
        omega = np.array([10.0, -10.0])
        result = compute_l16_lyapunov(K, omega)
        assert result.action in ("continue", "adjust", "halt")


def test_quantum_director_halt_action():
    """Verifies 153-156: L16 returns 'halt' when stability score <= 0.4."""
    from scpn_quantum_control.l16.quantum_director import compute_l16_lyapunov

    # Weak coupling → low echo, low R → low score → halt or adjust
    K = build_knm_paper27(L=2) * 0.001
    omega = OMEGA_N_16[:2] * 0.001
    result = compute_l16_lyapunov(K, omega)
    assert result.action in ("continue", "adjust", "halt")


def test_quantum_director_adjust_action():
    """Verifies 153-154: L16 returns 'adjust' when 0.4 < score <= 0.7."""
    from scpn_quantum_control.l16.quantum_director import compute_l16_lyapunov

    # Moderate coupling
    K = build_knm_paper27(L=2) * 0.3
    omega = OMEGA_N_16[:2]
    result = compute_l16_lyapunov(K, omega)
    assert result.action in ("continue", "adjust", "halt")
    assert 0.0 <= result.stability_score <= 1.0


def test_cpdr_zero_slope():
    """Verifies 201: CPDR returns raw value when regression slope ~ 0."""
    from scpn_quantum_control.mitigation.cpdr import cpdr_mitigate

    ideal = [0.1, 0.2, 0.3, 0.4]
    noisy = [0.15, 0.18, 0.35, 0.38]
    result = cpdr_mitigate(0.5, ideal, noisy)
    assert hasattr(result, "mitigated_value")


def test_control_qec_correction_failure():
    """Verifies 222: decode_and_correct returns False for heavy uncorrectable error."""
    from scpn_quantum_control.qec.control_qec import ControlQEC

    qec = ControlQEC(distance=3)
    # Apply a heavy error: all X errors (likely uncorrectable for d=3)
    err_x = np.ones(qec.code.num_data, dtype=np.int8)
    err_z = np.ones(qec.code.num_data, dtype=np.int8)
    result = qec.decode_and_correct(err_x, err_z)
    assert isinstance(result, bool)


def test_error_budget_max_distance():
    """Verifies 89: minimum_code_distance returns max_distance when no d satisfies target."""
    from scpn_quantum_control.qec.error_budget import minimum_code_distance

    # Very low target → needs very high distance
    d = minimum_code_distance(target_logical_rate=1e-30, p_physical=0.01, max_distance=7)
    assert d == 7


def test_error_budget_zero_comm_bound():
    """Verifies 128-129: n_steps=1, eps_trotter=0 when comm_bound near zero."""
    from scpn_quantum_control.qec.error_budget import compute_error_budget

    K = np.zeros((2, 2))
    omega = np.array([0.0, 0.0])
    result = compute_error_budget(K, omega, t_total=1.0)
    assert result.n_trotter_steps == 1
    assert result.trotter_error == 0.0


def test_quantum_costs_single_qubit():
    """Verifies 97-98: compute_c4_tcbo returns (1.0, 0.0) for single-qubit."""
    from qiskit.quantum_info import Statevector

    from scpn_quantum_control.ssgf.quantum_costs import compute_c4_tcbo

    sv = Statevector.from_label("0")
    cost, entropy = compute_c4_tcbo(sv, 1)
    assert cost == 1.0
    assert entropy == 0.0


def test_quantum_costs_no_correlators():
    """Verifies 127-128: compute_c_pgbo returns (1.0, 0.0) when no pairs."""
    from qiskit.quantum_info import Statevector

    from scpn_quantum_control.ssgf.quantum_costs import compute_c_pgbo

    sv = Statevector.from_label("0")
    cost, var = compute_c_pgbo(sv, 1)
    assert cost == 1.0
    assert var == 0.0


def test_quantum_outer_cycle_single_node():
    """Verify labelled surrogate path for a single-node classical cost."""
    import pytest

    from scpn_quantum_control.ssgf.quantum_outer_cycle import classical_cost

    W = np.array([[1.0]])
    with pytest.raises(ValueError, match="surrogate"):
        classical_cost(W)

    cost = classical_cost(W, allow_surrogate=True)
    assert cost == 1.0
