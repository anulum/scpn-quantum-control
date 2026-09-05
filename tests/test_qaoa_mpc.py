# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Qaoa Mpc
"""Tests for control/qaoa_mpc.py."""

import numpy as np
import pytest

from scpn_quantum_control.control.qaoa_mpc import QAOA_MPC
from scpn_quantum_control.hardware.classical import classical_brute_mpc


def test_build_cost_hamiltonian() -> None:
    """Build one cost-Hamiltonian qubit per control timestep."""
    B = np.array([[1.0, 0.0], [0.0, 1.0]])
    target = np.array([0.5, 0.5])
    mpc = QAOA_MPC(B, target, horizon=3, p_layers=1)
    H = mpc.build_cost_hamiltonian()
    assert H.num_qubits == 3


def test_optimize_returns_binary() -> None:
    """Return a horizon-shaped binary action sequence."""
    B = np.array([[1.0]])
    target = np.array([1.0])
    mpc = QAOA_MPC(B, target, horizon=4, p_layers=1)
    actions = mpc.optimize()
    assert actions.shape == (4,)
    assert set(np.unique(actions)).issubset({0, 1})


def test_cost_hamiltonian_hermitian() -> None:
    """Construct a Hermitian diagonal cost Hamiltonian."""
    B = np.eye(2)
    target = np.array([1.0, 0.0])
    mpc = QAOA_MPC(B, target, horizon=3, p_layers=1)
    H = mpc.build_cost_hamiltonian()
    mat = H.to_matrix()
    np.testing.assert_allclose(mat, mat.conj().T, atol=1e-12)


def test_hamiltonian_matches_classical_cost() -> None:
    """QAOA Hamiltonian diagonal must match classical_brute_mpc cost on each bitstring."""
    B = np.eye(2)
    target = np.array([0.8, 0.6])
    horizon = 4

    from scpn_quantum_control.hardware.classical import classical_brute_mpc

    mpc = QAOA_MPC(B, target, horizon=horizon, p_layers=1)
    H = mpc.build_cost_hamiltonian()
    H_mat = np.real(np.diag(np.array(H.to_matrix())))

    classical_costs = classical_brute_mpc(B, target, horizon)["all_costs"]

    for idx in range(2**horizon):
        assert abs(H_mat[idx] - classical_costs[idx]) < 1e-10, (
            f"bitstring {idx:04b}: H={H_mat[idx]:.6f}, classical={classical_costs[idx]:.6f}"
        )


def test_optimal_bitstring_matches_brute_force() -> None:
    """QAOA Hamiltonian minimum eigenvalue must correspond to brute-force optimal."""
    B = np.eye(2)
    target = np.array([0.8, 0.6])
    horizon = 3

    mpc = QAOA_MPC(B, target, horizon=horizon, p_layers=1)
    H = mpc.build_cost_hamiltonian()
    H_diag = np.real(np.diag(np.array(H.to_matrix())))

    brute = classical_brute_mpc(B, target, horizon=horizon)
    qaoa_min_idx = int(np.argmin(H_diag))
    qaoa_min_actions = np.array([(qaoa_min_idx >> bit) & 1 for bit in range(horizon)])

    np.testing.assert_array_equal(qaoa_min_actions, brute["optimal_actions"])


def test_optimize_before_build_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """RuntimeError fires when auto-build is sabotaged."""
    B = np.array([[1.0]])
    target = np.array([1.0])
    mpc = QAOA_MPC(B, target, horizon=2, p_layers=1)
    monkeypatch.setattr(mpc, "build_cost_hamiltonian", lambda: None)
    with pytest.raises(RuntimeError, match="cost Hamiltonian construction failed"):
        mpc.optimize()


def test_optimize_seeded_deterministic() -> None:
    """Seeded optimize produces identical action sequences."""
    B = np.array([[1.0]])
    target = np.array([1.0])
    mpc = QAOA_MPC(B, target, horizon=4, p_layers=1)
    r1 = mpc.optimize(seed=42)
    mpc._cost_ham = None  # reset to re-run from scratch
    r2 = mpc.optimize(seed=42)
    np.testing.assert_array_equal(r1, r2)


def test_qaoa_mpc_basic() -> None:
    """Optimize a two-step controller with a deterministic seed."""
    B = np.eye(2, dtype=np.float64)
    target = np.array([1.0, 0.0])
    mpc = QAOA_MPC(B, target, horizon=2, p_layers=1)
    result = mpc.optimize(seed=42)
    assert len(result) == 2


def test_qaoa_mpc_3d() -> None:
    """Optimize a three-step controller for a three-dimensional target."""
    B = np.eye(3, dtype=np.float64)
    target = np.ones(3)
    mpc = QAOA_MPC(B, target, horizon=3, p_layers=1)
    result = mpc.optimize(seed=0)
    assert len(result) == 3


def test_qaoa_mpc_result_binary() -> None:
    """Represent every optimized action as an integer binary value."""
    B = np.eye(2, dtype=np.float64)
    target = np.array([1.0, 0.0])
    mpc = QAOA_MPC(B, target, horizon=4, p_layers=1)
    result = mpc.optimize(seed=42)
    for a in result:
        assert a in (0, 1) or isinstance(a, (int, np.integer))


def test_qaoa_mpc_horizon_1() -> None:
    """Support the minimum positive control horizon."""
    B = np.array([[1.0]])
    target = np.array([1.0])
    mpc = QAOA_MPC(B, target, horizon=1, p_layers=1)
    result = mpc.optimize(seed=42)
    assert len(result) == 1


@pytest.mark.parametrize("horizon", [0, -3])
def test_rejects_non_positive_horizon(horizon: int) -> None:
    """Reject zero and negative control horizons."""
    with pytest.raises(ValueError, match="horizon must be positive"):
        QAOA_MPC(np.eye(2), np.ones(2), horizon=horizon)


def test_rejects_non_positive_layer_count() -> None:
    """Reject a controller without a QAOA layer."""
    with pytest.raises(ValueError, match="p_layers must be positive"):
        QAOA_MPC(np.eye(2), np.ones(2), horizon=2, p_layers=0)


def test_optimize_reuses_prebuilt_cost_hamiltonian() -> None:
    """Optimize through the public API with an already built cost operator."""
    mpc = QAOA_MPC(np.array([[1.0]]), np.array([1.0]), horizon=1, p_layers=1)
    expected = mpc.build_cost_hamiltonian()
    result = mpc.optimize(seed=42)
    assert mpc._cost_ham is expected
    assert result.shape == (1,)


def _qaoa_reference_costs(b_matrix: np.ndarray, target: np.ndarray, horizon: int) -> np.ndarray:
    """Enumerate the documented MPC cost straight from its definition."""
    actuation = np.asarray(b_matrix, dtype=float).sum(axis=1)
    goal = np.asarray(target, dtype=float)
    costs = []
    for index in range(2**horizon):
        total = 0.0
        for step in range(horizon):
            residual = ((index >> step) & 1) * actuation - goal
            total += float(np.dot(residual, residual))
        costs.append(total)
    return np.array(costs)


@pytest.mark.parametrize(
    ("b_matrix", "target", "horizon"),
    [
        (np.array([[1.0]]), np.array([-1.0]), 1),
        (np.eye(2), np.array([0.8, 0.6]), 3),
        (np.array([[0.6, -0.8], [0.8, 0.6]]), np.array([-0.3, 1.1]), 2),
    ],
)
def test_cost_hamiltonian_diagonal_equals_the_documented_cost(
    b_matrix: np.ndarray, target: np.ndarray, horizon: int
) -> None:
    """The Ising mapping must reproduce the cost it claims to encode."""
    controller = QAOA_MPC(b_matrix, target, horizon=horizon, p_layers=1)
    diagonal = np.real(np.diag(controller.build_cost_hamiltonian().to_matrix()))
    np.testing.assert_allclose(
        diagonal, _qaoa_reference_costs(b_matrix, target, horizon), atol=1e-9
    )


def test_cost_hamiltonian_separates_a_sign_flipped_target() -> None:
    """A norm-only mapping would give these two targets the same Hamiltonian."""
    positive = QAOA_MPC(np.array([[1.0]]), np.array([1.0]), horizon=1, p_layers=1)
    negative = QAOA_MPC(np.array([[1.0]]), np.array([-1.0]), horizon=1, p_layers=1)
    positive_diagonal = np.real(np.diag(positive.build_cost_hamiltonian().to_matrix()))
    negative_diagonal = np.real(np.diag(negative.build_cost_hamiltonian().to_matrix()))
    np.testing.assert_allclose(positive_diagonal, [1.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(negative_diagonal, [1.0, 4.0], atol=1e-9)


def test_cost_hamiltonian_agrees_with_the_brute_force_kernel() -> None:
    """The quantum mapping and the classical enumerator share one cost."""
    from scpn_quantum_control.hardware.classical import classical_brute_mpc

    b_matrix = np.array([[0.6, -0.8], [0.8, 0.6]])
    target = np.array([-0.3, 1.1])
    controller = QAOA_MPC(b_matrix, target, horizon=3, p_layers=1)
    diagonal = np.real(np.diag(controller.build_cost_hamiltonian().to_matrix()))
    brute = classical_brute_mpc(b_matrix, target, 3)
    np.testing.assert_allclose(diagonal, brute["all_costs"], atol=1e-9)
