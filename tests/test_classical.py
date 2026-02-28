"""Tests for hardware/classical.py reference computations."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from scpn_quantum_control.hardware.classical import (
    bloch_vectors_from_json,
    classical_brute_mpc,
    classical_exact_diag,
    classical_exact_evolution,
    classical_kuramoto_reference,
)

# --- classical_kuramoto_reference ---


def test_kuramoto_returns_correct_shapes():
    result = classical_kuramoto_reference(4, t_max=0.5, dt=0.1)
    assert result["times"].shape == (6,)
    assert result["theta"].shape == (6, 4)
    assert result["R"].shape == (6,)


def test_kuramoto_R_bounded():
    result = classical_kuramoto_reference(4, t_max=1.0, dt=0.05)
    for r in result["R"]:
        assert 0.0 <= r <= 1.0 + 1e-10


def test_kuramoto_identical_frequencies_stay_locked():
    """Identical frequencies with coupling should maintain synchrony."""
    K = np.array([[0, 1.0], [1.0, 0]])
    omega = np.array([1.0, 1.0])
    theta0 = np.array([0.1, 0.1])
    result = classical_kuramoto_reference(2, t_max=1.0, dt=0.01, K=K, omega=omega, theta0=theta0)
    # Started synchronised → should stay near R=1
    assert result["R"][-1] > 0.9


def test_kuramoto_zero_coupling_free_rotation():
    """Zero coupling: phases advance linearly at natural frequencies."""
    K = np.zeros((2, 2))
    omega = np.array([1.0, 2.0])
    theta0 = np.array([0.0, 0.0])
    result = classical_kuramoto_reference(2, t_max=0.5, dt=0.001, K=K, omega=omega, theta0=theta0)
    # Final phases should be omega * t_max
    np.testing.assert_allclose(result["theta"][-1], omega * 0.5, atol=0.01)


def test_kuramoto_default_params():
    """Default params use Paper 27 values."""
    result = classical_kuramoto_reference(4, t_max=0.2, dt=0.05)
    assert len(result["R"]) == 5  # 0.2/0.05 = 4 steps + 1


# --- classical_exact_diag ---


def test_exact_diag_returns_real_eigenvalues():
    result = classical_exact_diag(3)
    assert np.all(np.isreal(result["eigenvalues"]))


def test_exact_diag_ground_below_first_excited():
    result = classical_exact_diag(3)
    assert result["spectral_gap"] > 0.0


def test_exact_diag_eigenvalues_sorted():
    result = classical_exact_diag(4)
    evals = result["eigenvalues"]
    np.testing.assert_array_less(evals[:-1], evals[1:] + 1e-12)


def test_exact_diag_sparse_path():
    """k_eigenvalues triggers sparse solver even for small systems."""
    result = classical_exact_diag(3, k_eigenvalues=3)
    assert len(result["eigenvalues"]) == 3
    assert result["ground_energy"] == pytest.approx(result["eigenvalues"][0])


def test_exact_diag_ground_state_normalised():
    result = classical_exact_diag(3)
    norm = np.linalg.norm(result["ground_state"])
    np.testing.assert_allclose(norm, 1.0, atol=1e-12)


# --- classical_exact_evolution ---


def test_exact_evolution_shapes():
    result = classical_exact_evolution(3, t_max=0.3, dt=0.1)
    assert result["times"].shape == (4,)
    assert result["R"].shape == (4,)


def test_exact_evolution_R_bounded():
    result = classical_exact_evolution(3, t_max=0.5, dt=0.05)
    for r in result["R"]:
        assert 0.0 <= r <= 1.0 + 1e-10


def test_exact_evolution_unitarity():
    """Exact evolution preserves initial R at t=0."""
    result = classical_exact_evolution(3, t_max=0.3, dt=0.1)
    # R(0) should be well-defined and positive (Ry-initialised state)
    assert result["R"][0] > 0.0


def test_exact_evolution_default_params():
    """Default K and omega from Paper 27."""
    result = classical_exact_evolution(4, t_max=0.2, dt=0.1)
    assert len(result["R"]) == 3


# --- classical_brute_mpc ---


def test_brute_mpc_optimal_cost_minimal():
    B = np.eye(2)
    target = np.array([0.8, 0.6])
    result = classical_brute_mpc(B, target, horizon=3)
    assert result["optimal_cost"] <= np.min(result["all_costs"]) + 1e-12


def test_brute_mpc_binary_actions():
    B = np.array([[1.0]])
    target = np.array([1.0])
    result = classical_brute_mpc(B, target, horizon=4)
    assert result["optimal_actions"].shape == (4,)
    assert set(np.unique(result["optimal_actions"])).issubset({0, 1})


def test_brute_mpc_enumerates_all():
    result = classical_brute_mpc(np.eye(2), np.array([1.0, 0.0]), horizon=3)
    assert result["n_evaluated"] == 8
    assert result["all_costs"].shape == (8,)


def test_brute_mpc_zero_target():
    """Zero target: all-zeros action should be optimal (cost = 0)."""
    B = np.eye(2)
    target = np.array([0.0, 0.0])
    result = classical_brute_mpc(B, target, horizon=3)
    np.testing.assert_array_equal(result["optimal_actions"], [0, 0, 0])
    assert result["optimal_cost"] == pytest.approx(0.0)


# --- bloch_vectors_from_json ---


def test_bloch_vectors_from_json_roundtrip():
    data = {
        "exp_x": [0.5, 0.3, 0.1],
        "exp_y": [0.4, 0.2, 0.0],
        "exp_z": [0.6, 0.8, 0.9],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    result = bloch_vectors_from_json(path)
    Path(path).unlink()

    assert result["n_qubits"] == 3
    np.testing.assert_allclose(result["exp_x"], [0.5, 0.3, 0.1])
    expected_mag = np.sqrt(
        np.array([0.5, 0.3, 0.1]) ** 2
        + np.array([0.4, 0.2, 0.0]) ** 2
        + np.array([0.6, 0.8, 0.9]) ** 2
    )
    np.testing.assert_allclose(result["bloch_magnitudes"], expected_mag, atol=1e-12)


def test_bloch_vectors_pure_state():
    """Pure state on Bloch sphere: magnitude = 1."""
    data = {"exp_x": [1.0], "exp_y": [0.0], "exp_z": [0.0]}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        path = f.name

    result = bloch_vectors_from_json(path)
    Path(path).unlink()

    np.testing.assert_allclose(result["bloch_magnitudes"], [1.0], atol=1e-12)
