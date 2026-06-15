# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the quantum/classical co-simulation package
"""Tests for cosimulation/knm_partition.py and cosimulation/quantum_classical.py."""

import numpy as np
import pytest
import scipy.linalg as sla
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.cosimulation import (
    KnmPartition,
    cosimulate,
    partition_knm,
)
from scpn_quantum_control.cosimulation import quantum_classical as qc

try:
    import scpn_quantum_engine as _engine

    _HAS_RUST = hasattr(_engine, "cosim_classical_substep")
except ImportError:  # pragma: no cover - engine optional
    _engine = None
    _HAS_RUST = False


def _two_scale_network(n_core: int = 6, n_total: int = 60, seed: int = 0):
    """A strong core embedded in a weakly coupled ring."""
    rng = np.random.default_rng(seed)
    K = np.zeros((n_total, n_total))
    for i in range(n_core):
        for j in range(i + 1, n_core):
            K[i, j] = K[j, i] = 0.9
    for i in range(n_total):
        j = (i + 1) % n_total
        K[i, j] = K[j, i] = max(K[i, j], 0.05)
    omega = rng.standard_normal(n_total) * 0.3
    return K, omega


# --------------------------------------------------------------------------- #
# Partitioning
# --------------------------------------------------------------------------- #
def test_partition_selects_strong_core():
    K, omega = _two_scale_network(n_core=6, n_total=40)
    part = partition_knm(K, omega, max_quantum_nodes=6)
    assert part.quantum_indices == (0, 1, 2, 3, 4, 5)
    assert part.n_classical == 34
    assert part.quantum_coupling.shape == (6, 6)
    assert part.cross_coupling.shape == (6, 34)


def test_partition_conservation_is_edge_exact():
    K, omega = _two_scale_network(n_core=5, n_total=30)
    part = partition_knm(K, omega, max_quantum_nodes=5)
    c = part.conservation
    assert c.is_exact
    assert c.residual < 1e-9
    # The three buckets reconstruct the full upper-triangular coupling budget.
    triu_total = float(np.triu(np.abs(0.5 * (K + K.T)), k=1).sum())
    assert c.total_abs_coupling == pytest.approx(triu_total)
    assert c.quantum_internal_abs + c.classical_internal_abs + c.cross_abs == pytest.approx(
        triu_total
    )
    assert 0.0 <= c.cross_fraction <= 1.0


def test_partition_is_deterministic():
    K, omega = _two_scale_network()
    a = partition_knm(K, omega, max_quantum_nodes=6)
    b = partition_knm(K, omega, max_quantum_nodes=6)
    assert a.quantum_indices == b.quantum_indices
    assert a.provenance["growth_order"] == b.provenance["growth_order"]


def test_partition_records_asymmetry_and_symmetrises():
    K, omega = _two_scale_network(n_core=4, n_total=12)
    K[0, 1] += 0.4  # break symmetry
    part = partition_knm(K, omega, max_quantum_nodes=4)
    assert part.provenance["symmetrised"] is True
    assert part.provenance["input_asymmetry"] > 0.0
    assert np.allclose(part.quantum_coupling, part.quantum_coupling.T)


def test_partition_threshold_limits_core_growth():
    K, omega = _two_scale_network(n_core=6, n_total=30)
    # Threshold above the weak-ring coupling stops growth at the strong core.
    part = partition_knm(K, omega, max_quantum_nodes=10, coupling_threshold=0.5)
    assert set(part.quantum_indices) == {0, 1, 2, 3, 4, 5}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"K": np.zeros((3, 4)), "omega": np.zeros(3)},
        {"K": np.zeros((1, 1)), "omega": np.zeros(1)},
        {"K": np.zeros((4, 4)), "omega": np.zeros(3)},
        {"K": np.full((4, 4), np.nan), "omega": np.zeros(4)},
        {"K": np.zeros((4, 4)), "omega": np.zeros(4), "max_quantum_nodes": 0},
        {"K": np.zeros((4, 4)), "omega": np.zeros(4), "max_quantum_nodes": 99},
        {"K": np.zeros((4, 4)), "omega": np.zeros(4), "max_quantum_nodes": 5},
        {"K": np.zeros((4, 4)), "omega": np.zeros(4), "coupling_threshold": -1.0},
    ],
)
def test_partition_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        partition_knm(**kwargs)


# --------------------------------------------------------------------------- #
# Rust / Python classical substep parity
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_RUST, reason="cosim kernel not built")
@settings(max_examples=40, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=12),
    seed=st.integers(min_value=0, max_value=10_000),
)
def test_classical_substep_rust_parity(n, seed):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(-np.pi, np.pi, size=n)
    omega = rng.standard_normal(n)
    K = rng.standard_normal((n, n))
    K = 0.5 * (K + K.T)
    da = rng.standard_normal(n)
    db = rng.standard_normal(n)
    rust = np.asarray(
        _engine.cosim_classical_substep(theta, omega, np.ascontiguousarray(K), da, db, 0.01)
    )
    python = qc._classical_substep_python(theta, omega, K, da, db, 0.01)
    assert np.allclose(rust, python, rtol=0, atol=1e-12)


@pytest.mark.skipif(not _HAS_RUST, reason="cosim kernel not built")
def test_classical_substep_rejects_bad_args():
    with pytest.raises(ValueError):
        _engine.cosim_classical_substep(
            np.array([0.0]),
            np.array([0.0]),
            np.zeros((1, 1)),
            np.array([0.0]),
            np.array([0.0]),
            0.0,
        )
    with pytest.raises(ValueError):
        _engine.cosim_classical_substep(
            np.array([0.0, 1.0]),
            np.array([0.0]),
            np.zeros((2, 2)),
            np.array([0.0]),
            np.array([0.0]),
            0.01,
        )


# --------------------------------------------------------------------------- #
# Co-simulation physics
# --------------------------------------------------------------------------- #
def test_internal_propagator_is_unitary():
    K, omega = _two_scale_network(n_core=6, n_total=20)
    part = partition_knm(K, omega, max_quantum_nodes=6)
    half = qc._internal_half_propagator(part.quantum_coupling, part.quantum_omega, 0.05)
    assert np.allclose(half @ half.conj().T, np.eye(half.shape[0]), atol=1e-10)


def test_xy_hamiltonian_python_matches_rust():
    if not (_HAS_RUST and hasattr(_engine, "build_xy_hamiltonian_dense")):
        pytest.skip("XY Hamiltonian kernel not built")
    K, omega = _two_scale_network(n_core=4, n_total=8)
    part = partition_knm(K, omega, max_quantum_nodes=4)
    py = qc._xy_hamiltonian_dense_python(part.quantum_coupling, part.quantum_omega)
    flat = np.asarray(
        _engine.build_xy_hamiltonian_dense(
            np.ascontiguousarray(part.quantum_coupling).ravel(),
            np.ascontiguousarray(part.quantum_omega),
            4,
        )
    ).reshape(16, 16)
    assert np.allclose(py, flat, atol=1e-12)


def test_order_parameters_bounded():
    K, omega = _two_scale_network(n_core=6, n_total=50)
    res = cosimulate(K, omega, dt=0.02, n_steps=40, max_quantum_nodes=6, seed=3)
    for arr in (res.quantum_order, res.classical_order, res.global_order):
        assert np.all(arr >= -1e-9)
        assert np.all(arr <= 1.0 + 1e-9)
    assert res.classical_phases.shape == (41, res.partition.n_classical)
    assert res.quantum_expectation_x.shape == (41, res.partition.n_quantum)
    assert res.times.shape == (41,)


def test_decoupled_core_matches_exact_quantum():
    # Zero cross coupling -> the quantum core must evolve as an isolated system.
    n = 7
    K = np.zeros((n, n))
    for i in range(4):
        for j in range(i + 1, 4):
            K[i, j] = K[j, i] = 0.7
    for i in range(4, 7):
        for j in range(i + 1, 7):
            K[i, j] = K[j, i] = 0.3
    omega = np.array([0.5, -0.3, 0.2, 0.1, 0.4, -0.2, 0.3])
    part = partition_knm(K, omega, max_quantum_nodes=4)
    assert np.abs(part.cross_coupling).sum() == 0.0

    dt, n_steps = 0.01, 60
    res = cosimulate(
        K, omega, dt=dt, n_steps=n_steps, partition=part, theta0_classical=np.zeros(3)
    )
    hamiltonian = qc._xy_hamiltonian_dense_python(part.quantum_coupling, part.quantum_omega)
    state = qc._initial_quantum_state(4, None)
    step = sla.expm(-1j * hamiltonian * dt)
    exp_x = np.empty((n_steps + 1, 4))
    exp_y = np.empty((n_steps + 1, 4))
    for k in range(n_steps + 1):
        ex, ey = qc._expectation_xy(state, 4)
        exp_x[k] = ex
        exp_y[k] = ey
        state = step @ state
    assert np.allclose(res.quantum_expectation_x, exp_x, atol=1e-6)
    assert np.allclose(res.quantum_expectation_y, exp_y, atol=1e-6)


def test_decoupled_classical_matches_full_kuramoto():
    # Zero cross coupling -> the classical bath order parameter equals its
    # isolated all-classical baseline.
    n = 8
    K = np.zeros((n, n))
    for i in range(3):
        for j in range(i + 1, 3):
            K[i, j] = K[j, i] = 0.8
    for i in range(3, 8):
        j = (i + 1) if i + 1 < 8 else 3
        K[i, j] = K[j, i] = 0.25
    omega = np.linspace(-0.4, 0.4, n)
    part = partition_knm(K, omega, max_quantum_nodes=3)
    assert np.abs(part.cross_coupling).sum() == 0.0
    theta0 = np.array([0.1, -0.2, 0.3, 0.0, 0.4])
    res = cosimulate(K, omega, dt=0.01, n_steps=50, partition=part, theta0_classical=theta0)
    assert np.allclose(res.classical_order, res.baseline_classical_order, atol=1e-9)


def test_reproducible_with_seed():
    K, omega = _two_scale_network(n_core=5, n_total=24)
    a = cosimulate(K, omega, dt=0.02, n_steps=20, max_quantum_nodes=5, seed=7)
    b = cosimulate(K, omega, dt=0.02, n_steps=20, max_quantum_nodes=5, seed=7)
    assert np.array_equal(a.classical_phases, b.classical_phases)
    assert np.array_equal(a.quantum_expectation_x, b.quantum_expectation_x)


def test_provenance_and_claim_boundary():
    K, omega = _two_scale_network(n_core=4, n_total=16)
    res = cosimulate(K, omega, dt=0.02, n_steps=10, max_quantum_nodes=4, seed=1)
    assert res.provenance["quantum_integrator"] == "second_order_trotter_exact_internal"
    assert res.provenance["classical_integrator"] == "explicit_euler"
    assert "mean-field" in res.provenance["claim_boundary"]
    assert isinstance(res.partition, KnmPartition)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dt": 0.0, "n_steps": 5},
        {"dt": -0.1, "n_steps": 5},
        {"dt": 0.01, "n_steps": 0},
    ],
)
def test_cosimulate_rejects_bad_args(kwargs):
    K, omega = _two_scale_network(n_core=4, n_total=12)
    with pytest.raises(ValueError):
        cosimulate(K, omega, max_quantum_nodes=4, **kwargs)


def test_cosimulate_rejects_bad_initial_state():
    K, omega = _two_scale_network(n_core=4, n_total=12)
    part = partition_knm(K, omega, max_quantum_nodes=4)
    with pytest.raises(ValueError):
        cosimulate(K, omega, dt=0.01, n_steps=5, partition=part, quantum_state0=np.zeros(8))
    with pytest.raises(ValueError):
        cosimulate(K, omega, dt=0.01, n_steps=5, partition=part, quantum_state0=np.zeros(16))


@settings(max_examples=10, deadline=None)
@given(seed=st.integers(min_value=0, max_value=1000))
def test_global_order_property(seed):
    K, omega = _two_scale_network(n_core=4, n_total=20, seed=seed)
    res = cosimulate(K, omega, dt=0.02, n_steps=15, max_quantum_nodes=4, seed=seed)
    assert np.all(np.isfinite(res.global_order))
    assert np.all(res.global_order <= 1.0 + 1e-9)
    assert res.baseline_deviation >= 0.0
