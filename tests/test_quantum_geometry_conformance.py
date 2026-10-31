# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — conformance of the quantum Fisher metric and natural-gradient use
"""The QFI convention, its rank and its regularisation are explicit and independently checked.

Oracles do not use the spectral sum rule: the XY Hamiltonian is rebuilt here
from Kronecker products, the ground-state projector is differentiated by
central differences, and ``F_ab = 2 Tr(d_a P d_b P)`` is gauge free.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.analysis.qfi import QFI_CONVENTION, QFI_PSD_TOLERANCE, compute_qfi
from scpn_quantum_control.analysis.quantum_fisher_information import QuantumFisherInformation
from scpn_quantum_control.differentiable import GradientResult, natural_gradient
from scpn_quantum_control.phase.natural_gradient import (
    parameter_shift_natural_gradient_descent,
    solve_natural_gradient_direction,
    validate_natural_gradient_training,
)

Matrix = NDArray[np.complex128]
Pairs = list[tuple[int, int]]

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.diag([1.0, -1.0]).astype(np.complex128)


def _on(operators: dict[int, Matrix], n: int) -> Matrix:
    """Place single-qubit operators in little-endian order (qubit 0 rightmost)."""
    product: NDArray[np.complex128] = np.ones((1, 1), dtype=np.complex128)
    for qubit in reversed(range(n)):
        product = np.asarray(np.kron(product, operators.get(qubit, _I)), dtype=np.complex128)
    return np.asarray(product, dtype=np.complex128)


def _hamiltonian(couplings: dict[tuple[int, int], float], omega: NDArray[np.float64]) -> Matrix:
    """``H = -sum K_ij (X_i X_j + Y_i Y_j) - sum omega_i Z_i`` from Kronecker products."""
    n = omega.size
    total = np.zeros((2**n, 2**n), dtype=np.complex128)
    for (i, j), value in couplings.items():
        total -= value * (_on({i: _X, j: _X}, n) + _on({i: _Y, j: _Y}, n))
    for qubit in range(n):
        total -= omega[qubit] * _on({qubit: _Z}, n)
    return total


def _projector(couplings: dict[tuple[int, int], float], omega: NDArray[np.float64]) -> Matrix:
    """Ground-state projector of :func:`_hamiltonian`."""
    _, vectors = np.linalg.eigh(_hamiltonian(couplings, omega))
    ground = vectors[:, :1]
    return np.asarray(ground @ ground.conj().T, dtype=np.complex128)


def _projector_qfi(
    couplings: dict[tuple[int, int], float], omega: NDArray[np.float64], pairs: Pairs
) -> NDArray[np.float64]:
    """``F_ab = 2 Tr(d_a P d_b P)`` with central differences of the projector."""
    step = 1e-5
    derivatives = []
    for pair in pairs:
        plus, minus = dict(couplings), dict(couplings)
        plus[pair] = plus.get(pair, 0.0) + step
        minus[pair] = minus.get(pair, 0.0) - step
        derivatives.append((_projector(plus, omega) - _projector(minus, omega)) / (2 * step))
    return np.array(
        [[2.0 * float(np.real(np.trace(a @ b))) for b in derivatives] for a in derivatives]
    )


def _matrix(couplings: dict[tuple[int, int], float], n: int) -> NDArray[np.float64]:
    """Symmetric coupling matrix of a pair dictionary."""
    k = np.zeros((n, n))
    for (i, j), value in couplings.items():
        k[i, j] = k[j, i] = value
    return k


def test_quantum_geometry_conformance_01_spectral_qfi_equals_the_projector_derivative() -> None:
    """The public spectral QFI equals ``2 Tr(dP dP)`` from an independently built Hamiltonian."""
    couplings = {(0, 1): 1.0}
    omega = np.array([0.3, 0.7])
    result = compute_qfi(_matrix(couplings, 2), omega)
    reference = _projector_qfi(couplings, omega, [(0, 1)])
    np.testing.assert_allclose(result.qfi_matrix, reference, rtol=1e-6)
    assert result.qfi_matrix[0, 0] > 1e-3 and result.convention == QFI_CONVENTION
    observable = QuantumFisherInformation()(
        coupling_matrix=_matrix(couplings, 2), natural_frequencies=omega
    )
    assert observable["qfi"] == pytest.approx(float(reference[0, 0]), rel=1e-6)
    assert observable["qfi_rank"] == 1.0 and observable["qfi_min_eigenvalue"] > 0.0
    # Three qubits and three pairs: the full matrix, off-diagonal terms included.
    ring = {(0, 1): 0.9, (1, 2): 0.6, (0, 2): 0.4}
    omega3 = np.array([0.2, 0.5, 0.9])
    pairs: Pairs = [(0, 1), (1, 2), (0, 2)]
    np.testing.assert_allclose(
        compute_qfi(_matrix(ring, 3), omega3, pairs=pairs).qfi_matrix,
        _projector_qfi(ring, omega3, pairs),
        rtol=1e-5,
        atol=1e-7,
    )


def test_quantum_geometry_conformance_02_gauge_rank_and_degeneracy() -> None:
    """Eigenvector phases do not change the metric; null directions and degeneracy are explicit."""
    couplings = {(0, 1): 1.0}
    omega = np.array([0.3, 0.7])
    energies, vectors = np.linalg.eigh(_hamiltonian(couplings, omega))
    generator = -(_on({0: _X, 1: _X}, 2) + _on({0: _Y, 1: _Y}, 2))
    rng = np.random.default_rng(3)
    for _ in range(3):
        rephased = vectors * np.exp(1j * rng.uniform(0, 2 * np.pi, size=vectors.shape[1]))
        ground = rephased[:, 0]
        total = sum(
            abs(rephased[:, m].conj() @ generator @ ground) ** 2 / (energies[m] - energies[0]) ** 2
            for m in range(1, energies.size)
        )
        assert 4.0 * total == pytest.approx(
            compute_qfi(_matrix(couplings, 2), omega).qfi_matrix[0, 0], rel=1e-12
        )
    # The same coupling listed twice is one physical direction: rank 1 of 2, reported.
    doubled = compute_qfi(_matrix(couplings, 2), omega, pairs=[(0, 1), (0, 1)])
    assert doubled.metric_rank == 1 and doubled.qfi_matrix.shape == (2, 2)
    assert abs(doubled.metric_eigenvalues[0]) <= QFI_PSD_TOLERANCE * doubled.metric_eigenvalues[1]
    gradient = GradientResult(
        value=0.0,
        gradient=np.array([1.0, 0.5]),
        method="test",
        shift=None,
        coefficient=None,
        evaluations=0,
        parameter_names=("a", "b"),
        trainable=(True, True),
    )
    with pytest.raises(ValueError, match="positive definite|ill-conditioned"):
        natural_gradient(gradient, doubled.qfi_matrix)
    solved = solve_natural_gradient_direction(
        gradient.gradient, doubled.qfi_matrix, damping=0.0, max_condition_number=1e8
    )
    assert solved.metric_nullity == 1 and solved.diagonal_shift > 0.0
    assert solved.regularization_reason == "condition_number_limited"
    assert solved.condition_number <= 1e8
    with pytest.raises(ValueError, match="more than rounding"):
        solve_natural_gradient_direction(
            gradient.gradient, doubled.qfi_matrix, damping=0.0, max_condition_number=1.0 + 1e-15
        )
    # omega = (1, 1), K = 1: |00> and the symmetric singlet-sector state share the ground energy.
    degenerate = {(0, 1): 1.0}
    flat = np.array([1.0, 1.0])
    spectrum = np.linalg.eigvalsh(_hamiltonian(degenerate, flat))
    assert spectrum[1] - spectrum[0] == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError, match="ground state is degenerate"):
        compute_qfi(_matrix(degenerate, 2), flat)
    with pytest.raises(ValueError, match="degeneracy_tolerance"):
        compute_qfi(_matrix(couplings, 2), omega, degeneracy_tolerance=0.0)


def test_quantum_geometry_conformance_03_metric_is_symmetric_psd_within_its_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Seeded admitted fixtures give a symmetric metric with no eigenvalue below the declared bound.

    Parameters
    ----------
    monkeypatch
        Forces a negative eigenvalue to show it is refused.

    """
    rng = np.random.default_rng(11)
    for _ in range(4):
        k = rng.uniform(0.3, 1.2, size=(3, 3))
        k = np.triu(k, 1) + np.triu(k, 1).T
        omega = rng.uniform(0.1, 1.0, size=3)
        result = compute_qfi(k, omega)
        np.testing.assert_array_equal(result.qfi_matrix, result.qfi_matrix.T)
        np.testing.assert_allclose(
            result.metric_eigenvalues, np.linalg.eigvalsh(result.qfi_matrix), atol=1e-12
        )
        largest = float(np.max(np.abs(result.metric_eigenvalues)))
        assert result.metric_eigenvalues[0] >= -QFI_PSD_TOLERANCE * largest
    real_eigvalsh: Callable[..., NDArray[np.float64]] = np.linalg.eigvalsh

    def shifted(matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return eigenvalues with the smallest pushed below the bound."""
        values = np.asarray(real_eigvalsh(matrix), dtype=np.float64)
        values[0] = -1e-3 * float(np.max(np.abs(values)))
        return values

    monkeypatch.setattr(np.linalg, "eigvalsh", shifted)
    with pytest.raises(ValueError, match="not a metric"):
        compute_qfi(_matrix({(0, 1): 1.0}, 2), np.array([0.3, 0.7]))


def test_quantum_geometry_conformance_03_decisions_do_not_depend_on_energy_units() -> None:
    """Scaling ``K`` and ``omega`` by ``s`` keeps the ground state, so rank and refusals are equal.

    The QFI scales as ``1 / s^2`` and the gap as ``s``; no decision may change.
    """
    k = np.array(
        [[0.0, 0.9, 0.4, 0.0], [0.9, 0.0, 0.6, 0.3], [0.4, 0.6, 0.0, 0.8], [0.0, 0.3, 0.8, 0.0]]
    )
    omega = np.array([0.2, 0.5, 0.9, 0.35])
    reference = compute_qfi(k, omega)
    assert reference.metric_rank == len(reference.coupling_pairs) == 5
    for scale in (1e-10, 1e5):
        scaled = compute_qfi(scale * k, scale * omega)
        assert scaled.coupling_pairs == reference.coupling_pairs
        assert scaled.metric_rank == reference.metric_rank
        np.testing.assert_allclose(
            scaled.qfi_matrix * scale**2, reference.qfi_matrix, rtol=1e-6, atol=1e-9
        )
        assert scaled.spectral_gap == pytest.approx(scale * reference.spectral_gap, rel=1e-6)
        with pytest.raises(ValueError, match="ground state is degenerate"):
            compute_qfi(scale * _matrix({(0, 1): 1.0}, 2), scale * np.array([1.0, 1.0]))
    # The default pair selection is relative to the strongest coupling.
    weak = compute_qfi(1e-12 * k, 1e-12 * omega)
    assert weak.coupling_pairs == reference.coupling_pairs


def test_quantum_geometry_conformance_04_regularisation_changes_the_step_and_is_recorded() -> None:
    """Damping shrinks each eigen-component by ``mu / (mu + lambda)`` and is reported."""
    metric = np.diag([4.0, 0.01])
    gradient = np.array([1.0, 1.0])
    exact = solve_natural_gradient_direction(gradient, metric, damping=0.0)
    np.testing.assert_allclose(exact.direction, [0.25, 100.0])
    assert exact.regularization_reason == "none" and exact.diagonal_shift == 0.0
    for damping in (1e-3, 1e-1):
        damped = solve_natural_gradient_direction(gradient, metric, damping=damping)
        np.testing.assert_allclose(damped.direction, gradient / (np.diag(metric) + damping))
        bias = damped.direction / exact.direction
        np.testing.assert_allclose(bias, np.diag(metric) / (np.diag(metric) + damping))
        assert damped.damping == damping and damped.regularization_reason == "damped"
        assert damped.to_dict()["regularization_policy"] == {
            "damping": damping,
            "eigenvalue_floor": 0.0,
            "max_condition_number": 1e12,
            "degeneracy_tolerance": 1e-10,
        }
    record = GradientResult(
        value=0.0,
        gradient=gradient,
        method="test",
        shift=None,
        coefficient=None,
        evaluations=0,
        parameter_names=("a", "b"),
        trainable=(True, True),
    )
    assert natural_gradient(record, metric, damping=0.5).damping == 0.5


def _ry_rz_state(theta: NDArray[np.float64]) -> NDArray[np.complex128]:
    """``RZ(theta_1) RY(theta_0) |0>``."""
    a, b = float(theta[0]), float(theta[1])
    after_ry = np.array([np.cos(a / 2), np.sin(a / 2)], dtype=np.complex128)
    return np.array([np.exp(-1j * b / 2), np.exp(1j * b / 2)]) * after_ry


def test_quantum_geometry_conformance_05_costs_are_compared_by_evaluations() -> None:
    """Objective and metric calls are counted independently and match the reported costs.

    The Fubini-Study metric of ``RZ RY |0>`` is ``diag(1, sin^2 theta_0) / 4``.
    Natural and identity-metric descent are compared by objective plus metric
    evaluations, not by step count; the result says nothing about other ansatze.
    """
    observable = np.array([[0.3, 1.0], [1.0, -0.3]], dtype=np.complex128)
    calls = {"objective": 0, "metric": 0}

    def energy(theta: NDArray[np.float64]) -> float:
        """Return the counted expectation of the observable."""
        calls["objective"] += 1
        state = _ry_rz_state(theta)
        return float(np.real(state.conj() @ observable @ state))

    def metric(theta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the counted Fubini-Study metric of the ansatz."""
        calls["metric"] += 1
        return np.diag([0.25, 0.25 * np.sin(theta[0]) ** 2 + 1e-3])

    start = np.array([0.4, 0.3])
    costs: dict[str, tuple[int, int, float]] = {}
    for name, provider in (("natural", metric), ("identity", None)):
        calls.update(objective=0, metric=0)
        result = parameter_shift_natural_gradient_descent(
            energy, start, metric_tensor=provider, learning_rate=0.2, max_steps=40
        )
        assert result.evaluations == calls["objective"]
        assert result.metric_evaluations == calls["metric"]
        certificate = validate_natural_gradient_training(result)
        assert certificate.metric_evaluations == result.metric_evaluations
        assert certificate.to_dict()["metric_evaluations"] == result.metric_evaluations
        assert result.to_dict()["metric_evaluations"] == result.metric_evaluations
        costs[name] = (result.evaluations, result.metric_evaluations, result.best_value)
    # One metric call per visited point, never one per internal solve.
    natural = costs["natural"]
    assert 0 < natural[1] <= natural[0]
    assert costs["identity"][1] == 0
    minimum = float(np.min(np.linalg.eigvalsh(observable)))
    assert min(natural[2], costs["identity"][2]) <= minimum + 0.05


def test_reporting_companions_keep_their_labels() -> None:
    """The labelled QFI proxy and an untoleranced target keep their documented meaning."""
    proxy = QuantumFisherInformation()(
        counts={"00": 60, "11": 40}, sync_order=0.5, allow_proxy_estimate=True
    )
    assert proxy["is_quantum_fisher_information"] == 0.0 and proxy["sync_order_input"] == 0.5
    derived = QuantumFisherInformation()(
        counts={"00": 60, "11": 40}, dla_asymmetry=10.0, allow_proxy_estimate=True
    )
    assert derived["dla_asymmetry_input"] == 10.0 and derived["qfi_available"] == 0.0
    result = parameter_shift_natural_gradient_descent(
        lambda theta: float(np.sum(theta**2)), np.array([0.5]), max_steps=3
    )
    certificate = validate_natural_gradient_training(result, target_value=result.best_value)
    assert certificate.within_target_value_tolerance is True
