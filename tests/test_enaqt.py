# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — bounded ENAQT transport tests
"""Physical, validation, compatibility, and branch tests for ENAQT scans."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

import scpn_quantum_control.analysis.enaqt as enaqt_module
from scpn_quantum_control.analysis.enaqt import (
    DEFAULT_GAMMA_GRID,
    ENAQTResult,
    enaqt_scan,
)
from scpn_quantum_control.dense_budget import DenseAllocationError


def _array(values: object) -> NDArray[np.float64]:
    """Return one float array for concise test fixtures."""
    return np.asarray(values, dtype=np.float64)


def _chain(site_energies: tuple[float, ...]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return a unit-coupled chain and selected site energies."""
    sites = len(site_energies)
    coupling = np.zeros((sites, sites), dtype=np.float64)
    for site in range(sites - 1):
        coupling[site, site + 1] = 1.0
        coupling[site + 1, site] = 1.0
    return coupling, _array(site_energies)


@pytest.fixture(scope="module")
def intermediate_result() -> ENAQTResult:
    """Run the frozen disordered-chain positive control once."""
    coupling, omega = _chain((0.0, 3.0, -2.0, 1.0))
    return enaqt_scan(
        coupling,
        omega,
        gamma_range=_array(DEFAULT_GAMMA_GRID),
        t_evolve=10.0,
    )


def test_disordered_chain_has_reproducible_intermediate_optimum(
    intermediate_result: ENAQTResult,
) -> None:
    """Detect the preregistered interior optimum and both endpoint declines."""
    result = intermediate_result
    assert result.optimal_gamma == 3.0
    assert result.optimal_efficiency == pytest.approx(0.176564980145, abs=1e-10)
    assert result.coherent_efficiency == pytest.approx(0.052273864249, abs=1e-10)
    assert result.high_noise_efficiency == pytest.approx(0.011466613693, abs=1e-10)
    assert result.enhancement == pytest.approx(3.37769136991, abs=1e-9)
    assert result.has_intermediate_optimum is True
    assert result.source_site == 0
    assert result.target_site == 3


def test_uniform_chain_is_a_coherent_negative_control() -> None:
    """Refuse to invent an intermediate optimum when coherence wins."""
    coupling, omega = _chain((0.0, 0.0, 0.0))
    result = enaqt_scan(
        coupling,
        omega,
        gamma_range=_array([0.0, 0.01, 0.1, 1.0, 10.0, 30.0]),
        t_evolve=10.0,
    )
    assert result.optimal_gamma == 0.0
    assert result.enhancement == 1.0
    assert result.has_intermediate_optimum is False
    assert np.all(np.diff(result.efficiency_values) < 0.0)


def test_disconnected_target_is_a_zero_transport_negative_control() -> None:
    """Keep all efficiencies zero when no Hamiltonian path reaches the sink."""
    coupling = np.zeros((3, 3), dtype=np.float64)
    coupling[0, 1] = coupling[1, 0] = 1.0
    result = enaqt_scan(
        coupling,
        _array([0.0, 1.0, 2.0]),
        gamma_range=_array([0.0, 1.0, 10.0]),
        t_evolve=5.0,
    )
    np.testing.assert_allclose(result.efficiency_values, 0.0, atol=1e-14)
    assert result.has_intermediate_optimum is False
    assert result.enhancement == 0.0


def test_zero_gamma_is_evaluated_when_absent_and_grid_order_is_retained(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use exact zero and the largest gamma as endpoints without sorting output."""
    coupling, omega = _chain((0.0, 0.0))
    calls: list[float] = []

    def fake_efficiency(
        _hamiltonian: NDArray[np.complex128],
        gamma: float,
        _source_site: int,
        _target_site: int,
        _t_evolve: float,
        _n_steps: int,
        _sink_rate: float,
        _loss_rate: float,
    ) -> float:
        calls.append(gamma)
        return {0.0: 0.2, 1.0: 0.4, 3.0: 0.3}[gamma]

    monkeypatch.setattr(enaqt_module, "_transport_efficiency", fake_efficiency)
    result = enaqt_scan(coupling, omega, gamma_range=_array([3.0, 1.0, 1.0]))
    assert calls == [3.0, 1.0, 0.0]
    assert result.gamma_values.tolist() == [3.0, 1.0, 1.0]
    assert result.optimal_gamma == 1.0
    assert result.coherent_efficiency == 0.2
    assert result.high_noise_efficiency == 0.3
    assert result.has_intermediate_optimum is True


def test_exponential_segmentation_preserves_result() -> None:
    """Use the semigroup property to retain n_steps compatibility."""
    coupling, omega = _chain((0.0, 3.0, -2.0, 1.0))
    gammas = _array([0.0, 1.0, 3.0, 30.0])
    one = enaqt_scan(coupling, omega, gammas, t_evolve=10.0, n_steps=1)
    four = enaqt_scan(coupling, omega, gammas, t_evolve=10.0, n_steps=4)
    np.testing.assert_allclose(one.efficiency_values, four.efficiency_values, atol=1e-12)


def test_custom_source_target_and_rates_are_recorded() -> None:
    """Expose all scenario-defining transport parameters in the result."""
    coupling, omega = _chain((0.0, 0.0, 0.0))
    result = enaqt_scan(
        coupling,
        omega,
        gamma_range=_array([0.0, 2.0]),
        source_site=2,
        target_site=0,
        sink_rate=0.7,
        loss_rate=0.0,
        t_evolve=2.0,
    )
    assert result.source_site == 2
    assert result.target_site == 0
    assert result.sink_rate == 0.7
    assert result.loss_rate == 0.0
    assert all(0.0 <= value <= 1.0 for value in result.efficiency_values)


def test_generator_preserves_trace_and_adjoint_identity() -> None:
    """Check trace preservation and the Hilbert--Schmidt adjoint numerically."""
    coupling, omega = _chain((0.0, 1.0, -0.5))
    hamiltonian = enaqt_module._site_hamiltonian(coupling, omega)
    generator, trace = enaqt_module._transport_generator(
        hamiltonian,
        gamma=0.3,
        target_site=2,
        sink_rate=0.8,
        loss_rate=0.05,
    )
    dimension = 5
    rng = np.random.default_rng(7)
    rho = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
    observable = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(
        size=(dimension, dimension)
    )
    generated = generator.matvec(rho.reshape(-1)).reshape(dimension, dimension)
    adjointed = generator.rmatvec(observable.reshape(-1)).reshape(dimension, dimension)
    assert np.trace(generated) == pytest.approx(0.0, abs=1e-12)
    assert np.vdot(observable, generated) == pytest.approx(np.vdot(adjointed, rho), abs=1e-12)
    assert trace == pytest.approx(-0.3 * 3 * 4 - 5 * 0.8 - 5 * 3 * 0.05)


def test_result_compatibility_aliases_and_serialisation(
    intermediate_result: ENAQTResult,
) -> None:
    """Keep legacy readers functional while emitting honest field names."""
    result = intermediate_result
    assert result.optimal_r == result.optimal_efficiency
    assert result.r_values is result.efficiency_values
    assert result.coherent_r == result.coherent_efficiency
    assert result.classical_r == result.high_noise_efficiency
    assert result.gamma_values.flags.writeable is False
    assert result.efficiency_values.flags.writeable is False
    with pytest.raises(ValueError, match="read-only"):
        result.gamma_values[0] = 2.0
    payload = result.to_dict()
    assert payload["optimal_efficiency"] == result.optimal_efficiency
    assert "optimal_r" not in payload
    assert payload["has_intermediate_optimum"] is True


@pytest.mark.parametrize(
    ("coupling", "omega", "gammas", "source", "target", "message"),
    [
        (_array([1.0, 2.0]), _array([1.0, 2.0]), _array([0.0]), 0, 1, "square"),
        (np.zeros((1, 1)), _array([0.0]), _array([0.0]), 0, None, "at least two"),
        (np.zeros((2, 2)), _array([0.0]), _array([0.0]), 0, 1, "shape"),
        (np.full((2, 2), np.nan), _array([0.0, 0.0]), _array([0.0]), 0, 1, "finite"),
        (_array([[0.0, 1.0], [0.0, 0.0]]), _array([0.0, 0.0]), _array([0.0]), 0, 1, "symmetric"),
        (np.zeros((2, 2)), _array([0.0, 0.0]), _array([-1.0]), 0, 1, "non-negative"),
        (np.zeros((2, 2)), _array([0.0, 0.0]), _array([0.0]), True, 1, "source_site"),
        (np.zeros((2, 2)), _array([0.0, 0.0]), _array([0.0]), 0, 2, "target_site"),
        (np.zeros((2, 2)), _array([0.0, 0.0]), _array([0.0]), 0, 0, "must differ"),
    ],
)
def test_network_inputs_fail_closed(
    coupling: NDArray[np.float64],
    omega: NDArray[np.float64],
    gammas: NDArray[np.float64],
    source: int,
    target: int | None,
    message: str,
) -> None:
    """Reject malformed, non-Hermitian, or invalid site-network inputs."""
    with pytest.raises(ValueError, match=message):
        enaqt_scan(coupling, omega, gammas, source_site=source, target_site=target)


def test_complex_and_nonfinite_frequency_inputs_fail_closed() -> None:
    """Refuse lossy complex-to-real conversion and non-finite site energies."""
    with pytest.raises(ValueError, match="real-valued"):
        enaqt_scan(
            np.asarray([[0.0, 1j], [-1j, 0.0]]),
            _array([0.0, 0.0]),
        )
    with pytest.raises(ValueError, match="finite"):
        enaqt_scan(np.zeros((2, 2)), _array([0.0, np.inf]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"t_evolve": 0.0}, "t_evolve"),
        ({"sink_rate": -1.0}, "sink_rate"),
        ({"loss_rate": -1.0}, "loss_rate"),
        ({"minimum_improvement": -1.0}, "minimum_improvement"),
        ({"loss_rate": float("nan")}, "loss_rate"),
        ({"n_steps": 0}, "n_steps"),
        ({"n_steps": True}, "n_steps"),
    ],
)
def test_scalar_parameters_fail_closed(kwargs: dict[str, object], message: str) -> None:
    """Reject non-physical and invalid scan controls before evolution."""
    coupling, omega = _chain((0.0, 0.0))
    unchecked_scan = cast(Callable[..., ENAQTResult], enaqt_scan)
    with pytest.raises(ValueError, match=message):
        unchecked_scan(coupling, omega, gamma_range=_array([0.0]), **kwargs)


@pytest.mark.parametrize("gammas", [[], [[0.0]], [0.0, np.nan]])
def test_gamma_grid_shape_and_finiteness_fail_closed(gammas: object) -> None:
    """Require one finite non-empty gamma vector."""
    coupling, omega = _chain((0.0, 0.0))
    with pytest.raises(ValueError, match="gamma_range"):
        enaqt_scan(coupling, omega, gamma_range=cast(NDArray[np.float64], gammas))


def test_budget_rejects_before_hamiltonian_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the site-basis workspace gate before constructing the Hamiltonian."""
    coupling, omega = _chain(tuple(0.0 for _ in range(100)))

    def fail_if_called(
        _coupling: NDArray[np.float64], _omega: NDArray[np.float64]
    ) -> NDArray[np.complex128]:
        raise AssertionError("Hamiltonian allocated before budget gate")

    monkeypatch.setattr(enaqt_module, "_site_hamiltonian", fail_if_called)
    with pytest.raises(DenseAllocationError, match="site-basis density"):
        enaqt_scan(
            coupling,
            omega,
            gamma_range=_array([0.0]),
            max_dense_gib=1e-9,
        )


def test_invalid_budget_value_uses_shared_guard() -> None:
    """Retain the repository-wide positive budget contract."""
    coupling, omega = _chain((0.0, 0.0))
    with pytest.raises(ValueError, match="max_gib must be positive"):
        enaqt_scan(coupling, omega, max_dense_gib=0.0)


def test_default_grid_and_hamiltonian_diagonal() -> None:
    """Expose the fixed default grid and add site energies to coupling diagonals."""
    coupling = _array([[0.5, 1.0], [1.0, -0.5]])
    omega = _array([2.0, 3.0])
    hamiltonian = enaqt_module._site_hamiltonian(coupling, omega)
    np.testing.assert_allclose(hamiltonian, [[2.5, 1.0], [1.0, 2.5]])
    result = enaqt_scan(coupling, omega, t_evolve=0.1)
    assert result.gamma_values.tolist() == list(DEFAULT_GAMMA_GRID)


def test_efficiency_guard_rejects_nonphysical_solver_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail instead of clipping a materially non-physical propagated density."""
    coupling, omega = _chain((0.0, 0.0))
    hamiltonian = enaqt_module._site_hamiltonian(coupling, omega)

    def fake_exponential(
        _operator: object,
        vector: NDArray[np.complex128],
        *,
        traceA: float,
    ) -> NDArray[np.complex128]:
        del traceA
        result = np.zeros_like(vector)
        result[2 * 4 + 2] = 2.0
        return result

    monkeypatch.setattr(enaqt_module, "expm_multiply", fake_exponential)
    with pytest.raises(RuntimeError, match="escaped"):
        enaqt_module._transport_efficiency(
            hamiltonian,
            gamma=0.0,
            source_site=0,
            target_site=1,
            t_evolve=1.0,
            n_steps=1,
            sink_rate=1.0,
            loss_rate=0.0,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"gamma_values": _array([[0.0]])}, "gamma_values"),
        ({"efficiency_values": _array([0.1, 0.2])}, "equal shape"),
        ({"gamma_values": _array([-1.0] * len(DEFAULT_GAMMA_GRID))}, "non-negative"),
        (
            {"efficiency_values": _array([2.0] * len(DEFAULT_GAMMA_GRID))},
            "efficiency_values",
        ),
        ({"optimal_gamma": float("nan")}, "optimal_gamma"),
        ({"optimal_gamma": -1.0}, "optimal_gamma"),
        ({"optimal_efficiency": 2.0}, "optimal_efficiency"),
        ({"enhancement": -1.0}, "enhancement"),
        ({"has_intermediate_optimum": cast(bool, "yes")}, "must be boolean"),
        ({"source_site": -1}, "source_site"),
        ({"source_site": True}, "source_site"),
        ({"target_site": 0}, "must differ"),
        ({"t_evolve": 0.0}, "time and sink"),
        ({"sink_rate": 0.0}, "time and sink"),
        ({"loss_rate": -1.0}, "time and sink"),
        ({"optimal_gamma": 2.0}, "present"),
        ({"optimal_efficiency": 0.1}, "maximum scanned"),
        ({"coherent_efficiency": 0.1}, "gamma-zero"),
        ({"high_noise_efficiency": 0.1}, "largest gamma"),
    ],
)
def test_result_contract_fails_closed(
    intermediate_result: ENAQTResult,
    changes: dict[str, object],
    message: str,
) -> None:
    """Reject malformed result construction even outside the scanner."""
    unchecked_replace = cast(Callable[..., ENAQTResult], replace)
    with pytest.raises(ValueError, match=message):
        unchecked_replace(intermediate_result, **changes)
