# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for Lyapunov / contraction synchronisation certificates
"""Module-specific tests for :mod:`synchronisation_certificate`.

The contracts: the Lyapunov potential's gradient is the negative Kuramoto field, so its decrease rate
is exactly ``-‖θ̇‖²``; the contraction rate equals the algebraic connectivity (Fiedler value) of the
weighted Laplacian ``L(w)``; a phase-cohesive locked state is certified stable while a non-cohesive
anti-phase state is not; and the input contract is enforced.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.accel.networked_kuramoto import networked_kuramoto_force
from scpn_quantum_control.accel.synchronisation_certificate import (
    SynchronisationCertificate,
    certify_synchronisation,
    contraction_rate,
    phase_cohesiveness,
    potential_decrease_rate,
    synchronisation_potential,
)

_N = 8


def _network(seed: int = 0) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    raw = rng.uniform(0.3, 1.0, size=(_N, _N))
    coupling = 0.5 * (raw + raw.T)
    np.fill_diagonal(coupling, 0.0)
    omega = rng.standard_normal(_N) * 0.1
    return {"coupling": coupling, "omega": omega - omega.mean(), "rng": rng}


def _locked_state(network: dict[str, Any]) -> NDArray[np.float64]:
    phases = network["rng"].uniform(0.0, 0.3, size=_N)
    for _ in range(4000):
        phases = phases + 0.01 * (
            network["omega"] + networked_kuramoto_force(phases, network["coupling"])
        )
    return np.ascontiguousarray(phases, dtype=np.float64)


def test_potential_gradient_is_the_negative_field() -> None:
    network = _network()
    state = network["rng"].uniform(0.0, 0.5, size=_N)
    field = network["omega"] + networked_kuramoto_force(state, network["coupling"])
    gradient = np.zeros(_N, dtype=np.float64)
    eps = 1e-6
    for index in range(_N):
        plus = state.copy()
        minus = state.copy()
        plus[index] += eps
        minus[index] -= eps
        gradient[index] = (
            synchronisation_potential(plus, network["omega"], network["coupling"])
            - synchronisation_potential(minus, network["omega"], network["coupling"])
        ) / (2.0 * eps)
    assert gradient == pytest.approx(-field, abs=1e-7)
    # the decrease rate is exactly -||field||^2
    assert potential_decrease_rate(state, network["omega"], network["coupling"]) == pytest.approx(
        -float(field @ field)
    )


def test_contraction_rate_is_the_algebraic_connectivity() -> None:
    network = _network(1)
    state = _locked_state(network)
    difference = state[:, None] - state[None, :]
    weights = network["coupling"] * np.cos(difference)
    laplacian = np.diag(weights.sum(axis=1)) - weights
    fiedler = np.sort(np.linalg.eigvalsh(laplacian))[1]
    assert contraction_rate(state, network["coupling"]) == pytest.approx(fiedler)


def test_cohesive_locked_state_is_certified() -> None:
    network = _network(2)
    state = _locked_state(network)
    certificate = certify_synchronisation(state, network["omega"], network["coupling"])
    assert isinstance(certificate, SynchronisationCertificate)
    assert certificate.is_cohesive
    assert certificate.phase_cohesiveness < np.pi / 2.0
    assert certificate.is_contracting
    assert certificate.contraction_rate > 0.0
    assert certificate.is_certified
    assert certificate.lyapunov_decrease_rate <= 1e-9  # at the locked state the field vanishes


def test_anti_phase_state_is_not_certified() -> None:
    network = _network(3)
    anti_phase = np.zeros(_N)
    anti_phase[::2] = np.pi
    certificate = certify_synchronisation(anti_phase, network["omega"], network["coupling"])
    assert not certificate.is_cohesive
    assert certificate.phase_cohesiveness == pytest.approx(np.pi)
    assert certificate.contraction_rate < 0.0
    assert not certificate.is_certified


def test_phase_cohesiveness_value() -> None:
    coupling = np.ones((3, 3))
    np.fill_diagonal(coupling, 0.0)
    assert phase_cohesiveness(np.array([0.0, 0.1, 0.2]), coupling) == pytest.approx(0.2)
    # an isolated graph (no coupled pairs) is trivially cohesive
    assert phase_cohesiveness(np.array([0.0, 3.0]), np.zeros((2, 2))) == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("call", "kwargs", "message"),
    [
        ("cohesive", {"phases": np.zeros(1)}, "phases must be a one-dimensional"),
        ("cohesive", {"coupling": np.zeros((_N, _N + 1))}, "coupling must have shape"),
        ("cohesive", {"phases": np.full(_N, np.nan)}, "must be finite"),
        ("cohesive", {"coupling": np.triu(np.ones((_N, _N)), 1)}, "coupling must be symmetric"),
        ("potential", {"omega": np.zeros(_N + 1)}, "omega must have shape"),
        ("potential", {"omega": np.full(_N, np.inf)}, "omega must be finite"),
        ("certify", {"cohesiveness_threshold": 0.0}, "cohesiveness_threshold must lie in"),
        ("certify", {"cohesiveness_threshold": 4.0}, "cohesiveness_threshold must lie in"),
    ],
)
def test_validation_errors(call: str, kwargs: dict[str, Any], message: str) -> None:
    network = _network()
    base: dict[str, Any] = {
        "phases": np.zeros(_N),
        "omega": network["omega"],
        "coupling": network["coupling"],
        "cohesiveness_threshold": np.pi / 2.0,
    }
    base.update(kwargs)
    with pytest.raises(ValueError, match=message):
        if call == "cohesive":
            phase_cohesiveness(base["phases"], base["coupling"])
        elif call == "potential":
            synchronisation_potential(base["phases"], base["omega"], base["coupling"])
        else:
            certify_synchronisation(
                base["phases"],
                base["omega"],
                base["coupling"],
                cohesiveness_threshold=base["cohesiveness_threshold"],
            )
