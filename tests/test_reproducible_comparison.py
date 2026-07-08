# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the reproducible comparison artifact
"""Tests for the deterministic classical-vs-quantum Kuramoto comparison."""

from __future__ import annotations

import json
from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.benchmarks import reproducible_comparison as rc
from scpn_quantum_control.benchmarks.classical_baselines import ClassicalBaselineRun
from scpn_quantum_control.benchmarks.reproducible_comparison import (
    CLAIM_BOUNDARY,
    DETERMINISM,
    FAILURE_MODES,
    ComparisonMethodRow,
    ReproducibleKuramotoComparison,
    run_reproducible_kuramoto_comparison,
)


class _FakeQuantumKuramotoSolver:
    """Small deterministic stand-in for the statevector solver."""

    def __init__(
        self,
        n_oscillators: int,
        K: NDArray[np.float64],
        omega: NDArray[np.float64],
    ) -> None:
        self.n_oscillators = n_oscillators
        self.K = K
        self.omega = omega

    def run(
        self, *, t_max: float, dt: float, trotter_per_step: int
    ) -> dict[str, NDArray[np.float64]]:
        """Return a deterministic order-parameter trajectory."""
        del t_max, dt, trotter_per_step
        return {"R": np.array([0.41, 0.52], dtype=np.float64)}


@pytest.fixture(autouse=True)
def _deterministic_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep these artifact tests independent of Qiskit/NumPy coverage hooks."""

    def _fake_exact(
        n_osc: int,
        t_max: float,
        dt: float,
        K: NDArray[np.float64] | None = None,
        omega: NDArray[np.float64] | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        del n_osc, t_max, dt, K, omega
        return {"R": np.array([0.39, 0.5], dtype=np.float64)}

    def _fake_ode(
        K: NDArray[np.float64],
        omega: NDArray[np.float64],
        *,
        t_max: float,
        dt: float,
        theta0: NDArray[np.float64] | None = None,
    ) -> ClassicalBaselineRun:
        del t_max, dt, theta0
        return ClassicalBaselineRun(
            name="scipy_ode",
            backend="scipy.solve_ivp(RK45)",
            n_oscillators=K.shape[0],
            available=True,
            elapsed_ms=1.25,
            times=np.array([0.0, 1.0], dtype=np.float64),
            order_parameter=np.array([0.37, 0.48], dtype=np.float64),
            metadata={"omega_len": omega.shape[0]},
        )

    monkeypatch.setattr(rc, "classical_exact_evolution", _fake_exact)
    monkeypatch.setattr(rc, "QuantumKuramotoSolver", _FakeQuantumKuramotoSolver)
    monkeypatch.setattr(rc, "scipy_ode_baseline", _fake_ode)


def _run(
    n_oscillators: int,
    *,
    t_max: float = 0.6,
    dt: float = 0.2,
    trotter_per_step: int = 3,
    seed: int = 42,
    randomise_initial_phases: bool = False,
    K: NDArray[np.float64] | None = None,
    omega: NDArray[np.float64] | None = None,
) -> ReproducibleKuramotoComparison:
    """Run a small, fast comparison with documented defaults."""
    return run_reproducible_kuramoto_comparison(
        n_oscillators,
        t_max=t_max,
        dt=dt,
        trotter_per_step=trotter_per_step,
        seed=seed,
        randomise_initial_phases=randomise_initial_phases,
        K=K,
        omega=omega,
    )


def _reproducible_pairs(
    comparison: ReproducibleKuramotoComparison,
) -> list[tuple[str, float | None, float | None, bool]]:
    """Return only the seed-independent, timing-excluded quantities."""
    return [(r.method, r.r_final, r.r_error_vs_exact, r.available) for r in comparison.rows]


def test_default_run_has_three_methods_and_exact_reference() -> None:
    """A default run produces the exact, ODE and quantum rows."""
    comparison = _run(4)

    assert comparison.reference_method == "classical_exact"
    methods = {row.method for row in comparison.rows}
    assert methods == {"classical_exact", "classical_ode", "quantum_trotter"}
    assert comparison.n_oscillators == 4
    assert comparison.initial_condition == "omega_phase"
    assert comparison.metadata == {
        "coupling_source": "paper27",
        "omega_source": "omega_n_16",
        "statevector_boundary": 16,
    }


def test_reference_row_carries_no_self_error() -> None:
    """The exact reference row has no error against itself."""
    comparison = _run(3)

    reference = comparison.row("classical_exact")
    assert reference.r_error_vs_exact is None
    assert reference.r_final is not None


def test_quantum_and_ode_rows_carry_error_vs_exact() -> None:
    """The quantum and ODE rows report a non-negative error against exact."""
    comparison = _run(4)

    for method in ("classical_ode", "quantum_trotter"):
        row = comparison.row(method)
        assert row.available is True
        assert row.r_error_vs_exact is not None
        assert row.r_error_vs_exact >= 0.0


def test_reproducible_quantities_are_deterministic_across_runs() -> None:
    """Order-parameter values and errors repeat exactly; timing is excluded."""
    first = _run(4)
    second = _run(4)

    assert _reproducible_pairs(first) == _reproducible_pairs(second)


def test_quantum_row_matches_exact_on_resolvable_dynamics() -> None:
    """At small sizes the statevector Trotter route tracks the exact route."""
    comparison = _run(3)

    quantum = comparison.row("quantum_trotter")
    assert quantum.r_error_vs_exact is not None
    assert quantum.r_error_vs_exact < 0.05


def test_randomised_initial_phases_are_labelled_and_seed_reproducible() -> None:
    """The seeded random-phase mode is labelled and repeats for a fixed seed."""
    first = _run(3, seed=7, randomise_initial_phases=True)
    second = _run(3, seed=7, randomise_initial_phases=True)

    assert first.initial_condition == "seeded_uniform"
    assert first.seed == 7
    assert _reproducible_pairs(first) == _reproducible_pairs(second)


def test_caller_supplied_coupling_and_omega_are_recorded() -> None:
    """Caller-supplied K and omega flip the provenance metadata."""
    K = np.array([[0.0, 0.3, 0.0], [0.3, 0.0, 0.3], [0.0, 0.3, 0.0]], dtype=np.float64)
    omega = np.array([0.1, 0.2, 0.3], dtype=np.float64)

    comparison = _run(3, K=K, omega=omega)

    assert comparison.metadata == {
        "coupling_source": "caller",
        "omega_source": "caller",
        "statevector_boundary": 16,
    }


def test_to_dict_is_json_serialisable_and_complete() -> None:
    """The artifact serialises to JSON with all documented sections."""
    comparison = _run(3)
    payload = comparison.to_dict()

    text = json.dumps(payload)
    restored = json.loads(text)
    assert restored["reference_method"] == "classical_exact"
    assert restored["failure_modes"] == list(FAILURE_MODES)
    assert restored["claim_boundary"] == CLAIM_BOUNDARY
    assert restored["determinism"] == DETERMINISM
    assert restored["seed"] == 42
    assert len(restored["rows"]) == 3
    assert {row["method"] for row in restored["rows"]} == {
        "classical_exact",
        "classical_ode",
        "quantum_trotter",
    }


def test_method_row_to_dict_round_trips_fields() -> None:
    """A single method row serialises every field."""
    row = ComparisonMethodRow(
        method="classical_ode",
        backend="scipy.solve_ivp(RK45)",
        available=False,
        r_final=None,
        r_error_vs_exact=None,
        elapsed_ms=1.5,
        unavailable_reason="boom",
    )

    assert row.to_dict() == {
        "method": "classical_ode",
        "backend": "scipy.solve_ivp(RK45)",
        "available": False,
        "r_final": None,
        "r_error_vs_exact": None,
        "elapsed_ms": 1.5,
        "unavailable_reason": "boom",
    }


def test_row_lookup_raises_for_unknown_method() -> None:
    """Looking up a method that was not run raises ``KeyError``."""
    comparison = _run(3)

    with pytest.raises(KeyError, match="missing_method"):
        comparison.row("missing_method")


def test_unavailable_ode_propagates_none_error_and_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unavailable ODE row yields a ``None`` error and keeps its reason."""

    def _fake_ode(*_args: object, **_kwargs: object) -> ClassicalBaselineRun:
        return ClassicalBaselineRun(
            name="scipy_ode",
            backend="scipy.solve_ivp(RK45)",
            n_oscillators=3,
            available=False,
            elapsed_ms=0.0,
            unavailable_reason="forced unavailable",
        )

    monkeypatch.setattr(rc, "scipy_ode_baseline", _fake_ode)
    comparison = _run(3)

    ode = comparison.row("classical_ode")
    assert ode.available is False
    assert ode.r_final is None
    assert ode.r_error_vs_exact is None
    assert ode.unavailable_reason == "forced unavailable"


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (lambda: _run(1), "n_oscillators must be >= 2"),
        (lambda: _run(4, t_max=0.0), "t_max must be positive"),
        (lambda: _run(4, dt=0.0), "dt must be positive"),
        (lambda: _run(4, t_max=0.2, dt=0.5), "must not exceed t_max"),
        (lambda: _run(4, trotter_per_step=0), "trotter_per_step must be >= 1"),
        (lambda: _run(4, seed=-1), "seed must be non-negative"),
    ],
)
def test_invalid_arguments_are_rejected(
    call: Callable[[], ReproducibleKuramotoComparison], match: str
) -> None:
    """Every documented bound rejects out-of-range input."""
    with pytest.raises(ValueError, match=match):
        call()


def test_larger_than_16_comparison_is_classical_only() -> None:
    """N > 16 returns a scalable baseline and refuses statevector rows."""
    comparison = _run(20, t_max=0.2, dt=0.1)

    assert comparison.reference_method == "classical_ode"
    assert comparison.metadata == {
        "coupling_source": "paper27",
        "omega_source": "omega_n_16_periodic_extension",
        "statevector_boundary": 16,
    }
    ode = comparison.row("classical_ode")
    exact = comparison.row("classical_exact")
    quantum = comparison.row("quantum_trotter")
    assert ode.available is True
    assert ode.r_final is not None
    assert ode.r_error_vs_exact is None
    assert exact.available is False
    assert exact.r_final is None
    assert exact.unavailable_reason is not None
    assert "n_oscillators>16" in exact.unavailable_reason
    assert quantum.available is False
    assert quantum.r_final is None
    assert quantum.unavailable_reason is not None
    assert "n_oscillators>16" in quantum.unavailable_reason


def test_wrong_coupling_shape_is_rejected() -> None:
    """A coupling matrix of the wrong shape is rejected."""
    bad_K = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="K must have shape"):
        _run(3, K=bad_K)


def test_wrong_omega_shape_is_rejected() -> None:
    """A frequency vector of the wrong shape is rejected."""
    bad_omega = np.zeros(5, dtype=np.float64)
    with pytest.raises(ValueError, match="omega must have shape"):
        _run(3, omega=bad_omega)


def test_failure_modes_document_each_route() -> None:
    """The shared failure-mode list names each route and the scope caveat."""
    joined = " ".join(FAILURE_MODES).lower()
    assert "quantum_trotter" in joined
    assert "classical_exact" in joined
    assert "classical_ode" in joined
    assert "no quantum advantage" in joined
