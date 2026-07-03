# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — tests for the neural-operator advantage study
"""Contract tests for the surrogate-versus-direct-simulation advantage study.

These exercise the real torch training end-to-end on a tiny network, so the module skips without the
optional ``[torch]`` extra. They assert the honest contracts — the surrogate beats persistence on
held-out initial conditions, its error grows more slowly than persistence over the horizon, the
host-independent cost model is wired through, timings are optional and excluded from the reproducible
digest, and the study is deterministic on the content under fixed seeds.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

pytest.importorskip("torch")

from scpn_quantum_control.forecasting.neural_operator_advantage import (  # noqa: E402
    SCHEMA,
    HeldOutFidelity,
    NeuralOperatorAdvantage,
    evaluate_neural_operator_advantage,
)
from scpn_quantum_control.forecasting.neural_operator_cost_model import (  # noqa: E402
    SurrogateCostModel,
)

_N = 6
_DT = 0.05
_STEPS = 8  # horizon = 0.4
_FIXED_CLOCK = datetime(2026, 7, 3, 12, 0, 0, tzinfo=timezone.utc)


def _network(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    omega = rng.normal(0.0, 0.5, size=_N)
    coupling = np.full((_N, _N), 2.0 / _N, dtype=np.float64)
    np.fill_diagonal(coupling, 0.0)
    return omega, coupling


def _study(*, measure_wall_clock: bool = False) -> NeuralOperatorAdvantage:
    omega, coupling = _network(seed=11)
    return evaluate_neural_operator_advantage(
        omega,
        coupling,
        dt=_DT,
        n_steps=_STEPS,
        n_trajectories=64,
        n_eval=10,
        latent_dim=12,
        hidden_dim=24,
        epochs=150,
        learning_rate=3e-3,
        dataset_seed=3,
        train_seed=0,
        eval_seed=9999,
        measure_wall_clock=measure_wall_clock,
        clock=lambda: _FIXED_CLOCK,
    )


def test_surrogate_beats_persistence_on_held_out_initial_conditions() -> None:
    study = _study()
    assert isinstance(study.fidelity, HeldOutFidelity)
    assert study.fidelity.beats_persistence
    assert study.fidelity.surrogate_mean_error < study.fidelity.persistence_mean_error
    # training actually reduced the loss
    assert study.loss_final < study.loss_start
    # the ground-truth trajectories were produced by a real dispatched RK4 tier
    assert study.rk4_tier in {"rust", "python", "julia"}


def test_error_grows_more_slowly_than_persistence_over_the_horizon() -> None:
    study = _study()
    curve = study.fidelity.error_vs_horizon
    assert len(curve) == _STEPS + 1
    # persistence is exact at t=0 and grows monotonically; the surrogate wins by the horizon end
    times = [t for t, _, _ in curve]
    assert times[0] == 0.0
    assert curve[0][2] == pytest.approx(0.0, abs=1e-12)  # persistence error at t=0 is zero
    # by the terminal time the surrogate is clearly better than persistence
    assert study.fidelity.surrogate_terminal_error < study.fidelity.persistence_terminal_error
    # persistence error at the horizon exceeds its error near the start (it degrades with time)
    assert curve[-1][2] > curve[1][2]


def test_cost_model_is_wired_through() -> None:
    study = _study()
    assert isinstance(study.cost_model, SurrogateCostModel)
    assert study.cost_model.n_oscillators == _N
    assert study.cost_model.n_steps == _STEPS
    assert study.cost_model.rk4_right_hand_side_evaluations == 4 * _STEPS
    # the per-query FLOP ratio is a finite positive number
    assert study.cost_model.per_query_flop_ratio > 0.0


def test_wall_clock_is_optional_and_excluded_by_default() -> None:
    without = _study(measure_wall_clock=False)
    assert without.wall_clock_ms is None
    with_timing = _study(measure_wall_clock=True)
    assert with_timing.wall_clock_ms is not None
    assert set(with_timing.wall_clock_ms) == {
        "direct_full_trajectory_ms",
        "surrogate_single_query_ms",
    }
    assert all(value > 0.0 for value in with_timing.wall_clock_ms.values())
    # timings do not enter the reproducible payload digest
    assert without.payload_sha256 == with_timing.payload_sha256


def test_to_dict_is_json_ready_and_complete() -> None:
    import json

    study = _study()
    payload = study.to_dict()
    assert payload["schema"] == SCHEMA
    assert payload["generated_utc"] == _FIXED_CLOCK.isoformat()
    assert payload["production_claim_allowed"] is False
    assert len(payload["payload_sha256"]) == 64
    assert set(payload["fidelity"]) == {
        "n_eval",
        "horizon",
        "surrogate_mean_error",
        "persistence_mean_error",
        "surrogate_terminal_error",
        "persistence_terminal_error",
        "beats_persistence",
        "error_vs_horizon",
    }
    assert "cost_model" in payload and "provenance" in payload
    assert "claim_boundary" in payload and "determinism" in payload
    # the whole thing serialises
    json.dumps(payload)


def test_study_is_deterministic_on_content() -> None:
    first = _study()
    second = _study()
    assert first.payload_sha256 == second.payload_sha256
    assert first.fidelity.surrogate_mean_error == pytest.approx(
        second.fidelity.surrogate_mean_error
    )
    assert first.fidelity.error_vs_horizon == second.fidelity.error_vs_horizon
    assert first.to_dict() == second.to_dict()


def test_payload_digest_is_host_independent_of_provenance() -> None:
    # the digest covers only the bit-exact cost model and configuration, so it is stable regardless of
    # the captured host provenance or timings
    study = _study()
    assert isinstance(study.provenance, dict)
    assert "loadavg" in study.provenance  # host-bounded, deliberately outside the digest
    assert study.payload_sha256 == _study().payload_sha256


def test_non_positive_n_eval_is_rejected() -> None:
    omega, coupling = _network(seed=11)
    with pytest.raises(ValueError, match="n_eval"):
        evaluate_neural_operator_advantage(
            omega, coupling, dt=_DT, n_steps=_STEPS, n_trajectories=8, n_eval=0
        )
