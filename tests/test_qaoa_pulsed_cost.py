# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the FRC pulsed-shot QAOA cost
"""Tests for control/qaoa_pulsed_cost.py and control/frc_pulsed_qaoa.py."""

from typing import Any

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from scpn_quantum_control.control.frc_pulsed_qaoa import (
    classical_sqp_schedule,
    optimal_schedule,
    solve_frc_pulsed_qaoa,
)
from scpn_quantum_control.control.qaoa_pulsed_cost import (
    FRCPlasmaSurrogate,
    FRCQAOAObjective,
    _mrti_growth_numpy,
    decode_schedule_to_field,
    frc_pulsed_shot_cost,
)

_engine: Any = None
try:
    import scpn_quantum_engine as _loaded_engine

    _engine = _loaded_engine
    _HAS_RUST = hasattr(_engine, "frc_mrti_growth")
except ImportError:
    _HAS_RUST = False


def _objective(**kw: float) -> FRCQAOAObjective:
    base: dict[str, float] = dict(
        target_s_parameter=2.5,
        bank_energy_budget_J=5.0e5,
        mrti_amplitude_max_m=1.0e-2,
        tilt_margin_required=0.3,
    )
    base.update(kw)
    return FRCQAOAObjective(**base)


_TARGET8 = np.linspace(0.5, 4.0, 8)


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "kw",
    [
        {"target_s_parameter": 0.0},
        {"bank_energy_budget_J": -1.0},
        {"mrti_amplitude_max_m": 0.0},
        {"tilt_margin_required": -0.1},
        {"weight_s": -1.0},
    ],
)
def test_objective_rejects_bad_params(kw: dict[str, float]) -> None:
    """Reject non-positive objective targets and negative penalty weights."""
    with pytest.raises(ValueError):
        _objective(**kw)


@pytest.mark.parametrize(
    "kw", [{"atwood_number": 0.0}, {"atwood_number": 1.5}, {"elongation": 0.0}]
)
def test_surrogate_rejects_bad_params(kw: dict[str, float]) -> None:
    """Reject invalid Atwood numbers and non-positive surrogate scales."""
    with pytest.raises(ValueError):
        FRCPlasmaSurrogate(**kw)


def test_decode_schedule_to_field() -> None:
    """Decode cumulative bank firings and reject an empty schedule."""
    field = decode_schedule_to_field(np.array([1, 0, 1, 1]), delta_field_T=0.5)
    assert np.allclose(field, [0.5, 0.5, 1.0, 1.5])
    with pytest.raises(ValueError):
        decode_schedule_to_field(np.array([]), delta_field_T=0.5)


def test_cost_shape_mismatch_raises() -> None:
    """Reject a target field whose length differs from the schedule."""
    with pytest.raises(ValueError, match="match the schedule length"):
        frc_pulsed_shot_cost(np.ones(8), np.ones(7), 1.0e6, _objective())


# --------------------------------------------------------------------------- #
# Physics behaviour
# --------------------------------------------------------------------------- #
def test_cost_components_present_and_finite() -> None:
    """Return finite diagnostics for a valid pulsed-shot schedule."""
    cost, comp = frc_pulsed_shot_cost(
        np.ones(8), _TARGET8, 1.0e6, _objective(), return_components=True
    )
    assert cost >= 0.0 and np.isfinite(cost)
    for key in ("s_achieved", "energy_used_J", "mrti_amplitude_m", "tilt_margin", "peak_field_T"):
        assert np.isfinite(comp[key])
    assert comp["peak_field_T"] == pytest.approx(4.0)  # 8 banks * 0.5 T


def test_energy_penalty_activates_over_budget() -> None:
    """Activate the energy penalty when fired banks exceed the budget."""
    obj = _objective(bank_energy_budget_J=3.0e5, weight_energy=1.0)  # 3 banks worth
    _, comp = frc_pulsed_shot_cost(np.ones(8), _TARGET8, 1.0e6, obj, return_components=True)
    assert comp["energy_penalty"] > 0.0  # 8 banks fired exceeds the 3-bank budget


def test_s_parameter_monotonic_in_peak_field() -> None:
    """Increase the surrogate s-parameter monotonically with peak field."""
    surr = FRCPlasmaSurrogate()
    assert surr.s_parameter(4.0) > surr.s_parameter(2.0) > surr.s_parameter(0.0) == 0.0


def test_tilt_margin_decreases_with_s() -> None:
    """Decrease the kinetic tilt margin as the s-parameter increases."""
    surr = FRCPlasmaSurrogate()
    assert surr.tilt_margin(2.0) > surr.tilt_margin(10.0)


@settings(max_examples=30, deadline=None)
@given(
    extra=st.floats(min_value=0.0, max_value=5.0),
    weight=st.sampled_from(["weight_s", "weight_energy", "weight_mrti", "weight_tilt"]),
)
def test_cost_monotonic_nondecreasing_in_each_weight(extra: float, weight: str) -> None:
    """Keep total cost non-decreasing in every non-negative penalty weight."""
    schedule = np.array([1, 1, 0, 1, 0, 1, 1, 0], dtype=float)
    low = frc_pulsed_shot_cost(schedule, _TARGET8, 1.0e6, _objective(**{weight: 0.5}))
    high = frc_pulsed_shot_cost(schedule, _TARGET8, 1.0e6, _objective(**{weight: 0.5 + extra}))
    assert high >= low - 1e-9


# --------------------------------------------------------------------------- #
# Optimisers
# --------------------------------------------------------------------------- #
def test_classical_sqp_reaches_bruteforce_optimum() -> None:
    """Keep the classical schedule within five percent of exhaustive search."""
    opt = optimal_schedule(_TARGET8, 1.0e6, _objective())
    sqp = classical_sqp_schedule(_TARGET8, 1.0e6, _objective(), seed=1)
    assert sqp.cost <= opt.cost * 1.05


_QAOA_CASES = [
    (np.linspace(0.5, 4.0, 8), _objective()),
    (np.linspace(1.0, 3.0, 7), _objective(target_s_parameter=2.0, mrti_amplitude_max_m=8.0e-3)),
    (np.linspace(0.2, 5.0, 6), _objective(target_s_parameter=3.0, bank_energy_budget_J=3.0e5)),
]


@pytest.mark.parametrize("idx", range(len(_QAOA_CASES)))
def test_qaoa_within_five_percent_of_optimal(idx: int) -> None:
    """Keep sampled QAOA schedules near the exhaustive optimum."""
    target, obj = _QAOA_CASES[idx]
    opt = optimal_schedule(target, 1.0e6, obj)
    qaoa = solve_frc_pulsed_qaoa(target, 1.0e6, obj, p_layers=4, restarts=8, seed=idx)
    assert qaoa.cost <= opt.cost * 1.05


def test_qaoa_beats_mean_schedule() -> None:
    """Beat the mean cost over the complete binary schedule space."""
    target, obj = _QAOA_CASES[0]
    import itertools

    all_costs = [
        frc_pulsed_shot_cost(np.array(s, dtype=float), target, 1.0e6, obj)
        for s in itertools.product([0, 1], repeat=target.size)
    ]
    qaoa = solve_frc_pulsed_qaoa(target, 1.0e6, obj, p_layers=4, restarts=8, seed=0)
    assert qaoa.cost < float(np.mean(all_costs))


# --------------------------------------------------------------------------- #
# Rust ↔ NumPy parity
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not _HAS_RUST, reason="scpn_quantum_engine frc kernel not built")
@settings(max_examples=40, deadline=None)
@given(
    fields=st.lists(st.floats(min_value=0.0, max_value=10.0), min_size=2, max_size=256),
)
def test_mrti_growth_rust_parity(fields: list[float]) -> None:
    """Match native MRTI growth to the NumPy reference over generated fields."""
    field = np.array(fields, dtype=np.float64)
    rust = float(
        _engine.frc_mrti_growth(np.ascontiguousarray(field), 1.0e-6, 125.0, 0.9, 200.0, 2.0, 700.0)
    )
    python = _mrti_growth_numpy(field, 1.0e-6, 125.0, 0.9, 200.0, 2.0)
    assert rust == pytest.approx(python, rel=1e-12, abs=1e-12)


@pytest.mark.skipif(not _HAS_RUST, reason="scpn_quantum_engine frc kernel not built")
def test_mrti_growth_dispatch_used_in_cost() -> None:
    """Match the native dispatch result to the NumPy MRTI implementation."""
    surr = FRCPlasmaSurrogate()
    field = decode_schedule_to_field(np.ones(8), delta_field_T=0.5)
    k = 2.0 * np.pi / surr.perturbation_wavelength_m
    rust = float(
        _engine.frc_mrti_growth(
            np.ascontiguousarray(field),
            1.0e-6,
            k,
            surr.atwood_number,
            surr.areal_mass_kg_per_m2,
            surr.plasma_mass_density_kg_per_m3,
            700.0,
        )
    )
    python = _mrti_growth_numpy(
        field,
        1.0e-6,
        k,
        surr.atwood_number,
        surr.areal_mass_kg_per_m2,
        surr.plasma_mass_density_kg_per_m3,
    )
    assert rust == pytest.approx(python, rel=1e-12)
