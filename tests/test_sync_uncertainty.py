# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Synchronisation uncertainty tests
"""Tests for shot-noise uncertainty quantification of synchronisation metrics."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from scpn_quantum_control.analysis.sync_order_parameter import SyncOrderParameter
from scpn_quantum_control.analysis.sync_uncertainty import (
    UncertaintyInterval,
    metric_bootstrap,
    order_parameter_bootstrap,
    order_parameter_estimate,
    order_parameter_shot_noise,
)


def _binary_counts(n_qubits: int, p_one: float, n_shots: int, seed: int) -> dict[str, int]:
    """Sample counts where each qubit is independently ``1`` with probability ``p_one``."""
    rng = np.random.default_rng(seed)
    counts: dict[str, int] = {}
    for _ in range(n_shots):
        bits = "".join("1" if rng.random() < p_one else "0" for _ in range(n_qubits))
        counts[bits] = counts.get(bits, 0) + 1
    return counts


# --- point estimate fidelity ---------------------------------------------------


def test_point_estimate_matches_sync_order_parameter() -> None:
    """The interval point estimate equals the public Z-magnetisation proxy."""
    counts = {"000": 60, "111": 40, "010": 20}
    sync_result = SyncOrderParameter()(counts)
    expected = sync_result["sync_order_z_magnetisation"]
    assert order_parameter_estimate(counts) == pytest.approx(expected)
    assert sync_result["is_xy_kuramoto_order_parameter"] == 0.0


def test_point_estimate_all_aligned_is_one() -> None:
    """A fully aligned register has order parameter one."""
    assert order_parameter_estimate({"0000": 512}) == pytest.approx(1.0)


def test_point_estimate_balanced_is_zero() -> None:
    """Opposite aligned states in equal weight cancel to zero."""
    assert order_parameter_estimate({"000": 50, "111": 50}) == pytest.approx(0.0)


# --- analytic shot-noise interval ----------------------------------------------


def test_shot_noise_matches_hand_computation() -> None:
    """Standard error equals the closed-form Bessel-corrected shot-noise value."""
    counts = {"000": 75, "111": 25}  # per-shot magnetisation +1 (x75), -1 (x25)
    result = order_parameter_shot_noise(counts, coverage=0.95)

    mags = np.array([1.0, -1.0])
    weights = np.array([75, 25])
    n = 100
    mean = float(np.dot(mags, weights) / n)
    pop_var = float(np.dot(weights, (mags - mean) ** 2) / n)
    se = float(np.sqrt(pop_var * n / (n - 1) / n))

    assert result.point == pytest.approx(abs(mean))
    assert result.standard_error == pytest.approx(se)
    assert result.method == "shot-noise-delta"
    assert result.n_shots == 100
    assert result.n_resamples == 0


def test_shot_noise_interval_orders_low_point_high() -> None:
    """The coverage interval brackets the point estimate."""
    result = order_parameter_shot_noise({"000": 70, "111": 30})
    assert result.low <= result.point <= result.high
    assert result.width >= 0.0


def test_shot_noise_wider_coverage_is_wider_interval() -> None:
    """A higher coverage target yields a wider interval."""
    counts = {"000": 60, "111": 40}
    narrow = order_parameter_shot_noise(counts, coverage=0.80)
    wide = order_parameter_shot_noise(counts, coverage=0.99)
    assert wide.width > narrow.width


def test_shot_noise_straddling_zero_floors_at_zero() -> None:
    """When the mean interval straddles zero the order-parameter low bound is zero."""
    result = order_parameter_shot_noise({"000": 50, "111": 50}, coverage=0.95)
    assert result.low == 0.0
    assert result.high > 0.0


def test_shot_noise_zero_variance_has_zero_error() -> None:
    """A single observed magnetisation value has no shot-noise spread."""
    result = order_parameter_shot_noise({"0000": 256})
    assert result.standard_error == pytest.approx(0.0)
    assert result.low == pytest.approx(1.0)
    assert result.high == pytest.approx(1.0)


def test_shot_noise_rejects_single_shot() -> None:
    """A variance needs at least two shots."""
    with pytest.raises(ValueError, match="at least 2 shots"):
        order_parameter_shot_noise({"00": 1})


def test_shot_noise_rejects_bad_coverage() -> None:
    """Coverage must lie in the open unit interval."""
    with pytest.raises(ValueError, match="coverage must lie in"):
        order_parameter_shot_noise({"00": 10}, coverage=1.0)


# --- input validation ----------------------------------------------------------


def test_rejects_empty_counts() -> None:
    """Empty counts cannot be quantified."""
    with pytest.raises(ValueError, match="at least one measured bitstring"):
        order_parameter_estimate({})


def test_rejects_zero_total_shots() -> None:
    """A counts mapping that sums to zero shots is rejected."""
    with pytest.raises(ValueError, match="at least one shot"):
        order_parameter_estimate({"00": 0})


def test_rejects_negative_counts() -> None:
    """Negative shot weights are rejected."""
    with pytest.raises(ValueError, match="non-negative integers"):
        order_parameter_estimate({"00": -3})


def test_rejects_ragged_bitstrings() -> None:
    """Bitstrings of differing width are rejected."""
    with pytest.raises(ValueError, match="same qubit count"):
        order_parameter_estimate({"00": 10, "111": 5})


def test_rejects_non_binary_bitstrings() -> None:
    """Bitstrings must be made of '0' and '1' only."""
    with pytest.raises(ValueError, match="only '0' and '1'"):
        order_parameter_estimate({"0x": 10})


def test_rejects_empty_bitstring() -> None:
    """A zero-qubit bitstring is rejected."""
    with pytest.raises(ValueError, match="at least one qubit"):
        order_parameter_estimate({"": 10})


# --- bootstrap interval --------------------------------------------------------


def test_bootstrap_is_deterministic_for_fixed_seed() -> None:
    """A fixed seed reproduces the interval exactly."""
    counts = {"000": 70, "111": 30}
    first = order_parameter_bootstrap(counts, seed=7, n_resamples=500)
    second = order_parameter_bootstrap(counts, seed=7, n_resamples=500)
    assert (first.low, first.high, first.standard_error) == (
        second.low,
        second.high,
        second.standard_error,
    )


def test_bootstrap_point_matches_estimate_and_orders() -> None:
    """The bootstrap point matches the estimator and sits inside the interval."""
    counts = {"0000": 80, "1111": 20, "0011": 40}
    result = order_parameter_bootstrap(counts, n_resamples=1000)
    assert result.point == pytest.approx(order_parameter_estimate(counts))
    assert result.low <= result.point <= result.high
    assert result.method == "bootstrap-percentile"
    assert result.n_resamples == 1000


def test_bootstrap_rejects_non_positive_resamples() -> None:
    """At least one resample is required."""
    with pytest.raises(ValueError, match="n_resamples must be a positive integer"):
        order_parameter_bootstrap({"00": 10}, n_resamples=0)


def test_bootstrap_rejects_bad_coverage() -> None:
    """Coverage must lie in the open unit interval."""
    with pytest.raises(ValueError, match="coverage must lie in"):
        order_parameter_bootstrap({"00": 10}, coverage=0.0)


def test_bootstrap_agrees_with_delta_method_away_from_zero() -> None:
    """Far from zero, bootstrap and delta-method standard errors agree closely."""
    counts = _binary_counts(n_qubits=3, p_one=0.15, n_shots=4000, seed=11)
    delta = order_parameter_shot_noise(counts)
    boot = order_parameter_bootstrap(counts, n_resamples=4000, seed=3)
    assert boot.standard_error == pytest.approx(delta.standard_error, rel=0.20)


def test_bootstrap_empirical_coverage_is_close_to_target() -> None:
    """Across independent shot records the interval covers the truth near the target.

    This is the certified-coverage evidence: synthetic registers are drawn from a
    fixed single-qubit-flip probability, giving a known true order parameter, and
    the 90% bootstrap interval must cover it on roughly 90% of trials.
    """
    n_qubits, p_one, n_shots = 4, 0.2, 800
    true_value = abs(1.0 - 2.0 * p_one)  # |E[mean spin]| = |1 - 2 p_one|
    covered = 0
    trials = 120
    for trial in range(trials):
        counts = _binary_counts(n_qubits, p_one, n_shots, seed=1000 + trial)
        interval = order_parameter_bootstrap(counts, coverage=0.90, n_resamples=400, seed=trial)
        if interval.low <= true_value <= interval.high:
            covered += 1
    assert 0.80 <= covered / trials <= 0.99


# --- generic metric bootstrap (witnesses and any count metric) -----------------


def _fraction_all_zero(counts: Mapping[str, int]) -> float:
    """Fraction of shots in the all-zero outcome — a simple count-to-scalar metric."""
    total = sum(counts.values())
    if total == 0:
        return 0.0
    width = len(next(iter(counts)))
    return counts.get("0" * width, 0) / total


def test_metric_bootstrap_brackets_observed_value() -> None:
    """The generic bootstrap brackets the metric value on the observed counts."""
    counts = {"000": 70, "111": 20, "001": 10}
    result = metric_bootstrap(counts, _fraction_all_zero, n_resamples=800, seed=5)
    assert result.point == pytest.approx(0.70)
    assert result.low <= result.point <= result.high
    assert result.method == "bootstrap-percentile"


def test_metric_bootstrap_is_deterministic() -> None:
    """A fixed seed reproduces the generic bootstrap interval."""
    counts = {"00": 40, "11": 60}
    first = metric_bootstrap(counts, _fraction_all_zero, seed=2, n_resamples=300)
    second = metric_bootstrap(counts, _fraction_all_zero, seed=2, n_resamples=300)
    assert (first.low, first.high) == (second.low, second.high)


def test_metric_bootstrap_validates_inputs() -> None:
    """The generic bootstrap shares the coverage and resample guards."""
    with pytest.raises(ValueError, match="n_resamples must be a positive integer"):
        metric_bootstrap({"00": 10}, _fraction_all_zero, n_resamples=-1)
    with pytest.raises(ValueError, match="coverage must lie in"):
        metric_bootstrap({"00": 10}, _fraction_all_zero, coverage=2.0)


def test_uncertainty_interval_width_property() -> None:
    """The width property returns ``high - low``."""
    interval = UncertaintyInterval(0.5, 0.1, 0.4, 0.65, 0.95, "shot-noise-delta", 100, 0)
    assert interval.width == pytest.approx(0.25)
