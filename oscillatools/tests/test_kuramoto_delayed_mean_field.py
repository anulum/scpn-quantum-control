# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the time-delayed Kuramoto mean-field theory
r"""Tests for :mod:`oscillatools.accel.kuramoto_delayed_mean_field`.

The collective-frequency branches of the delayed mean-field model are checked against the
Yeung–Strogatz self-consistency ``Ω = ω₀ − K sin(Ω τ)``: every returned root cancels the
residual and lies in the analytic band ``[ω₀ − |K|, ω₀ + |K|]``; a single root at small ``K τ``
splits into several at large ``K τ`` (delay-induced multistability); and the stability indicator
``1 + K τ cos(Ω τ)`` partitions the branches into the interleaved stable / unstable sets.
"""

from __future__ import annotations

import numpy as np
import pytest

from oscillatools.accel.kuramoto_delayed_mean_field import (
    is_synchronised_branch_stable,
    stable_synchronised_frequencies,
    synchronised_branch_stability,
    synchronised_frequency_residual,
    synchronised_frequency_roots,
)

# --------------------------------------------------------------------------- residual


def test_residual_matches_definition() -> None:
    omega0, coupling, delay, frequency = 1.0, 2.0, 1.5, 0.4
    expected = omega0 - coupling * np.sin(frequency * delay) - frequency
    assert synchronised_frequency_residual(frequency, omega0, coupling, delay) == pytest.approx(
        expected
    )


def test_residual_vanishes_at_a_root() -> None:
    omega0, coupling, delay = 1.0, 3.0, 1.2
    for root in synchronised_frequency_roots(omega0, coupling, delay):
        assert abs(synchronised_frequency_residual(root, omega0, coupling, delay)) < 1e-9


# --------------------------------------------------------------------------- roots


def test_zero_coupling_has_single_root_at_natural_frequency() -> None:
    roots = synchronised_frequency_roots(2.5, 0.0, 1.0)
    assert roots.shape == (1,)
    assert roots[0] == pytest.approx(2.5)


def test_weak_coupling_has_a_single_root() -> None:
    # K·τ = 0.45 < 1: the self-consistency is monotone, one branch.
    roots = synchronised_frequency_roots(1.0, 0.3, 1.5)
    assert roots.shape == (1,)


def test_strong_coupling_is_multistable() -> None:
    # K·τ = 7.5 ≫ 1: several coexisting branches.
    roots = synchronised_frequency_roots(1.0, 5.0, 1.5)
    assert roots.size >= 3


def test_root_count_grows_with_delay_coupling_product() -> None:
    counts = [synchronised_frequency_roots(1.0, k, 1.5).size for k in (0.3, 1.0, 3.0, 5.0)]
    assert counts == sorted(counts)
    assert counts[0] == 1
    assert counts[-1] > counts[0]


def test_all_roots_lie_in_the_analytic_band() -> None:
    omega0, coupling, delay = 0.5, 4.0, 2.0
    roots = synchronised_frequency_roots(omega0, coupling, delay)
    assert np.all(roots >= omega0 - abs(coupling) - 1e-9)
    assert np.all(roots <= omega0 + abs(coupling) + 1e-9)


def test_roots_are_sorted_and_distinct() -> None:
    roots = synchronised_frequency_roots(1.0, 5.0, 1.5)
    assert np.all(np.diff(roots) > 0.0)


def test_negative_coupling_is_handled() -> None:
    omega0, coupling, delay = 1.0, -5.0, 1.5
    roots = synchronised_frequency_roots(omega0, coupling, delay)
    assert roots.size >= 3
    for root in roots:
        assert abs(synchronised_frequency_residual(root, omega0, coupling, delay)) < 1e-9


def test_roots_reject_non_positive_delay() -> None:
    with pytest.raises(ValueError, match="delay must be positive"):
        synchronised_frequency_roots(1.0, 2.0, 0.0)


def test_roots_reject_too_few_scan_points() -> None:
    with pytest.raises(ValueError, match="scan_points must be at least 2"):
        synchronised_frequency_roots(1.0, 2.0, 1.5, scan_points=1)


# --------------------------------------------------------------------------- stability


def test_stability_indicator_matches_definition() -> None:
    coupling, delay, frequency = 5.0, 1.5, 0.2
    expected = 1.0 + coupling * delay * np.cos(frequency * delay)
    assert synchronised_branch_stability(frequency, coupling, delay) == pytest.approx(expected)


def test_is_stable_agrees_with_indicator_sign() -> None:
    coupling, delay = 5.0, 1.5
    for frequency in np.linspace(-4.0, 4.0, 17):
        indicator = synchronised_branch_stability(frequency, coupling, delay)
        assert is_synchronised_branch_stable(frequency, coupling, delay) == (indicator > 0.0)


def test_weak_coupling_single_root_is_stable() -> None:
    # For K·τ < 1 the stability indicator 1 + K·τ·cos(Ω·τ) ≥ 1 − K·τ > 0 on every branch.
    omega0, coupling, delay = 1.0, 0.3, 1.5
    (root,) = synchronised_frequency_roots(omega0, coupling, delay)
    assert is_synchronised_branch_stable(root, coupling, delay)


# --------------------------------------------------------------------------- stable branches


def test_stable_branches_are_a_subset_of_all_roots() -> None:
    omega0, coupling, delay = 1.0, 5.0, 1.5
    roots = synchronised_frequency_roots(omega0, coupling, delay)
    stable = stable_synchronised_frequencies(omega0, coupling, delay)
    assert set(np.round(stable, 9)).issubset(set(np.round(roots, 9)))
    for frequency in stable:
        assert is_synchronised_branch_stable(frequency, coupling, delay)


def test_multistable_regime_has_several_stable_branches() -> None:
    stable = stable_synchronised_frequencies(1.0, 5.0, 1.5)
    assert stable.size >= 2


def test_stable_and_unstable_branches_interleave() -> None:
    omega0, coupling, delay = 1.0, 5.0, 1.5
    roots = synchronised_frequency_roots(omega0, coupling, delay)
    flags = [is_synchronised_branch_stable(r, coupling, delay) for r in roots]
    # Adjacent branches alternate stability (no two consecutive roots share a stability label).
    assert all(flags[i] != flags[i + 1] for i in range(len(flags) - 1))


def test_stable_branches_forward_scan_resolution() -> None:
    # A coarse scan still resolves the well-separated branches at this K·τ; the kwargs are wired.
    stable = stable_synchronised_frequencies(1.0, 5.0, 1.5, scan_points=2001, root_tolerance=1e-10)
    assert stable.size >= 2
    for frequency in stable:
        assert abs(synchronised_frequency_residual(frequency, 1.0, 5.0, 1.5)) < 1e-8


def test_weak_coupling_single_stable_branch() -> None:
    stable = stable_synchronised_frequencies(1.0, 0.3, 1.5)
    assert stable.shape == (1,)
