# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for Identity Coherence Budget
"""Tests for identity/coherence_budget.py."""

import pytest

from scpn_quantum_control.identity.coherence_budget import coherence_budget, fidelity_at_depth


def test_fidelity_at_zero_depth():
    assert fidelity_at_depth(0, 4) == 1.0


def test_fidelity_decreases_with_depth():
    f1 = fidelity_at_depth(10, 4)
    f2 = fidelity_at_depth(100, 4)
    f3 = fidelity_at_depth(500, 4)
    assert f1 > f2 > f3
    assert f3 > 0.0


def test_fidelity_decreases_with_qubits():
    f4 = fidelity_at_depth(100, 4)
    f8 = fidelity_at_depth(100, 8)
    f16 = fidelity_at_depth(100, 16)
    assert f4 > f8 > f16


def test_fidelity_bounded_zero_one():
    for depth in [1, 10, 100, 500, 1000]:
        f = fidelity_at_depth(depth, 4)
        assert 0.0 <= f <= 1.0


def test_negative_depth_raises():
    with pytest.raises(ValueError, match="non-negative"):
        fidelity_at_depth(-1, 4)


def test_zero_qubits_raises():
    with pytest.raises(ValueError, match="n_qubits"):
        fidelity_at_depth(10, 0)


def test_coherence_budget_returns_valid_depth():
    result = coherence_budget(4, fidelity_threshold=0.5)
    assert result["max_depth"] >= 0
    assert result["fidelity_at_max"] >= 0.5


def test_budget_depth_decreases_with_qubits():
    r4 = coherence_budget(4, fidelity_threshold=0.5)
    r8 = coherence_budget(8, fidelity_threshold=0.5)
    r16 = coherence_budget(16, fidelity_threshold=0.5)
    assert r4["max_depth"] >= r8["max_depth"] >= r16["max_depth"]


def test_budget_depth_decreases_with_higher_threshold():
    r_low = coherence_budget(4, fidelity_threshold=0.3)
    r_high = coherence_budget(4, fidelity_threshold=0.8)
    assert r_low["max_depth"] >= r_high["max_depth"]


def test_budget_fidelity_curve_populated():
    result = coherence_budget(4, fidelity_threshold=0.5)
    assert len(result["fidelity_curve"]) >= 2
    for _depth, fid in result["fidelity_curve"].items():
        assert 0.0 <= fid <= 1.0


def test_budget_hardware_params_present():
    result = coherence_budget(4)
    hp = result["hardware_params"]
    assert "t1_us" in hp
    assert "t2_us" in hp
    assert "cz_error" in hp


def test_invalid_threshold_raises():
    with pytest.raises(ValueError, match="fidelity_threshold"):
        coherence_budget(4, fidelity_threshold=0.0)
    with pytest.raises(ValueError, match="fidelity_threshold"):
        coherence_budget(4, fidelity_threshold=1.0)


def test_worse_hardware_gives_shorter_budget():
    r_good = coherence_budget(4, fidelity_threshold=0.5, cz_error=0.001)
    r_bad = coherence_budget(4, fidelity_threshold=0.5, cz_error=0.05)
    assert r_good["max_depth"] > r_bad["max_depth"]
