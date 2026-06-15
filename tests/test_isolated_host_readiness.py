# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for isolated-benchmark host readiness
"""Tests for benchmarks/isolated_host_readiness.py."""

import pytest

from scpn_quantum_control.benchmarks.isolated_host_readiness import (
    MAX_ISOLATED_LOAD,
    HostReadiness,
    assess_host_readiness,
    capture_host_readiness,
)


def _assess(**overrides):
    kwargs = {
        "reserved_core": 0,
        "governor": "performance",
        "frequency_mhz": 3200.0,
        "load_average": (0.2, 0.3, 0.25),
    }
    kwargs.update(overrides)
    return assess_host_readiness(**kwargs)


def test_ready_host_has_no_blockers():
    readiness = _assess()
    assert readiness.ready
    assert readiness.blockers == ()
    assert readiness.governor_is_stable
    assert readiness.load_is_low


def test_powersave_governor_blocks():
    readiness = _assess(governor="powersave")
    assert not readiness.ready
    assert not readiness.governor_is_stable
    assert any("performance" in blocker for blocker in readiness.blockers)


def test_missing_governor_blocks():
    readiness = _assess(governor=None)
    assert not readiness.ready
    assert any("unreadable" in blocker for blocker in readiness.blockers)


def test_high_load_blocks():
    readiness = _assess(load_average=(2.5, 1.8, 1.2))
    assert not readiness.ready
    assert not readiness.load_is_low
    assert any("exceeds the isolated threshold" in blocker for blocker in readiness.blockers)


def test_load_exactly_at_threshold_is_low():
    readiness = _assess(load_average=(MAX_ISOLATED_LOAD, 0.5, 0.5))
    assert readiness.load_is_low
    assert readiness.ready


def test_missing_load_blocks():
    readiness = _assess(load_average=None)
    assert not readiness.ready
    assert any("load average is unavailable" in blocker for blocker in readiness.blockers)


def test_missing_frequency_blocks():
    readiness = _assess(frequency_mhz=None)
    assert not readiness.ready
    assert any("frequency is unreadable" in blocker for blocker in readiness.blockers)


def test_multiple_blockers_accumulate():
    readiness = _assess(governor="ondemand", load_average=(3.0, 3.0, 3.0), frequency_mhz=None)
    assert not readiness.ready
    assert len(readiness.blockers) == 3


def test_reserved_core_must_be_non_negative():
    with pytest.raises(ValueError):
        _assess(reserved_core=-1)


def test_capture_host_readiness_runs_on_local_host():
    # Capture must succeed and return a structurally valid verdict regardless of
    # whether this host happens to be isolation-ready.
    readiness = capture_host_readiness(0)
    assert isinstance(readiness, HostReadiness)
    assert readiness.reserved_core == 0
    assert isinstance(readiness.blockers, tuple)
    assert readiness.ready == (not readiness.blockers)
