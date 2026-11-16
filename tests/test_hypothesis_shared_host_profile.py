# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Hypothesis shared-host profile tests
"""The suite's Hypothesis profile keeps host timing out of property-test verdicts."""

from __future__ import annotations

from hypothesis import HealthCheck, settings


def test_shared_host_profile_is_the_active_default() -> None:
    """The session runs with no deadline and with the too-slow health check suppressed."""
    active = settings()
    assert active.deadline is None
    assert HealthCheck.too_slow in active.suppress_health_check
    assert settings.get_profile("shared-host").deadline is None


def test_profile_keeps_the_example_count_of_hypothesis() -> None:
    """The profile changes timing only, so the default example count is unchanged."""
    assert settings().max_examples == settings.get_profile("default").max_examples


def test_per_test_settings_inherit_the_profile() -> None:
    """A test that sets only its example count keeps the profile's timing policy."""
    per_test = settings(max_examples=20)
    assert per_test.max_examples == 20
    assert per_test.deadline is None
    assert HealthCheck.too_slow in per_test.suppress_health_check
