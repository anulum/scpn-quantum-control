# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — Shared test fixtures
"""Shared fixtures for the oscillatools test suite."""

from __future__ import annotations

import numpy as np
import pytest

DT_VALUES = [0.01, 0.05, 0.1]


@pytest.fixture(params=DT_VALUES, ids=lambda dt: f"dt={dt}")
def dt(request: pytest.FixtureRequest) -> float:
    """Time step from {0.01, 0.05, 0.1}."""
    return float(request.param)


@pytest.fixture
def rng() -> np.random.Generator:
    """A reproducible NumPy random generator."""
    return np.random.default_rng(42)
