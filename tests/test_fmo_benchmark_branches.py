# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the FMO photosynthesis benchmark
"""Guard tests for the FMO complex benchmark helpers."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.applications.fmo_benchmark import (
    _finite_correlation,
    _validated_frequency_vector,
)


def test_frequency_vector_rejects_non_finite() -> None:
    """A non-finite frequency vector is rejected."""
    with pytest.raises(ValueError, match="must contain only finite values"):
        _validated_frequency_vector(np.array([0.1, np.inf], dtype=np.float64), 2, "freqs", "K")


def test_finite_correlation_maps_nan_to_zero() -> None:
    """A NaN correlation is mapped to zero."""
    assert _finite_correlation(float("nan")) == 0.0
