# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the ITER MHD benchmark
"""Guard tests for the ITER MHD mode-coupling benchmark."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.applications.iter_benchmark import (
    _validated_frequency_vector,
    iter_benchmark,
)


def test_frequency_vector_rejects_non_finite() -> None:
    """A non-finite frequency vector is rejected."""
    with pytest.raises(ValueError, match="must contain only finite values"):
        _validated_frequency_vector(np.array([0.1, np.inf], dtype=np.float64), 2, "freqs", "K")


def test_benchmark_rejects_unknown_reference_source_mode() -> None:
    """An unknown reference source mode is rejected for measured inputs."""
    eye = np.eye(2, dtype=np.float64)
    with pytest.raises(ValueError, match="reference_source_mode must be one of"):
        iter_benchmark(
            eye,
            np.zeros(2, dtype=np.float64),
            iter_coupling=eye,
            iter_frequencies=np.zeros(2, dtype=np.float64),
            reference_source_mode="bogus_mode",
        )
