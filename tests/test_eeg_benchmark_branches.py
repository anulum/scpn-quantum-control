# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the EEG benchmark
"""Guard and branch tests for the EEG functional-connectivity benchmark.

Covers the frequency-vector finiteness guard, the paired-input requirement and
the measured-data branch with supplied EEG coupling and frequencies.
"""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.applications.eeg_benchmark import (
    _validated_frequency_vector,
    eeg_benchmark,
)

_K = np.array([[0.0, 0.3], [0.3, 0.0]], dtype=np.float64)
_OMEGA = np.array([0.1, 0.2], dtype=np.float64)


def test_frequency_vector_rejects_non_finite() -> None:
    """A non-finite frequency vector is rejected."""
    with pytest.raises(ValueError, match="must contain only finite values"):
        _validated_frequency_vector(np.array([0.1, np.inf], dtype=np.float64), 2, "freqs", "K")


def test_benchmark_requires_paired_eeg_inputs() -> None:
    """Supplying EEG coupling without frequencies is rejected."""
    with pytest.raises(ValueError, match="must be supplied together"):
        eeg_benchmark(_K, _OMEGA, eeg_coupling=_K, eeg_frequencies=None)


def test_benchmark_measured_branch_with_supplied_inputs() -> None:
    """Supplying both EEG coupling and frequencies yields a measured-source result."""
    eeg_coupling = np.array([[0.0, 0.5], [0.5, 0.0]], dtype=np.float64)
    eeg_frequencies = np.array([10.0, 11.0], dtype=np.float64)
    result = eeg_benchmark(_K, _OMEGA, eeg_coupling=eeg_coupling, eeg_frequencies=eeg_frequencies)
    assert result.source_mode == "measured"
    assert result.publication_safe is True
