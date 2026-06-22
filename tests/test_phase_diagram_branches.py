# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the phase-diagram helpers
"""Guard tests for the critical-coupling and decoherence-temperature helpers."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.phase_diagram import (
    critical_coupling_finite_graph,
    decoherence_temperature,
)

_OMEGA = np.array([0.1, 0.5], dtype=np.float64)


def test_critical_coupling_rejects_non_finite_fiedler() -> None:
    """A non-finite Fiedler value is rejected."""
    with pytest.raises(ValueError, match="fiedler must be finite"):
        critical_coupling_finite_graph(_OMEGA, float("nan"))


def test_decoherence_temperature_rejects_nan_t1() -> None:
    """A NaN T1 decoherence time is rejected."""
    with pytest.raises(ValueError, match="t1 must be finite or infinite"):
        decoherence_temperature(float("nan"), 1.0)


def test_decoherence_temperature_rejects_nan_t2() -> None:
    """A NaN T2 decoherence time is rejected."""
    with pytest.raises(ValueError, match="t2 must be finite or infinite"):
        decoherence_temperature(1.0, float("nan"))
