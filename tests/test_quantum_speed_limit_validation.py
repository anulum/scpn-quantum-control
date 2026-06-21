# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Validation tests for the quantum speed limit
"""Parameter-validation tests for the quantum speed-limit estimator."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.analysis.quantum_speed_limit import _validate_qsl_parameters


def test_rejects_non_finite_t_target() -> None:
    """A non-finite target time is rejected."""
    with pytest.raises(ValueError, match="t_target must be finite"):
        _validate_qsl_parameters(np.inf, 0.1, 0.5)


def test_rejects_non_finite_dt() -> None:
    """A non-finite time step is rejected."""
    with pytest.raises(ValueError, match="dt must be finite"):
        _validate_qsl_parameters(1.0, np.inf, 0.5)


def test_rejects_non_finite_threshold() -> None:
    """A non-finite order-parameter threshold is rejected."""
    with pytest.raises(ValueError, match="R_threshold must be finite"):
        _validate_qsl_parameters(1.0, 0.1, np.inf)
