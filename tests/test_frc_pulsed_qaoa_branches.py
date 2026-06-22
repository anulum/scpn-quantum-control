# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the FRC pulsed QAOA solver
"""Guard tests for the FRC pulsed-QAOA scheduling solver."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from scpn_quantum_control.control.frc_pulsed_qaoa import (
    enumerate_costs,
    solve_frc_pulsed_qaoa,
)
from scpn_quantum_control.control.qaoa_pulsed_cost import FRCQAOAObjective


def _objective() -> FRCQAOAObjective:
    return FRCQAOAObjective(
        target_s_parameter=2.5,
        bank_energy_budget_J=5.0e5,
        mrti_amplitude_max_m=1.0e-2,
        tilt_margin_required=0.3,
    )


def _zero_cost(_schedule: NDArray[np.float64]) -> float:
    return 0.0


def test_enumerate_costs_rejects_out_of_range_horizon() -> None:
    """A horizon outside [1, 16] is rejected."""
    with pytest.raises(ValueError, match=r"horizon must be an integer in \[1, 16\]"):
        enumerate_costs(0, _zero_cost)


def test_solve_rejects_out_of_range_p_layers() -> None:
    """A QAOA depth outside [1, 8] is rejected."""
    with pytest.raises(ValueError, match=r"p_layers must lie in \[1, 8\]"):
        solve_frc_pulsed_qaoa(np.ones(2, dtype=np.float64), 1.0e6, _objective(), p_layers=0)


def test_solve_rejects_non_positive_restarts() -> None:
    """A non-positive restart count is rejected."""
    with pytest.raises(ValueError, match="restarts must be a positive integer"):
        solve_frc_pulsed_qaoa(np.ones(2, dtype=np.float64), 1.0e6, _objective(), restarts=0)
