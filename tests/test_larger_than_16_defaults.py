# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — larger-than-16 default input tests
"""Regression tests for scalable default Kuramoto inputs."""

from __future__ import annotations

from typing import Any

import numpy as np

from scpn_quantum_control.hardware.classical import classical_kuramoto_reference


def test_classical_kuramoto_default_frequencies_scale_past_16() -> None:
    """Default classical inputs build a full frequency vector for N > 16."""
    result: dict[str, Any] = classical_kuramoto_reference(20, t_max=0.1, dt=0.05)

    theta = np.asarray(result["theta"], dtype=np.float64)
    order_parameter = np.asarray(result["R"], dtype=np.float64)
    assert theta.shape == (3, 20)
    assert order_parameter.shape == (3,)
    assert np.all(np.isfinite(order_parameter))
