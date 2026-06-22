# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the K_nm partitioner
"""Guard tests for the quantum/classical K_nm partitioner."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.cosimulation.knm_partition import _validate, partition_knm


def test_validate_rejects_non_finite_omega() -> None:
    """A non-finite frequency vector is rejected."""
    with pytest.raises(ValueError, match="omega must be finite"):
        _validate(np.eye(2, dtype=np.float64), np.array([0.1, np.inf], dtype=np.float64))


def test_partition_rejects_negative_coupling_threshold() -> None:
    """A negative coupling threshold is rejected."""
    with pytest.raises(ValueError, match="coupling_threshold must be non-negative"):
        partition_knm(
            np.eye(2, dtype=np.float64),
            np.array([0.1, 0.2], dtype=np.float64),
            max_quantum_nodes=1,
            coupling_threshold=-1.0,
        )
