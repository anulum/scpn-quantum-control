# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the biological QEC batch pipeline
"""Guard tests for the biological QEC batch execution z-error matrix checks."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.qec.biological_pipeline import run_biological_qec_batch_execution

_K = np.array([[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0]], dtype=np.float64)


def test_batch_rejects_non_two_dimensional_matrix() -> None:
    """A one-dimensional z-error input is rejected."""
    with pytest.raises(ValueError, match="z_error_matrix must be a two-dimensional array"):
        run_biological_qec_batch_execution(_K, np.zeros(3, dtype=np.int8))


def test_batch_rejects_empty_matrix() -> None:
    """A z-error matrix with no rows is rejected."""
    with pytest.raises(ValueError, match="z_error_matrix must contain at least one row"):
        run_biological_qec_batch_execution(_K, np.zeros((0, 3), dtype=np.int8))
