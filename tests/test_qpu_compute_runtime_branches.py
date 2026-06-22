# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the QPU compute runtime
"""Guard tests for the deterministic QPU compute runtime."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from scpn_quantum_control.bridge import artifact_from_arrays
from scpn_quantum_control.qpu_compute_runtime import (
    deterministic_counts,
    execute_simulator_request,
)
from scpn_quantum_control.qpu_compute_types import QPUComputeRequest


def test_deterministic_counts_rejects_non_positive_shots() -> None:
    """A non-positive shot count is rejected."""
    with pytest.raises(ValueError, match="shots must be >= 1"):
        deterministic_counts({"00": 1.0}, 0)


def test_execute_rejects_non_statevector_backend_policy() -> None:
    """A request with a non-statevector backend policy is rejected.

    The request dataclass only admits the statevector policy, so a hash-matching
    stand-in exercises the runtime's own fail-closed backend-policy guard.
    """
    artifact = artifact_from_arrays(
        domain="unit",
        source_name="unit-source",
        source_mode="curated",
        K_nm=np.array([[0.0, 0.25], [0.25, 0.0]], dtype=np.float64),
        omega=np.array([0.1, 0.2], dtype=np.float64),
        normalization="unit",
        extraction_method="unit-test",
    )
    foreign = cast(
        QPUComputeRequest,
        SimpleNamespace(
            qpu_data_artifact_sha256=artifact.to_dict()["artifact_sha256"],
            backend_policy="hardware_submission",
        ),
    )
    with pytest.raises(ValueError, match="execute_simulator_request only accepts"):
        execute_simulator_request(artifact, foreign)
