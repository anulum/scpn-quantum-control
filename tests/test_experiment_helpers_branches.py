# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Branch tests for the hardware experiment helpers
"""Guard tests for the hardware experiment reduction helpers.

Each test drives one missing-data guard: absent counts or expectation values on
a job result and the count-consuming per-qubit, QAOA-cost and correlator
reductions.
"""

from __future__ import annotations

import pytest
from qiskit.quantum_info import SparsePauliOp

from scpn_quantum_control.hardware._experiment_helpers import (
    _correlator_from_counts,
    _expectation_per_qubit,
    _qaoa_cost_from_counts,
    _require_counts,
    _require_expectations,
)
from scpn_quantum_control.hardware.runner import JobResult


def _job_result() -> JobResult:
    return JobResult(
        job_id="job1",
        backend_name="sim",
        experiment_name="exp",
        metadata={},
    )


def test_require_counts_rejects_absent_counts() -> None:
    """A job result without counts is rejected."""
    with pytest.raises(ValueError, match="measurement counts are required"):
        _require_counts(_job_result())


def test_require_expectations_rejects_absent_values() -> None:
    """A job result without expectation values is rejected."""
    with pytest.raises(ValueError, match="expectation values are required"):
        _require_expectations(_job_result())


def test_expectation_per_qubit_requires_counts() -> None:
    """The per-qubit reduction requires measurement counts."""
    with pytest.raises(ValueError, match="measurement counts are required"):
        _expectation_per_qubit(None, 2)


def test_qaoa_cost_requires_counts() -> None:
    """The QAOA-cost reduction requires measurement counts."""
    with pytest.raises(ValueError, match="measurement counts are required"):
        _qaoa_cost_from_counts(None, SparsePauliOp("Z"), 1)


def test_correlator_requires_counts() -> None:
    """The correlator reduction requires measurement counts."""
    with pytest.raises(ValueError, match="measurement counts are required"):
        _correlator_from_counts(None, 0, 1)
