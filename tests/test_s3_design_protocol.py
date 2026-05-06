# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S3 design protocol
"""Tests for S3 pulse/ansatz design-readiness scoring."""

from __future__ import annotations

import numpy as np
import pytest

from scpn_quantum_control.benchmarks.s3_design_protocol import (
    default_s3_design_protocol,
    score_s3_candidates,
    validate_s3_design_rows,
)
from scpn_quantum_control.bridge.knm_hamiltonian import OMEGA_N_16, build_knm_paper27


def test_score_s3_candidates_emits_ansatz_and_pulse_rows() -> None:
    protocol = default_s3_design_protocol()
    rows = score_s3_candidates(
        protocol,
        build_knm_paper27(4),
        np.asarray(OMEGA_N_16[:4], dtype=np.float64),
    )

    assert {row.family for row in rows} == {"ansatz", "pulse"}
    assert all(row.status == "ok" for row in rows)
    assert all(row.score > 0.0 for row in rows)
    assert all(row.metrics["hardware_submission"] is False for row in rows)


def test_validate_s3_design_rows_rejects_missing_hardware_boundary() -> None:
    protocol = default_s3_design_protocol()
    rows = score_s3_candidates(
        protocol,
        build_knm_paper27(3),
        np.asarray(OMEGA_N_16[:3], dtype=np.float64),
    )
    bad = rows[0].to_dict()
    bad["metrics"] = {"depth": 3}

    with pytest.raises(ValueError, match="hardware_submission"):
        validate_s3_design_rows([bad], protocol=protocol)


def test_score_s3_candidates_rejects_invalid_problem_shape() -> None:
    with pytest.raises(ValueError, match="square"):
        score_s3_candidates(default_s3_design_protocol(), np.ones((2, 3)), np.ones(2))
