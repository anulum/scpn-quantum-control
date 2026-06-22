# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Guard tests for the S3 no-QPU design protocol
"""Validation and serialisation tests for the S3 no-QPU design-ranking protocol.

Covers the candidate field guards and serialisers, the scored-row validation
ladder and the problem-input shape and finiteness guards.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scpn_quantum_control.benchmarks.s3_design_protocol import (
    S3DesignCandidate,
    _validate_problem,
    default_s3_design_protocol,
    validate_s3_design_rows,
)

_PROTOCOL_ID = default_s3_design_protocol().protocol_id


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "protocol_id": _PROTOCOL_ID,
        "candidate_label": "candidate-1",
        "family": "ansatz",
        "status": "ok",
        "score": 0.5,
        "metrics": {"hardware_submission": False},
        "claim_boundary": "no-QPU analytic proxy",
    }
    row.update(overrides)
    return row


def test_candidate_rejects_empty_label() -> None:
    """A candidate with an empty label is rejected."""
    with pytest.raises(ValueError, match="label must be non-empty"):
        S3DesignCandidate(label="", family="ansatz", parameters={"depth": 2})


def test_candidate_rejects_empty_parameters() -> None:
    """A candidate without parameters is rejected."""
    with pytest.raises(ValueError, match="parameters must be non-empty"):
        S3DesignCandidate(label="c", family="ansatz", parameters={})


def test_candidate_round_trips_to_dict() -> None:
    """A candidate serialises to a JSON-compatible dictionary."""
    candidate = S3DesignCandidate(label="c", family="pulse", parameters={"depth": 3})
    payload = candidate.to_dict()
    assert payload["label"] == "c"
    assert payload["family"] == "pulse"
    assert payload["parameters"] == {"depth": 3}


def test_protocol_round_trips_to_dict() -> None:
    """The default protocol serialises with its candidate list."""
    payload = default_s3_design_protocol().to_dict()
    assert payload["protocol_id"] == _PROTOCOL_ID
    assert isinstance(payload["candidates"], list)
    assert payload["candidates"]


def test_validate_rows_rejects_empty_sequence() -> None:
    """An empty row sequence is rejected."""
    with pytest.raises(ValueError, match="S3 design rows must be non-empty"):
        validate_s3_design_rows([])


def test_validate_rows_rejects_mismatched_protocol_id() -> None:
    """A row carrying a foreign protocol id is rejected."""
    with pytest.raises(ValueError, match="row protocol_id does not match"):
        validate_s3_design_rows([_row(protocol_id="other")])


def test_validate_rows_rejects_empty_label() -> None:
    """A row with an empty candidate label is rejected."""
    with pytest.raises(ValueError, match="candidate_label must be non-empty text"):
        validate_s3_design_rows([_row(candidate_label="")])


def test_validate_rows_rejects_duplicate_label() -> None:
    """A repeated candidate label is rejected."""
    with pytest.raises(ValueError, match="duplicate candidate_label"):
        validate_s3_design_rows([_row(), _row()])


def test_validate_rows_rejects_unknown_family() -> None:
    """A row family outside the allowed set is rejected."""
    with pytest.raises(ValueError, match="family must be ansatz or pulse"):
        validate_s3_design_rows([_row(family="quantum")])


def test_validate_rows_rejects_non_ok_status() -> None:
    """A row that is not in the ok status is rejected."""
    with pytest.raises(ValueError, match="status='ok'"):
        validate_s3_design_rows([_row(status="blocked")])


def test_validate_rows_rejects_non_finite_score() -> None:
    """A non-finite score is rejected."""
    with pytest.raises(ValueError, match="score must be finite"):
        validate_s3_design_rows([_row(score=float("inf"))])


def test_validate_rows_rejects_empty_metrics() -> None:
    """An empty metrics mapping is rejected."""
    with pytest.raises(ValueError, match="metrics must be a non-empty mapping"):
        validate_s3_design_rows([_row(metrics={})])


def test_validate_problem_rejects_omega_length_mismatch() -> None:
    """The frequency vector length must match the coupling matrix."""
    with pytest.raises(ValueError, match="omega length must match k_matrix"):
        _validate_problem(np.zeros((2, 2), dtype=np.float64), np.zeros(3, dtype=np.float64))


def test_validate_problem_rejects_non_finite_inputs() -> None:
    """Non-finite problem inputs are rejected."""
    with pytest.raises(ValueError, match="S3 problem inputs must be finite"):
        _validate_problem(
            np.array([[0.0, np.inf], [np.inf, 0.0]], dtype=np.float64),
            np.zeros(2, dtype=np.float64),
        )
