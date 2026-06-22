# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Guard tests for the S2 advantage scaling protocol
"""Validation tests for the S2 advantage scaling protocol contracts.

Covers the baseline and protocol manifest field guards and the per-row
protocol-id, size and baseline accumulation messages.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from scpn_quantum_control.benchmarks.advantage_protocol import (
    default_s2_scaling_protocol,
    validate_scaling_rows,
)

_PROTOCOL = default_s2_scaling_protocol()
_BASELINE = _PROTOCOL.baselines[0]


def test_baseline_rejects_empty_label() -> None:
    """A baseline with an empty label is rejected."""
    with pytest.raises(ValueError, match="label must be non-empty"):
        dataclasses.replace(_BASELINE, label="")


def test_baseline_rejects_non_positive_max_qubits() -> None:
    """A non-positive max_qubits is rejected when provided."""
    with pytest.raises(ValueError, match="max_qubits must be positive when provided"):
        dataclasses.replace(_BASELINE, max_qubits=0)


def test_baseline_rejects_empty_claim_boundary() -> None:
    """A baseline without a claim boundary is rejected."""
    with pytest.raises(ValueError, match="claim_boundary must be non-empty"):
        dataclasses.replace(_BASELINE, claim_boundary="")


def test_protocol_rejects_empty_protocol_id() -> None:
    """A protocol with an empty id is rejected."""
    with pytest.raises(ValueError, match="protocol_id must be non-empty"):
        dataclasses.replace(_PROTOCOL, protocol_id="")


def test_protocol_rejects_non_positive_sizes() -> None:
    """A size set with a non-positive entry is rejected."""
    with pytest.raises(ValueError, match="sizes must contain positive integers"):
        dataclasses.replace(_PROTOCOL, sizes=())


def test_protocol_rejects_unsorted_sizes() -> None:
    """An unsorted size set is rejected."""
    with pytest.raises(ValueError, match="sizes must be sorted"):
        dataclasses.replace(_PROTOCOL, sizes=(8, 4))


def test_protocol_rejects_empty_baselines() -> None:
    """A protocol without baselines is rejected."""
    with pytest.raises(ValueError, match="baselines must be non-empty"):
        dataclasses.replace(_PROTOCOL, baselines=())


def test_protocol_rejects_empty_acceptance() -> None:
    """A protocol without acceptance criteria is rejected."""
    with pytest.raises(ValueError, match="acceptance must be non-empty"):
        dataclasses.replace(_PROTOCOL, acceptance=())


def test_protocol_rejects_empty_falsification() -> None:
    """A protocol without falsification criteria is rejected."""
    with pytest.raises(ValueError, match="falsification must be non-empty"):
        dataclasses.replace(_PROTOCOL, falsification=())


def test_protocol_rejects_empty_claim_boundary() -> None:
    """A protocol without a claim boundary is rejected."""
    with pytest.raises(ValueError, match="claim_boundary must be non-empty"):
        dataclasses.replace(_PROTOCOL, claim_boundary="")


def _row(**overrides: Any) -> dict[str, Any]:
    row = {
        "protocol_id": _PROTOCOL.protocol_id,
        "n_qubits": _PROTOCOL.sizes[0],
        "baseline": _BASELINE.label,
        "status": "ok",
        "metric_payload": {},
        "command": "test",
    }
    row.update(overrides)
    return row


def test_validate_rows_flags_protocol_id_mismatch() -> None:
    """A row carrying a foreign protocol id is flagged invalid."""
    validation = validate_scaling_rows(_PROTOCOL, [_row(protocol_id="other")])
    assert any("protocol_id must be" in item for item in validation.invalid_rows)


def test_validate_rows_flags_unknown_size() -> None:
    """A row with a size outside the protocol is flagged invalid."""
    validation = validate_scaling_rows(_PROTOCOL, [_row(n_qubits=999_999)])
    assert any("n_qubits must be one of" in item for item in validation.invalid_rows)


def test_validate_rows_flags_unknown_baseline() -> None:
    """A row naming an unknown baseline is flagged invalid."""
    validation = validate_scaling_rows(_PROTOCOL, [_row(baseline="not-a-baseline")])
    assert any("unknown baseline" in item for item in validation.invalid_rows)
