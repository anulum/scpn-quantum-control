# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — submit s1 IBM feedback pair tests
# SCPN Quantum Control -- Tests for S1 IBM feedback pair submission
"""Tests for the approval-gated S1 feedback-pair runner."""

from __future__ import annotations

import pytest

from scripts.submit_s1_ibm_feedback_pair import (
    _build_readiness_document,
    _manifest,
    _parse_args,
    _parse_physical_qubits,
    _selected_layout_payload,
)


def test_parse_physical_qubits_accepts_four_distinct_indices() -> None:
    assert _parse_physical_qubits("7,17,6,8") == (7, 17, 6, 8)


def test_parse_physical_qubits_rejects_wrong_length_and_duplicates() -> None:
    with pytest.raises(ValueError, match="exactly four"):
        _parse_physical_qubits("7,17,6")

    with pytest.raises(ValueError, match="duplicates"):
        _parse_physical_qubits("7,17,7,8")


def test_selected_layout_payload_records_logical_order() -> None:
    payload = _selected_layout_payload((7, 17, 6, 8))

    assert payload == {
        "system_qubits": [7, 17, 6],
        "monitor_qubit": 8,
        "logical_order": ["sys_0", "sys_1", "sys_2", "monitor"],
        "selection": "explicit_same_layout_repeat",
    }


def test_repeat_args_are_explicit_without_submit() -> None:
    args = _parse_args(
        [
            "--backend",
            "ibm_marrakesh",
            "--physical-qubits",
            "7,17,6,8",
            "--repeat-label",
            "s1_primary_same_layout_repeat_marrakesh_2026-05-20",
        ]
    )

    assert args.backend == "ibm_marrakesh"
    assert args.physical_qubits == "7,17,6,8"
    assert args.repeat_label == "s1_primary_same_layout_repeat_marrakesh_2026-05-20"
    assert args.submit is False


def test_readiness_document_exposes_layout_and_repeat_label_without_private_keys() -> None:
    arms = (
        {
            "label": "feedback",
            "estimated_qpu_seconds": 12.0,
            "transpiled_depth_max": 717,
        },
        {
            "label": "control",
            "estimated_qpu_seconds": 12.0,
            "transpiled_depth_max": 684,
        },
    )
    manifest = _manifest(
        {"package_id": "fixture"},
        {
            "arms": arms,
            "physical_layout": _selected_layout_payload((7, 17, 6, 8)),
            "repeat_label": "s1_primary_same_layout_repeat_marrakesh_2026-05-20",
        },
    )

    readiness = _build_readiness_document(
        backend_name="ibm_marrakesh",
        package_manifest=manifest,
        arm_summaries=arms,
        max_depth=1200,
        max_qpu_seconds=120.0,
        physical_qubits=(7, 17, 6, 8),
        repeat_label="s1_primary_same_layout_repeat_marrakesh_2026-05-20",
    )

    assert readiness["status"] == "ready_for_submission"
    assert readiness["physical_layout"]["system_qubits"] == [7, 17, 6]
    assert readiness["physical_layout"]["monitor_qubit"] == 8
    assert readiness["repeat_label"] == "s1_primary_same_layout_repeat_marrakesh_2026-05-20"
    assert "_physical_qubits" not in readiness
