# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for S1 IBM live readiness
"""Tests for no-submit S1 IBM live-readiness artefacts."""

from __future__ import annotations

import json

from scripts import prepare_s1_ibm_live_readiness as readiness


class _Config:
    num_qubits = 156
    basis_gates = ["cz", "id", "rz", "sx", "x"]
    max_shots = 100000
    max_experiments = 300
    dt = 2.222e-10
    dtm = 1.111e-9
    n_uchannels = 2


class _Target:
    operation_names = ["cz", "delay", "id", "if_else", "measure", "reset", "rz", "sx", "x"]
    meas_map = [[0], [1], [2], [3]]


class _Status:
    operational = True
    pending_jobs = 7
    status_msg = "active"


class _Backend:
    name = "ibm_ready_fake"
    num_qubits = 156
    target = _Target()

    def configuration(self) -> _Config:
        return _Config()

    def status(self) -> _Status:
        return _Status()


def test_build_live_readiness_document_is_no_submit_and_ready_for_pair_runner(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        readiness,
        "_transpile_summary",
        lambda backend, n_rounds: {
            "source": {
                "n_qubits": 4,
                "n_clbits": 6,
                "depth": 41,
                "operation_counts": {"measure": 6, "if_else": 6},
            },
            "transpiled": {
                "n_qubits": 156,
                "n_clbits": 6,
                "depth": 725,
                "operation_counts": {"cz": 189, "measure": 6, "if_else": 6},
            },
            "submission_performed": False,
        },
    )

    document = readiness.build_live_readiness_document(_Backend())

    assert document["hardware_submission"] is False
    assert document["credential_string_argument_supported"] is False
    assert document["capability_decision"]["status"] == "ready"
    assert document["openpulse_readiness_status"] == "ready"
    assert document["openpulse_blockers"] == []
    assert document["openpulse_readiness"]["ready"] is True
    assert document["readiness_status"] == "ready_for_pair_runner"
    assert document["backend_status"]["pending_jobs"] == 7
    assert document["transpilation"]["submission_performed"] is False
    assert document["blockers"] == []


def test_write_readiness_markdown_preserves_no_submission_boundary(tmp_path) -> None:
    document = {
        "date": "2026-05-06",
        "preregistration_date": "2026-05-06",
        "selected_backend": "ibm_ready_fake",
        "submission_state": "live_metadata_and_transpile_no_submission",
        "hardware_submission": False,
        "capability_decision": {"status": "ready"},
        "openpulse_readiness_status": "blocked",
        "openpulse_blockers": ["target is missing pulse native features: drive_channel_access"],
        "readiness_status": "blocked",
        "package_budget": {
            "shots_per_circuit": 1024,
            "repetitions": 12,
            "qpu_seconds_ceiling": 120.0,
        },
        "transpilation": {
            "transpiled": {
                "n_qubits": 156,
                "n_clbits": 6,
                "depth": 725,
                "operation_counts": {"cz": 189, "measure": 6},
            }
        },
        "blockers": ["provider submitter is not implemented"],
        "claim_boundary": "No hardware claim.",
    }
    path = tmp_path / "readiness.md"

    readiness.write_readiness_markdown(document, path)

    text = path.read_text(encoding="utf-8")
    assert "Hardware submission: `false`" in text
    assert "provider submitter is not implemented" in text
    assert "OpenPulse readiness status: `blocked`" in text
    assert "target is missing pulse native features: drive_channel_access" in text
    assert json.dumps({"cz": 189, "measure": 6}, indent=2, sort_keys=True) in text
