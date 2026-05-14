# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control -- Paper 0 HPC-UPDE derivation runner tests
"""Tests for the Paper 0 HPC-UPDE mathematical bridge fixture runner."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_paper0_hpc_upde_derivation_fixture import render_report, write_outputs


def test_hpc_upde_derivation_fixture_runner_writes_auditable_outputs(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "hpc_upde_derivation_result.json"
    report_path = tmp_path / "hpc_upde_derivation_result.md"

    payload = write_outputs(output_path=json_path, report_path=report_path)
    persisted = json.loads(json_path.read_text(encoding="utf-8"))
    report = render_report(payload)

    assert payload["hardware_status"] == "simulator_only_no_provider_submission"
    assert payload["source_ledger_span"] == ["P0R06615", "P0R06645"]
    assert payload["gradient_check_error"] < 1.0e-6
    assert payload["phase_locking_error_value"] >= 0.0
    assert payload["null_controls"]["unsupported_active_inference_evidence_rejection_label"] == 1.0
    assert "not empirical evidence" in payload["claim_boundary"]
    assert persisted == payload
    assert "# Paper 0 HPC-UPDE Mathematical Bridge Fixture" in report
    assert report_path.read_text(encoding="utf-8") == report
