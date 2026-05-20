# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — isolated provider smoke workflow tests
"""Static safety tests for isolated provider SDK smoke lanes."""

from __future__ import annotations

from pathlib import Path

from scpn_quantum_control.hardware.provider_smoke import isolated_provider_smoke_lanes

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "provider-isolated-smoke.yml"


def test_isolated_provider_workflow_matches_generated_lanes() -> None:
    matrix_by_extra = _workflow_matrix_rows(WORKFLOW.read_text(encoding="utf-8"))

    for lane in isolated_provider_smoke_lanes():
        row = matrix_by_extra[lane.extra]
        assert row["backend"] == lane.backend_ids[0]
        assert row["sdk_package"] == lane.sdk_packages[0]
        assert row["venv_path"] == lane.venv_path


def test_isolated_provider_workflow_is_manual_offline_and_non_secret() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "push:" not in text
    assert "pull_request:" not in text
    assert "permissions:\n  contents: read" in text
    assert "secrets." not in text
    assert "IBM" not in text
    assert "AWS_" not in text
    assert "AZURE_" not in text
    assert "TOKEN" not in text
    assert "--require-all" in text
    assert "scpn-provider-smoke" in text


def _workflow_matrix_rows(text: str) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("- extra: "):
            extra = line.removeprefix("- extra: ").strip()
            current = {"extra": extra}
            rows[extra] = current
        elif current is not None and ": " in line and not line.startswith("- "):
            key, value = line.split(": ", maxsplit=1)
            current[key] = value.strip().strip('"')
    return rows
