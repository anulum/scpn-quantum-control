# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- stable-core contract fixture tests
"""Tests for stable-core fixture comparison and gate helper wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.compare_stable_core_contract_fixtures import (
    compare_stable_core_contract_fixtures,
    write_stable_core_contract_fixtures,
)
from scripts.run_stable_core_contract_gate import (
    COMPARATOR_SCRIPT,
    build_stable_core_contract_gate_commands,
)


def _make_expected_payload(tmp_path: Path) -> tuple[Path, Path]:
    """Write committed-equivalent fixture artefacts into temporary paths."""

    expected_json = tmp_path / "expected_stable_core_contract_fixtures.json"
    expected_markdown = tmp_path / "expected_stable_core_contract_fixtures.md"
    write_stable_core_contract_fixtures(
        json_path=expected_json,
        markdown_path=expected_markdown,
    )
    return expected_json, expected_markdown


def test_compare_returns_valid_for_matching_expected_artifacts(tmp_path: Path) -> None:
    """Generated stable-core fixture payloads match an equivalent committed baseline."""

    expected_json, expected_markdown = _make_expected_payload(tmp_path)

    result = compare_stable_core_contract_fixtures(
        expected_json_path=expected_json,
        expected_markdown_path=expected_markdown,
    )

    assert result["valid"] is True
    assert result["blockers"] == ()
    assert result["digests"]["expected_json_sha256"] == result["digests"]["actual_json_sha256"]
    assert (
        result["digests"]["expected_markdown_sha256"]
        == result["digests"]["actual_markdown_sha256"]
    )


def test_compare_reports_json_drift(tmp_path: Path) -> None:
    """JSON drift is captured as a blocker in fixture comparison."""

    expected_json, expected_markdown = _make_expected_payload(tmp_path)
    payload = json.loads(expected_json.read_text(encoding="utf-8"))
    payload["hardware_submission"] = not payload["hardware_submission"]
    expected_json.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )

    result = compare_stable_core_contract_fixtures(
        expected_json_path=expected_json,
        expected_markdown_path=expected_markdown,
    )

    assert result["valid"] is False
    assert any("JSON artifacts differ" in blocker for blocker in result["blockers"])


def test_compare_reports_markdown_drift(tmp_path: Path) -> None:
    """Markdown drift is captured as a blocker in fixture comparison."""

    expected_json, expected_markdown = _make_expected_payload(tmp_path)
    expected_markdown.write_text(
        expected_markdown.read_text(encoding="utf-8") + "\n<!-- drift-marker -->\n",
        encoding="utf-8",
    )

    result = compare_stable_core_contract_fixtures(
        expected_json_path=expected_json,
        expected_markdown_path=expected_markdown,
    )

    assert result["valid"] is False
    assert any("Markdown artifacts differ" in blocker for blocker in result["blockers"])


def test_compare_rejects_asymmetric_actual_paths(tmp_path: Path) -> None:
    """Only accepting actual JSON+Markdown pairs avoids partial comparison drift."""

    expected_json, expected_markdown = _make_expected_payload(tmp_path)
    with pytest.raises(ValueError, match="--actual-json and --actual-markdown"):
        compare_stable_core_contract_fixtures(
            expected_json_path=expected_json,
            expected_markdown_path=expected_markdown,
            actual_json_path=tmp_path / "actual.json",
        )


def test_gate_helper_builds_comparator_command_with_explicit_expected_paths(
    tmp_path: Path,
) -> None:
    """The gate helper assembles deterministic comparator invocation arguments."""

    expected_json = tmp_path / "expected.json"
    expected_markdown = tmp_path / "expected.md"

    commands = build_stable_core_contract_gate_commands(
        comparator_script=COMPARATOR_SCRIPT,
        expected_json=expected_json,
        expected_markdown=expected_markdown,
    )

    assert commands == (
        (
            sys.executable,
            "scripts/compare_stable_core_contract_fixtures.py",
            "--expected-json",
            str(expected_json),
            "--expected-markdown",
            str(expected_markdown),
        ),
    )
