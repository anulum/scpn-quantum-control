# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 lane registry gate tests
"""Tests for Paper 0 lane registry comparator and gate wiring."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from scripts.compare_paper0_lane_registry import (
    compare_paper0_lane_registry,
    write_paper0_lane_registry,
)
from scripts.run_paper0_lane_registry_gate import (
    COMPARATOR_SCRIPT,
    build_paper0_lane_registry_gate_commands,
)


def _make_expected_payload(tmp_path: Path) -> tuple[Path, Path]:
    expected_json = tmp_path / "expected_paper0_lane_registry.json"
    expected_markdown = tmp_path / "expected_paper0_lane_registry.md"
    write_paper0_lane_registry(
        json_path=expected_json,
        markdown_path=expected_markdown,
    )
    return expected_json, expected_markdown


def test_compare_returns_valid_for_matching_expected_artifacts(tmp_path: Path) -> None:
    expected_json, expected_markdown = _make_expected_payload(tmp_path)

    result = compare_paper0_lane_registry(
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
    expected_json, expected_markdown = _make_expected_payload(tmp_path)
    payload = json.loads(expected_json.read_text(encoding="utf-8"))
    payload["lane_count"] = payload["lane_count"] + 1
    expected_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    result = compare_paper0_lane_registry(
        expected_json_path=expected_json,
        expected_markdown_path=expected_markdown,
    )

    assert result["valid"] is False
    assert any("JSON artifacts differ" in blocker for blocker in result["blockers"])


def test_compare_reports_markdown_drift(tmp_path: Path) -> None:
    expected_json, expected_markdown = _make_expected_payload(tmp_path)
    expected_markdown.write_text(
        expected_markdown.read_text(encoding="utf-8") + "\n<!-- drift-marker -->\n",
        encoding="utf-8",
    )

    result = compare_paper0_lane_registry(
        expected_json_path=expected_json,
        expected_markdown_path=expected_markdown,
    )

    assert result["valid"] is False
    assert any("Markdown artifacts differ" in blocker for blocker in result["blockers"])


def test_compare_rejects_asymmetric_actual_paths(tmp_path: Path) -> None:
    expected_json, expected_markdown = _make_expected_payload(tmp_path)
    with pytest.raises(ValueError, match="--actual-json and --actual-markdown"):
        compare_paper0_lane_registry(
            expected_json_path=expected_json,
            expected_markdown_path=expected_markdown,
            actual_json_path=tmp_path / "actual.json",
        )


def test_gate_helper_builds_comparator_command_with_explicit_expected_paths(
    tmp_path: Path,
) -> None:
    expected_json = tmp_path / "expected.json"
    expected_markdown = tmp_path / "expected.md"

    commands = build_paper0_lane_registry_gate_commands(
        comparator_script=COMPARATOR_SCRIPT,
        expected_json=expected_json,
        expected_markdown=expected_markdown,
    )

    assert commands == (
        (
            sys.executable,
            "scripts/compare_paper0_lane_registry.py",
            "--expected-json",
            str(expected_json),
            "--expected-markdown",
            str(expected_markdown),
        ),
    )
