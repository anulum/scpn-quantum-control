# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for mock and stub audit helper
"""Tests for mock/stub audit helper."""

from __future__ import annotations

from pathlib import Path

from tools.audit_mock_stub_usage import (
    audit_mock_stub_file,
    audit_mock_stub_tree,
    findings_to_json,
    format_findings,
)


def test_audit_detects_monkeypatch_and_mock_calls(tmp_path: Path):
    path = tmp_path / "test_boundary.py"
    path.write_text(
        "\n".join(
            [
                "from unittest.mock import MagicMock",
                "",
                "def test_backend_boundary(monkeypatch):",
                "    fake_backend = MagicMock(name='backend')",
                "    monkeypatch.setattr('pkg.backend', fake_backend)",
                "    assert fake_backend is not None",
            ]
        ),
        encoding="utf-8",
    )

    findings = audit_mock_stub_file(path)

    names = {item.name for item in findings}
    assert "MagicMock" in names
    assert "monkeypatch.setattr" in names
    assert any(item.appears_third_party_boundary for item in findings)


def test_audit_flags_result_term_context(tmp_path: Path):
    path = tmp_path / "test_result_stub.py"
    path.write_text(
        "\n".join(
            [
                "def fake_counts_result():",
                "    return {'00': 10}",
                "",
                "def test_counts_result():",
                "    counts = fake_counts_result()",
                "    assert counts == {'00': 10}",
            ]
        ),
        encoding="utf-8",
    )

    findings = audit_mock_stub_file(path)

    assert any(item.name == "fake_counts_result" for item in findings)
    assert any(item.touches_scientific_result_terms for item in findings)


def test_tree_audit_json_and_text_outputs_are_deterministic(tmp_path: Path):
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_one.py").write_text(
        "def stub_executor():\n    return 1\n",
        encoding="utf-8",
    )
    (tests_root / "helper.py").write_text(
        "def fake_ignored():\n    return 1\n",
        encoding="utf-8",
    )

    findings = audit_mock_stub_tree(tests_root)
    json_text = findings_to_json(findings)
    summary = format_findings(findings)

    assert len(findings) == 1
    assert "stub_executor" in json_text
    assert "Mock/stub audit summary:" in summary
