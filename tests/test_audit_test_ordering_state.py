# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for ordering-state audit helper
"""Tests for full-suite ordering-state audit helper."""

from __future__ import annotations

from pathlib import Path

from tools.audit_test_ordering_state import (
    audit_ordering_file,
    audit_ordering_tree,
    category_counts,
    findings_to_json,
    format_findings,
)


def test_detects_reload_environment_and_module_injection(tmp_path: Path):
    path = tmp_path / "test_ordering.py"
    path.write_text(
        "\n".join(
            [
                "import importlib",
                "import sys",
                "",
                "def test_state(monkeypatch, module):",
                "    importlib.reload(module)",
                "    monkeypatch.setenv('SCPN_IBM_BACKEND', 'sim')",
                "    monkeypatch.setitem(sys.modules, 'pkg.fake', module)",
            ]
        ),
        encoding="utf-8",
    )

    findings = audit_ordering_file(path)
    categories = {item.category for item in findings}

    assert "module_reload" in categories
    assert "environment_mutation" in categories
    assert "module_injection" in categories


def test_detects_random_seed_and_global_state_assignment(tmp_path: Path):
    path = tmp_path / "test_globals.py"
    path.write_text(
        "\n".join(
            [
                "import numpy as np",
                "",
                "def test_state(module):",
                "    np.random.seed(123)",
                "    module._HAS_RUST = False",
            ]
        ),
        encoding="utf-8",
    )

    findings = audit_ordering_file(path)
    counts = category_counts(findings)

    assert counts["random_seed_mutation"] == 1
    assert counts["global_state_assignment"] == 1


def test_tree_audit_json_and_summary_are_deterministic(tmp_path: Path):
    tests_root = tmp_path / "tests"
    tests_root.mkdir()
    (tests_root / "test_one.py").write_text(
        "def test_env(monkeypatch):\n    monkeypatch.delenv('SCPN_TEST_FLAG', raising=False)\n",
        encoding="utf-8",
    )
    (tests_root / "helper.py").write_text(
        "def ignored(monkeypatch):\n    monkeypatch.setenv('SCPN_TEST_FLAG', 'x')\n",
        encoding="utf-8",
    )

    findings = audit_ordering_tree(tests_root)
    json_text = findings_to_json(findings)
    summary = format_findings(findings)

    assert len(findings) == 1
    assert "environment_mutation" in json_text
    assert "Full-suite ordering-state audit summary:" in summary
