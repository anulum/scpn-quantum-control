# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — documentation surface audit tests
# SCPN Quantum Control -- Documentation surface audit tests
"""Tests for repository documentation-surface audit helper."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any


def _load_tool_module(module_name: str, filename: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_audit_documentation_surface = _load_tool_module(
    "audit_documentation_surface_for_tests",
    "audit_documentation_surface.py",
)
audit_files = _audit_documentation_surface.audit_files
audit_markdown_text = _audit_documentation_surface.audit_markdown_text
audit_python_text = _audit_documentation_surface.audit_python_text
candidate_markdown_files = _audit_documentation_surface.candidate_markdown_files
candidate_python_files = _audit_documentation_surface.candidate_python_files
filter_allowed_findings = _audit_documentation_surface.filter_allowed_findings
findings_to_json = _audit_documentation_surface.findings_to_json
format_findings = _audit_documentation_surface.format_findings
load_allowlist = _audit_documentation_surface.load_allowlist
main = _audit_documentation_surface.main


def test_python_audit_flags_public_missing_docstrings() -> None:
    findings = audit_python_text(
        Path("src/example.py"),
        '"""Module docs."""\n\nclass PublicClass:\n    def method(self):\n        return 1\n\ndef public_function():\n    return 2\n',
    )

    symbols = {item.symbol for item in findings}

    assert "PublicClass" in symbols
    assert "PublicClass.method" in symbols
    assert "public_function" in symbols


def test_python_audit_allows_private_missing_docstrings() -> None:
    findings = audit_python_text(
        Path("src/example.py"),
        '"""Module docs."""\n\nclass _PrivateClass:\n    def _method(self):\n        return 1\n\ndef _private_function():\n    return 2\n',
    )

    assert findings == ()


def test_markdown_audit_flags_missing_title_and_stale_status() -> None:
    findings = audit_markdown_text(
        Path("docs/page.md"),
        "Status Snapshot - 2026-04-29\n",
        current_date="2026-05-18",
    )

    kinds = {item.kind for item in findings}

    assert "markdown_title" in kinds
    assert "stale_status_snapshot" in kinds


def test_candidate_files_exclude_cache_and_site(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "site").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "ok.py").write_text('"""ok"""\n', encoding="utf-8")
    (tmp_path / "site" / "generated.py").write_text("def bad(): pass\n", encoding="utf-8")
    (tmp_path / "docs" / "page.md").write_text("# Page\n", encoding="utf-8")

    assert Path("src/ok.py") in candidate_python_files(tmp_path)
    assert Path("site/generated.py") not in candidate_python_files(tmp_path, ("site",))
    assert Path("docs/page.md") in candidate_markdown_files(tmp_path)


def test_json_and_text_reports_are_deterministic(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "src" / "module.py").write_text("def missing():\n    return 1\n", encoding="utf-8")
    (tmp_path / "docs" / "page.md").write_text("No title\n", encoding="utf-8")

    findings = audit_files(
        tmp_path,
        python_files=(Path("src/module.py"),),
        markdown_files=(Path("docs/page.md"),),
        current_date="2026-05-18",
    )
    decoded = json.loads(findings_to_json(findings))
    summary = format_findings(findings)

    assert decoded[0]["path"] <= decoded[-1]["path"]
    assert "Documentation surface audit findings:" in summary
    assert "by_kind:" in summary


def test_cli_can_report_without_failing(tmp_path: Path, capsys: Any) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("def missing():\n    return 1\n", encoding="utf-8")

    assert main(["--project-root", str(tmp_path), "--python-root", "src"]) == 0
    assert "total_findings" in capsys.readouterr().out


def test_cli_can_fail_on_findings(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "module.py").write_text("def missing():\n    return 1\n", encoding="utf-8")

    assert (
        main(["--project-root", str(tmp_path), "--python-root", "src", "--fail-on-findings"]) == 1
    )


def test_allowlist_filters_generated_script_main_findings(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "docs" / "internal").mkdir(parents=True)
    script = tmp_path / "scripts" / "build_paper0_generated_specs.py"
    script.write_text(
        '"""Generated spec builder."""\n\ndef main():\n    return None\n', encoding="utf-8"
    )
    allowlist = tmp_path / "docs" / "internal" / "documentation_surface_allowlist.json"
    allowlist.write_text(
        json.dumps(
            [
                {
                    "path_pattern": "scripts/build_paper0_*_specs.py",
                    "kind": "function",
                    "symbol": "main",
                    "reason": "generated Paper 0 spec builders share generator-level CLI documentation",
                }
            ]
        ),
        encoding="utf-8",
    )
    findings = audit_files(
        tmp_path,
        python_files=(Path("scripts/build_paper0_generated_specs.py"),),
        markdown_files=(),
        current_date="2026-05-18",
    )

    assert filter_allowed_findings(findings, load_allowlist(allowlist)) == ()


def test_allowlist_requires_non_empty_reason(tmp_path: Path) -> None:
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            [
                {
                    "path_pattern": "scripts/build_paper0_*_specs.py",
                    "kind": "function",
                    "symbol": "main",
                    "reason": "",
                }
            ]
        ),
        encoding="utf-8",
    )

    try:
        load_allowlist(allowlist)
    except ValueError as exc:
        assert "reason" in str(exc)
    else:  # pragma: no cover - assertion clarity
        raise AssertionError("allowlist entries without reasons must fail closed")


def test_cli_applies_allowlist_before_fail_on_findings(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    script = tmp_path / "scripts" / "build_paper0_generated_specs.py"
    script.write_text(
        '"""Generated spec builder."""\n\ndef main():\n    return None\n', encoding="utf-8"
    )
    allowlist = tmp_path / "allowlist.json"
    allowlist.write_text(
        json.dumps(
            [
                {
                    "path_pattern": "scripts/build_paper0_*_specs.py",
                    "kind": "function",
                    "symbol": "main",
                    "reason": "generated Paper 0 spec builders share generator-level CLI documentation",
                }
            ]
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "--project-root",
                str(tmp_path),
                "--python-root",
                "scripts",
                "--allowlist",
                str(allowlist),
                "--fail-on-findings",
            ]
        )
        == 0
    )
