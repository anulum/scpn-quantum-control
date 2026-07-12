# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — module-size policy audit tests
"""Tests for the tracked oversized-code responsibility audit."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_tool() -> ModuleType:
    path = Path(__file__).resolve().parents[1] / "tools" / "audit_module_size_policy.py"
    spec = importlib.util.spec_from_file_location("audit_module_size_policy_for_tests", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module-size policy audit from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_audit = _load_tool()


def _policy_payload(*files: dict[str, object], open_refactor_limit: int = 0) -> object:
    return {
        "threshold_lines": 3,
        "extensions": [".py", ".rs"],
        "open_refactor_limit": open_refactor_limit,
        "files": list(files),
    }


def _entry(
    path: str = "src/example.py",
    lines: int = 4,
    disposition: str = "cohesive",
    kind: str = "source",
    refactor_target: str = "",
) -> dict[str, object]:
    row: dict[str, object] = {
        "path": path,
        "lines": lines,
        "kind": kind,
        "disposition": disposition,
        "responsibility": "one bounded example lifecycle",
        "dependency_boundary": "depends on one leaf; consumed by one facade",
        "reassess_when": "an independent lifecycle appears",
    }
    if refactor_target:
        row["refactor_target"] = refactor_target
    return row


def test_count_physical_lines_handles_empty_and_unterminated_files(tmp_path: Path) -> None:
    empty = tmp_path / "empty.py"
    empty.write_bytes(b"")
    terminated = tmp_path / "terminated.py"
    terminated.write_bytes(b"a\nb\n")
    unterminated = tmp_path / "unterminated.py"
    unterminated.write_bytes(b"a\nb")

    assert _audit.count_physical_lines(empty) == 0
    assert _audit.count_physical_lines(terminated) == 2
    assert _audit.count_physical_lines(unterminated) == 2


def test_build_inventory_filters_suffix_and_threshold(tmp_path: Path) -> None:
    (tmp_path / "large.py").write_text("1\n2\n3\n4\n", encoding="utf-8")
    (tmp_path / "small.py").write_text("1\n2\n3\n", encoding="utf-8")
    (tmp_path / "large.md").write_text("1\n2\n3\n4\n", encoding="utf-8")

    inventory = _audit.build_inventory(
        tmp_path,
        ("large.md", "large.py", "missing.rs", "small.py"),
        3,
        frozenset({".py", ".rs"}),
    )

    assert inventory == (("large.py", 4),)


def test_parse_policy_rejects_duplicate_and_unsorted_paths() -> None:
    with pytest.raises(ValueError, match="duplicate paths"):
        _audit.parse_policy(_policy_payload(_entry(), _entry()))
    with pytest.raises(ValueError, match="sorted by path"):
        _audit.parse_policy(_policy_payload(_entry("src/z.py"), _entry("src/a.py")))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("kind", "bucket", "kind is unsupported"),
        ("disposition", "leave", "disposition is unsupported"),
        ("responsibility", "", "responsibility must be a non-empty string"),
        ("dependency_boundary", "", "dependency_boundary must be a non-empty string"),
        ("reassess_when", "", "reassess_when must be a non-empty string"),
    ],
)
def test_parse_policy_rejects_invalid_review_rows(field: str, value: str, message: str) -> None:
    row = _entry()
    row[field] = value

    with pytest.raises(ValueError, match=message):
        _audit.parse_policy(_policy_payload(row))


def test_parse_policy_requires_a_target_for_open_refactors() -> None:
    row = _entry(disposition="refactor_required")

    with pytest.raises(ValueError, match="need refactor_target"):
        _audit.parse_policy(_policy_payload(row, open_refactor_limit=1))


def test_audit_policy_reports_missing_stale_line_and_ratchet_drift() -> None:
    policy = _audit.parse_policy(_policy_payload(_entry(), open_refactor_limit=1))

    result = _audit.audit_policy(
        (("src/example.py", 5), ("src/new.rs", 8)),
        policy,
    )

    assert result.errors == (
        "unreviewed oversized file: src/new.rs (8 lines)",
        "line-count drift: src/example.py records 4, observed 5",
        "open-refactor ratchet drift: registry expects 1, observed 0",
    )


def test_audit_policy_reports_stale_registry_rows() -> None:
    policy = _audit.parse_policy(_policy_payload(_entry()))

    result = _audit.audit_policy((), policy)

    assert result.errors == ("stale oversized-file registry row: src/example.py",)


def test_open_refactors_pass_inventory_gate_but_fail_strict_mode() -> None:
    row = _entry(
        disposition="refactor_required",
        refactor_target="split parser from execution",
    )
    policy = _audit.parse_policy(_policy_payload(row, open_refactor_limit=1))
    result = _audit.audit_policy((("src/example.py", 4),), policy)

    assert result.errors == ()
    assert _audit.result_exit_code(result, strict=False) == 0
    assert _audit.result_exit_code(result, strict=True) == 1
    assert "OPEN src/example.py" in _audit.format_result(result)


def test_main_reports_registry_errors_without_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    registry = tmp_path / "invalid.json"
    registry.write_text("{}", encoding="utf-8")

    status = _audit.main(["--repo-root", str(tmp_path), "--registry", str(registry)])

    assert status == 2
    assert "module-size policy audit failed" in capsys.readouterr().out


def test_live_registry_matches_every_tracked_oversized_code_file() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry = repo_root / "tools" / "module_size_policy.json"

    result = _audit.audit_repository(repo_root, registry)
    policy = _audit.load_policy(registry)

    assert result.errors == ()
    assert len(result.open_refactors) == policy.open_refactor_limit
    assert result.inventory


def test_architecture_quotes_the_live_inventory_and_open_count() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry = repo_root / "tools" / "module_size_policy.json"
    result = _audit.audit_repository(repo_root, registry)
    architecture = (repo_root / "docs" / "architecture.md").read_text(encoding="utf-8")

    assert f"contains {len(result.inventory)} oversized tracked code files" in architecture
    assert f"and {len(result.open_refactors)} remain open" in architecture


def test_ci_and_preflight_run_the_inventory_gate() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    preflight = (repo_root / "tools" / "preflight.py").read_text(encoding="utf-8")

    assert "python tools/audit_module_size_policy.py" in workflow
    assert '"tools/audit_module_size_policy.py"' in preflight
    assert '"mypy-strict-module-size-policy"' in preflight


def test_registry_is_valid_json() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    registry = repo_root / "tools" / "module_size_policy.json"

    assert isinstance(json.loads(registry.read_text(encoding="utf-8")), dict)
