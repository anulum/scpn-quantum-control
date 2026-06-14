# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for the internal TODO policy guard
"""Tests for tools/check_internal_todo_policy.py.

Each case builds a throwaway git repository so the git-backed tracking check and
the working-tree scans are exercised against real state rather than mocks.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_TOOLS_DIR = _REPO_ROOT / "tools"


def _load_tool_module(module_name: str, filename: str) -> ModuleType:
    module_path = _TOOLS_DIR / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_guard = _load_tool_module("check_internal_todo_policy_for_tests", "check_internal_todo_policy.py")
check_internal_todo_policy = _guard.check_internal_todo_policy
competing_todo_files = _guard.competing_todo_files
format_findings = _guard.format_findings
main = _guard.main


def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@example.test"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "Test"], check=True)


def _write(path: Path, text: str = "x\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_clean_repo(tmp_path: Path) -> Path:
    _init_repo(tmp_path)
    _write(tmp_path / "docs" / "internal" / "TODO.md", "# canonical\n")
    # docs/internal stays untracked (mirrors the gitignored canonical queue)
    return tmp_path


def test_clean_repo_has_no_blocking_findings(tmp_path: Path) -> None:
    root = _make_clean_repo(tmp_path)

    findings = check_internal_todo_policy(root)

    assert findings == []
    assert main(["--root", str(root)]) == 0


def test_missing_canonical_todo_is_non_blocking(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    findings = check_internal_todo_policy(tmp_path)

    assert [f.category for f in findings] == ["missing-canonical-todo"]
    assert findings[0].blocking is False
    assert main(["--root", str(tmp_path)]) == 0


def test_tracked_internal_path_is_blocking(tmp_path: Path) -> None:
    root = _make_clean_repo(tmp_path)
    _write(root / "docs" / "internal" / "plan.md", "leaked\n")
    subprocess.run(["git", "-C", str(root), "add", "-f", "docs/internal/plan.md"], check=True)

    findings = check_internal_todo_policy(root)

    categories = [f.category for f in findings]
    assert "tracked-internal-path" in categories
    tracked = next(f for f in findings if f.category == "tracked-internal-path")
    assert tracked.path == "docs/internal/plan.md"
    assert tracked.blocking is True
    assert main(["--root", str(root)]) == 1


def test_tracked_coordination_path_is_blocking(tmp_path: Path) -> None:
    root = _make_clean_repo(tmp_path)
    _write(root / ".coordination" / "sessions" / "leak.md", "session\n")
    subprocess.run(
        ["git", "-C", str(root), "add", "-f", ".coordination/sessions/leak.md"], check=True
    )

    findings = check_internal_todo_policy(root)

    assert any(f.category == "tracked-internal-path" and f.blocking for f in findings)
    assert main(["--root", str(root)]) == 1


def test_competing_todo_queue_is_blocking(tmp_path: Path) -> None:
    root = _make_clean_repo(tmp_path)
    _write(root / "docs" / "notes" / "TODO.md", "# rogue queue\n")

    findings = check_internal_todo_policy(root)

    competing = [f for f in findings if f.category == "competing-todo"]
    assert [f.path for f in competing] == ["docs/notes/TODO.md"]
    assert competing[0].blocking is True
    assert main(["--root", str(root)]) == 1


@pytest.mark.parametrize("name", ["TODO", "TODO.md", "TODO.txt", "todo.md", "Todo.rst"])
def test_competing_detection_matches_todo_basenames(tmp_path: Path, name: str) -> None:
    root = _make_clean_repo(tmp_path)
    _write(root / "elsewhere" / name)

    competing = competing_todo_files(root)

    assert (root / "elsewhere" / name) in competing


@pytest.mark.parametrize("name", ["release_todo.md", "TODO_archive.md", "todos.md", "NOTES.md"])
def test_non_queue_filenames_are_not_competing(tmp_path: Path, name: str) -> None:
    root = _make_clean_repo(tmp_path)
    _write(root / "elsewhere" / name)

    assert competing_todo_files(root) == []


def test_canonical_todo_is_not_self_reported_as_competing(tmp_path: Path) -> None:
    root = _make_clean_repo(tmp_path)

    assert competing_todo_files(root) == []


def test_skipped_directories_are_not_scanned(tmp_path: Path) -> None:
    root = _make_clean_repo(tmp_path)
    _write(root / ".venv-linux" / "lib" / "TODO.md")
    _write(root / "node_modules" / "pkg" / "TODO.md")

    assert competing_todo_files(root) == []


def test_format_findings_reports_clean_state() -> None:
    assert "OK" in format_findings([])


def test_format_findings_tags_blocking_and_notice(tmp_path: Path) -> None:
    root = _make_clean_repo(tmp_path)
    _write(root / "docs" / "notes" / "TODO.md")

    rendered = format_findings(check_internal_todo_policy(root))

    assert "[FAIL] competing-todo" in rendered


def test_guard_passes_on_the_real_repository() -> None:
    # The live repository must itself satisfy the policy this guard enforces.
    assert main(["--root", str(_REPO_ROOT)]) == 0
