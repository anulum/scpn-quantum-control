# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Active queue status CLI contract tests
"""Exercise the queue checker through its CLI without requiring private records.

Temporary files are real checker inputs, not mocks. Run the CLI separately
against the canonical checkout for live private-queue acceptance. Public CI
must neither receive the private queue nor pretend its absence is a pass.
"""

from __future__ import annotations

import runpy
import subprocess
import sys
import tomllib
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import pytest

CHECKER = Path(__file__).resolve().parents[1] / "tools/check_active_queue_status.py"


def _root(
    tmp_path: Path,
    text: str,
    *,
    strict: bool = True,
    selection: str = 'select = ["D"]',
    ignore: str = "",
) -> Path:
    """Create a real queue/configuration pair consumed by the command."""
    (tmp_path / "docs/internal").mkdir(parents=True)
    (tmp_path / "docs/internal/TODO.md").write_text(text, encoding="utf-8")
    config = f"[tool.mypy]\nstrict = {str(strict).lower()}\n"
    config += f"[tool.ruff.lint]\n{selection}\n{ignore}\n"
    config += '[tool.ruff.lint.pydocstyle]\nconvention = "numpy"\n'
    (tmp_path / "pyproject.toml").write_text(config, encoding="utf-8")
    return tmp_path


def _run(root: Path) -> subprocess.CompletedProcess[str]:
    """Compare actual process output with the instrumentable script entrypoint."""
    completed = subprocess.run(
        [sys.executable, str(CHECKER), "--root", str(root)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    argv = sys.argv
    output = StringIO()
    try:
        sys.argv = [str(CHECKER), "--root", str(root)]
        with redirect_stdout(output), pytest.raises(SystemExit) as exit_info:
            runpy.run_path(str(CHECKER), run_name="__main__")
        assert exit_info.value.code == completed.returncode
        assert output.getvalue() == completed.stdout
    finally:
        sys.argv = argv
    return completed


@pytest.mark.parametrize(
    "body",
    [
        "- [ ] mypy is not strict",
        "- [~] Adopt `strict = true`",
        "- [ ] Review\n\n  mypy is not strict",
        "- [ ] Parent\n  - [x] Finished child\n\n  mypy is not strict",
        "- [x] Parent history\n  - [ ] mypy is not strict",
        "* [ ] Docstring enforcement absent",
        "+ [~] No ruff `D`",
        "- [ ] Review\n\n  Repository-wide Ruff `D` selection remains open debt.",
        "- [ ] No `pydocstyle` convention",
    ],
)
def test_contradictions_fail_through_cli(tmp_path: Path, body: str) -> None:
    """Detect original wordings, continuation paragraphs and independent children."""
    result = _run(_root(tmp_path, body))
    assert result.returncode == 1
    assert "is configured" in result.stdout
    assert result.stderr == ""


@pytest.mark.parametrize(
    "body",
    [
        "- [x] mypy is not strict\n  Historical body.",
        "- [ ] Parent current\n  - [x] mypy is not strict",
        "- [ ] Current work\n\n## History\nmypy is not strict",
        "- [ ] Current work\n  > mypy is not strict",
        "- [ ] Current work\n\n  ```md\n  - [ ] mypy is not strict\n  ```",
        "- [ ] Current work\n  ~~~~md\n  mypy is not strict\n  ~~~~",
        "- [ ] Current work\n  ```md\n  ~~~\n  mypy is not strict\n  ```",
        "- [ ] Raise the aggregate coverage gate from 90% to 95%.",
        "- [ ] mypy --strict passes on the changed files.",
        "- [ ] No Ruff diagnostics may remain in changed code.",
    ],
)
def test_history_examples_and_unrelated_work_are_not_active(tmp_path: Path, body: str) -> None:
    """Preserve completed prose and examples without treating them as live requests."""
    result = _run(_root(tmp_path, body))
    assert result.returncode == 0
    assert "No recognised" in result.stdout


@pytest.mark.parametrize(
    "selection", ['select = ["D"]', 'select = ["ALL"]', 'extend-select = ["D"]']
)
def test_effective_doc_selection_is_read(tmp_path: Path, selection: str) -> None:
    """Recognise direct, complete and additive doc-rule selection."""
    assert _run(_root(tmp_path, "- [ ] No ruff D", selection=selection)).returncode == 1


def test_true_absence_is_not_a_contradiction(tmp_path: Path) -> None:
    """A real configuration absence must not be misreported as a stale claim."""
    root = _root(
        tmp_path,
        "- [ ] mypy is not strict\n- [ ] No ruff D",
        strict=False,
        ignore='ignore = ["D"]',
    )
    assert _run(root).returncode == 0


@pytest.mark.parametrize("missing", ["pyproject.toml", "docs/internal/TODO.md"])
def test_missing_private_evidence_is_unavailable(tmp_path: Path, missing: str) -> None:
    """A fresh public checkout cannot claim to have validated private queue state."""
    root = _root(tmp_path, "- [ ] Current work")
    (root / missing).unlink()
    result = _run(root)
    assert result.returncode == 2
    assert "unavailable" in result.stdout
    assert "No recognised" not in result.stdout


@pytest.mark.parametrize(
    "config",
    [
        "bad TOML [",
        "[tool]\n",
        'tool = "wrong type"',
        '[tool.mypy]\nstrict=true\n[tool.ruff]\nlint="wrong type"',
    ],
)
def test_malformed_configuration_is_unavailable(tmp_path: Path, config: str) -> None:
    """Broken evidence is neither a queue contradiction nor successful validation."""
    root = _root(tmp_path, "- [ ] Current work")
    (root / "pyproject.toml").write_text(config, encoding="utf-8")
    assert _run(root).returncode == 2


def test_current_repository_configuration_matches_recorded_selection() -> None:
    """Verify only the recorded strict/NumPy selection, not universal compliance."""
    data = tomllib.loads((CHECKER.parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["tool"]["mypy"]["strict"] is True
    lint = data["tool"]["ruff"]["lint"]
    assert "D" in lint["select"]
    assert lint["pydocstyle"]["convention"] == "numpy"


def test_callable_checker_reports_exact_source_lines(tmp_path: Path) -> None:
    """Programmatic consumers receive stable source locations without private prose."""
    root = _root(tmp_path, "- [ ] Parent\n\n  mypy is not strict\n- [ ] No ruff D")
    namespace = runpy.run_path(str(CHECKER))
    assert namespace["contradictions"](root) == [
        "line 1: strict typing is configured",
        "line 4: NumPy docstring selection is configured",
    ]
