# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — documentation scope gate owner
"""Prove the scope gate fails on new debt and passes on the live tree.

The gate exists to stop a scope that reached zero from quietly re-opening, so
the case that matters is a newly added file with an undocumented function. That
is what these tests build and check, rather than asserting only that today's
tree is clean — a gate never seen to fail is not evidence.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from tools import audit_documentation_scopes as gate


def _write_module(root: Path, scope: str, name: str, body: str) -> None:
    """Place one module inside a scanned scope."""
    directory = root / scope
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(body, encoding="utf-8")


def test_documented_module_passes(tmp_path: Path) -> None:
    """A fully documented module leaves the scope clean."""
    _write_module(
        tmp_path,
        "src",
        "ok.py",
        '"""A module that documents itself."""\n\n\ndef f() -> int:\n    """Return one."""\n    return 1\n',
    )
    assert gate.unexempt(gate.scan(tmp_path, ("src",)), tmp_path) == []


def test_undocumented_function_is_reported(tmp_path: Path) -> None:
    """An undocumented public function re-opens the scope and is caught."""
    _write_module(
        tmp_path,
        "src",
        "bare.py",
        '"""A module whose function is undocumented."""\n\n\ndef f() -> int:\n    return 1\n',
    )
    offenders = gate.unexempt(gate.scan(tmp_path, ("src",)), tmp_path)
    assert len(offenders) == 1
    assert "bare.py" in offenders[0]


def test_exempt_file_is_not_reported(tmp_path: Path) -> None:
    """A recorded exemption suppresses its own findings and nothing else."""
    exempt_path = next(iter(gate.EXEMPT))
    _write_module(
        tmp_path,
        str(Path(exempt_path).parent),
        Path(exempt_path).name,
        '"""Frozen runner."""\n\n\ndef f() -> int:\n    return 1\n',
    )
    findings = gate.scan(tmp_path, ("data",))
    assert findings, "the scan must still see the file; the exemption filters, it does not hide"
    assert gate.unexempt(findings, tmp_path) == []


def test_every_exemption_states_a_reason() -> None:
    """An exemption without a reason is an exclusion, and is not allowed."""
    for path, reason in gate.EXEMPT.items():
        assert reason.strip(), path
        assert len(reason.split()) >= 4, f"{path}: the reason is too thin to review"


def test_the_live_tree_holds_every_scope_at_zero() -> None:
    """The repository itself satisfies the rule the gate enforces."""
    root = Path(__file__).resolve().parent.parent
    assert gate.unexempt(gate.scan(root, gate.ENFORCED_SCOPES), root) == []


@pytest.mark.parametrize("missing_scope", gate.ENFORCED_SCOPES)
def test_cli_refuses_each_missing_required_scope(tmp_path: Path, missing_scope: str) -> None:
    """Removing any required scope must fail admission, not reduce the inventory."""
    for scope in gate.ENFORCED_SCOPES:
        if scope != missing_scope:
            _write_module(tmp_path, scope, "documented.py", '"""Scope marker."""\n')
    result = subprocess.run(
        [sys.executable, str(Path(gate.__file__).resolve()), "--repo", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert missing_scope in result.stderr
    assert "scopes clean" not in result.stdout


def test_cli_refuses_absent_repository(tmp_path: Path) -> None:
    """A mistyped checkout cannot produce a successful documentation report."""
    result = subprocess.run(
        [sys.executable, str(Path(gate.__file__).resolve()), "--repo", str(tmp_path / "absent")],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert "repository" in result.stderr
    assert "scopes clean" not in result.stdout


def test_scan_refuses_empty_scope_selection(tmp_path: Path) -> None:
    """The scan API requires at least one explicit owner scope."""
    with pytest.raises(ValueError, match="at least one"):
        gate.scan(tmp_path, ())


def test_cli_refuses_file_instead_of_scope_directory(tmp_path: Path) -> None:
    """A same-named file cannot replace a required source directory."""
    for scope in gate.ENFORCED_SCOPES:
        if scope != "src":
            _write_module(tmp_path, scope, "documented.py", '"""Scope marker."""\n')
    (tmp_path / "src").write_text("not a source directory", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(Path(gate.__file__).resolve()), "--repo", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 2
    assert "src" in result.stderr
    assert "scopes clean" not in result.stdout


def test_empty_inventory_is_refused_but_non_python_scope_is_reported(tmp_path: Path) -> None:
    """No source is not success; a manuscript-only directory reports zero files."""
    for scope in gate.ENFORCED_SCOPES:
        (tmp_path / scope).mkdir(parents=True)
    command = [sys.executable, str(Path(gate.__file__).resolve()), "--repo", str(tmp_path)]
    empty = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    assert empty.returncode == 2
    assert "no Python files" in empty.stderr
    _write_module(tmp_path, "src", "documented.py", '"""Source inventory marker."""\n')
    present = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    assert present.returncode == 0, present.stderr
    assert "paper: 0 Python file(s)" in present.stdout
    assert "src: 1 Python file(s)" in present.stdout


def test_cli_reports_actual_scope_inventory(tmp_path: Path) -> None:
    """A complete documented input passes; new undocumented code then fails."""
    for scope in gate.ENFORCED_SCOPES:
        _write_module(tmp_path, scope, "documented.py", '"""Scope marker."""\n')
    command = [sys.executable, str(Path(gate.__file__).resolve()), "--repo", str(tmp_path)]
    clean = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    assert clean.returncode == 0, clean.stderr
    assert f"{len(gate.ENFORCED_SCOPES)} scopes inspected" in clean.stdout
    for scope in gate.ENFORCED_SCOPES:
        assert f"    {scope}: 1 Python file(s)\n" in clean.stdout
    _write_module(tmp_path, "src", "undocumented.py", "def undocumented():\n    return 1\n")
    debt = subprocess.run(command, capture_output=True, text=True, check=False, timeout=30)
    assert debt.returncode == 1
    assert "undocumented.py" in debt.stdout
    assert "scopes clean" not in debt.stdout


def test_main_reports_admission_debt_and_success(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The callable CLI returns distinct statuses for input errors, clean input and debt."""
    arguments = ["--repo", str(tmp_path)]
    assert gate.main(arguments) == 2
    assert "not a directory: src" in capsys.readouterr().err
    for scope in gate.ENFORCED_SCOPES:
        _write_module(tmp_path, scope, "documented.py", '"""Scope marker."""\n')
    assert gate.main(arguments) == 0
    assert "scopes inspected" in capsys.readouterr().out
    _write_module(tmp_path, "src", "undocumented.py", "def undocumented():\n    return 1\n")
    assert gate.main(arguments) == 1
    assert "undocumented.py" in capsys.readouterr().out


def test_main_refuses_inventory_with_no_python_files(tmp_path: Path) -> None:
    """An all-empty inventory produces an incomplete-scan exit code."""
    for scope in gate.ENFORCED_SCOPES:
        (tmp_path / scope).mkdir(parents=True)
    assert gate.main(["--repo", str(tmp_path)]) == 2
    assert gate.main(["--repo", str(tmp_path / "absent")]) == 2


@pytest.mark.parametrize(
    "command",
    [
        ("-m", "module_that_does_not_exist_for_gate_probe"),
        ("-c", "raise SystemExit(2)"),
        ("-c", "print('{}')"),
        ("-c", "print('[null]')"),
        ("-c", "print('[]'); raise SystemExit(1)"),
    ],
)
def test_main_reports_linter_execution_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, command: tuple[str, ...]
) -> None:
    """A linter process failure cannot be interpreted as an empty finding list."""
    for scope in gate.ENFORCED_SCOPES:
        _write_module(tmp_path, scope, "documented.py", '"""Scope marker."""\n')
    monkeypatch.setattr(gate, "SCAN", command)
    assert gate.main(["--repo", str(tmp_path)]) == 2


def test_outside_finding_is_not_treated_as_exempt(tmp_path: Path) -> None:
    """A linter finding outside the selected root remains visible to the caller."""
    _write_module(tmp_path, "src", "undocumented.py", "def undocumented():\n    return 1\n")
    findings = gate.scan(tmp_path, ("src",))
    outside = gate.unexempt(findings, tmp_path / "different-root")
    assert outside
    assert all(str(tmp_path / "src" / "undocumented.py") in line for line in outside)
