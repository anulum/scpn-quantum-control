# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — experimental LLM-QPU baseline evidence tests
"""Exercise real baseline refusal, blocker and failure-custody behavior."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.llm_qpu_baseline_gate import (
    dependency_report,
    run_baseline_command,
    verify_source_anchor,
)


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments],
        capture_output=True,
        text=True,
        timeout=15,
        check=True,
    )
    return result.stdout.strip()


def test_source_anchor_refuses_changed_head_and_audited_bytes(tmp_path: Path) -> None:
    """A real Git checkout refuses both forms of stale baseline evidence."""
    repo = tmp_path / "checkout"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Baseline Test")
    _git(repo, "config", "user.email", "baseline@example.invalid")
    source = repo / "module.py"
    source.write_text("VALUE = 1\n")
    _git(repo, "add", "module.py")
    _git(repo, "commit", "-qm", "baseline")
    pinned_head = _git(repo, "rev-parse", "HEAD")
    pinned_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    anchor = {"module.py": pinned_hash}
    assert (
        verify_source_anchor(repo, expected_head=pinned_head, source_sha256=anchor)["status"]
        == "PASS"
    )

    source.write_text("VALUE = 2\n")
    changed = verify_source_anchor(repo, expected_head=pinned_head, source_sha256=anchor)
    assert changed["status"] == "REFUSED"
    changed_failures = changed["failures"]
    assert isinstance(changed_failures, list)
    assert "audited source changed: module.py" in changed_failures
    _git(repo, "add", "module.py")
    _git(repo, "commit", "-qm", "drift")
    moved = verify_source_anchor(repo, expected_head=pinned_head, source_sha256=anchor)
    assert moved["status"] == "REFUSED"
    moved_failures = moved["failures"]
    assert isinstance(moved_failures, list)
    assert "HEAD differs from audited baseline" in moved_failures


def test_missing_optional_sdk_is_blocked_in_real_isolated_python() -> None:
    """An interpreter without site packages cannot promote Qiskit/IQM to PASS."""
    report = dependency_report(("qiskit", "iqm.iqm_client"), isolated=True)
    assert report["status"] == "BLOCKED"
    assert report["missing_modules"] == ["qiskit", "iqm.iqm_client"]
    assert dependency_report(("json",), isolated=True)["status"] == "PASS"


def test_failed_baseline_command_preserves_exit_and_cause(tmp_path: Path) -> None:
    """A real failing child process remains a FAIL in its durable report."""
    destination = tmp_path / "baseline" / "command.json"
    report = run_baseline_command(
        [sys.executable, "-c", "import sys; sys.stderr.write('original failure\\n'); sys.exit(7)"],
        cwd=tmp_path,
        report_path=destination,
    )
    recorded = json.loads(destination.read_text())
    assert report == recorded
    assert recorded["status"] == "FAIL"
    assert recorded["exit_code"] == 7
    assert "original failure" in recorded["stderr"]


def test_baseline_child_cannot_read_inherited_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The actual child sees neither poisoned provider key nor original HOME."""
    monkeypatch.setenv("IQM_TOKEN", "poison")
    monkeypatch.setenv("IBM_QUANTUM_TOKEN", "poison")
    original_home = str(Path.home())
    script = (
        "import os,pathlib;"
        "assert 'IQM_TOKEN' not in os.environ;"
        "assert 'IBM_QUANTUM_TOKEN' not in os.environ;"
        f"assert str(pathlib.Path.home()) != {original_home!r};"
        "assert not (pathlib.Path.home()/'.config/scpn-quantum-control/credentials.md').exists()"
    )
    result = run_baseline_command(
        [sys.executable, "-I", "-S", "-c", script],
        cwd=tmp_path,
        report_path=tmp_path / "offline.json",
    )
    assert result["status"] == "PASS"
