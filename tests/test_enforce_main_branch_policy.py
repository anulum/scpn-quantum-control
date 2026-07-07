# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — main-branch policy tests
"""Tests for the local main-branch-only Git policy guard."""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

from tools.enforce_main_branch_policy import (
    ZERO_OID,
    evaluate_pre_push,
    evaluate_reference_transaction,
    install_reference_transaction_hook,
    is_agentic_branch_name,
)


def test_reference_transaction_allows_main_updates_and_branch_deletions() -> None:
    """The hook must allow main updates and non-main branch cleanup."""
    findings = evaluate_reference_transaction(
        (
            f"{ZERO_OID} {'1' * 40} refs/heads/main",
            f"{'2' * 40} {ZERO_OID} refs/heads/codex/old-lane",
            f"{ZERO_OID} {ZERO_OID} refs/heads/codex/zero-delete",
        ),
        "prepared",
    )

    assert findings == ()


def test_reference_transaction_blocks_main_deletion() -> None:
    """The hook must reject attempts to delete the main branch."""
    findings = evaluate_reference_transaction(
        (f"{'1' * 40} {ZERO_OID} refs/heads/main",),
        "prepared",
    )

    assert len(findings) == 1
    assert findings[0].ref == "refs/heads/main"
    assert "may not be deleted" in findings[0].reason


def test_reference_transaction_allows_existing_ref_lock_for_cleanup() -> None:
    """The hook must allow Git's no-op ref lock before branch deletion."""
    existing_oid = "4" * 40
    findings = evaluate_reference_transaction(
        (f"{ZERO_OID} {existing_oid} refs/heads/codex/cleanup-target",),
        "prepared",
        lambda ref: existing_oid if ref == "refs/heads/codex/cleanup-target" else None,
    )

    assert findings == ()


def test_reference_transaction_rejects_non_main_and_agentic_branches() -> None:
    """The hook must block every non-main branch update before ref creation."""
    findings = evaluate_reference_transaction(
        (
            f"{ZERO_OID} {'1' * 40} refs/heads/feature/test",
            f"{ZERO_OID} {'2' * 40} refs/heads/codex/fuzz-lane",
        ),
        "prepared",
    )

    assert [finding.ref for finding in findings] == [
        "refs/heads/feature/test",
        "refs/heads/codex/fuzz-lane",
    ]
    assert "commit on main only" in findings[0].reason
    assert "agentic branch names" in findings[1].reason


def test_pre_push_rejects_non_main_branch_publication() -> None:
    """The pre-push adapter must allow main and deletion, not branch pushes."""
    findings = evaluate_pre_push(
        (
            f"refs/heads/main {'1' * 40} refs/heads/main {'2' * 40}",
            f"refs/heads/main {ZERO_OID} refs/heads/codex/old {'3' * 40}",
            f"refs/heads/codex/new {'4' * 40} refs/heads/codex/new {ZERO_OID}",
            f"refs/heads/main {ZERO_OID} refs/heads/main {'5' * 40}",
        )
    )

    assert len(findings) == 2
    assert findings[0].ref == "refs/heads/codex/new"
    assert "agentic branch names" in findings[0].reason
    assert findings[1].ref == "refs/heads/main"
    assert "may not be deleted" in findings[1].reason


def test_agentic_branch_token_detection() -> None:
    """Known agentic branch labels must be classified as forbidden."""
    assert is_agentic_branch_name("refs/heads/fable/waiting")
    assert is_agentic_branch_name("refs/heads/codex/pinv")
    assert not is_agentic_branch_name("refs/heads/main")


def test_install_writes_executable_reference_transaction_hook(tmp_path: Path) -> None:
    """Installer must create an executable local hook shim."""
    subprocess.run(
        ("git", "init", "--initial-branch=main", str(tmp_path)),
        check=True,
        capture_output=True,
        text=True,
    )
    hooks_dir = tmp_path / ".git" / "hooks"

    hook_path = install_reference_transaction_hook(tmp_path)

    assert hook_path == hooks_dir / "reference-transaction"
    hook_text = hook_path.read_text(encoding="utf-8")
    assert hook_text.startswith("#!/usr/bin/env sh")
    assert "--git-common-dir" in hook_text
    assert "skipping branch policy" in hook_text
    assert hook_path.stat().st_mode & stat.S_IXUSR


def _run_git(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one git command for the worktree-fallback hook tests."""
    return subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "policy-test",
            "GIT_AUTHOR_EMAIL": "policy-test@example.invalid",
            "GIT_COMMITTER_NAME": "policy-test",
            "GIT_COMMITTER_EMAIL": "policy-test@example.invalid",
        },
    )


def test_cli_passes_through_undocumented_reference_transaction_states(tmp_path: Path) -> None:
    """Unknown git hook phases (e.g. ``preparing``) must not abort ref updates."""
    script = Path(__file__).resolve().parents[1] / "tools" / "enforce_main_branch_policy.py"
    forbidden_line = f"{ZERO_OID} {'1' * 40} refs/heads/forbidden-branch\n"
    passthrough = subprocess.run(
        (sys.executable, str(script), "reference-transaction", "preparing"),
        input=forbidden_line,
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert passthrough.returncode == 0, passthrough.stderr
    prepared = subprocess.run(
        (sys.executable, str(script), "reference-transaction", "prepared"),
        input=forbidden_line,
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
    )
    assert prepared.returncode == 1
    assert "branch policy violation" in prepared.stderr


def test_hook_falls_back_to_primary_checkout_for_treeless_worktrees(tmp_path: Path) -> None:
    """Worktrees at commits without the script must still get a policy verdict."""
    primary = tmp_path / "primary"
    primary.mkdir()
    assert _run_git("init", "--initial-branch=main", str(primary), cwd=tmp_path).returncode == 0

    (primary / "seed.txt").write_text("seed\n", encoding="utf-8")
    assert _run_git("add", "seed.txt", cwd=primary).returncode == 0
    assert _run_git("commit", "-m", "seed without policy script", cwd=primary).returncode == 0

    tools_dir = primary / "tools"
    tools_dir.mkdir()
    script_source = Path(__file__).resolve().parents[1] / "tools" / "enforce_main_branch_policy.py"
    (tools_dir / "enforce_main_branch_policy.py").write_text(
        script_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    assert _run_git("add", "tools/enforce_main_branch_policy.py", cwd=primary).returncode == 0
    assert _run_git("commit", "-m", "add policy script", cwd=primary).returncode == 0
    install_reference_transaction_hook(primary)

    seed_commit = _run_git("rev-parse", "HEAD~1", cwd=primary).stdout.strip()
    worktree = tmp_path / "treeless-worktree"
    added = _run_git("worktree", "add", "--detach", str(worktree), seed_commit, cwd=primary)
    assert added.returncode == 0, added.stderr
    assert not (worktree / "tools" / "enforce_main_branch_policy.py").exists()

    (worktree / "change.txt").write_text("change\n", encoding="utf-8")
    assert _run_git("add", "change.txt", cwd=worktree).returncode == 0
    detached_commit = _run_git("commit", "-m", "detached commit under policy", cwd=worktree)
    assert detached_commit.returncode == 0, detached_commit.stderr

    branch_update = _run_git("branch", "forbidden-branch", cwd=worktree)
    assert branch_update.returncode != 0
    assert "branch policy violation" in (branch_update.stderr + branch_update.stdout)
