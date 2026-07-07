# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — main-branch policy tests
"""Tests for the local main-branch-only Git policy guard."""

from __future__ import annotations

import stat
import subprocess
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
    assert hook_path.read_text(encoding="utf-8").startswith("#!/usr/bin/env sh")
    assert hook_path.stat().st_mode & stat.S_IXUSR
