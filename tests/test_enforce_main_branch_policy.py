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

import pytest

from tools import enforce_main_branch_policy as policy
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


def test_zero_old_deletion_cannot_use_prune_exception() -> None:
    """A packed delete transaction has no old OID and must always be refused."""
    findings = evaluate_reference_transaction(
        [f"{ZERO_OID} {ZERO_OID} refs/heads/main"],
        "prepared",
        ref_prune_verifier=lambda ref, oid: True,
    )
    assert len(findings) == 1
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


@pytest.mark.parametrize("storage", ["loose", "packed", "mixed", "stale-packed"])
@pytest.mark.parametrize("compare_old", [False, True])
def test_pack_refs_preserves_main_and_deletion_remains_blocked(
    tmp_path: Path, storage: str, compare_old: bool
) -> None:
    """Real Git packing preserves refs while deletion fails in every storage state."""
    assert _run_git("init", "--initial-branch=main", str(tmp_path), cwd=tmp_path).returncode == 0
    assert _run_git("commit", "--allow-empty", "-m", "seed", cwd=tmp_path).returncode == 0
    tools_dir = tmp_path / "tools"
    tools_dir.mkdir()
    source = Path(__file__).resolve().parents[1] / "tools/enforce_main_branch_policy.py"
    (tools_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    install_reference_transaction_hook(tmp_path)
    before = _run_git("show-ref", cwd=tmp_path).stdout
    if storage != "loose":
        packed = _run_git("pack-refs", "--all", "--prune", cwd=tmp_path)
        assert packed.returncode == 0, packed.stderr
        assert _run_git("show-ref", cwd=tmp_path).stdout == before
        assert not (tmp_path / ".git/refs/heads/main").exists()
        if storage in ("mixed", "stale-packed"):
            assert _run_git("commit", "--allow-empty", "-m", "next", cwd=tmp_path).returncode == 0
            if storage == "mixed":
                assert _run_git("pack-refs", "--all", "--no-prune", cwd=tmp_path).returncode == 0
    before = _run_git("show-ref", cwd=tmp_path).stdout
    old_oid = _run_git("rev-parse", "HEAD", cwd=tmp_path).stdout.strip()
    arguments = (old_oid,) if compare_old else ()
    deleted = _run_git("update-ref", "-d", "refs/heads/main", *arguments, cwd=tmp_path)
    assert deleted.returncode != 0
    assert "main branch may not be deleted" in deleted.stderr
    assert _run_git("show-ref", cwd=tmp_path).stdout == before
    assert _run_git("rev-parse", "--verify", "HEAD", cwd=tmp_path).returncode == 0


@pytest.mark.parametrize(
    "contents",
    [
        None,
        b"",
        b"\xff",
        b"garbage refs/heads/main\n",
        b"# packed refs\n",
        (f"{'1' * 40} refs/heads/main\n" * 2).encode(),
        f"{'2' * 40} refs/heads/main\n".encode(),
    ],
)
def test_prune_verification_fails_closed_for_missing_or_ambiguous_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, contents: bytes | None
) -> None:
    """Unavailable, stale, duplicate and malformed packed copies cannot permit deletion."""
    packed = tmp_path / "packed-refs"
    if contents is not None:
        packed.write_bytes(contents)
    monkeypatch.setattr(policy, "_git_path", lambda root, name: packed)
    assert not policy._is_packed_ref_prune("refs/heads/main", "1" * 40)


def test_prune_verification_requires_unlocked_identical_packed_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Packed storage proof never permits a transaction holding its deletion lock."""
    packed = tmp_path / "packed-refs"
    packed.write_text(f"# pack-refs with: peeled\n{'1' * 40} refs/heads/main\n", encoding="ascii")
    monkeypatch.setattr(policy, "_git_path", lambda root, name: packed)
    line = f"{'1' * 40} {ZERO_OID} refs/heads/main"
    assert (
        evaluate_reference_transaction(
            [line], "prepared", ref_prune_verifier=policy._is_packed_ref_prune
        )
        == ()
    )
    (tmp_path / "packed-refs.lock").touch()
    assert evaluate_reference_transaction(
        [line], "prepared", ref_prune_verifier=policy._is_packed_ref_prune
    )


def test_prune_verification_rechecks_lock_after_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lock appearing during inspection prevents a positive storage verdict."""
    packed = tmp_path / "packed-refs"
    monkeypatch.setattr(policy, "_git_path", lambda root, name: packed)

    def read_with_lock(path: Path, encoding: str) -> str:
        """Model a concurrent packed-ref transaction taking its lock."""
        (tmp_path / "packed-refs.lock").touch()
        return f"{'1' * 40} refs/heads/main\n"

    monkeypatch.setattr(Path, "read_text", read_with_lock)
    assert not policy._is_packed_ref_prune("refs/heads/main", "1" * 40)


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
    before = _run_git("show-ref", cwd=worktree).stdout
    packed = _run_git("pack-refs", "--all", "--prune", cwd=worktree)
    assert packed.returncode == 0, packed.stderr
    assert _run_git("show-ref", cwd=worktree).stdout == before
