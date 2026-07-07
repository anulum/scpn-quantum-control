#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Main-branch-only Git policy
"""Enforce SCPN-QUANTUM-CONTROL's main-branch-only Git workflow.

The repository policy is intentionally stricter than normal Git collaboration:
local branches other than ``main`` must not be created or updated, and pushes
may only update ``main`` or delete non-main branches during cleanup. A
``reference-transaction`` hook blocks local branch creation/update before the
ref is written; the pre-push entry blocks accidental remote branch publication.
"""

from __future__ import annotations

import argparse
import os
import shutil
import stat
import subprocess  # nosec B404
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ZERO_OID = "0" * 40
MAIN_REF = "refs/heads/main"
RefOidResolver = Callable[[str], str | None]
FORBIDDEN_AGENTIC_TOKENS = (
    "agent",
    "codex",
    "copilot",
    "cursor",
    "devin",
    "fable",
    "gemini",
    "gpt",
    "llm",
    "openai",
    "windsurf",
)


@dataclass(frozen=True)
class BranchPolicyFinding:
    """One branch-policy decision emitted by a Git hook adapter."""

    ref: str
    reason: str


def is_agentic_branch_name(ref: str) -> bool:
    """Return whether ``ref`` contains a forbidden agentic branch token."""
    branch_name = ref.removeprefix("refs/heads/").lower()
    return any(token in branch_name for token in FORBIDDEN_AGENTIC_TOKENS)


def _is_branch_deletion(old_oid: str, new_oid: str) -> bool:
    """Return whether a reference transaction deletes a local branch."""
    return new_oid == ZERO_OID


def _is_branch_update(new_oid: str) -> bool:
    """Return whether a reference transaction keeps a branch alive."""
    return new_oid != ZERO_OID


def _is_existing_ref_lock(
    *,
    old_oid: str,
    new_oid: str,
    ref: str,
    ref_oid_resolver: RefOidResolver | None,
) -> bool:
    """Return whether Git is locking an existing ref without changing it.

    Git may report a zero ``old_oid`` for force-style reference transactions
    even when the ref already exists. During branch deletion this lock can
    precede the actual all-zero delete transaction. The guard allows only the
    exact no-op lock where the current ref already resolves to ``new_oid``; new
    branches and branch moves still fail.
    """
    if old_oid != ZERO_OID or new_oid == ZERO_OID or ref_oid_resolver is None:
        return False
    return ref_oid_resolver(ref) == new_oid


def evaluate_reference_transaction(
    lines: Iterable[str],
    state: str,
    ref_oid_resolver: RefOidResolver | None = None,
) -> tuple[BranchPolicyFinding, ...]:
    """Evaluate ``reference-transaction`` hook input lines."""
    if state != "prepared":
        return ()
    findings: list[BranchPolicyFinding] = []
    for raw_line in lines:
        parts = raw_line.strip().split()
        if len(parts) != 3:
            continue
        old_oid, new_oid, ref = parts
        if not ref.startswith("refs/heads/"):
            continue
        if ref == MAIN_REF and _is_branch_update(new_oid):
            continue
        if ref == MAIN_REF:
            findings.append(BranchPolicyFinding(ref=ref, reason="main branch may not be deleted"))
            continue
        if _is_branch_deletion(old_oid, new_oid):
            continue
        if _is_existing_ref_lock(
            old_oid=old_oid,
            new_oid=new_oid,
            ref=ref,
            ref_oid_resolver=ref_oid_resolver,
        ):
            continue
        if is_agentic_branch_name(ref):
            reason = "agentic branch names are forbidden and only main may be updated"
        else:
            reason = "local branches are forbidden; commit on main only"
        findings.append(BranchPolicyFinding(ref=ref, reason=reason))
    return tuple(findings)


def evaluate_pre_push(lines: Iterable[str]) -> tuple[BranchPolicyFinding, ...]:
    """Evaluate pre-push hook input lines."""
    findings: list[BranchPolicyFinding] = []
    for raw_line in lines:
        parts = raw_line.strip().split()
        if len(parts) != 4:
            continue
        local_ref, local_oid, remote_ref, _remote_oid = parts
        if local_oid == ZERO_OID:
            if remote_ref == MAIN_REF:
                findings.append(
                    BranchPolicyFinding(
                        ref=remote_ref,
                        reason="main branch may not be deleted",
                    )
                )
            continue
        if remote_ref == MAIN_REF and local_ref == MAIN_REF:
            continue
        if remote_ref.startswith("refs/heads/") or local_ref.startswith("refs/heads/"):
            if is_agentic_branch_name(remote_ref) or is_agentic_branch_name(local_ref):
                reason = "agentic branch names must not be pushed"
            else:
                reason = "only refs/heads/main may be pushed"
            findings.append(BranchPolicyFinding(ref=remote_ref, reason=reason))
    return tuple(findings)


def render_findings(findings: Sequence[BranchPolicyFinding]) -> str:
    """Render findings for Git hook stderr output."""
    lines = ["SCPN-QUANTUM-CONTROL branch policy violation:"]
    for finding in findings:
        lines.append(f"- {finding.ref}: {finding.reason}")
    return "\n".join(lines)


def _resolve_git_executable() -> str:
    """Return a trusted Git executable path."""
    located = shutil.which("git")
    if located is None:
        raise RuntimeError("git executable was not found on PATH")
    resolved = Path(located).resolve(strict=True)
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        raise RuntimeError(f"git executable is not executable: {resolved}")
    return str(resolved)


def _repo_root() -> Path:
    """Return the current Git worktree root."""
    completed = subprocess.run(  # nosec B603
        [_resolve_git_executable(), "rev-parse", "--show-toplevel"],
        capture_output=True,
        check=True,
        text=True,
    )
    return Path(completed.stdout.strip()).resolve(strict=True)


def _git_path(repo_root: Path, path: str) -> Path:
    """Return Git's resolved path for a repository-managed file."""
    completed = subprocess.run(  # nosec B603
        [_resolve_git_executable(), "-C", str(repo_root), "rev-parse", "--git-path", path],
        capture_output=True,
        check=True,
        text=True,
    )
    resolved = Path(completed.stdout.strip())
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved.resolve()


def _current_ref_oid(ref: str) -> str | None:
    """Return the currently stored object ID for ``ref`` when it exists."""
    completed = subprocess.run(  # nosec B603
        [_resolve_git_executable(), "rev-parse", "--verify", "--quiet", ref],
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        return None
    oid = completed.stdout.strip().splitlines()
    if not oid:
        return None
    return oid[0]


def install_reference_transaction_hook(repo_root: Path) -> Path:
    """Install the local reference-transaction hook for ``repo_root``."""
    hook_path = _git_path(repo_root, "hooks/reference-transaction")
    hook_path.parent.mkdir(parents=True, exist_ok=True)
    hook_path.write_text(
        "\n".join(
            (
                "#!/usr/bin/env sh",
                "# Installed by tools/enforce_main_branch_policy.py",
                'repo_root="$(git rev-parse --show-toplevel)" || exit 1',
                'python_bin="${SCPN_QC_PYTHON:-python3}"',
                'exec "$python_bin" "$repo_root/tools/enforce_main_branch_policy.py" '
                'reference-transaction "$@"',
                "",
            )
        ),
        encoding="utf-8",
    )
    hook_path.chmod(hook_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return hook_path


def _fail_if_needed(findings: Sequence[BranchPolicyFinding]) -> int:
    """Print findings and return a hook-compatible exit code."""
    if not findings:
        return 0
    print(render_findings(findings), file=sys.stderr)
    return 1


def main(argv: Sequence[str] | None = None) -> int:
    """Run the branch-policy command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    reference_parser = subparsers.add_parser("reference-transaction")
    reference_parser.add_argument("state", choices=("prepared", "committed", "aborted"))
    subparsers.add_parser("pre-push")
    subparsers.add_parser("install")
    args = parser.parse_args(argv)

    if args.command == "reference-transaction":
        return _fail_if_needed(
            evaluate_reference_transaction(sys.stdin, args.state, _current_ref_oid)
        )
    if args.command == "pre-push":
        return _fail_if_needed(evaluate_pre_push(sys.stdin))
    if args.command == "install":
        hook_path = install_reference_transaction_hook(_repo_root())
        print(f"installed {hook_path}")
        return 0
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
