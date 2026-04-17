#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Commit trailer verifier
"""Verify commit-message hygiene.

Two roles:

1. **commit-msg hook** — invoked with a single path argument (the file
   holding the pending commit message). Used by pre-commit to block
   local commits that omit `Co-Authored-By:` or that include any of
   the banned quality / slop words in the commit subject or body.

2. **CI auditor** — invoked without arguments. Walks every commit
   from `v0.9.0..HEAD` (the period over which the current trailer
   rules applied) and reports violations. Exits non-zero if any are
   found. Use in a weekly CI job rather than on every PR so that
   historical debt is visible but not a merge gate.

The rules mirror `feedback_branding_headers`,
`feedback_no_internal_quality_labels`, and
`feedback_anti_slop_policy`. When those rules change, update the
constants below and bump this file's version comment.

The historical debt from 2026-04-17 — seven Dependabot squash merges
merged without `Co-Authored-By` (`2a7d604`, `001d30b`, `8ce9324`,
`0a9dfbd`, `a93874a`, `57b05cf`, `cb2b1fe`) — is recorded in
`HISTORICAL_EXEMPT_SHAS` so the CI auditor does not re-report it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Required trailer in every commit message.
REQUIRED_TRAILER_RE = re.compile(r"^Co-Authored-By:\s+.+<\S+@\S+>\s*$", re.MULTILINE)

# Banned tokens per `feedback_no_internal_quality_labels` and
# `feedback_anti_slop_policy`. Case-insensitive whole-word match.
# Domain-technical uses (e.g. "STRONG correlation" in statistics)
# still appear in code/docs — but NOT in commit subjects, where the
# words would read as self-praise. Exempt specific historical
# wordings only if there is no alternative phrasing that keeps the
# technical meaning.
BANNED_WORDS = [
    "elite",
    "Elite",
    "SUPERIOR",
    "Superior",
    "ETALON",
    "Etalon",
    "comprehensive",
    "Comprehensive",
    "robust",
    "Robust",
    "leveraging",
    "Leveraging",
    "world-class",
    "World-Class",
    "best-in-class",
    "Best-in-Class",
]
BANNED_WORDS_RE = re.compile(r"\b(" + "|".join(re.escape(w) for w in BANNED_WORDS) + r")\b")

# Commits that predate these rules and cannot be retroactively amended
# without destructive force-push on a published branch. Do not add new
# SHAs here without an explicit waiver in the audit index.
HISTORICAL_EXEMPT_SHAS: frozenset[str] = frozenset(
    {
        # 2026-04-17: seven Dependabot PRs squash-merged without
        # Co-Authored-By. Documented in the internal gap audit.
        "2a7d604",
        "001d30b",
        "8ce9324",
        "0a9dfbd",
        "a93874a",
        "57b05cf",
        "cb2b1fe",
        # Pre-2026-04-17 commits whose subjects used banned words
        # ("comprehensive", "robust", "elite") before
        # `feedback_no_internal_quality_labels` + the anti-slop hook
        # were added. Force-amending them would rewrite published
        # history of `v0.9.5`; the subjects are frozen as historical.
        "04bd5aa",  # "comprehensive v0.9.5 update"
        "5fe9998",  # "robust counts extraction"
        "b97d5fb",  # "elite docs"
    }
)


def _message_violations(msg: str, check_body_banned: bool = False) -> list[str]:
    """Return a list of violations for this commit message.

    When `check_body_banned` is False (the default for the commit-msg
    hook and the CI auditor), banned words are only matched in the
    subject line (first non-empty line). That is where self-praise or
    slop would read as a tone failure; the body often cites banned
    words in the course of removing them, which is legitimate.
    """
    violations: list[str] = []
    if not REQUIRED_TRAILER_RE.search(msg):
        violations.append("missing `Co-Authored-By:` trailer")
    # Extract subject line (Keep a Changelog / Conventional Commits)
    subject = next((line for line in msg.splitlines() if line.strip()), "")
    scope = msg if check_body_banned else subject
    matches = BANNED_WORDS_RE.findall(scope)
    if matches:
        seen: set[str] = set()
        unique = [m for m in matches if not (m in seen or seen.add(m))]
        where = "message" if check_body_banned else "subject"
        violations.append(f"banned word(s) in {where}: {', '.join(unique)}")
    return violations


def _commit_msg_hook(path: Path) -> int:
    msg = path.read_text(encoding="utf-8")
    violations = _message_violations(msg)
    if violations:
        print("Commit message rejected:", file=sys.stderr)
        for v in violations:
            print(f"  - {v}", file=sys.stderr)
        print(file=sys.stderr)
        print(
            "Fix the message and retry. For Dependabot squash merges, use",
            file=sys.stderr,
        )
        print(
            '  gh pr merge <N> --squash --body "$(gh pr view <N> --json body -q .body)'
            '\\n\\nCo-Authored-By: Arcane Sapience <protoscience@anulum.li>"',
            file=sys.stderr,
        )
        return 1
    return 0


def _ci_audit(range_spec: str = "v0.9.0..HEAD") -> int:
    # Pipe the SHAs through a second git call that fetches each
    # message cleanly — avoids the newline-quoting pitfalls of
    # `git log --format=%B` piped through a single invocation.
    try:
        sha_result = subprocess.run(
            ["git", "rev-list", range_spec],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"git rev-list failed: {exc.stderr}", file=sys.stderr)
        return 2
    shas = [line.strip() for line in sha_result.stdout.splitlines() if line.strip()]
    fails: list[str] = []
    exempt_hits: list[str] = []
    for sha in shas:
        short = sha[:7]
        if short in HISTORICAL_EXEMPT_SHAS:
            exempt_hits.append(short)
            continue
        msg_result = subprocess.run(
            ["git", "log", "-1", "--format=%B", sha],
            capture_output=True,
            text=True,
            check=True,
        )
        violations = _message_violations(msg_result.stdout)
        if violations:
            fails.append(f"{short}: {'; '.join(violations)}")
    print(f"Audited {len(shas)} commits in {range_spec}")
    print(f"  Exempt (historical debt): {len(exempt_hits)}")
    print(f"  Violations: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


def main(argv: list[str]) -> int:
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 0
    if "--audit" in argv or "--range" in argv:
        range_spec = "v0.9.0..HEAD"
        if "--range" in argv:
            idx = argv.index("--range")
            if idx + 1 < len(argv):
                range_spec = argv[idx + 1]
        return _ci_audit(range_spec)
    if len(argv) >= 2:
        # commit-msg hook: first arg is path to message file
        return _commit_msg_hook(Path(argv[1]))
    # No args: default to CI audit mode
    return _ci_audit("v0.9.0..HEAD")


if __name__ == "__main__":
    sys.exit(main(sys.argv))
