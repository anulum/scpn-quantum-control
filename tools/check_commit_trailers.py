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
   local commits that omit `Seat` / `Authored by` trailers or that
   include any of the banned quality / slop words in the commit subject.

2. **CI auditor** — invoked without arguments. Walks every commit
   from `v0.9.6..HEAD` (the first public tag whose post-rule history is
   clean) and reports violations. Exits non-zero if any are found. Use
   in a weekly CI job rather than on every PR so that historical debt is
   visible but not a merge gate.

The rules mirror `feedback_branding_headers`,
`feedback_no_internal_quality_labels`, and
`feedback_anti_slop_policy`. When those rules change, update the
constants below and bump this file's version comment.

The historical debt from 2026-04-17 — seven Dependabot squash merges
merged without `Co-Authored-By` (`2a7d604`, `001d30b`, `8ce9324`,
`0a9dfbd`, `a93874a`, `57b05cf`, `cb2b1fe`) — is recorded in
`HISTORICAL_EXEMPT_SHAS` so the CI auditor does not re-report it.
The same set also records any later immutable protected-branch merge
that cannot be rewritten without violating branch protection.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess  # nosec B404
import sys
from datetime import datetime
from pathlib import Path

# Required authorship line in every new commit message. This is deliberately
# not a Git `Co-Authored-By` trailer; the project policy is forward-only from
# AUTHORSHIP_POLICY_EFFECTIVE_UTC and historical trailers remain accepted only
# for commits whose immutable commit timestamp predates that boundary.
REQUIRED_AUTHORSHIP_LINE = "Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)"
AUTHORSHIP_POLICY_EFFECTIVE_UTC = datetime.fromisoformat("2026-05-31T21:46:25+00:00")
LEGACY_COAUTHOR_TRAILER_RE = re.compile(
    r"^Co-Authored-By:\s+Arcane Sapience\s+<protoscience@anulum\.li>\s*$",
    re.MULTILINE,
)
SEAT_TRAILER_RE = re.compile(r"^Seat:\s+([A-Za-z0-9][A-Za-z0-9_-]{0,63})\s*$")
SEAT_TRAILER_PREFIX_RE = re.compile(r"^\s*Seat:")
FORBIDDEN_SEAT_PREFIXES = ("claude-", "codex-")

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
        # 2026-05-12: Dependabot coverage squash merge landed through
        # protected branch UI without the required trailer; branch protection
        # blocks retroactive repair by force-push.
        "4bbcc87",
        # 2026-05-18: five Dependabot squash merges were created through
        # the protected branch UI with a literal escaped "\n\n" before the
        # trailer, so Git did not parse a real authorship line.
        # Force-amending them would rewrite published main history; the
        # waiver is recorded in docs/internal/AUDIT_INDEX.md.
        "d07a7ea",
        "5f4df4a",
        "57b6545",
        "d44bad7",
        "9e70460",
        # Pre-2026-04-17 commits whose subjects used banned words
        # ("comprehensive", "robust", "elite") before
        # `feedback_no_internal_quality_labels` + the anti-slop hook
        # were added. Force-amending them would rewrite published
        # history of `v0.9.5`; the subjects are frozen as historical.
        "04bd5aa",  # "comprehensive v0.9.5 update"
        "5fe9998",  # "robust counts extraction"
        "b97d5fb",  # "elite docs"
        # 2026-07-01: the actions/cache 4.2.4->6.1.0 Dependabot PR (#117)
        # was squash-merged with the auto-generated GitHub message, which
        # omitted the required authorship line. `main` has
        # `allow_force_pushes` disabled, so the merge commit cannot be
        # amended without a destructive, protection-defeating rewrite under
        # a live co-agent. Recorded as immutable debt; the remaining
        # Dependabot merges pass the trailer via `--body`. Waiver in
        # docs/internal/AUDIT_INDEX.md.
        "9bf247c",  # "build(deps): bump actions/cache from 4.2.4 to 6.1.0 (#117)"
        # 2026-05-20: five roadmap/readiness-gate commits (S7–S11) were
        # authored before the local authorship-trailer hook was armed on
        # this workstation. They are already on protected `main`, which
        # disallows the force-push that retroactive repair would require.
        "cc15eb0",  # "Add S7 logical DLA parity roadmap gate"
        "03c4aef",  # "Add S8 adaptive branching readiness gate"
        "b9cb117",  # "Add S9 quantum thermodynamics readiness gate"
        "261074d",  # "Add S10 analog-native readiness gate"
        "c542990",  # "Add S11 quantum sensing readiness gate"
        # 2026-05-25/26: ten Dependabot squash merges (#79–#86 plus Rust
        # crate bumps) landed through the protected-branch UI without the
        # trailer.
        "adfd598",  # "Bump actions/stale 10.2.0 -> 10.3.0 (#84)"
        "ce098dd",  # "Bump codecov/codecov-action 6.0.0 -> 6.0.1 (#79)"
        "fda13e6",  # "Bump github/codeql-action 4.35.5 -> 4.36.0 (#83)"
        "34af5fd",  # "Bump pip-audit 2.9.0 -> 2.10.0 (#80)"
        "9d232e5",  # "Bump pydantic-settings 2.14.0 -> 2.14.1 (#82)"
        "3f360a6",  # "Bump qiskit-ibm-runtime 0.46.1 -> 0.47.0 (#81)"
        "6dfd700",  # "Bump reqwest 0.12.28 -> 0.13.3"
        "093457e",  # "Bump ruff 0.15.12 -> 0.15.14 (#86)"
        "cf5c069",  # "Bump serde_json 1.0.149 -> 1.0.150"
        "908e340",  # "Bump mypy 1.20.1 -> 2.1.0 (#85)"
        # 2026-06-15: seven build(deps) Dependabot squash merges without
        # the trailer (protected-branch UI).
        "b87216d",  # "build(deps): bump chrono to 0.4.45"
        "9d4e6cf",  # "build(deps): bump codecov-action to 7.0.0"
        "b233a8c",  # "build(deps): bump github/codeql-action to 4.36.2"
        "c0c3fb6",  # "build(deps): bump requests to 2.34.2"
        "2858406",  # "build(deps): bump sc-neurocore to 3.15.33"
        "6c4c85a",  # "build(deps-dev): bump ruff to 0.15.16"
        "001682c",  # "build(deps-dev): bump structlog to 26.1.0"
        # 2026-07-07: `fix: repair ci gate drift` predates a trailer-hook
        # re-arm; already on `main`, unamendable under a live co-agent.
        "2fd17e6",  # "fix: repair ci gate drift"
        # 2026-07-14: the setuptools 83.0.0 revert was created with
        # `git revert --no-edit`, which does not carry the trailer;
        # forward-fixed here rather than by rewriting pushed history.
        "9af0b25",  # Revert "build(deps): bump setuptools 81.0.0 -> 83.0.0"
    }
)


def _has_required_authorship_line(msg: str) -> bool:
    """Return True when the exact required authorship line is present."""
    return any(line.strip() == REQUIRED_AUTHORSHIP_LINE for line in msg.splitlines())


def _seat_trailer_violations(msg: str) -> list[str]:
    """Return violations for the forward-only agent seat trailer."""
    lines = msg.splitlines()
    seat_indices = [
        index for index, line in enumerate(lines) if SEAT_TRAILER_PREFIX_RE.match(line)
    ]
    if not seat_indices:
        return ["missing `Seat: <seat-id>` trailer"]
    violations: list[str] = []
    if len(seat_indices) != 1:
        violations.append("expected exactly one `Seat: <seat-id>` trailer")
        return violations

    seat_index = seat_indices[0]
    seat_line = lines[seat_index].strip()
    match = SEAT_TRAILER_RE.match(seat_line)
    if match is None:
        violations.append("invalid `Seat: <seat-id>` trailer")
        return violations

    seat_id = match.group(1).lower()
    if any(seat_id.startswith(prefix) for prefix in FORBIDDEN_SEAT_PREFIXES):
        violations.append("vendor-prefixed `Seat:` trailer is forbidden")

    authorship_indices = [
        index for index, line in enumerate(lines) if line.strip() == REQUIRED_AUTHORSHIP_LINE
    ]
    if len(authorship_indices) == 1:
        authorship_index = authorship_indices[0]
        between = lines[seat_index + 1 : authorship_index]
        if seat_index >= authorship_index or any(line.strip() for line in between):
            violations.append("`Seat:` trailer must immediately precede the authorship line")
    return violations


def _message_violations(
    msg: str,
    check_body_banned: bool = False,
    allow_legacy_trailer: bool = False,
    require_seat_trailer: bool = False,
) -> list[str]:
    """Return a list of violations for this commit message.

    When `check_body_banned` is False (the default for the commit-msg
    hook and the CI auditor), banned words are only matched in the
    subject line (first non-empty line). That is where self-praise or
    slop would read as a tone failure; the body often cites banned
    words in the course of removing them, which is legitimate.
    """
    violations: list[str] = []
    has_current_line = _has_required_authorship_line(msg)
    has_legacy_line = bool(LEGACY_COAUTHOR_TRAILER_RE.search(msg))
    if not has_current_line and not (allow_legacy_trailer and has_legacy_line):
        violations.append(f"missing `{REQUIRED_AUTHORSHIP_LINE}` authorship line")
    if require_seat_trailer:
        violations.extend(_seat_trailer_violations(msg))
    # Extract subject line (Keep a Changelog / Conventional Commits)
    subject = next((line for line in msg.splitlines() if line.strip()), "")
    scope = msg if check_body_banned else subject
    matches = BANNED_WORDS_RE.findall(scope)
    if matches:
        seen: set[str] = set()
        unique: list[str] = []
        for match in matches:
            if match in seen:
                continue
            seen.add(match)
            unique.append(match)
        where = "message" if check_body_banned else "subject"
        violations.append(f"banned word(s) in {where}: {', '.join(unique)}")
    return violations


def _commit_msg_hook(path: Path) -> int:
    """Run the forward-only commit-message hook against one message file."""
    msg = path.read_text(encoding="utf-8")
    violations = _message_violations(msg, require_seat_trailer=True)
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
            (
                '  gh pr merge <N> --squash --body "$(gh pr view <N> '
                "--json body -q .body)\\n\\nSeat: <seat-id>\\n\\n"
                f'{REQUIRED_AUTHORSHIP_LINE}"'
            ),
            file=sys.stderr,
        )
        return 1
    return 0


DEFAULT_AUDIT_RANGE = "v0.9.6..HEAD"


def _resolve_git_executable() -> str | None:
    """Return an absolute executable path for git when it is available."""
    located = shutil.which("git")
    if located is None:
        return None
    try:
        resolved = Path(located).resolve(strict=True)
    except (OSError, ValueError):
        return None
    if not resolved.is_file() or not os.access(resolved, os.X_OK):
        return None
    return str(resolved)


def _run_git(git_executable: str, *args: str) -> subprocess.CompletedProcess[str]:
    """Run an admitted git command without shell expansion."""
    return subprocess.run(  # nosec B603
        [git_executable, *args],
        capture_output=True,
        text=True,
        check=True,
        shell=False,
    )


def _ci_audit(range_spec: str = DEFAULT_AUDIT_RANGE) -> int:
    # Pipe the SHAs through a second git call that fetches each
    # message cleanly — avoids the newline-quoting pitfalls of
    # `git log --format=%B` piped through a single invocation.
    git_executable = _resolve_git_executable()
    if git_executable is None:
        print("git executable unavailable", file=sys.stderr)
        return 2
    try:
        sha_result = _run_git(git_executable, "rev-list", range_spec)
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
        msg_result = _run_git(git_executable, "log", "-1", "--format=%B", sha)
        date_result = _run_git(git_executable, "log", "-1", "--format=%cI", sha)
        committed_at = datetime.fromisoformat(date_result.stdout.strip())
        violations = _message_violations(
            msg_result.stdout,
            allow_legacy_trailer=committed_at < AUTHORSHIP_POLICY_EFFECTIVE_UTC,
        )
        if violations:
            fails.append(f"{short}: {'; '.join(violations)}")
    print(f"Audited {len(shas)} commits in {range_spec}")
    print(f"  Exempt (historical debt): {len(exempt_hits)}")
    print(f"  Violations: {len(fails)}")
    for f in fails:
        print(f"  - {f}")
    return 1 if fails else 0


def main(argv: list[str]) -> int:
    """Validate commit message trailers for the configured revision range."""
    if "--help" in argv or "-h" in argv:
        print(__doc__)
        return 0
    if "--audit" in argv or "--range" in argv:
        range_spec = DEFAULT_AUDIT_RANGE
        if "--range" in argv:
            idx = argv.index("--range")
            if idx + 1 < len(argv):
                range_spec = argv[idx + 1]
        return _ci_audit(range_spec)
    if len(argv) >= 2:
        # commit-msg hook: first arg is path to message file
        return _commit_msg_hook(Path(argv[1]))
    # No args: default to CI audit mode
    return _ci_audit(DEFAULT_AUDIT_RANGE)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
