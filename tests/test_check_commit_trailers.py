# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for commit authorship checker
"""Tests for the commit-message authorship checker."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml


def _load_tool_module(module_name: str, filename: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_check_commit_trailers = _load_tool_module(
    "check_commit_trailers_for_tests",
    "check_commit_trailers.py",
)

REQUIRED_AUTHORSHIP_LINE = "Authored by Anulum Fortis & Arcane Sapience (protoscience@anulum.li)"
LEGACY_COAUTHOR_TRAILER = "Co-Authored-By: " + "Arcane Sapience <protoscience@anulum.li>"
VALID_SEAT_TRAILER = "Seat: 14753"
TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "check_commit_trailers.py"


def _message_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(text, encoding="utf-8")
    return path


def test_commit_message_hook_accepts_required_authorship_line(tmp_path: Path) -> None:
    """One neutral seat immediately before the sole authorship line is admitted."""
    path = _message_file(
        tmp_path,
        "\n".join(
            [
                "Add release audit coverage",
                "",
                VALID_SEAT_TRAILER,
                "",
                REQUIRED_AUTHORSHIP_LINE,
            ]
        ),
    )

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 0


def test_commit_message_hook_rejects_missing_authorship_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """An otherwise ordinary message must name the required authorship line."""
    path = _message_file(tmp_path, "Add release audit coverage\n")

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 1
    assert f"missing `{REQUIRED_AUTHORSHIP_LINE}` authorship line" in capsys.readouterr().err


def test_commit_message_hook_rejects_banned_subject_word(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A valid authorship trailer does not excuse a prohibited subject label."""
    path = _message_file(
        tmp_path,
        "\n".join(
            [
                "Add comprehensive release audit",
                "",
                VALID_SEAT_TRAILER,
                "",
                REQUIRED_AUTHORSHIP_LINE,
            ]
        ),
    )

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 1
    assert "banned word(s) in subject: comprehensive" in capsys.readouterr().err


def test_message_violations_allow_banned_words_in_body_by_default() -> None:
    """Discussion of prohibited wording in the body is distinct from subject branding."""
    message = "\n".join(
        [
            "Add release audit coverage",
            "",
            "This removes a prior comprehensive wording from docs.",
            "",
            REQUIRED_AUTHORSHIP_LINE,
        ]
    )

    assert _check_commit_trailers._message_violations(message) == []
    assert _check_commit_trailers._message_violations(message, check_body_banned=True) == [
        "banned word(s) in message: comprehensive"
    ]


def test_commit_message_hook_rejects_missing_seat_trailer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The commit-msg hook requires a forward-only seat trailer."""
    path = _message_file(
        tmp_path,
        "\n".join(
            [
                "Add release audit coverage",
                "",
                REQUIRED_AUTHORSHIP_LINE,
            ]
        ),
    )

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 1
    assert "missing `Seat: <seat-id>` trailer" in capsys.readouterr().err


def test_message_violations_do_not_require_seat_by_default() -> None:
    """Historical CI audit mode does not apply the forward-only seat policy."""
    message = "\n".join(
        [
            "Add release audit coverage",
            "",
            REQUIRED_AUTHORSHIP_LINE,
        ]
    )

    assert _check_commit_trailers._message_violations(message) == []


def test_commit_message_hook_rejects_duplicate_seat_trailers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Exactly one seat trailer is accepted."""
    path = _message_file(
        tmp_path,
        "\n".join(
            [
                "Add release audit coverage",
                "",
                "Seat: 14753",
                "Seat: rf01",
                "",
                REQUIRED_AUTHORSHIP_LINE,
            ]
        ),
    )

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 1
    assert "expected exactly one `Seat: <seat-id>` trailer" in capsys.readouterr().err


@pytest.mark.parametrize(
    "seat",
    [
        "Seat:",
        "Seat: codex-14753",
        "Seat: claude-rf01",
        "Seat: grok-a7f3",
        "Seat: gemini-9d2f",
        "Seat: kimi-3dcd",
        "Seat: openai-x",
        "Seat: anthropic-y",
        "Seat: xai-z",
        "Seat: grok",
        "Seat: claude",
        "Seat: codex",
    ],
)
def test_commit_message_hook_rejects_malformed_or_vendor_prefixed_seat(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], seat: str
) -> None:
    """Seat IDs must be syntactically valid and vendor-neutral."""
    path = _message_file(
        tmp_path,
        "\n".join(
            [
                "Add release audit coverage",
                "",
                seat,
                "",
                REQUIRED_AUTHORSHIP_LINE,
            ]
        ),
    )

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 1
    err = capsys.readouterr().err
    assert (
        "invalid `Seat: <seat-id>` trailer" in err
        or "vendor-prefixed `Seat:` trailer is forbidden" in err
        or "vendor-token `Seat:` trailer is forbidden" in err
    )


def test_commit_message_hook_accepts_bare_seat_id(
    tmp_path: Path,
) -> None:
    """Bare seat suffixes (fleet form) must pass the commit-msg hook."""
    path = _message_file(
        tmp_path,
        "\n".join(
            [
                "Add release audit coverage",
                "",
                "Seat: a7f3",
                "",
                REQUIRED_AUTHORSHIP_LINE,
            ]
        ),
    )

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 0


def test_commit_message_hook_rejects_non_adjacent_seat_trailer(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The seat trailer belongs in the trailer block with the authorship line."""
    path = _message_file(
        tmp_path,
        "\n".join(
            [
                "Add release audit coverage",
                "",
                VALID_SEAT_TRAILER,
                "",
                "Body text splits the trailer block.",
                "",
                REQUIRED_AUTHORSHIP_LINE,
            ]
        ),
    )

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 1
    assert (
        "`Seat:` trailer must immediately precede the authorship line" in capsys.readouterr().err
    )


def test_message_violations_deduplicates_repeated_banned_words() -> None:
    """Repeated subject violations produce one diagnostic per distinct token."""
    message = "\n".join(
        [
            "Add robust robust release audit",
            "",
            REQUIRED_AUTHORSHIP_LINE,
        ]
    )

    assert _check_commit_trailers._message_violations(message) == [
        "banned word(s) in subject: robust"
    ]


def test_legacy_coauthor_trailer_is_transition_only() -> None:
    """Only explicitly selected historical compatibility permits the former trailer."""
    message = "\n".join(
        [
            "Add release audit coverage",
            "",
            LEGACY_COAUTHOR_TRAILER,
        ]
    )

    assert (
        _check_commit_trailers._message_violations(
            message,
            allow_legacy_trailer=True,
        )
        == []
    )
    assert _check_commit_trailers._message_violations(message) == [
        f"missing `{REQUIRED_AUTHORSHIP_LINE}` authorship line"
    ]


def test_commit_trailer_checker_help_returns_zero(capsys: pytest.CaptureFixture[str]) -> None:
    """Help describes both entry modes without requiring a repository audit."""
    assert _check_commit_trailers.main(["check_commit_trailers.py", "--help"]) == 0
    assert "Verify commit-message hygiene" in capsys.readouterr().out


def test_ci_audit_default_range_starts_at_clean_public_tag() -> None:
    """The historical audit retains its established lower revision boundary."""
    assert _check_commit_trailers.DEFAULT_AUDIT_RANGE == "v0.9.6..HEAD"


def test_ci_audit_fails_closed_when_git_executable_is_unavailable(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CI audit refuses to launch a partial git command."""
    monkeypatch.setenv("PATH", "")

    assert _check_commit_trailers._ci_audit("HEAD") == 2
    assert "git executable unavailable" in capsys.readouterr().err


def test_resolve_git_executable_rejects_unresolvable_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale path from PATH lookup is not admitted as git."""
    missing = tmp_path / "missing-git"
    monkeypatch.setattr(_check_commit_trailers.shutil, "which", lambda _name: str(missing))

    assert _check_commit_trailers._resolve_git_executable() is None


def test_resolve_git_executable_rejects_non_executable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-executable file is not admitted as git."""
    candidate = tmp_path / "git"
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    candidate.chmod(0o644)
    monkeypatch.setattr(_check_commit_trailers.shutil, "which", lambda _name: str(candidate))

    assert _check_commit_trailers._resolve_git_executable() is None


def test_resolve_git_executable_and_run_git_admit_current_git() -> None:
    """The resolver returns an executable path that can run a harmless git command."""
    git_executable = _check_commit_trailers._resolve_git_executable()

    assert git_executable is not None
    result = _check_commit_trailers._run_git(git_executable, "--version")
    assert result.stdout.startswith("git version ")


def test_ci_audit_reports_rev_list_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A failing rev-list call returns the CI-audit infrastructure error code."""
    monkeypatch.setattr(_check_commit_trailers, "_resolve_git_executable", lambda: sys.executable)

    def fake_run_git(_git_executable: str, *args: str) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(128, ["git", *args], stderr="bad revision")

    monkeypatch.setattr(_check_commit_trailers, "_run_git", fake_run_git)

    assert _check_commit_trailers._ci_audit("bad..range") == 2
    assert "git rev-list failed: bad revision" in capsys.readouterr().err


def test_ci_audit_reports_exempt_and_failing_commits(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CI audit counts historical exemptions and reports current violations."""
    monkeypatch.setattr(_check_commit_trailers, "_resolve_git_executable", lambda: sys.executable)
    clean_sha = "abc1234567890"
    exempt_sha = "2a7d604000000"
    failing_sha = "bad9999000000"

    def fake_run_git(_git_executable: str, *args: str) -> subprocess.CompletedProcess[str]:
        if args == ("rev-list", "HEAD"):
            return subprocess.CompletedProcess(
                args, 0, stdout=f"{clean_sha}\n{exempt_sha}\n{failing_sha}\n"
            )
        if args == ("log", "-1", "--format=%B", clean_sha):
            return subprocess.CompletedProcess(
                args, 0, stdout=f"Add audit\n\n{REQUIRED_AUTHORSHIP_LINE}\n"
            )
        if args == ("log", "-1", "--format=%B", failing_sha):
            return subprocess.CompletedProcess(args, 0, stdout="Add audit\n")
        if args[0:3] == ("log", "-1", "--format=%cI"):
            return subprocess.CompletedProcess(args, 0, stdout="2026-06-01T00:00:00+00:00\n")
        raise AssertionError(f"unexpected git args: {args}")

    monkeypatch.setattr(_check_commit_trailers, "_run_git", fake_run_git)

    assert _check_commit_trailers._ci_audit("HEAD") == 1
    out = capsys.readouterr().out
    assert "Audited 3 commits in HEAD" in out
    assert "Exempt (historical debt): 1" in out
    assert "Violations: 1" in out
    assert f"{failing_sha[:7]}: missing `{REQUIRED_AUTHORSHIP_LINE}` authorship line" in out


def test_pr_183_immutable_merge_is_explicitly_exempt() -> None:
    """The protected PR merge remains an explicit, auditable exception."""
    assert "2ff0114" in _check_commit_trailers.HISTORICAL_EXEMPT_SHAS


def test_modularity_fix_forward_authorship_debt_is_exactly_exempt() -> None:
    """Keep the protected-main waiver bounded to the three published commits."""
    assert {"af96351", "ea9adea", "43f0c93"}.issubset(
        _check_commit_trailers.HISTORICAL_EXEMPT_SHAS
    )


def test_main_dispatches_audit_ranges(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI dispatcher passes the requested audit range through."""
    seen: list[str] = []

    def fake_ci_audit(range_spec: str = _check_commit_trailers.DEFAULT_AUDIT_RANGE) -> int:
        seen.append(range_spec)
        return 0

    monkeypatch.setattr(_check_commit_trailers, "_ci_audit", fake_ci_audit)

    assert _check_commit_trailers.main(["check_commit_trailers.py", "--audit"]) == 0
    assert _check_commit_trailers.main(["check_commit_trailers.py", "--range", "HEAD"]) == 0
    assert _check_commit_trailers.main(["check_commit_trailers.py", "--range"]) == 2
    assert _check_commit_trailers.main(["check_commit_trailers.py"]) == 0
    assert seen == [
        _check_commit_trailers.DEFAULT_AUDIT_RANGE,
        "HEAD",
        _check_commit_trailers.DEFAULT_AUDIT_RANGE,
    ]


def _vendor_token() -> str:
    """Return one vendor token, taken from the tool so no literal lives here."""
    token: str = _check_commit_trailers.VENDOR_ATTRIBUTION_TOKENS[0]
    return token


def test_every_vendor_token_is_rejected_in_a_coauthor_trailer() -> None:
    """Each policed vendor token is caught in a `Co-Authored-By` trailer."""
    for token in _check_commit_trailers.VENDOR_ATTRIBUTION_TOKENS:
        message = (
            f"fix(scope): summary\n\n{VALID_SEAT_TRAILER}\n\n{REQUIRED_AUTHORSHIP_LINE}\n\n"
            f"Co-Authored-By: Model <noreply@{token}.example>\n"
        )
        violations = _check_commit_trailers._vendor_attribution_violations(message)
        assert violations, f"{token} passed unnoticed"
        assert token in violations[0]


def test_vendor_session_trailer_is_rejected() -> None:
    """A `<Vendor>-Session:` trailer is rejected on its own, without a co-author line."""
    token = _vendor_token()
    message = (
        f"fix(scope): summary\n\n{VALID_SEAT_TRAILER}\n\n{REQUIRED_AUTHORSHIP_LINE}\n\n"
        f"{token.capitalize()}-Session: https://example.invalid/session_1\n"
    )
    assert _check_commit_trailers._vendor_attribution_violations(message)


def test_legacy_arcane_coauthor_trailer_is_not_vendor_attribution() -> None:
    """The former project trailer is not vendor attribution; sole-authorship checks reject it."""
    message = (
        f"fix(scope): summary\n\n{VALID_SEAT_TRAILER}\n\n"
        f"{REQUIRED_AUTHORSHIP_LINE}\n\n{LEGACY_COAUTHOR_TRAILER}\n"
    )
    assert _check_commit_trailers._vendor_attribution_violations(message) == []


def test_vendor_prefixed_seat_trailer_is_left_to_the_seat_check() -> None:
    """A vendor-prefixed seat trailer is reported once, by the seat check only."""
    token = _vendor_token()
    message = f"fix(scope): summary\n\nSeat: {token}-fcb0\n\n{REQUIRED_AUTHORSHIP_LINE}\n"
    assert _check_commit_trailers._vendor_attribution_violations(message) == []
    assert _check_commit_trailers._seat_trailer_violations(message)


def test_clean_message_has_no_vendor_attribution_violation() -> None:
    """A compliant message reports nothing."""
    message = f"fix(scope): summary\n\n{VALID_SEAT_TRAILER}\n\n{REQUIRED_AUTHORSHIP_LINE}\n"
    assert _check_commit_trailers._vendor_attribution_violations(message) == []


def test_commit_msg_hook_rejects_vendor_attribution(tmp_path: Path) -> None:
    """The forward-only hook refuses a message carrying a vendor trailer."""
    token = _vendor_token()
    path = _message_file(
        tmp_path,
        f"fix(scope): summary\n\n{VALID_SEAT_TRAILER}\n\n{REQUIRED_AUTHORSHIP_LINE}\n\n"
        f"Co-Authored-By: Model <noreply@{token}.example>\n",
    )
    assert _check_commit_trailers._commit_msg_hook(path) == 1


def test_auditor_leaves_vendor_attribution_off() -> None:
    """The auditor walks published commits, so the check stays off by default."""
    token = _vendor_token()
    message = (
        f"fix(scope): summary\n\n{VALID_SEAT_TRAILER}\n\n{REQUIRED_AUTHORSHIP_LINE}\n\n"
        f"Co-Authored-By: Model <noreply@{token}.example>\n"
    )
    assert _check_commit_trailers._message_violations(message) == []
    assert _check_commit_trailers._message_violations(message, check_vendor_attribution=True)


def test_benign_trailer_shaped_lines_pass_through() -> None:
    """The repository's own `Parity:`/`Benchmark:` body lines are not attribution."""
    message = (
        "fix(scope): summary\n\n"
        "Parity: N/A - Python only. Benchmark: N/A - no benchmarked path.\n"
        "Refs: docs/test_infrastructure.md\n\n"
        f"{VALID_SEAT_TRAILER}\n\n{REQUIRED_AUTHORSHIP_LINE}\n"
    )
    assert _check_commit_trailers._vendor_attribution_violations(message) == []


@pytest.mark.parametrize("extra", [LEGACY_COAUTHOR_TRAILER, REQUIRED_AUTHORSHIP_LINE])
def test_cli_rejects_additional_authorship(tmp_path: Path, extra: str) -> None:
    """A valid seat cannot hide an extra coauthor or duplicate authorship line."""
    path = _message_file(
        tmp_path,
        f"fix: validate input\n\n{VALID_SEAT_TRAILER}\n\n{REQUIRED_AUTHORSHIP_LINE}\n\n{extra}\n",
    )
    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), str(path)],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 1
    assert "Commit message rejected" in result.stderr


@pytest.mark.parametrize("violation", ["missing_seat", "legacy", "vendor", "duplicate", "none"])
def test_strict_audit_uses_real_git_objects_without_date_escape(
    tmp_path: Path, violation: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI rejects invalid messages even when object timestamps predate the policy."""
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Contract Test",
        "GIT_COMMITTER_NAME": "Contract Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
        "GIT_AUTHOR_DATE": "2020-01-01T00:00:00+00:00",
        "GIT_COMMITTER_DATE": "2020-01-01T00:00:00+00:00",
    }
    subprocess.run(
        ["git", "init", "--bare", str(tmp_path / "objects.git")],
        check=True,
        capture_output=True,
        timeout=10,
    )
    git_args = ["git", f"--git-dir={tmp_path / 'objects.git'}"]
    tree = subprocess.run(
        [*git_args, "mktree"], input="", capture_output=True, text=True, check=True, timeout=10
    ).stdout.strip()
    seat = "" if violation == "missing_seat" else VALID_SEAT_TRAILER
    extra = {
        "missing_seat": "",
        "none": "",
        "legacy": LEGACY_COAUTHOR_TRAILER,
        "vendor": f"{_vendor_token()}-Session: https://example.invalid/session",
        "duplicate": REQUIRED_AUTHORSHIP_LINE,
    }[violation]
    message = f"fix: validate input\n\n{seat}\n\n{REQUIRED_AUTHORSHIP_LINE}\n\n{extra}\n"
    commit = subprocess.run(
        [*git_args, "commit-tree", tree],
        input=message,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.strip()
    result = subprocess.run(
        [
            sys.executable,
            str(TOOL_PATH),
            "--strict",
            "--range",
            f"{commit}^!",
        ],
        cwd=tmp_path / "objects.git",
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == (0 if violation == "none" else 1), result.stdout + result.stderr
    assert "Audited 1 commits" in result.stdout
    monkeypatch.chdir(tmp_path / "objects.git")
    monkeypatch.setattr(_check_commit_trailers, "HISTORICAL_EXEMPT_SHAS", {commit[:7]})
    assert _check_commit_trailers.main([str(TOOL_PATH), "--strict", "--range", f"{commit}^!"]) == (
        0 if violation == "none" else 1
    )
    assert not subprocess.run(
        [*git_args, "for-each-ref", "--format=%(refname)"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["--strict"],
        ["--strict", "--range"],
        ["--range", ""],
        ["--range", "--strict"],
        ["--strict", "--range", "HEAD..HEAD"],
    ],
)
def test_strict_scope_is_explicit_and_nonempty(arguments: list[str]) -> None:
    """Missing, malformed or empty strict scopes fail rather than changing the audit target."""
    assert _check_commit_trailers.main([str(TOOL_PATH), *arguments]) == 2


@pytest.mark.parametrize(
    ("event", "before", "expected_returncode"),
    [
        ("push", "0" * 40, 0),
        ("push", "HEAD~1", 0),
        ("push", "f" * 40, 1),
        ("pull_request", "0" * 40, 0),
        ("schedule", "0" * 40, 0),
        ("workflow_dispatch", "0" * 40, 0),
    ],
)
def test_workflow_selects_and_executes_strict_range(
    tmp_path: Path, event: str, before: str, expected_returncode: int
) -> None:
    """The actual range-selection shell binds event and scheduled audits to strict mode."""
    repository = tmp_path / "objects.git"
    subprocess.run(
        ["git", "init", "--bare", str(repository)], check=True, capture_output=True, timeout=10
    )
    git = ["git", f"--git-dir={repository}"]
    tree = subprocess.run(
        [*git, "mktree"], input="", text=True, capture_output=True, check=True, timeout=10
    ).stdout.strip()
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Contract Test",
        "GIT_COMMITTER_NAME": "Contract Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    parent: list[str] = []
    for _ in range(2):
        commit = subprocess.run(
            [*git, "commit-tree", tree, *parent],
            input="test: range selection\n",
            env=environment,
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        parent = ["-p", commit]
    subprocess.run([*git, "update-ref", "HEAD", commit], check=True, timeout=10)
    workflow_path = TOOL_PATH.parents[1] / ".github" / "workflows" / "commit-trailers.yml"
    steps = yaml.safe_load(workflow_path.read_text(encoding="utf-8"))["jobs"]["audit"]["steps"]
    selection = next(step["run"] for step in steps if step.get("id") == "range")
    substitutions = {
        "github.event_name": event,
        "github.actor": "maintainer",
        "github.event.before": before,
        "github.event.after": "HEAD",
        "github.event.pull_request.base.sha": "HEAD~1",
        "github.event.pull_request.head.sha": "HEAD",
    }
    for expression, value in substitutions.items():
        selection = selection.replace("${{ " + expression + " }}", value)
    output = tmp_path / "github-output"
    completed = subprocess.run(
        ["bash", "-e", "-c", selection],
        cwd=repository,
        env={**os.environ, "GITHUB_OUTPUT": str(output), "GITHUB_HEAD_REF": "main"},
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert completed.returncode == expected_returncode, completed.stderr
    if expected_returncode:
        assert "before commit is unavailable" in completed.stderr
        assert not output.exists()
        return
    outputs = dict(line.split("=", 1) for line in output.read_text(encoding="utf-8").splitlines())
    expected = {
        "push": "HEAD" if before == "0" * 40 else "HEAD~1..HEAD",
        "pull_request": "HEAD~1..HEAD",
        "schedule": "a1760207032178e3b926c2dd25a5d83367daf76b..HEAD",
        "workflow_dispatch": "a1760207032178e3b926c2dd25a5d83367daf76b..HEAD",
    }[event]
    assert outputs["strict-range"] == expected
    runner = steps[-1]
    assert runner["env"]["STRICT_RANGE"] == "${{ steps.range.outputs.strict-range }}"
    assert (
        'python tools/check_commit_trailers.py --strict --range "$STRICT_RANGE"' in runner["run"]
    )


def _seat_identifier(suffix: str = "f636", vendor_index: int = 0) -> str:
    """Return a seat identifier built from the tool's own vendor list.

    Parameters
    ----------
    suffix : str
        Hexadecimal seat suffix.
    vendor_index : int
        Position in ``SEAT_IDENTIFIER_VENDORS``.

    Returns
    -------
    str
        ``<vendor>-<suffix>``, so no vendor literal lives in this public file.

    """
    vendor: str = _check_commit_trailers.SEAT_IDENTIFIER_VENDORS[vendor_index]
    return f"{vendor}-{suffix}"


def _record_name(prefix: tuple[str, ...], stamp: str) -> str:
    """Return a coordination record name joined from its parts.

    Parameters
    ----------
    prefix : tuple[str, ...]
        Capitalised prefix segments.
    stamp : str
        Dashed date or compact UTC stamp.

    Returns
    -------
    str
        The segments and the stamp joined by underscores. The public test file
        therefore names no record literally.

    """
    return "_".join((*prefix, stamp))


def _reviewer() -> str:
    """Return the internal reviewer name from the tool, so none lives here.

    Returns
    -------
    str
        The tool's ``INTERNAL_REVIEWER_NAME``.

    """
    name: str = _check_commit_trailers.INTERNAL_REVIEWER_NAME
    return name


def _identifier_message(text: str = "", subject: str = "fix: validate input") -> str:
    """Return a message that is valid apart from the given body text.

    Parameters
    ----------
    text : str
        Body text placed between the subject and the trailers.
    subject : str
        The subject line.

    Returns
    -------
    str
        Subject, body, seat trailer and authorship line.

    """
    return f"{subject}\n\n{text}\n\n{VALID_SEAT_TRAILER}\n{REQUIRED_AUTHORSHIP_LINE}\n"


def _hook_violations(message: str) -> list[str]:
    """Return what the commit-msg hook policy reports for one message.

    Parameters
    ----------
    message : str
        The full commit message.

    Returns
    -------
    list[str]
        Violations under the hook's flags.

    """
    violations: list[str] = _check_commit_trailers._message_violations(
        message,
        require_seat_trailer=True,
        check_vendor_attribution=True,
        check_identifiers=True,
    )
    return violations


@pytest.mark.parametrize(
    ("text", "label", "name"),
    [
        (
            f"Found while gating for {_seat_identifier()}.",
            "agent seat identifier",
            _seat_identifier(),
        ),
        (
            f"Routed by {_seat_identifier('157251', 2).upper()}.",
            "agent seat identifier",
            _seat_identifier("157251", 2).upper(),
        ),
        (
            f"See {_record_name(('HANDOVER',), '20261118T181923Z')}.",
            "coordination record name",
            _record_name(("HANDOVER",), "20261118T181923Z"),
        ),
        (
            f"See {_record_name(('REVIEW',), '2026-11-20')} for the verdict.",
            "coordination record name",
            _record_name(("REVIEW",), "2026-11-20"),
        ),
        (
            f"Per {_record_name(('LANDING',), '2026-11-21T1342Z')}.",
            "coordination record name",
            _record_name(("LANDING",), "2026-11-21T1342Z"),
        ),
        (
            f"As {_record_name(('NOTE', 'FROM', 'PEER'), '20261120T212421Z')} says.",
            "coordination record name",
            _record_name(("NOTE", "FROM", "PEER"), "20261120T212421Z"),
        ),
        (f"Approved by {_reviewer()}.", "internal reviewer name", _reviewer()),
        ("Asked on hub 399765.", "hub event citation", "hub 399765"),
        ("Answered in seq #401897.", "hub event citation", "seq #401897"),
        ("Recorded as event 12345.", "hub event citation", "event 12345"),
    ],
    ids=[
        "seat",
        "upper-case-seat",
        "compact-stamp-record",
        "dashed-date-record",
        "dated-stamp-record",
        "multi-segment-record",
        "reviewer",
        "hub",
        "seq",
        "event",
    ],
)
def test_an_internal_identifier_in_the_body_is_rejected(text: str, label: str, name: str) -> None:
    """Each identifier form is refused once, under its own label."""
    assert _hook_violations(_identifier_message(text)) == [f"{label} in the message: {name}"]


@pytest.mark.parametrize(
    "text",
    [
        "Runs on Python 3.12 and keeps PEP_8 names.",
        "Reads the snapshot_2026-11-20 fixture and the UTF_8 codec.",
        "Retries seq 4321 and event 999, below the citation width.",
        f"A {_reviewer().lower()}ous check, and the {_seat_identifier('code')} integration.",
        "Pins the wheel digest 9e4e4d08 of HANDOVER and REVIEW forms.",
    ],
    ids=["version-and-pep", "lower-case-fixture", "short-numbers", "near-names", "bare-prefixes"],
)
def test_text_that_only_resembles_an_identifier_is_allowed(text: str) -> None:
    """Version numbers, lower-case names, short numbers and bare prefixes pass."""
    assert _hook_violations(_identifier_message(text)) == []


def test_an_identifier_in_the_subject_is_rejected() -> None:
    """The subject line is part of the checked message."""
    record = _record_name(("REVIEW",), "20261121T133713Z")

    assert _hook_violations(_identifier_message(subject=f"fix: land {record}")) == [
        f"coordination record name in the message: {record}"
    ]


def test_each_identifier_is_reported_once_in_first_order() -> None:
    """Repeated names are listed once, in the order they first appear."""
    seat = _seat_identifier("b0f9")
    message = _identifier_message(f"hub 400312, then {seat}, hub 400312 again and {seat}.")

    assert _hook_violations(message) == [
        f"agent seat identifier in the message: {seat}",
        "hub event citation in the message: hub 400312",
    ]


def test_an_identifier_in_a_hash_line_is_rejected() -> None:
    """A line starting with ``#`` is checked, because ``git commit -F`` keeps it."""
    record = _record_name(("HANDOVER",), "20261118T181923Z")
    message = _identifier_message() + f"# Drafted from {record}\n"

    assert _hook_violations(message) == [f"coordination record name in the message: {record}"]


def test_the_seat_trailer_is_judged_by_its_own_rule_only() -> None:
    """A vendor-shaped seat is refused once, by the seat rule, not again as an identifier."""
    message = _identifier_message().replace(VALID_SEAT_TRAILER, f"Seat: {_seat_identifier()}")

    assert _hook_violations(message) == ["vendor-prefixed `Seat:` trailer is forbidden"]


def test_identifier_rule_is_off_unless_requested() -> None:
    """The historical policy and the default flags do not judge identifiers."""
    message = _identifier_message(f"Approved by {_reviewer()} on hub 399765.")

    assert _check_commit_trailers._message_violations(message) == []
    assert (
        _check_commit_trailers._message_violations(
            message, require_seat_trailer=True, check_vendor_attribution=True
        )
        == []
    )


def test_commit_msg_hook_rejects_an_internal_identifier(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The real hook entry point refuses the message and names the identifier."""
    path = _message_file(tmp_path, _identifier_message(f"Requested by {_seat_identifier()}."))

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 1
    error = capsys.readouterr().err
    assert "Commit message rejected" in error
    assert f"agent seat identifier in the message: {_seat_identifier()}" in error


def test_this_tool_carries_the_identifier_policy_marker() -> None:
    """Commits made from this tool's version on are checked by the strict audit."""
    assert _check_commit_trailers.IDENTIFIER_POLICY_MARKER == "commit-body-policy-v2"
    assert _check_commit_trailers.IDENTIFIER_POLICY_MARKER in TOOL_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("tool_text", "expected_returncode", "checked"),
    [
        (None, 0, 0),
        ("# an earlier copy without the marker\n", 0, 0),
        ("# carries commit-body-policy-v2\n", 1, 1),
    ],
    ids=["no-tool-copy", "copy-without-marker", "copy-with-marker"],
)
def test_strict_audit_applies_the_identifier_rule_by_the_commit_tool_copy(
    tmp_path: Path,
    tool_text: str | None,
    expected_returncode: int,
    checked: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Only a commit whose own tool copy carries the marker is judged by the rule."""
    repository = tmp_path / "objects.git"
    subprocess.run(
        ["git", "init", "--bare", str(repository)], check=True, capture_output=True, timeout=10
    )
    git = ["git", f"--git-dir={repository}"]
    entries = ""
    if tool_text is not None:
        blob = subprocess.run(
            [*git, "hash-object", "-w", "--stdin"],
            input=tool_text,
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        tools_tree = subprocess.run(
            [*git, "mktree"],
            input=f"100644 blob {blob}\tcheck_commit_trailers.py\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=10,
        ).stdout.strip()
        entries = f"040000 tree {tools_tree}\ttools\n"
    tree = subprocess.run(
        [*git, "mktree"], input=entries, text=True, capture_output=True, check=True, timeout=10
    ).stdout.strip()
    environment = {
        **os.environ,
        "GIT_AUTHOR_NAME": "Contract Test",
        "GIT_COMMITTER_NAME": "Contract Test",
        "GIT_AUTHOR_EMAIL": "test@example.invalid",
        "GIT_COMMITTER_EMAIL": "test@example.invalid",
    }
    record = _record_name(("HANDOVER",), "20261118T181923Z")
    commit = subprocess.run(
        [*git, "commit-tree", tree],
        input=_identifier_message(f"Continues {record}."),
        env=environment,
        text=True,
        capture_output=True,
        check=True,
        timeout=10,
    ).stdout.strip()
    result = subprocess.run(
        [sys.executable, str(TOOL_PATH), "--strict", "--range", f"{commit}^!"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == expected_returncode, result.stdout + result.stderr
    assert f"Identifier rule (commit-body-policy-v2): {checked} commits" in result.stdout
    assert (f"coordination record name in the message: {record}" in result.stdout) is bool(checked)
    monkeypatch.chdir(repository)
    assert (
        _check_commit_trailers.main([str(TOOL_PATH), "--strict", "--range", f"{commit}^!"])
        == expected_returncode
    )
    assert f"Identifier rule (commit-body-policy-v2): {checked} commits" in capsys.readouterr().out
