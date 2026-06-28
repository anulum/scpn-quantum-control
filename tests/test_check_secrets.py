# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for vault-pattern secret scanner
"""Tests for the vault-pattern secret scanner."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest


def _load_tool_module(module_name: str, filename: str) -> ModuleType:
    module_path = Path(__file__).resolve().parents[1] / "tools" / filename
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_check_secrets = _load_tool_module("check_secrets_for_tests", "check_secrets.py")


def _vault_with_candidate(tmp_path: Path, candidate: str = "CredentialCandidate123!") -> Path:
    vault = tmp_path / "CREDENTIALS.md"
    vault.write_text(f"api credential: {candidate}\n", encoding="utf-8")
    return vault


def test_extract_vault_tokens_keeps_credential_shaped_values_and_ignores_docs(
    tmp_path: Path,
) -> None:
    vault = tmp_path / "CREDENTIALS.md"
    candidate = "CredentialCandidate123!"
    vault.write_text(
        "\n".join(
            [
                "https://github.com/anulum/scpn-quantum-control",
                "plainalphabeticcredential",
                f"api credential: {candidate}",
            ]
        ),
        encoding="utf-8",
    )

    tokens = _check_secrets.extract_vault_tokens(vault)

    assert candidate in tokens
    assert "https://github.com/anulum/scpn-quantum-control" not in tokens
    assert "plainalphabeticcredential" not in tokens


def test_extract_vault_tokens_returns_empty_for_missing_vault(tmp_path: Path) -> None:
    assert _check_secrets.extract_vault_tokens(tmp_path / "missing.md") == set()


def test_ignorable_token_boundaries() -> None:
    assert _check_secrets._shannon_entropy("") == 0.0
    assert _check_secrets._is_ignorable("short")
    assert _check_secrets._is_ignorable("https://example.test/abc123XYZ")
    assert _check_secrets._is_ignorable("plainalphabeticcredential")
    assert _check_secrets._is_ignorable("aaa111aaa111aaa1")
    assert _check_secrets._is_ignorable("!!!!@@@@####$$$$")
    assert _check_secrets._is_ignorable("aaaabbbbcccc1111")


def test_scan_for_tokens_checks_only_added_diff_lines() -> None:
    candidate_value = "CredentialCandidate123!"
    diff = "\n".join(
        [
            "diff --git a/file.py b/file.py",
            f"-old_value = '{candidate_value}'",
            "+++ b/file.py",
            "+safe = 'placeholder'",
        ]
    )

    assert _check_secrets.scan_for_tokens(diff, {candidate_value}) == []
    assert _check_secrets.scan_for_tokens(
        f"+new_value = '{candidate_value}'", {candidate_value}
    ) == [(candidate_value, 1)]


def test_scan_for_tokens_returns_empty_for_missing_inputs() -> None:
    assert _check_secrets.scan_for_tokens("", {"CredentialCandidate123!"}) == []
    assert _check_secrets.scan_for_tokens("+x", set()) == []


def test_resolve_git_executable_returns_none_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing PATH lookup is not admitted as git."""
    monkeypatch.setattr(_check_secrets.shutil, "which", lambda _name: None)

    assert _check_secrets._resolve_git_executable() is None


def test_resolve_git_executable_rejects_stale_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale PATH lookup result is not admitted as git."""
    missing = tmp_path / "missing-git"
    monkeypatch.setattr(_check_secrets.shutil, "which", lambda _name: str(missing))

    assert _check_secrets._resolve_git_executable() is None


def test_resolve_git_executable_rejects_non_executable_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-executable file is not admitted as git."""
    candidate = tmp_path / "git"
    candidate.write_text("#!/bin/sh\n", encoding="utf-8")
    candidate.chmod(0o644)
    monkeypatch.setattr(_check_secrets.shutil, "which", lambda _name: str(candidate))

    assert _check_secrets._resolve_git_executable() is None


def test_resolve_git_executable_and_run_git_admit_current_git() -> None:
    """The resolver returns an executable path that can run a harmless git command."""
    git_executable = _check_secrets._resolve_git_executable()

    assert git_executable is not None
    result = _check_secrets._run_git(git_executable, "--version")
    assert result.stdout.startswith("git version ")


def test_get_staged_diff_falls_back_without_admitted_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Staged-diff scanning does not launch a partial git command."""
    monkeypatch.setattr(_check_secrets, "_resolve_git_executable", lambda: None)

    assert _check_secrets.get_staged_diff() == ""


def test_get_staged_diff_returns_git_stdout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_check_secrets, "_resolve_git_executable", lambda: "/usr/bin/git")

    def fake_run_git(
        _git_executable: str,
        *args: str,
    ) -> subprocess.CompletedProcess[str]:
        assert args == ("diff", "--cached", "--unified=0")
        return subprocess.CompletedProcess(args, 0, stdout="+secret = literal-value\n")

    monkeypatch.setattr(_check_secrets, "_run_git", fake_run_git)

    assert _check_secrets.get_staged_diff() == "+secret = literal-value\n"


def test_get_staged_diff_returns_empty_when_git_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_check_secrets, "_resolve_git_executable", lambda: "/usr/bin/git")

    def fake_run_git(_git_executable: str, *args: str) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    monkeypatch.setattr(_check_secrets, "_run_git", fake_run_git)

    assert _check_secrets.get_staged_diff() == ""


def test_get_working_tree_content_falls_back_without_admitted_git(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full-tree scanning treats unavailable git as empty scanner input."""
    monkeypatch.setattr(_check_secrets, "_resolve_git_executable", lambda: None)

    assert _check_secrets.get_working_tree_content() == ""


def test_get_working_tree_content_reads_tracked_text_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(_check_secrets, "_resolve_git_executable", lambda: "/usr/bin/git")
    (tmp_path / "tracked.txt").write_text("plain text\n", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")
    (tmp_path / "unreadable.txt").write_text("hidden\n", encoding="utf-8")
    (tmp_path / "directory").mkdir()
    original_read_text = Path.read_text

    def fake_read_text(
        self: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if self.name == "unreadable.txt":
            raise OSError("cannot read")
        return original_read_text(self, encoding=encoding, errors=errors)

    def fake_run_git(
        _git_executable: str,
        *args: str,
    ) -> subprocess.CompletedProcess[str]:
        assert args == ("ls-files",)
        return subprocess.CompletedProcess(
            args,
            0,
            stdout="tracked.txt\nimage.png\nmissing.txt\nunreadable.txt\ndirectory\n",
        )

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    monkeypatch.setattr(_check_secrets, "_run_git", fake_run_git)

    assert _check_secrets.get_working_tree_content() == "plain text\n"


def test_get_working_tree_content_returns_empty_when_git_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_check_secrets, "_resolve_git_executable", lambda: "/usr/bin/git")

    def fake_run_git(_git_executable: str, *args: str) -> subprocess.CompletedProcess[str]:
        raise FileNotFoundError

    monkeypatch.setattr(_check_secrets, "_run_git", fake_run_git)

    assert _check_secrets.get_working_tree_content() == ""


def test_scan_keyword_passwords_flags_literals_and_skips_placeholders() -> None:
    key_password = "pass" + "word"
    key_api = "api" + "_key"
    key_token = "tok" + "en"
    key_secret = "se" + "cret"
    content = "\n".join(
        [
            f"+{key_password} = '<password>'",
            f"+{key_api} = os.environ['API_KEY']",
            f"+{key_token} = fake_token_for_tests",
            f"+{key_secret} = literal-value",
        ]
    )

    hits = _check_secrets.scan_keyword_passwords(content)

    assert hits == [(key_secret, "literal-value", 4)]


def test_scan_keyword_passwords_handles_empty_and_diff_metadata() -> None:
    key_secret = "se" + "cret"
    content = "\n".join(
        [
            "diff --git a/file.py b/file.py",
            f"-{key_secret} = removed-value",
            "+++ b/file.py",
            f"+// {key_secret} = literal-value",
        ]
    )

    assert _check_secrets.scan_keyword_passwords("") == []
    assert _check_secrets.scan_keyword_passwords(content) == [(key_secret, "literal-value", 4)]


def test_scan_keyword_passwords_skips_variable_references_and_identifiers() -> None:
    content = "\n".join(
        [
            "+token = args.token",
            "+password = password_value",
            "+access_key = cfg.access_key",
            "+passwd = $FTP_PASS",
        ]
    )

    assert _check_secrets.scan_keyword_passwords(content) == []


def test_scan_keyword_passwords_skips_calls_and_exact_markdown_states() -> None:
    key_secret = "se" + "cret"
    content = "\n".join(
        [
            f"+{key_secret} = build_secret()",
            f"+{key_secret} = config['SECRET']",
            f"+{key_secret} = pending",
        ]
    )

    assert _check_secrets.scan_keyword_passwords(content) == []


def test_scan_keyword_passwords_keeps_non_identifier_dotted_literals() -> None:
    key_secret = "se" + "cret"

    assert _check_secrets.scan_keyword_passwords(f"+{key_secret} = 123.abc") == [
        (key_secret, "123.abc", 1)
    ]


def test_redact_preserves_prefix_and_length_only() -> None:
    assert _check_secrets.redact("abcdef123456") == "abc***<12ch>"
    assert _check_secrets.redact("abc") == "<3ch>"


def test_main_skips_missing_vault(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["check_secrets.py", "--vault", str(tmp_path / "missing.md")])

    assert _check_secrets.main() == 0
    assert "vault not found or empty" in capsys.readouterr().err


def test_main_show_count(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vault = _vault_with_candidate(tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_secrets.py", "--vault", str(vault), "--show-count"])

    assert _check_secrets.main() == 0
    assert "extracted 1 candidate tokens" in capsys.readouterr().out


def test_main_returns_zero_when_selected_scope_is_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    vault = _vault_with_candidate(tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_secrets.py", "--vault", str(vault), "--all"])
    monkeypatch.setattr(_check_secrets, "get_working_tree_content", lambda: "")

    assert _check_secrets.main() == 0


def test_main_returns_zero_when_no_hits(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    vault = _vault_with_candidate(tmp_path)
    monkeypatch.setattr(sys, "argv", ["check_secrets.py", "--vault", str(vault)])
    monkeypatch.setattr(_check_secrets, "get_staged_diff", lambda: "+safe = 'placeholder'\n")

    assert _check_secrets.main() == 0


def test_main_reports_vault_and_keyword_hits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    candidate = "CredentialCandidate123!"
    vault = _vault_with_candidate(tmp_path, candidate)
    key_secret = "se" + "cret"
    monkeypatch.setattr(sys, "argv", ["check_secrets.py", "--vault", str(vault), "--all"])
    monkeypatch.setattr(
        _check_secrets,
        "get_working_tree_content",
        lambda: "\n".join(
            [
                f"first = '{candidate}'",
                f"second = '{candidate}'",
                f"{key_secret} = literal-value",
                f"{key_secret} = literal-value",
            ]
        ),
    )

    assert _check_secrets.main() == 1
    out = capsys.readouterr().out
    assert "2 vault token match(es) in working tree" in out
    assert "2 keyword-password match(es) in working tree" in out
    assert "CredentialCandidate123!" not in out


def test_main_reports_vault_hits_without_keyword_hits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    candidate = "CredentialCandidate123!"
    vault = _vault_with_candidate(tmp_path, candidate)
    monkeypatch.setattr(sys, "argv", ["check_secrets.py", "--vault", str(vault)])
    monkeypatch.setattr(_check_secrets, "get_staged_diff", lambda: f"+value = '{candidate}'\n")

    assert _check_secrets.main() == 1
    out = capsys.readouterr().out
    assert "1 vault token match(es) in staged changes" in out
    assert "keyword-password" not in out


def test_main_reports_keyword_hits_without_vault_hits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    vault = _vault_with_candidate(tmp_path)
    key_secret = "se" + "cret"
    monkeypatch.setattr(sys, "argv", ["check_secrets.py", "--vault", str(vault)])
    monkeypatch.setattr(
        _check_secrets, "get_staged_diff", lambda: f"+{key_secret} = literal-value\n"
    )

    assert _check_secrets.main() == 1
    out = capsys.readouterr().out
    assert "1 keyword-password match(es) in staged changes" in out
    assert "vault token" not in out
