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
import sys
from pathlib import Path
from types import ModuleType


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


def test_extract_vault_tokens_keeps_credential_shaped_values_and_ignores_docs(tmp_path: Path):
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


def test_scan_for_tokens_checks_only_added_diff_lines():
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


def test_scan_keyword_passwords_flags_literals_and_skips_placeholders():
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


def test_scan_keyword_passwords_skips_variable_references_and_identifiers():
    content = "\n".join(
        [
            "+token = args.token",
            "+password = password_value",
            "+access_key = cfg.access_key",
            "+passwd = $FTP_PASS",
        ]
    )

    assert _check_secrets.scan_keyword_passwords(content) == []


def test_redact_preserves_prefix_and_length_only():
    assert _check_secrets.redact("abcdef123456") == "abc***<12ch>"
    assert _check_secrets.redact("abc") == "<3ch>"
