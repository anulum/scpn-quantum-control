# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Tests for commit trailer checker
"""Tests for the commit-message trailer checker."""

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


_check_commit_trailers = _load_tool_module(
    "check_commit_trailers_for_tests",
    "check_commit_trailers.py",
)


def _message_file(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(text, encoding="utf-8")
    return path


def test_commit_message_hook_accepts_required_trailer(tmp_path: Path):
    path = _message_file(
        tmp_path,
        "\n".join(
            [
                "Add release audit coverage",
                "",
                "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>",
            ]
        ),
    )

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 0


def test_commit_message_hook_rejects_missing_trailer(tmp_path: Path, capsys):
    path = _message_file(tmp_path, "Add release audit coverage\n")

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 1
    assert "missing `Co-Authored-By:` trailer" in capsys.readouterr().err


def test_commit_message_hook_rejects_banned_subject_word(tmp_path: Path, capsys):
    path = _message_file(
        tmp_path,
        "\n".join(
            [
                "Add comprehensive release audit",
                "",
                "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>",
            ]
        ),
    )

    assert _check_commit_trailers.main(["check_commit_trailers.py", str(path)]) == 1
    assert "banned word(s) in subject: comprehensive" in capsys.readouterr().err


def test_message_violations_allow_banned_words_in_body_by_default():
    message = "\n".join(
        [
            "Add release audit coverage",
            "",
            "This removes a prior comprehensive wording from docs.",
            "",
            "Co-Authored-By: Arcane Sapience <protoscience@anulum.li>",
        ]
    )

    assert _check_commit_trailers._message_violations(message) == []
    assert _check_commit_trailers._message_violations(message, check_body_banned=True) == [
        "banned word(s) in message: comprehensive"
    ]


def test_commit_trailer_checker_help_returns_zero(capsys):
    assert _check_commit_trailers.main(["check_commit_trailers.py", "--help"]) == 0
    assert "Verify commit-message hygiene" in capsys.readouterr().out


def test_ci_audit_default_range_starts_at_clean_public_tag():
    assert _check_commit_trailers.DEFAULT_AUDIT_RANGE == "v0.9.6..HEAD"
