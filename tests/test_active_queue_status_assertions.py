# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Active-queue status assertion tests
"""No open queue item may assert an enforcement that is in fact configured.

The internal queue is append-only and keeps its history, which is right: a
completed item's original wording is evidence. The hazard is that an item left
unticked still reads as work to do. Two June items said strict MyPy and Ruff
``D`` were absent long after `pyproject.toml` adopted both, so anyone picking up
the queue would have started work that was already done.

This is a cross-check, not a linter for prose. It reads what the configuration
actually enforces and fails only when an *unticked* item contradicts it, so
historical text stays exactly as written.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import tomllib

TODO = Path("docs/internal/TODO.md")
"""The canonical internal queue."""

PYPROJECT = Path("pyproject.toml")
"""The configuration the queue's claims are checked against."""

UNTICKED = re.compile(r"^\s*-\s*\[[ ~]\]\s*(?P<text>.+)$")
"""An item that still reads as open, including the partial ``[~]`` marker."""

STRICT_TYPING_ABSENT = re.compile(
    r"mypy is not strict|not strict.{0,20}mypy|adopt\s+`?strict\s*=\s*true`?",
    re.IGNORECASE,
)
"""Claims that strict typing is not configured."""

DOCSTRING_ENFORCEMENT_ABSENT = re.compile(
    r"docstring enforcement absent"
    r"|no\s+ruff\s+`?D`?|no\s+`?pydocstyle`?"
    r"|repository-wide ruff `?D`? selection remains open",
    re.IGNORECASE,
)
"""Claims that docstring enforcement is not configured."""


def _configuration() -> dict[str, object]:
    """Return what the repository configuration enforces today.

    Returns
    -------
    dict
        ``strict_typing`` and ``docstring_enforcement`` booleans, each read from
        `pyproject.toml` rather than assumed.

    """
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    tools = data["tool"]
    lint = tools["ruff"].get("lint", {})
    selected = set(lint.get("select", ()))
    convention = lint.get("pydocstyle", {}).get("convention")
    return {
        "strict_typing": bool(tools["mypy"].get("strict")),
        "docstring_enforcement": "D" in selected and convention == "numpy",
    }


def _unticked_items() -> list[tuple[int, str]]:
    """Return every queue item that still reads as open, with its whole body.

    A checkbox item is not one line. The claim that repository-wide Ruff ``D``
    remained open debt sat on a continuation line, so a first-line-only reader
    could not see it; this joins the checkbox line with the indented lines that
    belong to it.

    Returns
    -------
    list
        One ``(checkbox line number, full item text)`` pair per unticked or
        partial checkbox.

    """
    lines = TODO.read_text(encoding="utf-8").splitlines()
    items: list[tuple[int, str]] = []
    index = 0
    while index < len(lines):
        match = UNTICKED.match(lines[index])
        if match is None:
            index += 1
            continue
        start = index
        body = [match.group("text")]
        indent = len(lines[index]) - len(lines[index].lstrip())
        index += 1
        while index < len(lines):
            following = lines[index]
            if not following.strip():
                break
            if len(following) - len(following.lstrip()) <= indent:
                break
            body.append(following.strip())
            index += 1
        items.append((start + 1, " ".join(body)))
    return items


class TestConfigurationIsWhatWeThink:
    """Read the configuration first, so the cross-check means something."""

    def test_strict_typing_is_configured(self) -> None:
        """``[tool.mypy] strict`` is the fact the June item denied."""
        assert _configuration()["strict_typing"] is True

    def test_docstring_enforcement_is_configured(self) -> None:
        """Ruff selects ``D`` and pydocstyle uses the NumPy convention."""
        assert _configuration()["docstring_enforcement"] is True


class TestNoOpenItemContradictsIt:
    """The acceptance: no active claim that either is absent."""

    def test_no_open_item_says_strict_typing_is_absent(self) -> None:
        """An unticked item claiming this would send someone to redo it."""
        if not _configuration()["strict_typing"]:
            pytest.skip("strict typing is not configured, so such a claim would be true")
        offenders = [
            f"line {number}: {text}"
            for number, text in _unticked_items()
            if STRICT_TYPING_ABSENT.search(text)
        ]
        assert offenders == []

    def test_no_open_item_says_docstring_enforcement_is_absent(self) -> None:
        """Same for Ruff ``D`` and the pydocstyle convention."""
        if not _configuration()["docstring_enforcement"]:
            pytest.skip("docstring enforcement is not configured, so such a claim would be true")
        offenders = [
            f"line {number}: {text}"
            for number, text in _unticked_items()
            if DOCSTRING_ENFORCEMENT_ABSENT.search(text)
        ]
        assert offenders == []


class TestTheCheckCanFail:
    """A cross-check that cannot fail is decoration."""

    @pytest.mark.parametrize(
        ("text", "pattern"),
        [
            ("mypy is not strict — `[tool.mypy]` carries only check_untyped_defs", "typing"),
            ("Adopt `strict = true` (part of the in-progress strict rollout)", "typing"),
            ("Docstring enforcement absent — no ruff `D`, no `pydocstyle` convention", "docs"),
            ("Repository-wide Ruff `D` selection remains open debt.", "docs"),
        ],
    )
    def test_the_recorded_wordings_are_recognised(self, text: str, pattern: str) -> None:
        """The exact sentences this card reconciled must still be detected.

        Parameters
        ----------
        text
            Wording taken verbatim from the reconciled June items.
        pattern
            Which detector should match it.

        """
        detector = STRICT_TYPING_ABSENT if pattern == "typing" else DOCSTRING_ENFORCEMENT_ABSENT
        assert detector.search(text) is not None

    @pytest.mark.parametrize(
        "text",
        [
            "Raise the aggregate coverage gate from 90% to the ≥95 minimum.",
            "Add a dedicated owner for the phase module.",
            "mypy --strict passes on the changed files.",
        ],
    )
    def test_ordinary_items_are_not_flagged(self, text: str) -> None:
        """The detectors must not fire on unrelated open work.

        Parameters
        ----------
        text
            An open item that says nothing about missing enforcement.

        """
        assert STRICT_TYPING_ABSENT.search(text) is None
        assert DOCSTRING_ENFORCEMENT_ABSENT.search(text) is None

    def test_the_queue_still_carries_open_items(self) -> None:
        """If nothing is unticked the cross-check is vacuous, so say so."""
        assert _unticked_items() != []
