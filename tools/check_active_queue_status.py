# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Active queue configuration assertion checker
"""Compare open local queue assertions with actual strict typing/doc configuration.

Run with ``--root`` pointing to the canonical checkout. Private TODO content is
never required in public CI: fixture tests prove checker behaviour, while the
operator runs this command against the actual local queue. Exit 0 means no
recognised contradiction, 1 means a contradiction, and 2 means unreadable input.
This recognises named historical claims, not arbitrary natural-language truth.
"""

from __future__ import annotations

import argparse
import re
import tomllib
from pathlib import Path

CHECKBOX = re.compile(r"^(\s*)[-*+]\s+\[([ xX~])\]\s*(.*)$")
STRICT_ABSENT = re.compile(
    r"mypy is not strict|not strict.{0,20}mypy|adopt\s+`?strict\s*=\s*true`?", re.I
)
DOCS_ABSENT = re.compile(
    r"docstring enforcement absent|no\s+ruff\s+`?D\b`?|no\s+`?pydocstyle`?"
    r"|repository-wide ruff `?D`? selection remains open",
    re.I,
)


def open_items(text: str) -> list[tuple[int, str]]:
    """Read open checkbox bodies, excluding closed children and fenced examples.

    Parameters
    ----------
    text
        Markdown queue; indented paragraphs remain attached across blank lines.

    Returns
    -------
    list[tuple[int, str]]
        One-based source line and prose for each open or partial checkbox.
        Nested checkboxes retain their own state, independent of their parent.
    """
    items: list[tuple[int, list[str]]] = []
    stack: list[tuple[int, list[str]]] = []
    fence = ""
    for number, line in enumerate(text.expandtabs(4).splitlines(), 1):
        stripped = line.lstrip()
        marker = re.match(r"(`{3,}|~{3,})", stripped)
        if marker:
            token = marker[0]
            if not fence:
                fence = token
            elif token[0] == fence[0] and len(token) >= len(fence):
                fence = ""
            continue
        if fence or not stripped.strip():
            continue
        indent = len(line) - len(stripped)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        match = CHECKBOX.match(line)
        if match:
            body = [match[3]]
            stack.append((indent, body))
            if match[2] in (" ", "~"):
                items.append((number, body))
        elif stack and not stripped.startswith(">"):
            stack[-1][1].append(stripped)
    return [(number, " ".join(body)) for number, body in items]


def contradictions(root: Path) -> list[str]:
    """Check recognised absence assertions against this checkout's configuration.

    Parameters
    ----------
    root
        Checkout containing pyproject.toml and the private docs/internal/TODO.md.

    Returns
    -------
    list[str]
        Contradiction categories and line numbers without echoing private prose.

    Raises
    ------
    OSError, ValueError, KeyError, TypeError, AttributeError
        Required input is unavailable, malformed or structurally incomplete.

    Notes
    -----
    Ruff selection is not evidence that all files are documented; per-file
    exceptions, renderer success and documentation debt remain separate.
    """
    data = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    tools = data["tool"]
    lint = tools["ruff"]["lint"]
    selected = set(lint.get("select", ())) | set(lint.get("extend-select", ()))
    ignored = set(lint.get("ignore", ()))
    strict = tools["mypy"].get("strict") is True
    docs = (
        bool(selected & {"D", "ALL"})
        and "D" not in ignored
        and lint.get("pydocstyle", {}).get("convention") == "numpy"
    )
    items = open_items((root / "docs/internal/TODO.md").read_text(encoding="utf-8"))
    findings: list[str] = []
    for number, text in items:
        if strict and STRICT_ABSENT.search(text):
            findings.append(f"line {number}: strict typing is configured")
        if docs and DOCS_ABSENT.search(text):
            findings.append(f"line {number}: NumPy docstring selection is configured")
    return findings


def main(argv: list[str] | None = None) -> int:
    """Report contradictions or unavailable evidence with distinct exit statuses.

    Parameters
    ----------
    argv
        Optional command arguments; defaults to the process command line.

    Returns
    -------
    int
        Zero for no recognised contradictions, one for findings, two for input
        failure. An absent private queue never produces a success claim.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    try:
        findings = contradictions(args.root)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        print(f"queue status check unavailable: {type(exc).__name__}")
        return 2
    if findings:
        print("\n".join(findings))
        return 1
    print("No recognised active queue/configuration contradictions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
