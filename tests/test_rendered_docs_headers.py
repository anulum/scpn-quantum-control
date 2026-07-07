# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Rendered-docs header-standard guards
"""Guard rendered markdown against code-style header blocks.

Per the fleet rendered-docs header standard (2026-07-04): a rendered markdown
page must open with real content — the code-file SPDX block renders as a wall
of H1 headings (the live site literally opened with six licence headings).
Licence lives in ``LICENSE``, ``REUSE.toml`` annotations, and the MkDocs
footer; the only acceptable in-file form is an invisible HTML comment.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _rendered_markdown_files() -> list[Path]:
    """Return every markdown file that renders publicly.

    Returns
    -------
    list of Path
        All ``docs/**/*.md`` outside ``docs/internal`` plus root-level
        ``*.md`` files.
    """
    docs = [path for path in (_REPO_ROOT / "docs").rglob("*.md") if "internal" not in path.parts]
    return sorted(docs) + sorted(_REPO_ROOT.glob("*.md"))


def test_rendered_markdown_opens_with_content_not_code_header() -> None:
    """No rendered page starts with a ``#``-prefixed or bare SPDX header."""
    offenders: list[str] = []
    for path in _rendered_markdown_files():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() == "":
                continue
            if line.startswith("# SPDX-License-Identifier") or line.startswith(
                "SPDX-License-Identifier"
            ):
                offenders.append(str(path.relative_to(_REPO_ROOT)))
            break
    assert offenders == [], f"rendered markdown opens with a code-style SPDX header: {offenders}"


def test_mkdocs_footer_carries_the_licence_line() -> None:
    """The MkDocs site declares the licence in its footer copyright."""
    mkdocs = (_REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert "copyright:" in mkdocs and "AGPL-3.0-or-later" in mkdocs, (
        "mkdocs.yml lost the footer copyright/licence line"
    )


def test_reuse_annotations_cover_rendered_docs() -> None:
    """REUSE.toml declares the licence for header-less docs pages."""
    reuse = (_REPO_ROOT / "REUSE.toml").read_text(encoding="utf-8")
    assert "docs/**/*.md" in reuse, (
        "REUSE.toml lost the docs/**/*.md annotation that replaces in-file headers"
    )
