# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — Differentiable claim Markdown rendering
"""Render bounded differentiable claim evidence as deterministic Markdown."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Protocol

NON_PROMOTIONAL_BOUNDARY = (
    "No hardware, provider, QPU, GPU, production-performance, or isolated_affinity claim."
)


class _ClaimRow(Protocol):
    """Read-only claim fields required by the Markdown renderers."""

    @property
    def claim_id(self) -> str:
        """Return the stable claim identity."""
        ...

    @property
    def claim_text(self) -> str:
        """Return the governed claim statement."""
        ...

    @property
    def evidence_artifact_ids(self) -> tuple[str, ...]:
        """Return supporting evidence identities."""
        ...

    @property
    def benchmark_artifact_ids(self) -> tuple[str, ...]:
        """Return benchmark evidence identities."""
        ...

    @property
    def known_gaps(self) -> tuple[str, ...]:
        """Return explicit claim limitations."""
        ...

    @property
    def promotion_status(self) -> str:
        """Return the internal promotion status."""
        ...

    @property
    def claim_boundary(self) -> str:
        """Return the exact public boundary."""
        ...


class _SupportSurfaceAlignment(Protocol):
    """Read-only alignment fields required by its Markdown renderer."""

    @property
    def schema(self) -> str:
        """Return the alignment schema."""
        ...

    @property
    def artifact_id(self) -> str:
        """Return the alignment artefact identity."""
        ...

    @property
    def passed(self) -> bool:
        """Return whether the alignment passed."""
        ...

    @property
    def claim_boundary(self) -> str:
        """Return the alignment interpretation boundary."""
        ...

    @property
    def checked_claim_ids(self) -> tuple[str, ...]:
        """Return audited claim identities."""
        ...

    @property
    def checked_paths(self) -> tuple[str, ...]:
        """Return audited repository paths."""
        ...

    @property
    def errors(self) -> tuple[str, ...]:
        """Return ordered alignment errors."""
        ...


def render_claim_ledger_markdown(rows: Iterable[_ClaimRow]) -> str:
    """Render a compact claim-ledger summary for reviewers.

    Parameters
    ----------
    rows
        Ordered claim rows.

    Returns
    -------
    str
        Deterministic Markdown with table-control characters escaped.
    """
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Differentiable Phase-QNode Claim Ledger",
        "-->",
        "",
        "# Differentiable Phase-QNode Claim Ledger",
        "",
        "| Claim | Status | Artefact IDs | Benchmark IDs | Known gaps |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            "| {claim} | {status} | {artefacts} | {benchmarks} | {gaps} |".format(
                claim=_markdown_cell(row.claim_id),
                status=_markdown_cell(row.promotion_status),
                artefacts=", ".join(
                    _markdown_cell(artifact_id) for artifact_id in row.evidence_artifact_ids
                )
                or "none",
                benchmarks=", ".join(
                    _markdown_cell(artifact_id) for artifact_id in row.benchmark_artifact_ids
                )
                or "none",
                gaps="<br>".join(_markdown_cell(gap) for gap in row.known_gaps),
            )
        )
    lines.extend(
        [
            "",
            "Bounded language: the differentiable lane remains a promotion candidate unless "
            "isolated CI benchmark evidence and external comparison artefacts pass.",
        ]
    )
    return "\n".join(lines) + "\n"


def render_differentiable_support_surface_alignment_markdown(
    alignment: _SupportSurfaceAlignment,
) -> str:
    """Render support-surface alignment evidence for reviewers.

    Parameters
    ----------
    alignment
        Validated alignment evidence.

    Returns
    -------
    str
        Deterministic Markdown with safe code spans and table cells.
    """
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Differentiable support-surface alignment",
        "-->",
        "",
        "# Differentiable Support-Surface Alignment",
        "",
        f"- Schema: {_markdown_code(alignment.schema)}",
        f"- Artifact ID: {_markdown_code(alignment.artifact_id)}",
        f"- `passed`: {_markdown_code(str(alignment.passed))}",
        f"- Claim boundary: {_markdown_cell(alignment.claim_boundary)}",
        "",
        "## Checked Claims",
        "",
        "| Claim ID |",
        "|---|",
    ]
    for claim_id in alignment.checked_claim_ids:
        lines.append(f"| {_markdown_code(claim_id)} |")
    lines.extend(["", "## Checked Paths", "", "| Path |", "|---|"])
    for path in alignment.checked_paths:
        lines.append(f"| {_markdown_code(path)} |")
    if alignment.errors:
        lines.extend(["", "## Errors", "", "| Error |", "|---|"])
        for error in alignment.errors:
            lines.append(f"| {_markdown_cell(error)} |")
    return "\n".join(lines) + "\n"


def render_public_claim_table(rows: Iterable[_ClaimRow]) -> str:
    """Render public-safe differentiable claim wording from ordered rows.

    Parameters
    ----------
    rows
        Ordered claim rows.

    Returns
    -------
    str
        Deterministic public Markdown derived only from ledger fields.
    """
    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "SCPN Quantum Control — Differentiable Public Claim Table",
        "-->",
        "",
        "# Differentiable Public Claim Table",
        "",
        "This table is generated from the differentiable claim ledger. It is the",
        "public wording boundary for current differentiable-programming claims.",
        "",
        "| Claim ID | Public status | Public-safe wording | Do not claim yet | Evidence |",
        "|---|---|---|---|---|",
    ]
    lines.extend(_render_public_claim_row(row) for row in rows)
    lines.extend(
        [
            "",
            "Global boundary: no differentiable row is promoted until the claim ledger,",
            "external comparison rows, and isolated CI benchmark artefacts all pass.",
        ]
    )
    return "\n".join(lines) + "\n"


def _public_status(row: _ClaimRow) -> str:
    """Map an internal promotion status to the public table vocabulary."""
    if row.promotion_status == "promoted":
        return "promoted"
    if row.promotion_status in {"hard_gap", "blocked"}:
        return "blocked"
    return "bounded-candidate"


def _public_safe_wording(row: _ClaimRow) -> str:
    """Return public-safe wording for a row's current promotion status."""
    if row.promotion_status == "promoted":
        return f"{row.claim_text} Claim boundary: {row.claim_boundary}"
    return (
        "Evidence-backed candidate surface for the bounded differentiable lane; "
        "use the listed artefacts and claim boundary when discussing scope."
    )


def _public_blocked_wording(row: _ClaimRow) -> str:
    """Return the boundary that blocks overclaiming for a row."""
    if row.promotion_status == "promoted":
        return "Do not extend beyond the exact claim boundary without a new ledger row."
    return NON_PROMOTIONAL_BOUNDARY


def _public_evidence_wording(row: _ClaimRow) -> str:
    """Render evidence identities as safe Markdown code spans."""
    artefacts = ", ".join(_markdown_code(artifact) for artifact in row.evidence_artifact_ids)
    benchmarks = ", ".join(_markdown_code(artifact) for artifact in row.benchmark_artifact_ids)
    return f"Artefacts: {artefacts}; benchmark IDs: {benchmarks}."


def _render_public_claim_row(row: _ClaimRow) -> str:
    """Render one canonical public claim-table row."""
    return (
        f"| {_markdown_code(row.claim_id)} | {_markdown_code(_public_status(row))} | "
        f"{_markdown_cell(_public_safe_wording(row))} | "
        f"{_markdown_cell(_public_blocked_wording(row))} | "
        f"{_public_evidence_wording(row)} |"
    )


def _markdown_cell(value: str) -> str:
    """Escape table delimiters and fold physical lines into one cell."""
    return " ".join(value.splitlines()).replace("|", r"\|")


def _markdown_code(value: str) -> str:
    """Render a table-safe code span around arbitrary single-line text."""
    safe = _markdown_cell(value)
    longest_run = max((len(run) for run in re.findall(r"`+", safe)), default=0)
    fence = "`" * (longest_run + 1)
    padding = " " if safe.startswith("`") or safe.endswith("`") else ""
    return f"{fence}{padding}{safe}{padding}{fence}"


__all__ = [
    "NON_PROMOTIONAL_BOUNDARY",
    "render_claim_ledger_markdown",
    "render_differentiable_support_surface_alignment_markdown",
    "render_public_claim_table",
]
