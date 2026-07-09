#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# oscillatools — evidence-grounded "why oscillatools" page renderer
"""Render the public ``Why oscillatools`` comparison page.

The page is a compact, user-facing decision aid generated from evidence already
committed with the standalone package:

* the live facade capability map, which defines the documented public surface;
* the CI tier-benchmark artefact, which records per-primitive Rust, Julia, and
  Python measurements, parity drift, and the production-claim boundary;
* the local tier-benchmark artefact, which documents workstation reproduction;
* the external-comparison methodology page, which keeps third-party solver
  claims fail-closed until a harness artefact exists.

The renderer does not execute benchmarks. It formats existing artefacts so the
public page cannot silently outgrow the measured evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import oscillatools as kuramoto

TIER_ORDER = ("rust", "julia", "python")


@dataclass(frozen=True)
class CapabilitySummary:
    """Summary of the exported facade used by the page."""

    groups: int
    symbols: int
    hard_dependencies: str
    optional_tiers: tuple[str, ...]


@dataclass(frozen=True)
class TierEvidenceSummary:
    """Summary of the committed tier-benchmark artefacts."""

    ci_rows: int
    local_rows: int
    fastest_counts: Mapping[str, int]
    median_non_python_speedup: float | None
    parity_max_abs_diff: float | None
    production_claim_allowed: bool
    ci_generated_utc: str
    local_generated_utc: str


def load_artifact(path: Path | None) -> dict[str, Any] | None:
    """Load a JSON artefact or return ``None`` when the path is absent.

    Parameters
    ----------
    path:
        Path to the artefact. ``None`` and missing paths both mean the
        environment is unavailable.

    Returns
    -------
    dict[str, Any] | None
        Parsed JSON object when available; otherwise ``None``.
    """

    if path is None or not path.exists():
        return None
    return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))


def capability_summary() -> CapabilitySummary:
    """Collect the live facade size and dependency posture.

    Returns
    -------
    CapabilitySummary
        Export count, group count, and the dependency posture advertised by the
        package entry point.
    """

    capability_map = kuramoto.capabilities()
    symbols = sum(len(symbols_for_group) for symbols_for_group in capability_map.values())
    return CapabilitySummary(
        groups=len(capability_map),
        symbols=symbols,
        hard_dependencies="NumPy + SciPy",
        optional_tiers=("Rust engine", "Julia", "JAX", "PyTorch", "matplotlib", "scikit-learn"),
    )


def tier_evidence_summary(
    ci: Mapping[str, Any] | None, local: Mapping[str, Any] | None
) -> TierEvidenceSummary:
    """Summarize committed tier-benchmark artefacts.

    Parameters
    ----------
    ci:
        CI artefact mapping, or ``None`` when unavailable.
    local:
        Local artefact mapping, or ``None`` when unavailable.

    Returns
    -------
    TierEvidenceSummary
        Counts, fastest-tier distribution, median same-algorithm speedup, worst
        recorded parity deviation, and production-claim policy.
    """

    ci_results = _results(ci)
    fastest_counts = {tier: 0 for tier in TIER_ORDER}
    speedups: list[float] = []
    parity_values: list[float] = []

    for result in ci_results:
        fastest = result.get("fastest_backend")
        if isinstance(fastest, str) and fastest in fastest_counts:
            fastest_counts[fastest] += 1
        rows = _rows_by_backend(result)
        python_stats = _stats(rows.get("python"))
        fastest_stats = _stats(rows.get(fastest)) if isinstance(fastest, str) else None
        if fastest != "python" and python_stats is not None and fastest_stats is not None:
            python_p50 = float(python_stats["p50_us"])
            fastest_p50 = float(fastest_stats["p50_us"])
            if fastest_p50 > 0.0:
                speedups.append(python_p50 / fastest_p50)
        parity_value = result.get("parity_max_abs_diff")
        if isinstance(parity_value, int | float):
            parity_values.append(float(parity_value))

    return TierEvidenceSummary(
        ci_rows=len(ci_results),
        local_rows=len(_results(local)),
        fastest_counts=fastest_counts,
        median_non_python_speedup=_median(speedups) if speedups else None,
        parity_max_abs_diff=max(parity_values) if parity_values else None,
        production_claim_allowed=bool(ci.get("production_claim_allowed")) if ci else False,
        ci_generated_utc=str(ci.get("generated_utc", "—")) if ci else "—",
        local_generated_utc=str(local.get("generated_utc", "—")) if local else "—",
    )


def render(
    capability: CapabilitySummary,
    tier_evidence: TierEvidenceSummary,
    *,
    competitive_page: Path,
) -> str:
    """Render the full Markdown document.

    Parameters
    ----------
    capability:
        Facade capability summary.
    tier_evidence:
        Tier-benchmark evidence summary.
    competitive_page:
        Relative path to the competitive-benchmark methodology page.

    Returns
    -------
    str
        Markdown document without a trailing newline.
    """

    optional_tiers = ", ".join(capability.optional_tiers)
    fastest_counts = ", ".join(
        f"{tier}={tier_evidence.fastest_counts.get(tier, 0)}" for tier in TIER_ORDER
    )
    median_speedup = (
        f"{tier_evidence.median_non_python_speedup:.2f}x"
        if tier_evidence.median_non_python_speedup is not None
        else "not available"
    )
    parity = (
        f"{tier_evidence.parity_max_abs_diff:.2e}"
        if tier_evidence.parity_max_abs_diff is not None
        else "not available"
    )
    production_boundary = "`true`" if tier_evidence.production_claim_allowed else "`false`"

    lines = [
        "<!--",
        "SPDX-License-Identifier: AGPL-3.0-or-later",
        "Commercial license available",
        "© Concepts 1996–2026 Miroslav Šotek. All rights reserved.",
        "© Code 2020–2026 Miroslav Šotek. All rights reserved.",
        "ORCID: 0009-0009-3560-0851",
        "Contact: www.anulum.li | protoscience@anulum.li",
        "Generated by tools/render_why_oscillatools.py — do not edit by hand.",
        "-->",
        "# Why oscillatools",
        "",
        "`oscillatools` is a standalone coupled-phase-oscillator package with a",
        "small required dependency floor and optional acceleration tiers. This page",
        "summarizes when that shape is useful, and keeps every comparison tied to",
        "the committed evidence pages.",
        "",
        "## Evidence Snapshot",
        "",
        "| Evidence surface | Current value | Source |",
        "|---|---:|---|",
        f"| Public facade groups | {capability.groups} | `oscillatools.capabilities()` |",
        f"| Public facade symbols | {capability.symbols} | `oscillatools.capabilities()` |",
        f"| Required dependency floor | {capability.hard_dependencies} | package metadata and docs |",
        f"| Optional tiers | {optional_tiers} | package extras and docs |",
        f"| CI tier-benchmark rows | {tier_evidence.ci_rows} | [tier benchmark](tier_benchmarks.md) |",
        f"| Local tier-benchmark rows | {tier_evidence.local_rows} | [tier benchmark](tier_benchmarks.md) |",
        f"| Fastest CI tier counts | {fastest_counts} | [tier benchmark](tier_benchmarks.md) |",
        f"| Median non-Python same-algorithm speedup | {median_speedup} | [tier benchmark](tier_benchmarks.md) |",
        f"| Worst recorded cross-tier parity drift | {parity} | [tier benchmark](tier_benchmarks.md) |",
        f"| Production latency claim allowed | {production_boundary} | committed tier artefact |",
        "",
        "## Decision Matrix",
        "",
        "| Need | What this package offers | Evidence boundary |",
        "|---|---|---|",
        "| Installable oscillator modelling | NumPy/SciPy floor for Kuramoto-family models, observables, integrators, adjoints, diagnostics, and control helpers. | The facade inventory is generated from `oscillatools.capabilities()`; see the handbook and capability snapshot. |",
        "| Optional acceleration without changing the public API | Rust, Julia, JAX, PyTorch, visualization, and scikit-learn routes are optional extras layered over the same public facade. | The tier benchmark records which tiers were measured in CI and locally; missing tiers render as unavailable, not estimated. |",
        "| Reproducible same-algorithm tier comparison | CI and local artefacts report per-primitive P50 latency, fastest tier, provenance, and parity drift. | These numbers compare in-package tiers only; they do not rank third-party solvers. |",
        f"| External solver comparison | The external-comparison methodology is documented in [{competitive_page.stem}]({competitive_page.as_posix()}). | No external package verdict is made until a harness artefact records package versions, tolerances, host details, and raw rows. |",
        "| User-facing claim discipline | Production latency and external-ranking claims are blocked by the committed artefact policy. | `production_claim_allowed` remains false in the tier artefact summarized above. |",
        "",
        "## Reading The Numbers",
        "",
        "The median same-algorithm speedup is a dispatch and regression signal for the",
        "measured CI host, not a portable latency promise. The parity drift is the",
        "largest absolute cross-tier numerical difference recorded by the CI artefact;",
        "it is a reproducibility check, not a proof for every possible equation,",
        "backend, graph, or hardware target.",
        "",
        "Use the [multi-language tier benchmark](tier_benchmarks.md) for the raw rows,",
        "the [competitive benchmark](competitive_benchmark.md) for external-comparison",
        "methodology, and the [gradient coverage matrix](gradient_coverage_matrix.md)",
        "for the derivative-surface inventory.",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Render the page and write it to ``--output``.

    Parameters
    ----------
    argv:
        Optional command-line argument vector for tests.

    Returns
    -------
    int
        Process exit code. ``0`` means the page was written.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ci", type=Path, required=True, help="CI tier artefact JSON")
    parser.add_argument("--local", type=Path, required=True, help="local tier artefact JSON")
    parser.add_argument("--output", type=Path, required=True, help="Markdown output path")
    parser.add_argument(
        "--competitive-page",
        type=Path,
        default=Path("competitive_benchmark.md"),
        help="relative link target for the competitive-benchmark methodology page",
    )
    args = parser.parse_args(list(sys.argv[1:] if argv is None else argv))

    ci = load_artifact(args.ci)
    local = load_artifact(args.local)
    if ci is None:
        parser.error("--ci must point to an existing tier artefact")
    if local is None:
        parser.error("--local must point to an existing tier artefact")

    document = render(
        capability_summary(),
        tier_evidence_summary(ci, local),
        competitive_page=args.competitive_page,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document + "\n", encoding="utf-8")
    print(f"[render] wrote {args.output}")
    return 0


def _results(artifact: Mapping[str, Any] | None) -> list[Mapping[str, Any]]:
    """Return the result rows from an artefact."""

    if artifact is None:
        return []
    return [
        cast(Mapping[str, Any], result)
        for result in artifact.get("results", [])
        if isinstance(result, Mapping)
    ]


def _rows_by_backend(result: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    """Return measured and unavailable backend rows indexed by backend label."""

    rows: dict[str, Mapping[str, Any]] = {}
    for row in result.get("rows", []):
        if isinstance(row, Mapping):
            backend = row.get("backend")
            if isinstance(backend, str):
                rows[backend] = cast(Mapping[str, Any], row)
    return rows


def _stats(row: Mapping[str, Any] | None) -> Mapping[str, Any] | None:
    """Return a backend stats mapping, if the backend was measured."""

    if row is None:
        return None
    stats = row.get("stats")
    if isinstance(stats, Mapping):
        return cast(Mapping[str, Any], stats)
    return None


def _median(values: Sequence[float]) -> float:
    """Return the median of a non-empty numeric sequence."""

    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return 0.5 * (ordered[midpoint - 1] + ordered[midpoint])


if __name__ == "__main__":
    raise SystemExit(main())
