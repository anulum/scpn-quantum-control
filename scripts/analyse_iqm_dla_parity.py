#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IQM DLA parity raw-count reproducer
"""Analyse IQM DLA/parity counts against the IBM Phase 2 protocol boundary."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scipy.stats import combine_pvalues, fisher_exact

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IQM_INPUT = (
    REPO_ROOT
    / "data"
    / "iqm_paper_replication"
    / "iqm_dla_parity_minimal_2026-05-13_sanitized.json"
)
DEFAULT_IBM_INPUT = (
    REPO_ROOT / "data" / "phase2_dla_parity" / "phase2_reduced_ag_summary_2026-05-05.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "iqm_paper_replication"


@dataclass(frozen=True)
class WilsonInterval:
    low: float
    high: float


@dataclass(frozen=True)
class IQMDepthComparison:
    depth: int
    iqm_leakage_even: float
    iqm_leakage_odd: float
    iqm_leakage_even_ci95: WilsonInterval
    iqm_leakage_odd_ci95: WilsonInterval
    iqm_asymmetry_relative: float
    iqm_leakage_difference: float
    iqm_fisher_odds_ratio: float
    iqm_fisher_p: float
    iqm_even_shots: int
    iqm_odd_shots: int
    iqm_even_leakage_counts: int
    iqm_odd_leakage_counts: int
    ibm_phase2_leakage_even: float | None
    ibm_phase2_leakage_odd: float | None
    ibm_phase2_asymmetry_relative: float | None
    ibm_phase2_welch_p: float | None
    sign_matches_ibm_phase2: bool | None
    standard_error_difference: float
    z_difference: float


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> WilsonInterval:
    if total <= 0:
        raise ValueError("total must be positive")
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = (phat + z * z / (2.0 * total)) / denom
    half_width = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * total)) / total) / denom
    return WilsonInterval(low=max(0.0, centre - half_width), high=min(1.0, centre + half_width))


def _relative_asymmetry(even: float, odd: float) -> float:
    return (even - odd) / max(odd, 1e-12)


def _load_iqm_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("provider") != "iqm":
        raise ValueError(f"unexpected provider in {path}: {payload.get('provider')}")
    if payload.get("tier") != "dla_parity_minimal":
        raise ValueError(f"unexpected IQM tier in {path}: {payload.get('tier')}")
    rows = payload.get("records")
    if not isinstance(rows, list) or not rows:
        raise ValueError("IQM payload has no records")
    if any("job_id" in row for row in rows):
        raise ValueError("sanitised IQM input must not contain raw job_id fields")
    return rows


def _load_ibm_phase2_by_depth(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("backend") != "ibm_kingston":
        raise ValueError(f"unexpected IBM backend in {path}: {payload.get('backend')}")
    return {int(row["depth"]): row for row in payload["depth_summaries"]}


def analyse(
    iqm_input: Path = DEFAULT_IQM_INPUT, ibm_input: Path = DEFAULT_IBM_INPUT
) -> dict[str, Any]:
    """Return IQM minimal DLA/parity analysis with IBM Phase 2 sign comparison."""
    rows = _load_iqm_rows(iqm_input)
    ibm_by_depth = _load_ibm_phase2_by_depth(ibm_input)

    buckets: dict[tuple[int, str], dict[str, Any]] = {}
    for row in rows:
        meta = row["meta"]
        depth = int(meta["depth"])
        sector = str(meta["sector"])
        if sector not in {"even", "odd"}:
            raise ValueError(f"unexpected IQM sector: {sector}")
        key = (depth, sector)
        if key in buckets:
            raise ValueError(f"duplicate IQM minimal row for depth={depth}, sector={sector}")
        shots = int(row["shots"])
        leakage = int(row["leakage_counts"])
        if not 0 <= leakage <= shots:
            raise ValueError(f"invalid leakage count for {row['circuit_name']}")
        buckets[key] = row

    depth_rows: list[IQMDepthComparison] = []
    p_values = []
    for depth in sorted({key[0] for key in buckets}):
        even_row = buckets.get((depth, "even"))
        odd_row = buckets.get((depth, "odd"))
        if even_row is None or odd_row is None:
            raise ValueError(f"incomplete IQM even/odd pair at depth {depth}")

        even_shots = int(even_row["shots"])
        odd_shots = int(odd_row["shots"])
        even_leak = int(even_row["leakage_counts"])
        odd_leak = int(odd_row["leakage_counts"])
        even_rate = even_leak / even_shots
        odd_rate = odd_leak / odd_shots
        odds_ratio, fisher_p = fisher_exact(
            [
                [even_leak, even_shots - even_leak],
                [odd_leak, odd_shots - odd_leak],
            ],
            alternative="two-sided",
        )
        p_values.append(float(fisher_p))

        ibm = ibm_by_depth.get(depth)
        ibm_asymmetry = float(ibm["asymmetry_relative"]) if ibm is not None else None
        iqm_asymmetry = _relative_asymmetry(even_rate, odd_rate)
        sign_matches = None
        if ibm_asymmetry is not None:
            sign_matches = (iqm_asymmetry == 0.0 and ibm_asymmetry == 0.0) or (
                iqm_asymmetry * ibm_asymmetry > 0.0
            )

        depth_rows.append(
            IQMDepthComparison(
                depth=depth,
                iqm_leakage_even=even_rate,
                iqm_leakage_odd=odd_rate,
                iqm_leakage_even_ci95=_wilson_interval(even_leak, even_shots),
                iqm_leakage_odd_ci95=_wilson_interval(odd_leak, odd_shots),
                iqm_asymmetry_relative=iqm_asymmetry,
                iqm_leakage_difference=even_rate - odd_rate,
                iqm_fisher_odds_ratio=float(odds_ratio),
                iqm_fisher_p=float(fisher_p),
                iqm_even_shots=even_shots,
                iqm_odd_shots=odd_shots,
                iqm_even_leakage_counts=even_leak,
                iqm_odd_leakage_counts=odd_leak,
                ibm_phase2_leakage_even=float(ibm["leakage_even"]) if ibm is not None else None,
                ibm_phase2_leakage_odd=float(ibm["leakage_odd"]) if ibm is not None else None,
                ibm_phase2_asymmetry_relative=ibm_asymmetry,
                ibm_phase2_welch_p=float(ibm["welch_p"]) if ibm is not None else None,
                sign_matches_ibm_phase2=sign_matches,
                standard_error_difference=math.sqrt(
                    even_rate * (1.0 - even_rate) / even_shots
                    + odd_rate * (1.0 - odd_rate) / odd_shots
                ),
                z_difference=(even_rate - odd_rate)
                / max(
                    math.sqrt(
                        even_rate * (1.0 - even_rate) / even_shots
                        + odd_rate * (1.0 - odd_rate) / odd_shots
                    ),
                    1e-12,
                ),
            )
        )

    fisher_chi2, fisher_p = combine_pvalues(p_values, method="fisher")
    sign_matches = [row.sign_matches_ibm_phase2 for row in depth_rows]
    matched = sum(match is True for match in sign_matches)
    compared = sum(match is not None for match in sign_matches)

    return {
        "schema": "scpn_iqm_dla_parity_minimal_analysis_v1",
        "provider": "iqm",
        "iqm_input": str(iqm_input.relative_to(REPO_ROOT)),
        "ibm_reference": str(ibm_input.relative_to(REPO_ROOT)),
        "tier": "dla_parity_minimal",
        "depth_summaries": [asdict(row) for row in depth_rows],
        "fisher_chi2": float(fisher_chi2),
        "fisher_p": float(fisher_p),
        "n_depths_tested": len(depth_rows),
        "n_depths_iqm_p_below_0_05": sum(row.iqm_fisher_p < 0.05 for row in depth_rows),
        "n_signs_matching_ibm_phase2": matched,
        "n_depths_compared_to_ibm_phase2": compared,
        "claim_boundary": (
            "IQM minimal tier is suitable for cross-provider sanity evidence and "
            "protocol debugging, but has one 256-shot replicate per sector and is "
            "not sufficient to upgrade manuscript claims. The next IQM run should "
            "use fixed paired layouts or an explicit randomised layout block, not "
            "blind automatic layout, before paper-core replication or repeated "
            "statistics are interpreted."
        ),
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    rows = summary["depth_summaries"]
    lines = [
        "# IQM DLA Parity Minimal Analysis",
        "",
        "This report reuses the IBM DLA/parity leakage metric: leakage is the fraction of shots whose measured bitstring has parity opposite to the prepared initial parity.",
        "",
        f"- IQM input: `{summary['iqm_input']}`",
        f"- IBM reference: `{summary['ibm_reference']}`",
        f"- Depths tested: `{summary['n_depths_tested']}`",
        f"- IQM Fisher combined p-value: `{summary['fisher_p']:.6g}`",
        f"- Depths with IQM Fisher p < 0.05: `{summary['n_depths_iqm_p_below_0_05']}`",
        f"- Sign matches vs IBM Phase 2: `{summary['n_signs_matching_ibm_phase2']} / {summary['n_depths_compared_to_ibm_phase2']}`",
        "",
        "| Depth | IQM even leak | IQM odd leak | IQM asymmetry | IQM p | IBM Phase 2 asymmetry | Sign match |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        sign = "yes" if row["sign_matches_ibm_phase2"] else "no"
        lines.append(
            f"| {row['depth']} | {row['iqm_leakage_even']:.6f} | "
            f"{row['iqm_leakage_odd']:.6f} | {row['iqm_asymmetry_relative']:+.6f} | "
            f"{row['iqm_fisher_p']:.6g} | {row['ibm_phase2_asymmetry_relative']:+.6f} | {sign} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The IQM minimal tier does not reproduce the positive IBM Phase 2 DLA/parity asymmetry sign at the tested depths. Each IQM depth has only one even/odd pair at 256 shots, so this is a low-statistics cross-provider diagnostic rather than a manuscript-grade replication.",
        "",
        "A second technical boundary applies: the first IQM minimal run used automatic transpiler layout. Follow-up inspection showed that IQM can choose different physical qubits for the even and odd circuits at the same depth, which confounds sector with layout/calibration. Repeated statistics should therefore pin the same physical layout for paired even/odd circuits or use an explicitly randomised layout block.",
        "",
        "## Claim Boundary",
        "",
        summary["claim_boundary"],
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iqm-input", type=Path, default=DEFAULT_IQM_INPUT)
    parser.add_argument("--ibm-input", type=Path, default=DEFAULT_IBM_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = analyse(args.iqm_input, args.ibm_input)
    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "iqm_dla_parity_minimal_analysis_2026-05-13.json"
    md_path = args.output_dir / "iqm_dla_parity_minimal_analysis_2026-05-13.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(md_path, summary)
    print(f"wrote_json={json_path}")
    print(f"wrote_md={md_path}")
    print(f"fisher_p={summary['fisher_p']:.6g}")
    print(
        "sign_matches="
        f"{summary['n_signs_matching_ibm_phase2']}/{summary['n_depths_compared_to_ibm_phase2']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
