#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IQM layout-pinned DLA repeat analysis
"""Analyse one layout-pinned IQM DLA/parity repeat record."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from scipy.stats import combine_pvalues, fisher_exact

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IBM_INPUT = (
    REPO_ROOT / "data" / "phase2_dla_parity" / "phase2_reduced_ag_summary_2026-05-05.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "iqm_paper_replication"


@dataclass(frozen=True)
class LayoutDepthSummary:
    depth: int
    iqm_leakage_even: float
    iqm_leakage_odd: float
    iqm_leakage_difference: float
    iqm_asymmetry_relative: float
    iqm_fisher_odds_ratio: float
    iqm_fisher_p: float
    standard_error_difference: float
    z_difference: float
    ibm_phase2_asymmetry_relative: float
    sign_matches_ibm_phase2: bool
    even_job_id_sha256: str
    odd_job_id_sha256: str
    even_transpiled_depth: int
    odd_transpiled_depth: int


def _relative_asymmetry(even: float, odd: float) -> float:
    return (even - odd) / max(odd, 1e-12)


def _layout_slug(layout: list[int]) -> str:
    return "q" + "-".join(str(qubit) for qubit in layout)


def analyse_layout_repeat(input_path: Path, ibm_input: Path = DEFAULT_IBM_INPUT) -> dict[str, Any]:
    """Analyse a sanitized layout-pinned IQM repeat record."""
    input_path = input_path.resolve()
    ibm_input = ibm_input.resolve()
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    if payload.get("provider") != "iqm":
        raise ValueError(f"unexpected provider: {payload.get('provider')}")
    if payload.get("tier") != "dla_parity_minimal_layout_pinned_repeat":
        raise ValueError(f"unexpected tier: {payload.get('tier')}")
    if any("job_id" in row for row in payload["records"]):
        raise ValueError("input must be sanitized and must not contain raw job_id fields")

    ibm = json.loads(ibm_input.read_text(encoding="utf-8"))
    ibm_by_depth = {int(row["depth"]): row for row in ibm["depth_summaries"]}
    by_depth_sector: dict[tuple[int, str], dict[str, Any]] = {}
    readout_rows = []
    for row in payload["records"]:
        if row["kind"] == "dla_parity":
            meta = row["meta"]
            by_depth_sector[(int(meta["depth"]), str(meta["sector"]))] = row
        elif row["kind"] == "readout_baseline":
            readout_rows.append(row)

    depth_summaries: list[LayoutDepthSummary] = []
    p_values = []
    for depth in sorted({key[0] for key in by_depth_sector}):
        even = by_depth_sector[(depth, "even")]
        odd = by_depth_sector[(depth, "odd")]
        even_shots = int(even["shots"])
        odd_shots = int(odd["shots"])
        even_leak = int(even["leakage_counts"])
        odd_leak = int(odd["leakage_counts"])
        even_rate = even_leak / even_shots
        odd_rate = odd_leak / odd_shots
        odds_ratio, fisher_p = fisher_exact(
            [[even_leak, even_shots - even_leak], [odd_leak, odd_shots - odd_leak]],
            alternative="two-sided",
        )
        p_values.append(float(fisher_p))
        se = math.sqrt(
            even_rate * (1.0 - even_rate) / even_shots + odd_rate * (1.0 - odd_rate) / odd_shots
        )
        asymmetry = _relative_asymmetry(even_rate, odd_rate)
        ibm_asymmetry = float(ibm_by_depth[depth]["asymmetry_relative"])
        depth_summaries.append(
            LayoutDepthSummary(
                depth=depth,
                iqm_leakage_even=even_rate,
                iqm_leakage_odd=odd_rate,
                iqm_leakage_difference=even_rate - odd_rate,
                iqm_asymmetry_relative=asymmetry,
                iqm_fisher_odds_ratio=float(odds_ratio),
                iqm_fisher_p=float(fisher_p),
                standard_error_difference=se,
                z_difference=(even_rate - odd_rate) / max(se, 1e-12),
                ibm_phase2_asymmetry_relative=ibm_asymmetry,
                sign_matches_ibm_phase2=asymmetry * ibm_asymmetry > 0.0,
                even_job_id_sha256=str(even["job_id_sha256"]),
                odd_job_id_sha256=str(odd["job_id_sha256"]),
                even_transpiled_depth=int(even["transpiled_depth"]),
                odd_transpiled_depth=int(odd["transpiled_depth"]),
            )
        )

    fisher_chi2, fisher_p = combine_pvalues(p_values, method="fisher")
    return {
        "schema": "scpn_iqm_dla_layout_pinned_repeat_analysis_v1",
        "provider": "iqm",
        "source": str(input_path.relative_to(REPO_ROOT)),
        "ibm_reference": str(ibm_input.relative_to(REPO_ROOT)),
        "requested_initial_layout": payload["requested_initial_layout"],
        "total_circuits": int(payload["total_circuits"]),
        "total_shots": int(payload["total_shots"]),
        "depth_summaries": [asdict(row) for row in depth_summaries],
        "readout_baselines": [
            {
                "circuit_name": row["circuit_name"],
                "initial": row["meta"]["initial"],
                "sector": row["meta"]["sector"],
                "shots": int(row["shots"]),
                "parity_leakage": float(row["parity_leakage"]),
                "initial_state_retention": float(row["initial_state_retention"]),
                "job_id_sha256": row["job_id_sha256"],
            }
            for row in readout_rows
        ],
        "fisher_chi2": float(fisher_chi2),
        "fisher_p": float(fisher_p),
        "n_depths_iqm_p_below_0_05": sum(row.iqm_fisher_p < 0.05 for row in depth_summaries),
        "n_signs_matching_ibm_phase2": sum(row.sign_matches_ibm_phase2 for row in depth_summaries),
        "n_depths_compared_to_ibm_phase2": len(depth_summaries),
        "claim_boundary": (
            "Layout-pinned IQM repeat is controlled diagnostic hardware evidence. "
            "It is not sufficient for manuscript claim upgrade until repeated "
            "statistics, calibration treatment, and cross-layout consistency are established."
        ),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    """Write a compact claim-boundary report."""
    lines = [
        "# IQM Layout-Pinned DLA Minimal Repeat Analysis",
        "",
        f"- Source: `{summary['source']}`",
        f"- IBM reference: `{summary['ibm_reference']}`",
        f"- Requested physical layout: `{summary['requested_initial_layout']}`",
        f"- Total circuits: `{summary['total_circuits']}`",
        f"- Total shots: `{summary['total_shots']}`",
        f"- Combined Fisher p-value: `{summary['fisher_p']:.6g}`",
        f"- Sign matches vs IBM Phase 2: `{summary['n_signs_matching_ibm_phase2']} / {summary['n_depths_compared_to_ibm_phase2']}`",
        "",
        "| Depth | IQM even leak | IQM odd leak | IQM asymmetry | IQM p | IBM asymmetry | Sign match |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary["depth_summaries"]:
        lines.append(
            f"| {row['depth']} | {row['iqm_leakage_even']:.6f} | "
            f"{row['iqm_leakage_odd']:.6f} | {row['iqm_asymmetry_relative']:+.6f} | "
            f"{row['iqm_fisher_p']:.6g} | {row['ibm_phase2_asymmetry_relative']:+.6f} | "
            f"{'yes' if row['sign_matches_ibm_phase2'] else 'no'} |"
        )
    lines += [
        "",
        "## Same-Layout Readout Baselines",
        "",
        "| Initial | Sector | Parity leakage | Initial-state retention |",
        "|---|---|---:|---:|",
    ]
    for row in summary["readout_baselines"]:
        lines.append(
            f"| `{row['initial']}` | {row['sector']} | "
            f"{row['parity_leakage']:.6f} | {row['initial_state_retention']:.6f} |"
        )
    lines += ["", "## Claim Boundary", "", summary["claim_boundary"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--ibm-input", type=Path, default=DEFAULT_IBM_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    summary = analyse_layout_repeat(args.input, args.ibm_input)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = _layout_slug(summary["requested_initial_layout"])
    json_path = args.output_dir / f"iqm_dla_layout_pinned_repeat_{slug}_analysis_2026-05-13.json"
    md_path = args.output_dir / f"iqm_dla_layout_pinned_repeat_{slug}_analysis_2026-05-13.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, summary)
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
