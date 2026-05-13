#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — IQM layout campaign summary
"""Summarise completed and cancelled IQM layout-pinned DLA blocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "iqm_paper_replication"
DEFAULT_ANALYSES = (
    DEFAULT_OUTPUT_DIR / "iqm_dla_layout_pinned_repeat_analysis_2026-05-13.json",
    DEFAULT_OUTPUT_DIR / "iqm_dla_layout_pinned_repeat_q13-8-9-14_analysis_2026-05-13.json",
    DEFAULT_OUTPUT_DIR / "iqm_dla_layout_pinned_repeat_q11-6-5-10_analysis_2026-05-13.json",
    DEFAULT_OUTPUT_DIR / "iqm_dla_layout_pinned_repeat_q17-18-19-15_analysis_2026-05-13.json",
    DEFAULT_OUTPUT_DIR / "iqm_dla_layout_pinned_repeat_q0-1-4-3_analysis_2026-05-13.json",
    DEFAULT_OUTPUT_DIR / "iqm_dla_layout_pinned_repeat_q2-7-12-13_analysis_2026-05-13.json",
)
DEFAULT_CANCELLED = (
    DEFAULT_OUTPUT_DIR / "iqm_dla_layout_pinned_repeat_q11-6-5-10_2026-05-13_cancelled.json",
)


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT))


def summarise_campaign(
    analyses: tuple[Path, ...] = DEFAULT_ANALYSES,
    cancelled: tuple[Path, ...] = DEFAULT_CANCELLED,
) -> dict[str, Any]:
    """Build a campaign-level summary from sanitized public artefacts."""
    completed = []
    for path in analyses:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        layout = payload["requested_initial_layout"]
        completed.append(
            {
                "analysis_path": _relative(path),
                "layout": layout,
                "total_circuits": payload["total_circuits"],
                "total_shots": payload["total_shots"],
                "combined_fisher_p": payload["fisher_p"],
                "sign_matches_vs_ibm_phase2": (
                    f"{payload['n_signs_matching_ibm_phase2']}/"
                    f"{payload['n_depths_compared_to_ibm_phase2']}"
                ),
                "depth_asymmetries": {
                    str(row["depth"]): row["iqm_asymmetry_relative"]
                    for row in payload["depth_summaries"]
                },
                "readout_parity_leakage": {
                    row["initial"]: row["parity_leakage"] for row in payload["readout_baselines"]
                },
            }
        )

    cancelled_rows = []
    for path in cancelled:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        cancelled_rows.append(
            {
                "path": _relative(path),
                "layout": payload["requested_initial_layout"],
                "submitted_circuit": payload["submitted_circuit"],
                "submitted_shots": payload["submitted_shots"],
                "status_before_cancel": payload["status_before_cancel"],
                "status_after_cancel": payload["status_after_cancel"],
                "job_id_sha256": payload["job_id_sha256"],
                "unsubmitted_circuits": payload["unsubmitted_circuits"],
            }
        )

    return {
        "schema": "scpn_iqm_layout_campaign_summary_v1",
        "provider": "iqm",
        "platform": "IQM Resonance Garnet",
        "completed_layout_blocks": completed,
        "cancelled_layout_blocks": cancelled_rows,
        "completed_blocks": len(completed),
        "completed_circuits": sum(row["total_circuits"] for row in completed),
        "completed_shots": sum(row["total_shots"] for row in completed),
        "cancelled_submitted_circuits": len(cancelled_rows),
        "claim_boundary": (
            "The current IQM layout campaign is controlled diagnostic evidence. "
            "Six completed pinned layouts and one earlier queued/cancelled "
            "third-layout attempt do not support manuscript claim upgrade; they "
            "motivate a larger scheduled credit-grant campaign with repeated "
            "layout statistics."
        ),
    }


def write_markdown(path: Path, summary: dict[str, Any]) -> None:
    """Write a concise human-readable campaign summary."""
    lines = [
        "# IQM Layout Campaign Summary",
        "",
        f"- Completed layout blocks: `{summary['completed_blocks']}`",
        f"- Completed circuits: `{summary['completed_circuits']}`",
        f"- Completed shots: `{summary['completed_shots']}`",
        f"- Cancelled submitted circuits: `{summary['cancelled_submitted_circuits']}`",
        "",
        "| Layout | Shots | Fisher p | Sign matches | d4 asym | d6 asym | d10 asym |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["completed_layout_blocks"]:
        asym = row["depth_asymmetries"]
        lines.append(
            f"| `{row['layout']}` | {row['total_shots']} | {row['combined_fisher_p']:.6g} | "
            f"{row['sign_matches_vs_ibm_phase2']} | {asym['4']:+.6f} | "
            f"{asym['6']:+.6f} | {asym['10']:+.6f} |"
        )
    lines += [
        "",
        "## Cancelled / Not Executed",
        "",
        "| Layout | Submitted circuit | Status before | Status after | Unsubmitted circuits | Job hash |",
        "|---|---|---|---|---:|---|",
    ]
    for row in summary["cancelled_layout_blocks"]:
        lines.append(
            f"| `{row['layout']}` | `{row['submitted_circuit']}` | "
            f"{row['status_before_cancel']} | {row['status_after_cancel']} | "
            f"{row['unsubmitted_circuits']} | `{row['job_id_sha256']}` |"
        )
    lines += ["", "## Claim Boundary", "", summary["claim_boundary"]]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    summary = summarise_campaign()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "iqm_layout_campaign_summary_2026-05-13.json"
    md_path = args.output_dir / "iqm_layout_campaign_summary_2026-05-13.md"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(md_path, summary)
    print(f"wrote_json={json_path}")
    print(f"wrote_md={md_path}")
    print(f"completed_blocks={summary['completed_blocks']}")
    print(f"completed_shots={summary['completed_shots']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
