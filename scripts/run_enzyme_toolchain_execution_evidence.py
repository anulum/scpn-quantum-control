#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — real Enzyme/LLVM toolchain AD execution evidence writer.
"""Capture and write the real Enzyme/LLVM reverse-mode AD execution evidence artefact.

Drives the installed Enzyme/LLVM toolchain over the scalar, vector and matrix battery,
checks each toolchain gradient against the analytic reference, and writes a dated JSON
evidence artefact plus a Markdown summary. When the toolchain is absent the artefact is
written as a gated hard gap rather than fabricated.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from scpn_quantum_control.compiler import (
    run_enzyme_toolchain_execution_evidence,
)


def _render_markdown(payload: dict[str, object]) -> str:
    """Return a Markdown summary for the captured Enzyme execution evidence."""

    cases = payload["cases"]
    assert isinstance(cases, list)
    lines = [
        "# Real Enzyme/LLVM reverse-mode AD execution evidence",
        "",
        f"- artifact_id: `{payload['artifact_id']}`",
        f"- toolchain_available: {payload['toolchain_available']}",
        f"- toolchain: {payload['toolchain']}",
        f"- beyond_scalar_executed: {payload['beyond_scalar_executed']}",
        f"- executed_operation_families: {payload['executed_operation_families']}",
        f"- max_gradient_error: {payload['max_gradient_error']:.3e}"
        f" (tolerance {payload['gradient_parity_tolerance']:.3e})",
        "",
        "| case | family | dim | status | gradient_error |",
        "|------|--------|-----|--------|----------------|",
    ]
    for case in cases:
        assert isinstance(case, dict)
        gradient_error = case["gradient_error"]
        gradient_text = "—" if gradient_error is None else f"{float(gradient_error):.2e}"
        lines.append(
            f"| {case['case_id']} | {case['operation_family']} | {case['operand_dimension']}"
            f" | {case['status']} | {gradient_text} |"
        )
    lines.append("")
    lines.append(f"Claim boundary: {payload['claim_boundary']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Write the Enzyme toolchain execution evidence artefact from CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/differentiable_phase_qnode"))
    parser.add_argument("--stamp", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--gradient-parity-tolerance", type=float, default=1e-9)
    args = parser.parse_args()

    evidence = run_enzyme_toolchain_execution_evidence(
        artifact_id=f"enzyme-toolchain-ad-execution-{args.stamp}",
        gradient_parity_tolerance=args.gradient_parity_tolerance,
    )
    payload = evidence.to_dict()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"enzyme_toolchain_ad_execution_evidence_{args.stamp}.json"
    md_path = args.output_dir / f"enzyme_toolchain_ad_execution_evidence_{args.stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    if not payload["toolchain_available"]:
        print("note: Enzyme toolchain unavailable; artefact written as a gated hard gap")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
