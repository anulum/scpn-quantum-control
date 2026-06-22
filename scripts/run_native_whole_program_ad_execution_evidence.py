#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — native LLVM/JIT whole-program AD execution evidence writer.
"""Capture and write the native LLVM/JIT whole-program AD execution evidence artefact.

Runs the fixed whole-program AD execution battery through the native LLVM/JIT path,
checks each value and gradient against the interpreted Program AD reference, and writes
a dated JSON evidence artefact plus a Markdown summary under the evidence directory.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from scpn_quantum_control.compiler import (
    run_native_whole_program_ad_execution_evidence,
)


def _render_markdown(payload: dict[str, object]) -> str:
    """Return a Markdown summary for the captured execution evidence."""

    cases = payload["cases"]
    assert isinstance(cases, list)
    lines = [
        "# Native LLVM/JIT whole-program AD execution evidence",
        "",
        f"- artifact_id: `{payload['artifact_id']}`",
        f"- beyond_scalar_executed: {payload['beyond_scalar_executed']}",
        f"- executed_operation_families: {payload['executed_operation_families']}",
        f"- max_value_error: {payload['max_value_error']:.3e}",
        f"- max_gradient_error: {payload['max_gradient_error']:.3e}"
        f" (tolerance {payload['gradient_parity_tolerance']:.3e})",
        f"- fail_closed_boundaries: {payload['fail_closed_boundaries']}",
        "",
        "| case | family | dim | status | value_error | gradient_error |",
        "|------|--------|-----|--------|-------------|----------------|",
    ]
    for case in cases:
        assert isinstance(case, dict)
        value_error = case["value_error"]
        gradient_error = case["gradient_error"]
        value_text = "—" if value_error is None else f"{float(value_error):.2e}"
        gradient_text = "—" if gradient_error is None else f"{float(gradient_error):.2e}"
        lines.append(
            f"| {case['case_id']} | {case['operation_family']} | {case['operand_dimension']}"
            f" | {case['status']} | {value_text} | {gradient_text} |"
        )
    lines.append("")
    lines.append(f"Claim boundary: {payload['claim_boundary']}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    """Write the native execution evidence artefact from CLI arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("data/differentiable_phase_qnode"))
    parser.add_argument("--stamp", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--gradient-parity-tolerance", type=float, default=1e-6)
    args = parser.parse_args()

    evidence = run_native_whole_program_ad_execution_evidence(
        artifact_id=f"native-whole-program-ad-execution-{args.stamp}",
        gradient_parity_tolerance=args.gradient_parity_tolerance,
    )
    payload = evidence.to_dict()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"native_whole_program_ad_execution_evidence_{args.stamp}.json"
    md_path = args.output_dir / f"native_whole_program_ad_execution_evidence_{args.stamp}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
