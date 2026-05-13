#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 NTHS spin-glass fixture runner
"""Run the Paper 0 NTHS spin-glass simulator fixture."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from scpn_quantum_control.paper0.nths_spin_glass_validation import (
    SpinGlassValidationResult,
    validate_nths_spin_glass_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"


def default_problem() -> tuple[np.ndarray, np.ndarray]:
    """Return the deterministic finite NTHS spin-glass validation fixture."""
    J_ij = np.array(
        [
            [0.0, 0.84, 0.71, -0.42, -0.16, -0.21],
            [0.84, 0.0, 0.63, -0.38, -0.28, -0.18],
            [0.71, 0.63, 0.0, -0.44, -0.24, -0.33],
            [-0.42, -0.38, -0.44, 0.0, 0.78, 0.69],
            [-0.16, -0.28, -0.24, 0.78, 0.0, 0.73],
            [-0.21, -0.18, -0.33, 0.69, 0.73, 0.0],
        ],
        dtype=np.float64,
    )
    h_i = np.array([0.09, -0.04, 0.03, -0.07, 0.05, -0.02], dtype=np.float64)
    return J_ij, h_i


def run_default_fixture() -> SpinGlassValidationResult:
    """Run the default Paper 0 NTHS spin-glass validation fixture."""
    J_ij, h_i = default_problem()
    return validate_nths_spin_glass_fixture(J_ij, h_i)


def render_report(payload: dict[str, Any]) -> str:
    """Render a concise Markdown report for the NTHS spin-glass fixture."""
    lines = [
        "# Paper 0 NTHS Spin-Glass Fixture",
        "",
        f"- Spec key: `{payload['spec_key']}`",
        f"- Protocol: `{payload['validation_protocol']}`",
        f"- Hardware status: `{payload['hardware_status']}`",
        f"- Source equations: `{', '.join(payload['source_equation_ids'])}`",
        f"- Source ledgers: `{', '.join(payload['source_ledger_ids'])}`",
        f"- Exact state count: `{payload['state_count']}`",
        f"- Ground-state energy: `{payload['ground_state_energy']}`",
        f"- Mean energy: `{payload['mean_energy']}`",
        f"- Ground-state magnetisation: `{payload['ground_state_magnetisation']}`",
        f"- Edwards-Anderson q_EA: `{payload['edwards_anderson_q']}`",
        f"- Ultrametric violation: `{payload['ultrametric_violation']}`",
        f"- Runtime: `{payload['runtime_ms']}` ms",
        "",
        "## Null Controls",
        "",
    ]
    for key, value in sorted(payload["null_controls"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(
        [
            "",
            "## Policy",
            "",
            "No provider submission is represented. This is a simulator-only "
            "finite spin-glass fixture for the source-anchored Paper 0 NTHS "
            "Hamiltonian.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    result: SpinGlassValidationResult,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
    runtime_ms: float | None = None,
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the NTHS spin-glass fixture."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_nths_spin_glass_fixture_result_{date_tag}.json"
    report_path = output_dir / f"paper0_nths_spin_glass_fixture_report_{date_tag}.md"
    payload = _json_ready_payload(result, runtime_ms=0.0 if runtime_ms is None else runtime_ms)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _json_ready_payload(
    result: SpinGlassValidationResult,
    *,
    runtime_ms: float,
) -> dict[str, Any]:
    return {
        "spec_key": result.spec_key,
        "validation_protocol": result.validation_protocol,
        "hardware_status": result.hardware_status,
        "source_equation_ids": list(result.source_equation_ids),
        "source_ledger_ids": list(result.source_ledger_ids),
        "state_count": result.state_count,
        "ground_state": list(result.ground_state),
        "ground_state_energy": result.ground_state_energy,
        "ground_state_magnetisation": result.ground_state_magnetisation,
        "mean_energy": result.mean_energy,
        "edwards_anderson_q": result.edwards_anderson_q,
        "ultrametric_violation": result.ultrametric_violation,
        "null_controls": dict(result.null_controls),
        "problem_metadata": dict(result.problem_metadata),
        "runtime_ms": runtime_ms,
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args(argv)

    start = time.perf_counter()
    result = run_default_fixture()
    runtime_ms = (time.perf_counter() - start) * 1000.0
    paths = write_outputs(
        result, output_dir=args.output_dir, date_tag=args.date_tag, runtime_ms=runtime_ms
    )
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(_json_ready_payload(result, runtime_ms=runtime_ms), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
