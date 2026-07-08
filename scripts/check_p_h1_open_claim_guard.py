#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — p_h1 Open-Claim Guard Export
"""Run the public p_h1 open-claim guard and write its report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scpn_quantum_control.analysis.p_h1_open_guard import run_p_h1_open_guard

DATE = "2026-07-08"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "data" / "p_h1_open_guard" / f"p_h1_open_guard_{DATE}.json"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    """Parse p_h1 open-claim guard options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    """Run the p_h1 open-claim guard and write a JSON report."""
    args = parse_args()
    report = run_p_h1_open_guard(REPO_ROOT)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {_display_path(args.output)}")
    if not report.passed:
        for violation in report.violations:
            print(f"{violation.path}: {violation.reason}: {violation.excerpt}")
        return 1
    print(f"checked {len(report.checked_paths)} p_h1 public surfaces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
