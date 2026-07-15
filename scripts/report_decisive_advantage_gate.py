#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# SCPN Quantum Control — decisive advantage-gate report script
"""Report the decisive advantage gate: preregistration and, if rows are given, the decision.

Without ``--rows`` the script emits the preregistered protocol manifest (the
public commitment before any run). With ``--rows`` it additionally validates the
measured rows and prints the fail-closed decision label.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from scpn_quantum_control.benchmarks.decisive_advantage_protocol import (
    DecisiveAdvantageProtocol,
    default_decisive_advantage_protocol,
    evaluate_decision,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "decisive_advantage_gate"


def _rows_from_payload(payload: Any) -> list[dict[str, Any]]:
    """Extract a list of row mappings from a loaded JSON payload.

    Parameters
    ----------
    payload
        Parsed JSON: either a list of rows or a mapping with a ``rows`` list.

    Returns
    -------
    list of dict
        The rows as plain dictionaries.

    Raises
    ------
    ValueError
        If the payload is neither a list nor a mapping carrying a ``rows`` list.
    """
    if isinstance(payload, list):
        return [dict(row) for row in payload]
    if isinstance(payload, Mapping) and isinstance(payload.get("rows"), list):
        return [dict(row) for row in payload["rows"]]
    raise ValueError("rows payload must be a list or contain a rows list")


def build_report(
    protocol: DecisiveAdvantageProtocol,
    rows: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Assemble the preregistration (and decision, when rows are supplied) report.

    Parameters
    ----------
    protocol
        The decisive advantage protocol.
    rows
        Measured rows to decide on, or ``None`` for a preregistration-only report.

    Returns
    -------
    dict
        A JSON-ready report with the protocol manifest and, if rows are given,
        the validation result and decision outcome.
    """
    report: dict[str, Any] = {"protocol": protocol.to_dict()}
    if rows is not None:
        report["validation"] = protocol.validate_rows(rows).to_dict()
        report["decision"] = evaluate_decision(protocol, rows).to_dict()
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv
        Argument vector, or ``None`` to read from ``sys.argv``.

    Returns
    -------
    argparse.Namespace
        Parsed ``rows`` and ``out_dir`` options.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows",
        type=Path,
        default=None,
        help="Optional measured-rows JSON; omit for a preregistration-only report.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the decisive advantage-gate report.

    Parameters
    ----------
    argv
        Argument vector, or ``None`` to read from ``sys.argv``.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    args = _parse_args(argv)
    protocol = default_decisive_advantage_protocol()

    rows: list[dict[str, Any]] | None = None
    if args.rows is not None:
        rows = _rows_from_payload(json.loads(Path(args.rows).read_text(encoding="utf-8")))

    report = build_report(protocol, rows)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{protocol.protocol.protocol_id}.json"
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print(f"protocol: {protocol.protocol.protocol_id}")
    print(f"qpu_time_estimate_s: {protocol.qpu_time_estimate_s:.3f}")
    if "decision" in report:
        print(f"decision: {report['decision']['label']}")
        for reason in report["decision"]["reasons"]:
            print(f"  - {reason}")
    else:
        print("preregistration only (no rows supplied)")
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
