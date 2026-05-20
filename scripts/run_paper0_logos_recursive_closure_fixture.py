#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Logos recursive closure runner
"""Run the Paper 0 Logos recursive-closure validation fixture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

from scpn_quantum_control.paper0.logos_recursive_closure_validation import (
    validate_logos_recursive_closure_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return value


def render_report(payload: dict[str, Any]) -> str:
    """Render a compact Markdown report for the Logos fixture."""
    return "\n".join(
        [
            "# Paper 0 Logos Recursive Closure Fixture",
            "",
            f"- Source span: {payload['source_ledger_span'][0]} - "
            f"{payload['source_ledger_span'][1]}",
            f"- Hardware status: {payload['hardware_status']}",
            f"- Claim boundary: {payload['claim_boundary']}",
            f"- Axioms: {payload['axiom_count']}",
            f"- H_int roles: {payload['hint_role_count']}",
            f"- Blank separators: {payload['blank_separator_count']}",
            f"- Next boundary: {payload['next_source_boundary']}",
            "",
            "## H_int Axiom Roles",
            *(f"- `{key}`: {value}" for key, value in sorted(payload["hint_axiom_roles"].items())),
            "",
            "## Null Controls",
            *(f"- `{key}`: {value}" for key, value in sorted(payload["null_controls"].items())),
            "",
        ]
    )


def write_outputs(
    *,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    """Run the fixture and write JSON plus Markdown outputs."""
    result = validate_logos_recursive_closure_fixture()
    payload = cast(dict[str, Any], _json_ready(result.as_dict()))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    """Run the Logos recursive-closure fixture and write artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR
        / "paper0_logos_recursive_closure_fixture_result_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "paper0_logos_recursive_closure_fixture_report_2026-05-13.md",
    )
    args = parser.parse_args()
    payload = write_outputs(output_path=args.output, report_path=args.report)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
