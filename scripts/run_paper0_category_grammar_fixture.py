#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 category grammar fixture runner
"""Run the Paper 0 category grammar formal-consistency fixture."""

from __future__ import annotations

import argparse
import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

from scpn_quantum_control.paper0.category_grammar_validation import (
    CategoryGrammarConfig,
    validate_category_grammar_fixture,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"


def _json_ready(value: Any) -> Any:
    if isinstance(value, MappingProxyType):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(payload: dict[str, Any]) -> str:
    """Render a compact Markdown fixture report."""
    return (
        "# Paper 0 Category Grammar Fixture\n\n"
        f"- Source span: {payload['source_ledger_span'][0]} - {payload['source_ledger_span'][1]}\n"
        f"- Hardware status: {payload['hardware_status']}\n"
        f"- Layer count: {payload['layer_count']}\n"
        f"- Identity law: {payload['category_laws']['identity_law']}\n"
        f"- Associativity law: {payload['category_laws']['associativity_law']}\n"
        f"- Functorial mapping valid: {payload['functorial_mapping_valid']}\n"
        f"- Truth values: {', '.join(payload['truth_values'])}\n"
        f"- Natural transformation boundary: {payload['natural_transformation_boundary']}\n"
        f"- Claim boundary: {payload['claim_boundary']}\n"
    )


def write_outputs(
    *,
    output_path: Path,
    report_path: Path,
    spec_bundle_path: Path | None = None,
) -> dict[str, Any]:
    """Run fixture and write JSON plus Markdown artefacts."""
    result = validate_category_grammar_fixture(
        CategoryGrammarConfig(spec_bundle_path=spec_bundle_path)
    )
    payload = cast(dict[str, Any], _json_ready(result))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(payload), encoding="utf-8")
    return payload


def main() -> int:
    """Run the default category grammar fixture."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR / "paper0_category_grammar_fixture_result_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR / "paper0_category_grammar_fixture_report_2026-05-13.md",
    )
    parser.add_argument("--spec-bundle", type=Path, default=None)
    args = parser.parse_args()

    payload = write_outputs(
        output_path=args.output,
        report_path=args.report,
        spec_bundle_path=args.spec_bundle,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
