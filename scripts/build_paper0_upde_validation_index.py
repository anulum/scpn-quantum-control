#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# © Concepts 1996–2026 Miroslav Šotek. All rights reserved.
# © Code 2020–2026 Miroslav Šotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control — Paper 0 UPDE aggregate validation index
"""Build an aggregate validation index over all Paper 0 UPDE fixtures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_SPECS_PATH = DEFAULT_EXTRACTION_DIR / "paper0_upde_validation_specs_2026-05-13.json"
DEFAULT_RESULT_GLOB = "paper0_upde_*_fixture_result_2026-05-13.json"
SIMULATOR_ONLY_STATUS = "simulator_only_no_provider_submission"


def build_validation_index(
    *,
    specs_path: Path = DEFAULT_SPECS_PATH,
    result_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Build the aggregate UPDE validation index with complete fixture coverage."""
    specs_payload = _load_json(specs_path)
    specs = list(specs_payload.get("specs", []))
    if not specs:
        raise ValueError(f"no specs found in {specs_path}")
    results = [_load_json(path) for path in _default_result_paths() if result_paths is None]
    if result_paths is not None:
        results = [_load_json(path) for path in result_paths]

    specs_by_key = {str(spec["key"]): spec for spec in specs}
    results_by_key = {str(result["spec_key"]): result for result in results}
    expected_keys = set(specs_by_key)
    actual_keys = set(results_by_key)
    missing = sorted(expected_keys - actual_keys)
    extra = sorted(actual_keys - expected_keys)
    if missing:
        raise ValueError(f"missing fixture results for promoted specs: {missing}")
    if extra:
        raise ValueError(f"fixture results without promoted specs: {extra}")

    fixtures: list[dict[str, Any]] = []
    total_runtime = 0.0
    for key in sorted(expected_keys):
        spec = specs_by_key[key]
        result = results_by_key[key]
        if result.get("hardware_status") != SIMULATOR_ONLY_STATUS:
            raise ValueError(
                f"non-simulator fixture result for {key}: {result.get('hardware_status')}"
            )
        runtime_ms = float(result.get("runtime_ms", 0.0))
        if runtime_ms < 0.0:
            raise ValueError(f"runtime_ms must be non-negative for {key}")
        total_runtime += runtime_ms
        null_controls = dict(result.get("null_controls", {}))
        fixtures.append(
            {
                "spec_key": key,
                "validation_protocol": str(result["validation_protocol"]),
                "hardware_status": str(result["hardware_status"]),
                "source_equation_ids": list(result.get("source_equation_ids", [])),
                "source_ledger_ids": list(result.get("source_ledger_ids", [])),
                "declared_null_controls": list(spec.get("null_controls", [])),
                "measured_null_controls": null_controls,
                "null_control_count": len(null_controls),
                "runtime_ms": runtime_ms,
                "result_fields": sorted(result.keys()),
                "implementation_status": str(spec.get("implementation_status", "unknown")),
            }
        )

    all_simulator_only = all(
        fixture["hardware_status"] == SIMULATOR_ONLY_STATUS for fixture in fixtures
    )
    summary = {
        "spec_count": len(specs),
        "fixture_result_count": len(results),
        "coverage_match": expected_keys == actual_keys,
        "spec_keys": [fixture["spec_key"] for fixture in fixtures],
        "hardware_status": SIMULATOR_ONLY_STATUS if all_simulator_only else "mixed",
        "all_simulator_only": all_simulator_only,
        "total_runtime_ms": total_runtime,
        "next_recommended_family": "paper0.next_mechanism_family",
        "policy": (
            "This index aggregates simulator fixtures only. No provider submission "
            "is represented and no hardware claim follows from these results."
        ),
    }
    return {"summary": summary, "fixtures": fixtures}


def render_validation_report(index: dict[str, Any]) -> str:
    """Render a concise Markdown report from the aggregate index."""
    summary = index["summary"]
    status = "match" if summary["coverage_match"] else "mismatch"
    lines = [
        "# Paper 0 UPDE Aggregate Validation Index",
        "",
        f"- Spec count: `{summary['spec_count']}`",
        f"- Fixture result count: `{summary['fixture_result_count']}`",
        f"- Coverage status: `{status}`",
        f"- Hardware status: `{summary['hardware_status']}`",
        f"- Total local fixture runtime: `{summary['total_runtime_ms']}` ms",
        "",
        "## Fixtures",
        "",
    ]
    for fixture in index["fixtures"]:
        equations = ", ".join(fixture["source_equation_ids"])
        ledgers = ", ".join(fixture["source_ledger_ids"])
        lines.extend(
            [
                f"### {fixture['spec_key']}",
                "",
                f"- Protocol: `{fixture['validation_protocol']}`",
                f"- Source equations: `{equations}`",
                f"- Source ledgers: `{ledgers}`",
                f"- Measured null controls: `{fixture['null_control_count']}`",
                f"- Runtime: `{fixture['runtime_ms']}` ms",
                f"- Hardware status: `{fixture['hardware_status']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "No provider submission is represented by this aggregate index. "
            "The records are simulator-only validation fixtures for source-anchored "
            "Paper 0 equations.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    index: dict[str, Any],
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write aggregate JSON and Markdown outputs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_upde_aggregate_validation_index_{date_tag}.json"
    report_path = output_dir / f"paper0_upde_aggregate_validation_index_{date_tag}.md"
    json_path.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_validation_report(index), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _default_result_paths() -> list[Path]:
    return sorted(DEFAULT_EXTRACTION_DIR.glob(DEFAULT_RESULT_GLOB))


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"required JSON file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}") from exc


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specs", type=Path, default=DEFAULT_SPECS_PATH)
    parser.add_argument("--result", action="append", type=Path, dest="results")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args(argv)

    index = build_validation_index(specs_path=args.specs, result_paths=args.results)
    paths = write_outputs(index, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(index["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
