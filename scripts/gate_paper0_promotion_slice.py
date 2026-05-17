#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 promotion gate
"""Gate generated Paper 0 promotion artefacts before acceptance."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

INTERNAL_AGENT_TERMS = (
    "Co" + "dex",
    "Open" + "AI",
    "Clau" + "de",
    "Anthro" + "pic",
    "Gem" + "ini",
    "Google " + "AI",
)


@dataclass(frozen=True, slots=True)
class PromotionGateResult:
    """Gate result for one generated Paper 0 promotion slice."""

    passed: bool
    source_start: str
    source_end: str
    source_record_count: int
    spec_count: int
    failures: tuple[str, ...]
    warnings: tuple[str, ...]


def gate_promotion_slice(
    *,
    spec_bundle_path: Path,
    fixture_path: Path | None = None,
    reconciliation_path: Path | None = None,
    public_paths: tuple[Path, ...] = (),
) -> PromotionGateResult:
    """Gate generated spec, fixture, reconciliation, and public surface artefacts."""
    failures: list[str] = []
    warnings: list[str] = []
    spec_payload = _read_json(spec_bundle_path)
    summary = spec_payload.get("summary", {})
    span = tuple(summary.get("source_ledger_span", ()))
    if len(span) != 2:
        failures.append("spec summary must define a two-item source_ledger_span")
        source_start = ""
        source_end = ""
        expected_count = 0
    else:
        source_start, source_end = str(span[0]), str(span[1])
        expected_count = _ledger_number(source_end) - _ledger_number(source_start) + 1
        if summary.get("source_record_count") != expected_count:
            failures.append("spec source_record_count must equal numeric source span length")

    if summary.get("consumed_source_record_count") != summary.get("source_record_count"):
        failures.append("consumed_source_record_count must equal source_record_count")
    if summary.get("coverage_match") is not True:
        failures.append("spec coverage_match must be true")
    if summary.get("unconsumed_source_ledger_ids") not in (
        [],
        (),
    ):  # JSON versus direct payload compatibility.
        failures.append("unconsumed_source_ledger_ids must be empty")
    if summary.get("spec_count", 0) < 1:
        failures.append("spec_count must be positive")
    if "not validation evidence" not in str(summary.get("claim_boundary", "")):
        failures.append(
            "spec claim_boundary must state that the artefact is not validation evidence"
        )
    if summary.get("hardware_status") != "source_methodology_no_experiment":
        failures.append("hardware_status must remain source_methodology_no_experiment")

    specs = tuple(spec_payload.get("specs", ()))
    if len(specs) != summary.get("spec_count"):
        failures.append("spec list length must equal summary spec_count")
    required_ids = (
        tuple(
            f"P0R{number:05d}"
            for number in range(_ledger_number(source_start), _ledger_number(source_end) + 1)
        )
        if span
        else ()
    )
    for index, spec in enumerate(specs):
        prefix = f"spec[{index}]"
        if tuple(spec.get("source_ledger_ids", ())) != required_ids:
            failures.append(
                f"{prefix} source_ledger_ids must exactly equal the contiguous source span"
            )
        if spec.get("claim_boundary") != summary.get("claim_boundary"):
            failures.append(f"{prefix} claim_boundary must match summary claim_boundary")
        if spec.get("hardware_status") != summary.get("hardware_status"):
            failures.append(f"{prefix} hardware_status must match summary hardware_status")
        if not spec.get("canonical_statement"):
            failures.append(f"{prefix} canonical_statement must be non-empty")
        if not spec.get("null_controls"):
            failures.append(f"{prefix} null_controls must be non-empty")

    if fixture_path is not None:
        _gate_fixture(
            fixture_path=fixture_path,
            summary=summary,
            expected_count=expected_count,
            failures=failures,
        )
    else:
        warnings.append("fixture artefact was not provided")

    if reconciliation_path is not None:
        _gate_reconciliation(reconciliation_path=reconciliation_path, failures=failures)
    else:
        warnings.append("reconciliation artefact was not provided")

    for path in public_paths:
        _gate_public_text(path=path, failures=failures)

    return PromotionGateResult(
        passed=not failures,
        source_start=source_start,
        source_end=source_end,
        source_record_count=int(summary.get("source_record_count", 0) or 0),
        spec_count=int(summary.get("spec_count", 0) or 0),
        failures=tuple(failures),
        warnings=tuple(warnings),
    )


def write_gate_result(result: PromotionGateResult, output_path: Path) -> Path:
    """Write a JSON gate result for auditability."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def main() -> int:
    """Run the Paper 0 promotion gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec-bundle", type=Path, required=True)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--reconciliation", type=Path)
    parser.add_argument("--public-path", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = gate_promotion_slice(
        spec_bundle_path=args.spec_bundle,
        fixture_path=args.fixture,
        reconciliation_path=args.reconciliation,
        public_paths=tuple(args.public_path),
    )
    if args.output is not None:
        write_gate_result(result, args.output)
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0 if result.passed else 1


def _gate_fixture(
    *,
    fixture_path: Path,
    summary: dict[str, Any],
    expected_count: int,
    failures: list[str],
) -> None:
    fixture = _read_json(fixture_path)
    if tuple(fixture.get("source_ledger_span", ())) != tuple(
        summary.get("source_ledger_span", ())
    ):
        failures.append("fixture source_ledger_span must match spec summary")
    if fixture.get("source_record_count") != expected_count:
        failures.append("fixture source_record_count must equal numeric source span length")
    if fixture.get("component_count") != summary.get("spec_count"):
        failures.append("fixture component_count must equal spec_count")
    if fixture.get("hardware_status") != summary.get("hardware_status"):
        failures.append("fixture hardware_status must match spec summary")
    if fixture.get("claim_boundary") != summary.get("claim_boundary"):
        failures.append("fixture claim_boundary must match spec summary")
    if fixture.get("next_source_boundary") != summary.get("next_source_boundary"):
        failures.append("fixture next_source_boundary must match spec summary")


def _gate_reconciliation(*, reconciliation_path: Path, failures: list[str]) -> None:
    reconciliation = _read_json(reconciliation_path)
    summary = reconciliation.get("summary", {})
    if summary.get("missing_surface_count") != 0:
        failures.append("reconciliation missing_surface_count must be zero")
    if summary.get("overlap_count") != 0:
        failures.append("reconciliation overlap_count must be zero")
    if summary.get("promoted_surface_integrity") is not True:
        failures.append("reconciliation promoted_surface_integrity must be true")


def _gate_public_text(*, path: Path, failures: list[str]) -> None:
    text = path.read_text(encoding="utf-8")
    for term in INTERNAL_AGENT_TERMS:
        if term in text:
            failures.append(f"public path {path} contains internal agent/vendor term {term}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ledger_number(ledger_id: str) -> int:
    return int(ledger_id.removeprefix("P0R"))


if __name__ == "__main__":
    raise SystemExit(main())
