#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Domain Interfaces and Renormalisation spec builder
"""Promote Paper 0 Domain Interfaces and Renormalisation records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = (
    REPO_ROOT
    / "paper"
    / "gotm_scpn_master_publications"
    / "gotm-scpn_paper-00_the_foundational_framework"
    / "source_validation_artifacts"
)
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = (
    "P0R05633",
    "P0R05634",
    "P0R05635",
    "P0R05636",
    "P0R05637",
    "P0R05638",
    "P0R05639",
    "P0R05640",
)
CLAIM_BOUNDARY = "source-bounded domain interfaces and renormalisation source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "domain_interfaces_and_renormalisation.domain_interfaces_and_renormalisation": {
        "context_id": "domain_interfaces_and_renormalisation",
        "validation_protocol": "paper0.domain_interfaces_and_renormalisation.domain_interfaces_and_renormalisation",
        "canonical_statement": "The source-bounded component 'Domain Interfaces and Renormalisation' preserves Paper 0 records P0R05633-P0R05634 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05633:domain_interfaces_and_renormalisation",
            "P0R05634:domain_interfaces_and_renormalisation",
        ),
        "source_formulae": (
            "P0R05633: Domain Interfaces and Renormalisation",
            "P0R05634: The transition between the five Domains involves significant transformations of information and timescale separation, formalised using Renormalisation Group (RG) concepts and Impedance Matching.",
        ),
        "test_protocols": (
            "preserve Domain Interfaces and Renormalisation source-accounting boundary",
        ),
        "null_results": (
            "Domain Interfaces and Renormalisation is not empirical validation evidence",
        ),
        "variables": ("domain_interfaces_and_renormalisation",),
        "validation_targets": ("preserve records P0R05633-P0R05634",),
        "null_controls": (
            "domain_interfaces_and_renormalisation must remain source-bounded accounting",
        ),
    },
    "domain_interfaces_and_renormalisation.1_renormalisation_group_rg_flow_across_domains": {
        "context_id": "1_renormalisation_group_rg_flow_across_domains",
        "validation_protocol": "paper0.domain_interfaces_and_renormalisation.1_renormalisation_group_rg_flow_across_domains",
        "canonical_statement": "The source-bounded component '1. Renormalisation Group (RG) Flow Across Domains:' preserves Paper 0 records P0R05635-P0R05640 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05635:1_renormalisation_group_rg_flow_across_domains",
            "P0R05636:1_renormalisation_group_rg_flow_across_domains",
            "P0R05637:1_renormalisation_group_rg_flow_across_domains",
            "P0R05638:1_renormalisation_group_rg_flow_across_domains",
            "P0R05639:1_renormalisation_group_rg_flow_across_domains",
            "P0R05640:1_renormalisation_group_rg_flow_across_domains",
        ),
        "source_formulae": (
            "P0R05635: 1. Renormalisation Group (RG) Flow Across Domains:",
            "P0R05636: The transition across domains is viewed as an RG flow. The dynamics of a lower domain define the effective coupling constants for the higher domain.",
            "P0R05637: Example: The Biological-Organismal Interface (DI -> DII):",
            "P0R05638: The transition from L4 to L5 involves renormalising the microscopic degrees of freedom (oscillators) into the macroscopic order parameter (the _O field). The RG flow equations (derived from the UPDE) describe how effective couplings (Keff) change with scale (mu):",
            "P0R05639: $\\mu d\\mu dKeff = \\beta K(K,\\omega,\\ldots)$",
            "P0R05640: The emergence of the Self (L5) corresponds to reaching a fixed point in the RG flow where macroscopic coherence stabilises.",
        ),
        "test_protocols": (
            "preserve 1. Renormalisation Group (RG) Flow Across Domains: source-accounting boundary",
        ),
        "null_results": (
            "1. Renormalisation Group (RG) Flow Across Domains: is not empirical validation evidence",
        ),
        "variables": ("1_renormalisation_group_rg_flow_across_domains",),
        "validation_targets": ("preserve records P0R05635-P0R05640",),
        "null_controls": (
            "1_renormalisation_group_rg_flow_across_domains must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class DomainInterfacesAndRenormalisationSpec:
    """Spec promoted from Paper 0 source records."""

    key: str
    context_id: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    source_formulae: tuple[str, ...]
    test_protocols: tuple[str, ...]
    null_results: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class DomainInterfacesAndRenormalisationSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[DomainInterfacesAndRenormalisationSpec, ...]
    summary: dict[str, Any]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL ledger into dictionaries."""
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
    return records


def build_domain_interfaces_and_renormalisation_specs(
    source_records: list[dict[str, Any]],
) -> DomainInterfacesAndRenormalisationSpecBundle:
    """Build source-covered specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[DomainInterfacesAndRenormalisationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            DomainInterfacesAndRenormalisationSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 foundational extraction",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
                source_formulae=tuple(metadata["source_formulae"]),
                test_protocols=tuple(metadata["test_protocols"]),
                null_results=tuple(metadata["null_results"]),
                variables=tuple(metadata["variables"]),
                validation_targets=tuple(metadata["validation_targets"]),
                executable_validation_targets=tuple(metadata["validation_targets"]),
                null_controls=tuple(metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="promoted_source_accounting_fixture",
                domain_review_status="source_bounded_no_empirical_validation",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 " + "Domain Interfaces and Renormalisation" + " Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": sorted(set(SOURCE_LEDGER_IDS) - set(consumed)),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "category_counts": dict(sorted(category_counts.items())),
        "block_type_counts": dict(sorted(block_counts.items())),
        "math_ids": sorted(
            {math_id for record in anchors for math_id in record.get("math_ids", [])}
        ),
        "image_ids": sorted(
            {image_id for record in anchors for image_id in record.get("image_ids", [])}
        ),
        "table_ids": sorted(
            {str(record["table_id"]) for record in anchors if record.get("table_id") is not None}
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R05641",
    }
    return DomainInterfacesAndRenormalisationSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> DomainInterfacesAndRenormalisationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_domain_interfaces_and_renormalisation_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: DomainInterfacesAndRenormalisationSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Domain Interfaces and Renormalisation" + " Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### `{spec.key}`",
                "",
                spec.canonical_statement,
                "",
                f"- Context: `{spec.context_id}`",
                f"- Protocol: `{spec.validation_protocol}`",
                f"- Source equations: {', '.join(spec.source_equation_ids)}",
                f"- Null controls: {', '.join(spec.null_controls)}",
                "",
            ]
        )
    return "\n".join(lines)


def write_outputs(
    bundle: DomainInterfacesAndRenormalisationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_domain_interfaces_and_renormalisation_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_domain_interfaces_and_renormalisation_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 domain-interface renormalisation specs from the ledger."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    outputs = write_outputs(
        build_from_ledger(args.ledger), output_dir=args.output_dir, date_tag=args.date_tag
    )
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
