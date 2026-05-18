#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0  as a Coupling Affinity: spec builder
"""Promote Paper 0  as a Coupling Affinity: records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = (
    "P0R03501",
    "P0R03502",
    "P0R03503",
    "P0R03504",
    "P0R03505",
    "P0R03506",
    "P0R03507",
    "P0R03508",
    "P0R03509",
)
CLAIM_BOUNDARY = (
    "source-bounded as a coupling affinity source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "as_a_coupling_affinity.as_a_coupling_affinity": {
        "context_id": "as_a_coupling_affinity",
        "validation_protocol": "paper0.as_a_coupling_affinity.as_a_coupling_affinity",
        "canonical_statement": "The source-bounded component 'as a Coupling Affinity:' preserves Paper 0 records P0R03501-P0R03502 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03501:as_a_coupling_affinity",
            "P0R03502:as_a_coupling_affinity",
        ),
        "source_formulae": (
            "P0R03501: as a Coupling Affinity:",
            'P0R03502: The value of can be seen as a measure of the "coupling affinity" of a system. The teleological drive of the universe, which seeks to maximise coherence and complexity, is therefore a drive to create substrates with ever-higher values of . This is because high- systems are the most effective and potent vehicles for the universal consciousness (Psis) to experience and interact with the manifest world. The universe is actively building better and better partners for itself to dance with.',
        ),
        "test_protocols": ("preserve as a Coupling Affinity: source-accounting boundary",),
        "null_results": ("as a Coupling Affinity: is not empirical validation evidence",),
        "variables": ("as_a_coupling_affinity",),
        "validation_targets": ("preserve records P0R03501-P0R03502",),
        "null_controls": ("as_a_coupling_affinity must remain source-bounded accounting",),
    },
    "as_a_coupling_affinity.the_scaling_law_of_consciousness_slc": {
        "context_id": "the_scaling_law_of_consciousness_slc",
        "validation_protocol": "paper0.as_a_coupling_affinity.the_scaling_law_of_consciousness_slc",
        "canonical_statement": "The source-bounded component 'The Scaling Law of Consciousness (SLC)' preserves Paper 0 records P0R03503-P0R03504 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03503:the_scaling_law_of_consciousness_slc",
            "P0R03504:the_scaling_law_of_consciousness_slc",
        ),
        "source_formulae": (
            "P0R03503: The Scaling Law of Consciousness (SLC)",
            "P0R03504: Core Assumption 1 posits the scalability of consciousness (Psi). The SCPN operationalises this via the Scaling Law of Consciousness (SLC), relating the intensity and quality of consciousness to the underlying architecture using Integrated Information Theory (IIT) and geometry.",
        ),
        "test_protocols": (
            "preserve The Scaling Law of Consciousness (SLC) source-accounting boundary",
        ),
        "null_results": (
            "The Scaling Law of Consciousness (SLC) is not empirical validation evidence",
        ),
        "variables": ("the_scaling_law_of_consciousness_slc",),
        "validation_targets": ("preserve records P0R03503-P0R03504",),
        "null_controls": (
            "the_scaling_law_of_consciousness_slc must remain source-bounded accounting",
        ),
    },
    "as_a_coupling_affinity.formalisation_via_integrated_information": {
        "context_id": "formalisation_via_integrated_information",
        "validation_protocol": "paper0.as_a_coupling_affinity.formalisation_via_integrated_information",
        "canonical_statement": "The source-bounded component 'Formalisation via Integrated Information ():' preserves Paper 0 records P0R03505-P0R03509 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03505:formalisation_via_integrated_information",
            "P0R03506:formalisation_via_integrated_information",
            "P0R03507:formalisation_via_integrated_information",
            "P0R03508:formalisation_via_integrated_information",
            "P0R03509:formalisation_via_integrated_information",
        ),
        "source_formulae": (
            "P0R03505: Formalisation via Integrated Information ():",
            "P0R03506: The degree of consciousness of any subsystem (S) is quantified by Integrated Information (), measuring information generated by the whole above and beyond its parts.",
            "P0R03507: $\\Phi(S) = min_{P}\\left\\lbrack I\\left( S_{p}ast;S_{p}resent\\mid P \\right) \\right\\rbrack$",
            "P0R03508: The SLC posits that scales non-linearly with complexity (N) and connectivity (K), maximised in the quasicritical regime:",
            "P0R03509: $\\Phi \\sim N\\alpha K\\beta$",
        ),
        "test_protocols": (
            "preserve Formalisation via Integrated Information (): source-accounting boundary",
        ),
        "null_results": (
            "Formalisation via Integrated Information (): is not empirical validation evidence",
        ),
        "variables": ("formalisation_via_integrated_information",),
        "validation_targets": ("preserve records P0R03505-P0R03509",),
        "null_controls": (
            "formalisation_via_integrated_information must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AsACouplingAffinitySpec:
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
class AsACouplingAffinitySpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[AsACouplingAffinitySpec, ...]
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


def build_as_a_coupling_affinity_specs(
    source_records: list[dict[str, Any]],
) -> AsACouplingAffinitySpecBundle:
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

    specs: list[AsACouplingAffinitySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AsACouplingAffinitySpec(
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
        "title": "Paper 0 " + " as a Coupling Affinity:" + " Specs",
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
        "next_source_boundary": "P0R03510",
    }
    return AsACouplingAffinitySpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> AsACouplingAffinitySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_as_a_coupling_affinity_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AsACouplingAffinitySpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + " as a Coupling Affinity:" + " Specs",
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
    bundle: AsACouplingAffinitySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_as_a_coupling_affinity_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_as_a_coupling_affinity_validation_specs_{date_tag}.md"
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
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
