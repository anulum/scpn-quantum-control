#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Domain III Overview: Memory and Projection Control spec builder
"""Promote Paper 0 Domain III Overview: Memory and Projection Control records."""

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
    "P0R02237",
    "P0R02238",
    "P0R02239",
    "P0R02240",
    "P0R02241",
    "P0R02242",
    "P0R02243",
    "P0R02244",
    "P0R02245",
    "P0R02246",
    "P0R02247",
    "P0R02248",
)
CLAIM_BOUNDARY = "source-bounded domain iii overview memory and projection control source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "domain_iii_overview_memory_and_projection_control.domain_iii_overview_memory_and_projection_control": {
        "context_id": "domain_iii_overview_memory_and_projection_control",
        "validation_protocol": "paper0.domain_iii_overview_memory_and_projection_control.domain_iii_overview_memory_and_projection_control",
        "canonical_statement": "The source-bounded component 'Domain III Overview: Memory and Projection Control' preserves Paper 0 records P0R02237-P0R02241 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02237:domain_iii_overview_memory_and_projection_control",
            "P0R02238:domain_iii_overview_memory_and_projection_control",
            "P0R02239:domain_iii_overview_memory_and_projection_control",
            "P0R02240:domain_iii_overview_memory_and_projection_control",
            "P0R02241:domain_iii_overview_memory_and_projection_control",
        ),
        "source_formulae": (
            "P0R02237: Domain III Overview: Memory and Projection Control",
            "P0R02238: Domain III of the Sentient-Consciousness Projection Network (SCPN) encompasses the crucial functions of information storage and boundary regulation, represented by Layer 9 and Layer 10, respectively. This domain marks a critical transition from the embodied, real-time processing of the individual organism (Domain II) to the more abstract, informational structures that govern collective and trans-temporal consciousness.",
            'P0R02239: Layer 9, the Existential Holograph, is posited as the primary long-term memory engram for the Layer 5 Self. This layer functions as a high-dimensional, holographic repository of an organism\'s significant experiences, beliefs, and predictive models. The term "holograph" is used to imply a distributed, non-local storage mechanism where information is encoded in interference patterns, making the system resilient to partial damage. This layer is not merely a passive archive; it constitutes the deep structure of the individual\'s generative model, providing the foundational priors that shape all subsequent perception and action. As noted in the description of Layer 5, the nightly process of memory consolidation during NREM sleep is the specific neurobiological mechanism responsible for "writing" data from the temporary hippocampal buffer to this permanent holographic store.',
            "P0R02240: Within the Sentient-Consciousness Projection Network (SCPN), memory is managed by Layer 9 (Existential Holograph). It functions as a non-local, high-dimensional repository where significant experiences and predictive models are stored using holographic principles.",
            "P0R02241: The following equations govern the encoding, integrity, and retrieval of information within this holographic substrate.",
        ),
        "test_protocols": (
            "preserve Domain III Overview: Memory and Projection Control source-accounting boundary",
        ),
        "null_results": (
            "Domain III Overview: Memory and Projection Control is not empirical validation evidence",
        ),
        "variables": ("domain_iii_overview_memory_and_projection_control",),
        "validation_targets": ("preserve records P0R02237-P0R02241",),
        "null_controls": (
            "domain_iii_overview_memory_and_projection_control must remain source-bounded accounting",
        ),
    },
    "domain_iii_overview_memory_and_projection_control.p0r02242": {
        "context_id": "p0r02242",
        "validation_protocol": "paper0.domain_iii_overview_memory_and_projection_control.p0r02242",
        "canonical_statement": "The source-bounded component 'P0R02242' preserves Paper 0 records P0R02242-P0R02242 without empirical validation claims.",
        "source_equation_ids": ("P0R02242:p0r02242",),
        "source_formulae": ("P0R02242: P0R02242",),
        "test_protocols": ("preserve P0R02242 source-accounting boundary",),
        "null_results": ("P0R02242 is not empirical validation evidence",),
        "variables": ("p0r02242",),
        "validation_targets": ("preserve records P0R02242-P0R02242",),
        "null_controls": ("p0r02242 must remain source-bounded accounting",),
    },
    "domain_iii_overview_memory_and_projection_control.1_memory_retrieval_retrocausality_via_abl_rule": {
        "context_id": "1_memory_retrieval_retrocausality_via_abl_rule",
        "validation_protocol": "paper0.domain_iii_overview_memory_and_projection_control.1_memory_retrieval_retrocausality_via_abl_rule",
        "canonical_statement": "The source-bounded component '1. Memory Retrieval (Retrocausality via ABL Rule)' preserves Paper 0 records P0R02243-P0R02248 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02243:1_memory_retrieval_retrocausality_via_abl_rule",
            "P0R02244:1_memory_retrieval_retrocausality_via_abl_rule",
            "P0R02245:1_memory_retrieval_retrocausality_via_abl_rule",
            "P0R02246:1_memory_retrieval_retrocausality_via_abl_rule",
            "P0R02247:1_memory_retrieval_retrocausality_via_abl_rule",
            "P0R02248:1_memory_retrieval_retrocausality_via_abl_rule",
        ),
        "source_formulae": (
            "P0R02243: 1. Memory Retrieval (Retrocausality via ABL Rule)",
            "P0R02244: Retrieval from the Existential Holograph leverages the Two-State Vector Formalism (TSVF), where the probability of a present outcome (memory recall) is biased by a future boundary condition (the attractor state).",
            "P0R02245: Equation (Python Format):",
            "P0R02246: p_a_given_t = (abs(bra_phi_t @ projection_a @ ket_psi_t)**2) / sum(abs(bra_phi_t @ projection_j @ ket_psi_t)**2)",
            "P0R02247: Legend:",
            "P0R02248: p_a_given_t: Probability of outcome a at time t. | bra_phi_t: Backward-evolving state vector from a future boundary condition (e.g., Layer 15). | ket_psi_t: Standard forward-evolving quantum state vector from the past. | projection_a: Projection operator for the specific memory state a. | projection_j: Summation of projection operators for all possible intermediate outcomes.",
        ),
        "test_protocols": (
            "preserve 1. Memory Retrieval (Retrocausality via ABL Rule) source-accounting boundary",
        ),
        "null_results": (
            "1. Memory Retrieval (Retrocausality via ABL Rule) is not empirical validation evidence",
        ),
        "variables": ("1_memory_retrieval_retrocausality_via_abl_rule",),
        "validation_targets": ("preserve records P0R02243-P0R02248",),
        "null_controls": (
            "1_memory_retrieval_retrocausality_via_abl_rule must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class DomainIiiOverviewMemoryAndProjectionControlSpec:
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
class DomainIiiOverviewMemoryAndProjectionControlSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[DomainIiiOverviewMemoryAndProjectionControlSpec, ...]
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


def build_domain_iii_overview_memory_and_projection_control_specs(
    source_records: list[dict[str, Any]],
) -> DomainIiiOverviewMemoryAndProjectionControlSpecBundle:
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

    specs: list[DomainIiiOverviewMemoryAndProjectionControlSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            DomainIiiOverviewMemoryAndProjectionControlSpec(
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
        "title": "Paper 0 " + "Domain III Overview: Memory and Projection Control" + " Specs",
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
        "next_source_boundary": "P0R02249",
    }
    return DomainIiiOverviewMemoryAndProjectionControlSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> DomainIiiOverviewMemoryAndProjectionControlSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_domain_iii_overview_memory_and_projection_control_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: DomainIiiOverviewMemoryAndProjectionControlSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Domain III Overview: Memory and Projection Control" + " Specs",
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
    bundle: DomainIiiOverviewMemoryAndProjectionControlSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_domain_iii_overview_memory_and_projection_control_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_domain_iii_overview_memory_and_projection_control_validation_specs_{date_tag}.md"
    )
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
