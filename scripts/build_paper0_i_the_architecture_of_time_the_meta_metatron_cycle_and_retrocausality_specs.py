#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 I. The Architecture of Time: The Meta Metatron Cycle and Retrocausality spec builder
"""Promote Paper 0 I. The Architecture of Time: The Meta Metatron Cycle and Retrocausality records."""

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
    "P0R05928",
    "P0R05929",
    "P0R05930",
    "P0R05931",
    "P0R05932",
    "P0R05933",
    "P0R05934",
    "P0R05935",
)
CLAIM_BOUNDARY = "source-bounded i the architecture of time the meta metatron cycle and retrocausality source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality.i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality": {
        "context_id": "i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality",
        "validation_protocol": "paper0.i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality.i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality",
        "canonical_statement": "The source-bounded component 'I. The Architecture of Time: The Meta Metatron Cycle and Retrocausality' preserves Paper 0 records P0R05928-P0R05929 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05928:i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality",
            "P0R05929:i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality",
        ),
        "source_formulae": (
            "P0R05928: I. The Architecture of Time: The Meta Metatron Cycle and Retrocausality",
            'P0R05929: The SCPN operates within a temporal framework defined by the "Meta Metatron Cycle" (MMC, P3), which posits underlying cyclic, reversible dynamics from which the linear arrow of time emerges.',
        ),
        "test_protocols": (
            "preserve I. The Architecture of Time: The Meta Metatron Cycle and Retrocausality source-accounting boundary",
        ),
        "null_results": (
            "I. The Architecture of Time: The Meta Metatron Cycle and Retrocausality is not empirical validation evidence",
        ),
        "variables": ("i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality",),
        "validation_targets": ("preserve records P0R05928-P0R05929",),
        "null_controls": (
            "i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality must remain source-bounded accounting",
        ),
    },
    "i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality.1_the_cyclic_operator_and_reversibility": {
        "context_id": "1_the_cyclic_operator_and_reversibility",
        "validation_protocol": "paper0.i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality.1_the_cyclic_operator_and_reversibility",
        "canonical_statement": "The source-bounded component '1. The Cyclic Operator and Reversibility:' preserves Paper 0 records P0R05930-P0R05933 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05930:1_the_cyclic_operator_and_reversibility",
            "P0R05931:1_the_cyclic_operator_and_reversibility",
            "P0R05932:1_the_cyclic_operator_and_reversibility",
            "P0R05933:1_the_cyclic_operator_and_reversibility",
        ),
        "source_formulae": (
            "P0R05930: 1. The Cyclic Operator and Reversibility:",
            "P0R05931: The Source-Field (L13) dynamics are governed by a cyclic operator (O^MMC), implying periodic evolution.",
            "P0R05932: $O\\hat{}MMC \\mid \\Psi Source(t)\\rangle = \\mid \\Psi Source(t + TMMC)\\rangle$",
            "P0R05933: The fundamental evolution operator U(t) is unitary and periodic.",
        ),
        "test_protocols": (
            "preserve 1. The Cyclic Operator and Reversibility: source-accounting boundary",
        ),
        "null_results": (
            "1. The Cyclic Operator and Reversibility: is not empirical validation evidence",
        ),
        "variables": ("1_the_cyclic_operator_and_reversibility",),
        "validation_targets": ("preserve records P0R05930-P0R05933",),
        "null_controls": (
            "1_the_cyclic_operator_and_reversibility must remain source-bounded accounting",
        ),
    },
    "i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality.2_emergence_of_the_arrow_of_time": {
        "context_id": "2_emergence_of_the_arrow_of_time",
        "validation_protocol": "paper0.i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality.2_emergence_of_the_arrow_of_time",
        "canonical_statement": "The source-bounded component '2. Emergence of the Arrow of Time:' preserves Paper 0 records P0R05934-P0R05935 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05934:2_emergence_of_the_arrow_of_time",
            "P0R05935:2_emergence_of_the_arrow_of_time",
        ),
        "source_formulae": (
            "P0R05934: 2. Emergence of the Arrow of Time:",
            "P0R05935: The linear arrow of time emerges at lower layers (L1-L5) due to decoherence and coarse-graining. Interactions with complex biological substrates introduce effective non-unitary evolution (Lindblad dynamics) and the increase of von Neumann entropy (S=Tr(rhologrho)), defining the emergent thermodynamic arrow.",
        ),
        "test_protocols": (
            "preserve 2. Emergence of the Arrow of Time: source-accounting boundary",
        ),
        "null_results": (
            "2. Emergence of the Arrow of Time: is not empirical validation evidence",
        ),
        "variables": ("2_emergence_of_the_arrow_of_time",),
        "validation_targets": ("preserve records P0R05934-P0R05935",),
        "null_controls": (
            "2_emergence_of_the_arrow_of_time must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalitySpec:
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
class ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalitySpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalitySpec, ...]
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


def build_i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_specs(
    source_records: list[dict[str, Any]],
) -> ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalitySpecBundle:
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

    specs: list[ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalitySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalitySpec(
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
        "title": "Paper 0 "
        + "I. The Architecture of Time: The Meta Metatron Cycle and Retrocausality"
        + " Specs",
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
        "next_source_boundary": "P0R05936",
    }
    return ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalitySpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalitySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_specs(
        load_jsonl(ledger_path)
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(
    bundle: ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalitySpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "I. The Architecture of Time: The Meta Metatron Cycle and Retrocausality"
        + " Specs",
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
    bundle: ITheArchitectureOfTimeTheMetaMetatronCycleAndRetrocausalitySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_i_the_architecture_of_time_the_meta_metatron_cycle_and_retrocausality_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
