#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Localised: spec builder
"""Promote Paper 0 Localised: records."""

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
    "P0R01820",
    "P0R01821",
    "P0R01822",
    "P0R01823",
    "P0R01824",
    "P0R01825",
    "P0R01826",
    "P0R01827",
    "P0R01828",
    "P0R01829",
    "P0R01830",
)
CLAIM_BOUNDARY = "source-bounded localised source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "localised.localised": {
        "context_id": "localised",
        "validation_protocol": "paper0.localised.localised",
        "canonical_statement": "The source-bounded component 'Localised:' preserves Paper 0 records P0R01820-P0R01821 without empirical validation claims.",
        "source_equation_ids": ("P0R01820:localised", "P0R01821:localised"),
        "source_formulae": (
            "P0R01820: Localised:",
            'P0R01821: It is a finite-energy "lump," providing a specific location for interaction.',
        ),
        "test_protocols": ("preserve Localised: source-accounting boundary",),
        "null_results": ("Localised: is not empirical validation evidence",),
        "variables": ("localised",),
        "validation_targets": ("preserve records P0R01820-P0R01821",),
        "null_controls": ("localised must remain source-bounded accounting",),
    },
    "localised.persistent": {
        "context_id": "persistent",
        "validation_protocol": "paper0.localised.persistent",
        "canonical_statement": "The source-bounded component 'Persistent:' preserves Paper 0 records P0R01822-P0R01823 without empirical validation claims.",
        "source_equation_ids": ("P0R01822:persistent", "P0R01823:persistent"),
        "source_formulae": (
            "P0R01822: Persistent:",
            "P0R01823: It lasts over the organism's lifetime.",
        ),
        "test_protocols": ("preserve Persistent: source-accounting boundary",),
        "null_results": ("Persistent: is not empirical validation evidence",),
        "variables": ("persistent",),
        "validation_targets": ("preserve records P0R01822-P0R01823",),
        "null_controls": ("persistent must remain source-bounded accounting",),
    },
    "localised.the_nature_of_the_interaction": {
        "context_id": "the_nature_of_the_interaction",
        "validation_protocol": "paper0.localised.the_nature_of_the_interaction",
        "canonical_statement": "The source-bounded component 'The Nature of the Interaction:' preserves Paper 0 records P0R01824-P0R01825 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01824:the_nature_of_the_interaction",
            "P0R01825:the_nature_of_the_interaction",
        ),
        "source_formulae": (
            "P0R01824: The Nature of the Interaction:",
            'P0R01825: H_int therefore describes the coupling between the universal, background Psi-field (Psis) and this specific, individualised, charge-stabilised knot of itself (sigma = O). The physics of the Q-ball is the physics that makes an individual a distinct, addressable entity with which the rest of the universe can coherently interact. Without this charge-supported stability, there would be no persistent "Self" to couple to.',
        ),
        "test_protocols": ("preserve The Nature of the Interaction: source-accounting boundary",),
        "null_results": ("The Nature of the Interaction: is not empirical validation evidence",),
        "variables": ("the_nature_of_the_interaction",),
        "validation_targets": ("preserve records P0R01824-P0R01825",),
        "null_controls": ("the_nature_of_the_interaction must remain source-bounded accounting",),
    },
    "localised.the_metaphysical_stance_hierarchical_field_monism_hfm": {
        "context_id": "the_metaphysical_stance_hierarchical_field_monism_hfm",
        "validation_protocol": "paper0.localised.the_metaphysical_stance_hierarchical_field_monism_hfm",
        "canonical_statement": "The source-bounded component 'The Metaphysical Stance: Hierarchical Field Monism (HFM)' preserves Paper 0 records P0R01826-P0R01830 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01826:the_metaphysical_stance_hierarchical_field_monism_hfm",
            "P0R01827:the_metaphysical_stance_hierarchical_field_monism_hfm",
            "P0R01828:the_metaphysical_stance_hierarchical_field_monism_hfm",
            "P0R01829:the_metaphysical_stance_hierarchical_field_monism_hfm",
            "P0R01830:the_metaphysical_stance_hierarchical_field_monism_hfm",
        ),
        "source_formulae": (
            "P0R01826: The Metaphysical Stance: Hierarchical Field Monism (HFM)",
            "P0R01827: The framework's specific metaphysical stance is Hierarchical Field Monism (HFM), which defines the relationship between consciousness, mind, and matter.",
            "P0R01828: Monism: There is only one fundamental substance-the Psi-field.",
            "P0R01829: Hierarchy: This single substance manifests in a nested hierarchy of forms and dynamics across the 15 layers of the SCPN.",
            'P0R01830: Emergent Modalities: "Mind" and "Matter" are not two different substances. They are emergent, complementary descriptions of the Psi-field\'s dynamics at a specific level of the hierarchy. This stance aims to dissolve the traditional mind-body problem.',
        ),
        "test_protocols": (
            "preserve The Metaphysical Stance: Hierarchical Field Monism (HFM) source-accounting boundary",
        ),
        "null_results": (
            "The Metaphysical Stance: Hierarchical Field Monism (HFM) is not empirical validation evidence",
        ),
        "variables": ("the_metaphysical_stance_hierarchical_field_monism_hfm",),
        "validation_targets": ("preserve records P0R01826-P0R01830",),
        "null_controls": (
            "the_metaphysical_stance_hierarchical_field_monism_hfm must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class LocalisedSpec:
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
class LocalisedSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[LocalisedSpec, ...]
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


def build_localised_specs(source_records: list[dict[str, Any]]) -> LocalisedSpecBundle:
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

    specs: list[LocalisedSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            LocalisedSpec(
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
        "title": "Paper 0 " + "Localised:" + " Specs",
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
        "next_source_boundary": "P0R01831",
    }
    return LocalisedSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> LocalisedSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_localised_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: LocalisedSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Localised:" + " Specs",
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
    bundle: LocalisedSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_localised_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_localised_validation_specs_{date_tag}.md"
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
