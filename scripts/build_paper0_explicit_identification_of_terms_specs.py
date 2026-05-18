#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Explicit Identification of Terms: spec builder
"""Promote Paper 0 Explicit Identification of Terms: records."""

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
    "P0R04283",
    "P0R04284",
    "P0R04285",
    "P0R04286",
    "P0R04287",
    "P0R04288",
    "P0R04289",
    "P0R04290",
)
CLAIM_BOUNDARY = "source-bounded explicit identification of terms source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "explicit_identification_of_terms.explicit_identification_of_terms": {
        "context_id": "explicit_identification_of_terms",
        "validation_protocol": "paper0.explicit_identification_of_terms.explicit_identification_of_terms",
        "canonical_statement": "The source-bounded component 'Explicit Identification of Terms:' preserves Paper 0 records P0R04283-P0R04286 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04283:explicit_identification_of_terms",
            "P0R04284:explicit_identification_of_terms",
            "P0R04285:explicit_identification_of_terms",
            "P0R04286:explicit_identification_of_terms",
        ),
        "source_formulae": (
            "P0R04283: Explicit Identification of Terms:",
            "P0R04284: Psis: The universal field is represented by its phase component, the pseudoscalar field a (the ALP).",
            "P0R04285: sigma: The collective state variable sigma is the local electromagnetic field configuration, specifically the EB product, which is what the ALP couples to.",
            "P0R04286: lambda: The coupling constant lambda is identified with the fundamental axion-photon coupling constant, ga.",
        ),
        "test_protocols": (
            "preserve Explicit Identification of Terms: source-accounting boundary",
        ),
        "null_results": (
            "Explicit Identification of Terms: is not empirical validation evidence",
        ),
        "variables": ("explicit_identification_of_terms",),
        "validation_targets": ("preserve records P0R04283-P0R04286",),
        "null_controls": (
            "explicit_identification_of_terms must remain source-bounded accounting",
        ),
    },
    "explicit_identification_of_terms.the_nature_of_the_interaction": {
        "context_id": "the_nature_of_the_interaction",
        "validation_protocol": "paper0.explicit_identification_of_terms.the_nature_of_the_interaction",
        "canonical_statement": "The source-bounded component 'The Nature of the Interaction:' preserves Paper 0 records P0R04287-P0R04288 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04287:the_nature_of_the_interaction",
            "P0R04288:the_nature_of_the_interaction",
        ),
        "source_formulae": (
            "P0R04287: The Nature of the Interaction:",
            'P0R04288: H_int in this context is the Primakoff effect. The coupling is not a generic "informational force" but a specific, well-defined particle physics interaction. The proposed biological amplifiers (magnetite, CISS, etc.) are mechanisms that locally increase the effective value of the coupling constant lambda (ga), making the mind-matter interface far more efficient than it would be in a vacuum. This provides a concrete, testable, and physically plausible pathway for the Psi-field to interact with the brain.',
        ),
        "test_protocols": ("preserve The Nature of the Interaction: source-accounting boundary",),
        "null_results": ("The Nature of the Interaction: is not empirical validation evidence",),
        "variables": ("the_nature_of_the_interaction",),
        "validation_targets": ("preserve records P0R04287-P0R04288",),
        "null_controls": ("the_nature_of_the_interaction must remain source-bounded accounting",),
    },
    "explicit_identification_of_terms.the_psi_field_electromagnetic_interface_the_role_of_axion_like_particles": {
        "context_id": "the_psi_field_electromagnetic_interface_the_role_of_axion_like_particles",
        "validation_protocol": "paper0.explicit_identification_of_terms.the_psi_field_electromagnetic_interface_the_role_of_axion_like_particles",
        "canonical_statement": "The source-bounded component 'The Psi-Field-Electromagnetic Interface: The Role of Axion-Like Particles (ALPs)' preserves Paper 0 records P0R04289-P0R04290 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04289:the_psi_field_electromagnetic_interface_the_role_of_axion_like_particles",
            "P0R04290:the_psi_field_electromagnetic_interface_the_role_of_axion_like_particles",
        ),
        "source_formulae": (
            "P0R04289: The Psi-Field-Electromagnetic Interface: The Role of Axion-Like Particles (ALPs)",
            "P0R04290: While the primary interactions of the Psi-field are Geometric and Informational, its influence on the brain's electromagnetic (EM) activity requires a specific interface with the Standard Model EM field (F_munu). We propose that this coupling is mediated by Axion-Like Particles (ALPs). This chapter details a critical and concrete mechanism for the mind-matter interface within SCPN framework.",
        ),
        "test_protocols": (
            "preserve The Psi-Field-Electromagnetic Interface: The Role of Axion-Like Particles (ALPs) source-accounting boundary",
        ),
        "null_results": (
            "The Psi-Field-Electromagnetic Interface: The Role of Axion-Like Particles (ALPs) is not empirical validation evidence",
        ),
        "variables": ("the_psi_field_electromagnetic_interface_the_role_of_axion_like_particles",),
        "validation_targets": ("preserve records P0R04289-P0R04290",),
        "null_controls": (
            "the_psi_field_electromagnetic_interface_the_role_of_axion_like_particles must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ExplicitIdentificationOfTermsSpec:
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
class ExplicitIdentificationOfTermsSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ExplicitIdentificationOfTermsSpec, ...]
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


def build_explicit_identification_of_terms_specs(
    source_records: list[dict[str, Any]],
) -> ExplicitIdentificationOfTermsSpecBundle:
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

    specs: list[ExplicitIdentificationOfTermsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ExplicitIdentificationOfTermsSpec(
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
        "title": "Paper 0 " + "Explicit Identification of Terms:" + " Specs",
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
        "next_source_boundary": "P0R04291",
    }
    return ExplicitIdentificationOfTermsSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ExplicitIdentificationOfTermsSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_explicit_identification_of_terms_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ExplicitIdentificationOfTermsSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Explicit Identification of Terms:" + " Specs",
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
    bundle: ExplicitIdentificationOfTermsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_explicit_identification_of_terms_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_explicit_identification_of_terms_validation_specs_{date_tag}.md"
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
