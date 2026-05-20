#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The QEC Imperative and the Role of the Psi-Field spec builder
"""Promote Paper 0 The QEC Imperative and the Role of the Psi-Field records."""

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
    "P0R03051",
    "P0R03052",
    "P0R03053",
    "P0R03054",
    "P0R03055",
    "P0R03056",
    "P0R03057",
    "P0R03058",
)
CLAIM_BOUNDARY = "source-bounded the qec imperative and the role of the psi field source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_qec_imperative_and_the_role_of_the_psi_field.the_qec_imperative_and_the_role_of_the_psi_field": {
        "context_id": "the_qec_imperative_and_the_role_of_the_psi_field",
        "validation_protocol": "paper0.the_qec_imperative_and_the_role_of_the_psi_field.the_qec_imperative_and_the_role_of_the_psi_field",
        "canonical_statement": "The source-bounded component 'The QEC Imperative and the Role of the Psi-Field' preserves Paper 0 records P0R03051-P0R03057 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03051:the_qec_imperative_and_the_role_of_the_psi_field",
            "P0R03052:the_qec_imperative_and_the_role_of_the_psi_field",
            "P0R03053:the_qec_imperative_and_the_role_of_the_psi_field",
            "P0R03054:the_qec_imperative_and_the_role_of_the_psi_field",
            "P0R03055:the_qec_imperative_and_the_role_of_the_psi_field",
            "P0R03056:the_qec_imperative_and_the_role_of_the_psi_field",
            "P0R03057:the_qec_imperative_and_the_role_of_the_psi_field",
        ),
        "source_formulae": (
            "P0R03051: The QEC Imperative and the Role of the Psi-Field",
            'P0R03052: This section articulates the fundamental imperative for the Multi-Scale Quantum Error Correction (MS-QEC) architecture and posits a primary functional role for the Psi-field in its operation. The framework confronts the central challenge to quantum biology: the persistence of functional quantum coherence in "warm, wet, and noisy" biological environments. It asserts that MS-QEC is not an ancillary feature but a prerequisite for the physical viability of the entire SCPN.',
            "P0R03053: The core proposal is that the Psi-field acts as the master stabiliser for the nested QEC codes. It is the active organising principle that allows life to create and sustain pockets of quantum coherence against the overwhelming statistical pressure of environmental decoherence. This is framed not as a violation of the Second Law of Thermodynamics, but as a sophisticated form of entropy management. By coupling to the biological substrate, the Psi-field provides a continuous input of negentropy-analogous to work-that actively biases the system's dynamics towards coherence-preserving pathways. In this view, every observed instance of functional quantum biology, from photosynthesis to enzymatic action, serves as direct physical evidence of the Psi-field's ongoing role in stabilising the quantum foundations of life.",
            "P0R03054: This section tackles a huge puzzle: how can the delicate, fragile world of quantum mechanics possibly play a role in the warm, messy, and chaotic environment of a living cell? It's like trying to build a perfectly stable sandcastle in the middle of a hurricane. The answer is that life has a secret weapon.",
            "P0R03055: That secret weapon is the Psi-field itself. We propose that the fundamental job of the consciousness field is to act as the ultimate guardian of quantum coherence. It's like a focused, intelligent force field that creates tiny, calm pockets in the middle of the biological storm, allowing delicate quantum processes to happen without being instantly destroyed.",
            'P0R03056: This isn\'t magic or a violation of the laws of physics. It\'s a form of "entropy management." Think of it like a tiny, super-efficient refrigerator. A fridge uses energy to pump heat out and create a pocket of cold, ordered space inside. In the same way, the Psi-field "pumps" chaos out of biological systems, creating the pockets of quantum order that are essential for life to function. Every time a plant efficiently captures sunlight, that\'s evidence of the Psi-field at work, protecting the quantum magic of life.',
            "P0R03057: P0R03057",
        ),
        "test_protocols": (
            "preserve The QEC Imperative and the Role of the Psi-Field source-accounting boundary",
        ),
        "null_results": (
            "The QEC Imperative and the Role of the Psi-Field is not empirical validation evidence",
        ),
        "variables": ("the_qec_imperative_and_the_role_of_the_psi_field",),
        "validation_targets": ("preserve records P0R03051-P0R03057",),
        "null_controls": (
            "the_qec_imperative_and_the_role_of_the_psi_field must remain source-bounded accounting",
        ),
    },
    "the_qec_imperative_and_the_role_of_the_psi_field.meta_framework_integrations": {
        "context_id": "meta_framework_integrations",
        "validation_protocol": "paper0.the_qec_imperative_and_the_role_of_the_psi_field.meta_framework_integrations",
        "canonical_statement": "The source-bounded component 'Meta-Framework Integrations' preserves Paper 0 records P0R03058-P0R03058 without empirical validation claims.",
        "source_equation_ids": ("P0R03058:meta_framework_integrations",),
        "source_formulae": ("P0R03058: Meta-Framework Integrations",),
        "test_protocols": ("preserve Meta-Framework Integrations source-accounting boundary",),
        "null_results": ("Meta-Framework Integrations is not empirical validation evidence",),
        "variables": ("meta_framework_integrations",),
        "validation_targets": ("preserve records P0R03058-P0R03058",),
        "null_controls": ("meta_framework_integrations must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class TheQecImperativeAndTheRoleOfThePsiFieldSpec:
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
class TheQecImperativeAndTheRoleOfThePsiFieldSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheQecImperativeAndTheRoleOfThePsiFieldSpec, ...]
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


def build_the_qec_imperative_and_the_role_of_the_psi_field_specs(
    source_records: list[dict[str, Any]],
) -> TheQecImperativeAndTheRoleOfThePsiFieldSpecBundle:
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

    specs: list[TheQecImperativeAndTheRoleOfThePsiFieldSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheQecImperativeAndTheRoleOfThePsiFieldSpec(
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
        "title": "Paper 0 " + "The QEC Imperative and the Role of the Psi-Field" + " Specs",
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
        "next_source_boundary": "P0R03059",
    }
    return TheQecImperativeAndTheRoleOfThePsiFieldSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheQecImperativeAndTheRoleOfThePsiFieldSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_qec_imperative_and_the_role_of_the_psi_field_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheQecImperativeAndTheRoleOfThePsiFieldSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The QEC Imperative and the Role of the Psi-Field" + " Specs",
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
    bundle: TheQecImperativeAndTheRoleOfThePsiFieldSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_qec_imperative_and_the_role_of_the_psi_field_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_qec_imperative_and_the_role_of_the_psi_field_validation_specs_{date_tag}.md"
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
