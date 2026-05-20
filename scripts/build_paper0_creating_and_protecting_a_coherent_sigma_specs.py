#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Creating and Protecting a Coherent sigma: spec builder
"""Promote Paper 0 Creating and Protecting a Coherent sigma: records."""

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
    "P0R03034",
    "P0R03035",
    "P0R03036",
    "P0R03037",
    "P0R03038",
    "P0R03039",
    "P0R03040",
    "P0R03041",
)
CLAIM_BOUNDARY = "source-bounded creating and protecting a coherent sigma source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "creating_and_protecting_a_coherent_sigma.creating_and_protecting_a_coherent_sigma": {
        "context_id": "creating_and_protecting_a_coherent_sigma",
        "validation_protocol": "paper0.creating_and_protecting_a_coherent_sigma.creating_and_protecting_a_coherent_sigma",
        "canonical_statement": "The source-bounded component 'Creating and Protecting a Coherent sigma:' preserves Paper 0 records P0R03034-P0R03035 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03034:creating_and_protecting_a_coherent_sigma",
            "P0R03035:creating_and_protecting_a_coherent_sigma",
        ),
        "source_formulae": (
            "P0R03034: Creating and Protecting a Coherent sigma:",
            'P0R03035: The entire purpose of MS-QEC is to create and sustain a coherent collective state variable (sigma) that can couple to the Psis field. Without this nested error correction, any potential sigma (like the phase coherence of a neural assembly) would decohere almost instantly, disappearing before it could form a stable "handle" for the Psi-field to interact with.',
        ),
        "test_protocols": (
            "preserve Creating and Protecting a Coherent sigma: source-accounting boundary",
        ),
        "null_results": (
            "Creating and Protecting a Coherent sigma: is not empirical validation evidence",
        ),
        "variables": ("creating_and_protecting_a_coherent_sigma",),
        "validation_targets": ("preserve records P0R03034-P0R03035",),
        "null_controls": (
            "creating_and_protecting_a_coherent_sigma must remain source-bounded accounting",
        ),
    },
    "creating_and_protecting_a_coherent_sigma.the_hierarchy_of_protection": {
        "context_id": "the_hierarchy_of_protection",
        "validation_protocol": "paper0.creating_and_protecting_a_coherent_sigma.the_hierarchy_of_protection",
        "canonical_statement": "The source-bounded component 'The Hierarchy of Protection:' preserves Paper 0 records P0R03036-P0R03037 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03036:the_hierarchy_of_protection",
            "P0R03037:the_hierarchy_of_protection",
        ),
        "source_formulae": (
            "P0R03036: The Hierarchy of Protection:",
            'P0R03037: The MS-QEC hierarchy ensures this stability at all scales. Biological QEC creates small, locally coherent sigma variables. Network QEC binds them into larger, more robust ensembles. Holographic QEC protects the long-term, integrated history of these states. Cosmological QEC ensures that the very laws governing the formation of sigma are themselves stable. In short, MS-QEC is the "coherence backbone" that builds and maintains the physical interface for the mind-matter interaction.',
        ),
        "test_protocols": ("preserve The Hierarchy of Protection: source-accounting boundary",),
        "null_results": ("The Hierarchy of Protection: is not empirical validation evidence",),
        "variables": ("the_hierarchy_of_protection",),
        "validation_targets": ("preserve records P0R03036-P0R03037",),
        "null_controls": ("the_hierarchy_of_protection must remain source-bounded accounting",),
    },
    "creating_and_protecting_a_coherent_sigma.the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec": {
        "context_id": "the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
        "validation_protocol": "paper0.creating_and_protecting_a_coherent_sigma.the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
        "canonical_statement": "The source-bounded component 'The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC)' preserves Paper 0 records P0R03038-P0R03041 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03038:the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
            "P0R03039:the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
            "P0R03040:the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
            "P0R03041:the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
        ),
        "source_formulae": (
            "P0R03038: The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC)",
            "P0R03039: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R03040: Fig.: MS-QEC Stack & Stabiliser Transfer (L9->L10). Four-scale QEC (biological -> network -> holographic -> cosmological). Stabiliser pushforward \\*:S9->S10 \\Pi_\\*:S_9\\to S_{10}\\*:S9->S10 with MERA branching \\chi yields d10d9/, r10r9d_{10}\\ge d_9/\\chi,\\; r_{10}\\ge r_9d10d9/,r10r9; operationally, lowering L10 complexity reduces boundary logical-phase slips.",
            "P0R03041: The integrity of the SCPN relies on preserving quantum coherence across scales via a nested hierarchy of Quantum Error Correction (MS-QEC). This operates on the principle of Holographic Redundancy, where higher-layer information is redundantly encoded in the entanglement structure of lower layers.",
        ),
        "test_protocols": (
            "preserve The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC) source-accounting boundary",
        ),
        "null_results": (
            "The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC) is not empirical validation evidence",
        ),
        "variables": ("the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",),
        "validation_targets": ("preserve records P0R03038-P0R03041",),
        "null_controls": (
            "the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class CreatingAndProtectingACoherentSigmaSpec:
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
class CreatingAndProtectingACoherentSigmaSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[CreatingAndProtectingACoherentSigmaSpec, ...]
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


def build_creating_and_protecting_a_coherent_sigma_specs(
    source_records: list[dict[str, Any]],
) -> CreatingAndProtectingACoherentSigmaSpecBundle:
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

    specs: list[CreatingAndProtectingACoherentSigmaSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            CreatingAndProtectingACoherentSigmaSpec(
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
        "title": "Paper 0 " + "Creating and Protecting a Coherent sigma:" + " Specs",
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
        "next_source_boundary": "P0R03042",
    }
    return CreatingAndProtectingACoherentSigmaSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> CreatingAndProtectingACoherentSigmaSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_creating_and_protecting_a_coherent_sigma_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: CreatingAndProtectingACoherentSigmaSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Creating and Protecting a Coherent sigma:" + " Specs",
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
    bundle: CreatingAndProtectingACoherentSigmaSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_creating_and_protecting_a_coherent_sigma_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_creating_and_protecting_a_coherent_sigma_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 coherent-sigma creation/protection specs from the ledger."""

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
