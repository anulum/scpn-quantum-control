#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 3. The CSF and Glymphatic System: The Entropy Sink (L1-L4) spec builder
"""Promote Paper 0 3. The CSF and Glymphatic System: The Entropy Sink (L1-L4) records."""

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
    "P0R04871",
    "P0R04872",
    "P0R04873",
    "P0R04874",
    "P0R04875",
    "P0R04876",
    "P0R04877",
    "P0R04878",
)
CLAIM_BOUNDARY = "source-bounded section 3 the csf and glymphatic system the entropy sink l1 l4 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4.3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4": {
        "context_id": "3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4",
        "validation_protocol": "paper0.section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4.3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4",
        "canonical_statement": "The source-bounded component '3. The CSF and Glymphatic System: The Entropy Sink (L1-L4)' preserves Paper 0 records P0R04871-P0R04876 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04871:3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4",
            "P0R04872:3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4",
            "P0R04873:3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4",
            "P0R04874:3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4",
            "P0R04875:3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4",
            "P0R04876:3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4",
        ),
        "source_formulae": (
            "P0R04871: 3. The CSF and Glymphatic System: The Entropy Sink (L1-L4)",
            "P0R04872: The Cerebrospinal Fluid (CSF) provides buoyancy, chemical buffering, and waste clearance.",
            "P0R04873: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R04874: Fig.: CSF supports buoyancy, buffering, and volume transmission (L2); the glymphatic system clears waste (sleep-enhanced). Acts as the Entropy Sink of the CHE, sustaining L1 QEC and L4 criticality; failure raises noise \\eta and degrades L1.",
            "P0R04875: Volume Transmission (L2): CSF facilitates the global diffusion of signalling molecules, modulating the overall brain state (tuning UPDE parameters). | The Glymphatic System (Entropy Management): This system clears metabolic waste (e.g., Amyloid-beta) via perivascular flow, primarily during sleep. SCPN Mapping: The Glymphatic system acts as the primary Entropy Sink for the Consciousness Heat Engine (CHE), essential for maintaining the low-entropy state required for L1 QEC and L4 criticality. | Pathology (Glymphatic Failure): Impaired function leads to the accumulation of toxins, increasing noise () and degrading the L1 substrate, contributing to neurodegeneration (e.g., Alzheimer's Disease).",
            "P0R04876: P0R04876",
        ),
        "test_protocols": (
            "preserve 3. The CSF and Glymphatic System: The Entropy Sink (L1-L4) source-accounting boundary",
        ),
        "null_results": (
            "3. The CSF and Glymphatic System: The Entropy Sink (L1-L4) is not empirical validation evidence",
        ),
        "variables": ("3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4",),
        "validation_targets": ("preserve records P0R04871-P0R04876",),
        "null_controls": (
            "3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4 must remain source-bounded accounting",
        ),
    },
    "section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4.ii_neuro_vascular_coupling_and_hemodynamics_the_energetics_of_consciousn": {
        "context_id": "ii_neuro_vascular_coupling_and_hemodynamics_the_energetics_of_consciousn",
        "validation_protocol": "paper0.section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4.ii_neuro_vascular_coupling_and_hemodynamics_the_energetics_of_consciousn",
        "canonical_statement": "The source-bounded component 'II. Neuro-Vascular Coupling and Hemodynamics: The Energetics of Consciousness' preserves Paper 0 records P0R04877-P0R04878 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04877:ii_neuro_vascular_coupling_and_hemodynamics_the_energetics_of_consciousn",
            "P0R04878:ii_neuro_vascular_coupling_and_hemodynamics_the_energetics_of_consciousn",
        ),
        "source_formulae": (
            "P0R04877: II. Neuro-Vascular Coupling and Hemodynamics: The Energetics of Consciousness",
            "P0R04878: Consciousness is energetically expensive (RMetabolic). The brain requires precise regulation of cerebral blood flow (CBF) via Neurovascular Coupling (NVC).",
        ),
        "test_protocols": (
            "preserve II. Neuro-Vascular Coupling and Hemodynamics: The Energetics of Consciousness source-accounting boundary",
        ),
        "null_results": (
            "II. Neuro-Vascular Coupling and Hemodynamics: The Energetics of Consciousness is not empirical validation evidence",
        ),
        "variables": ("ii_neuro_vascular_coupling_and_hemodynamics_the_energetics_of_consciousn",),
        "validation_targets": ("preserve records P0R04877-P0R04878",),
        "null_controls": (
            "ii_neuro_vascular_coupling_and_hemodynamics_the_energetics_of_consciousn must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4Spec:
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
class Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4Spec, ...]
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


def build_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_specs(
    source_records: list[dict[str, Any]],
) -> Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4SpecBundle:
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

    specs: list[Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4Spec(
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
        + "3. The CSF and Glymphatic System: The Entropy Sink (L1-L4)"
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
        "next_source_boundary": "P0R04879",
    }
    return Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_specs(
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


def render_report(bundle: Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "3. The CSF and Glymphatic System: The Entropy Sink (L1-L4)" + " Specs",
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
    bundle: Section3TheCsfAndGlymphaticSystemTheEntropySinkL1L4SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_3_the_csf_and_glymphatic_system_the_entropy_sink_l1_l4_validation_specs_{date_tag}.md"
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
