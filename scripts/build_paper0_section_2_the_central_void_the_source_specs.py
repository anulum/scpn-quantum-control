#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 2. The Central Void (The Source): spec builder
"""Promote Paper 0 2. The Central Void (The Source): records."""

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
    "P0R02566",
    "P0R02567",
    "P0R02568",
    "P0R02569",
    "P0R02570",
    "P0R02571",
    "P0R02572",
    "P0R02573",
    "P0R02574",
    "P0R02575",
    "P0R02576",
    "P0R02577",
    "P0R02578",
    "P0R02579",
)
CLAIM_BOUNDARY = "source-bounded section 2 the central void the source source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_2_the_central_void_the_source.2_the_central_void_the_source": {
        "context_id": "2_the_central_void_the_source",
        "validation_protocol": "paper0.section_2_the_central_void_the_source.2_the_central_void_the_source",
        "canonical_statement": "The source-bounded component '2. The Central Void (The Source):' preserves Paper 0 records P0R02566-P0R02567 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02566:2_the_central_void_the_source",
            "P0R02567:2_the_central_void_the_source",
        ),
        "source_formulae": (
            "P0R02566: 2. The Central Void (The Source):",
            "P0R02567: The centre of the Torus represents the Source-Field (L13)-the ontological ground from which the architecture emerges and returns.",
        ),
        "test_protocols": (
            "preserve 2. The Central Void (The Source): source-accounting boundary",
        ),
        "null_results": (
            "2. The Central Void (The Source): is not empirical validation evidence",
        ),
        "variables": ("2_the_central_void_the_source",),
        "validation_targets": ("preserve records P0R02566-P0R02567",),
        "null_controls": ("2_the_central_void_the_source must remain source-bounded accounting",),
    },
    "section_2_the_central_void_the_source.3_the_surface_dynamics_criticality_and_coherence": {
        "context_id": "3_the_surface_dynamics_criticality_and_coherence",
        "validation_protocol": "paper0.section_2_the_central_void_the_source.3_the_surface_dynamics_criticality_and_coherence",
        "canonical_statement": "The source-bounded component '3. The Surface Dynamics (Criticality and Coherence):' preserves Paper 0 records P0R02568-P0R02569 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02568:3_the_surface_dynamics_criticality_and_coherence",
            "P0R02569:3_the_surface_dynamics_criticality_and_coherence",
        ),
        "source_formulae": (
            "P0R02568: 3. The Surface Dynamics (Criticality and Coherence):",
            "P0R02569: The dynamic surface exhibits scale-free avalanches and turbulent flow, representing the Quasicritical regime (sigma1). Stable geometric patterns on the surface represent the action of Multi-Scale Quantum Error Correction (MS-QEC), maintaining the integrity of the flow.",
        ),
        "test_protocols": (
            "preserve 3. The Surface Dynamics (Criticality and Coherence): source-accounting boundary",
        ),
        "null_results": (
            "3. The Surface Dynamics (Criticality and Coherence): is not empirical validation evidence",
        ),
        "variables": ("3_the_surface_dynamics_criticality_and_coherence",),
        "validation_targets": ("preserve records P0R02568-P0R02569",),
        "null_controls": (
            "3_the_surface_dynamics_criticality_and_coherence must remain source-bounded accounting",
        ),
    },
    "section_2_the_central_void_the_source.4_the_attractor_l15_l8": {
        "context_id": "4_the_attractor_l15_l8",
        "validation_protocol": "paper0.section_2_the_central_void_the_source.4_the_attractor_l15_l8",
        "canonical_statement": "The source-bounded component '4. The Attractor (L15/L8):' preserves Paper 0 records P0R02570-P0R02579 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02570:4_the_attractor_l15_l8",
            "P0R02571:4_the_attractor_l15_l8",
            "P0R02572:4_the_attractor_l15_l8",
            "P0R02573:4_the_attractor_l15_l8",
            "P0R02574:4_the_attractor_l15_l8",
            "P0R02575:4_the_attractor_l15_l8",
            "P0R02576:4_the_attractor_l15_l8",
            "P0R02577:4_the_attractor_l15_l8",
            "P0R02578:4_the_attractor_l15_l8",
            "P0R02579:4_the_attractor_l15_l8",
        ),
        "source_formulae": (
            "P0R02570: 4. The Attractor (L15/L8):",
            "P0R02571: The overall trajectory of the Torus is guided by the Cosmic Attractor (L8) and optimised by the Ethical Functional (L15), representing the Teleological (RG) Flow.",
            "P0R02572: [IMAGE:Ein Bild, das Text, Screenshot, Kreis, Design enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R02573: P0R02573",
            "P0R02574: The MIP-Fiber Parallel Transport Smoothing",
            "P0R02575: P0R02575",
            "P0R02576: When the SCPN state-trajectory crosses a Minimum Information Partition (MIP) boundary, the internal fiber space of the $\\Psi$-field undergoes a discrete reconfiguration. To prevent discontinuities in the Berry Phase ($\\Gamma$), we define the Fiber Smoothing Operator ($\\mathcal{S}_{MIP}$):",
            "P0R02577: $$\\mathcal{S}_{MIP} = \\lim_{\\epsilon \\to 0} \\oint_{\\gamma_{MIP}} A_\\mu dx^\\mu \\otimes \\mathbf{T}_{MIP}$$",
            "P0R02578: where $\\mathbf{T}_{MIP}$ is a transition tensor that maps the holonomy of the pre-partitioned state to the newly emergent manifold. This ensures that memory loops are preserved even during drastic shifts in the system's organizational scale.",
            "P0R02579: P0R02579",
        ),
        "test_protocols": ("preserve 4. The Attractor (L15/L8): source-accounting boundary",),
        "null_results": ("4. The Attractor (L15/L8): is not empirical validation evidence",),
        "variables": ("4_the_attractor_l15_l8",),
        "validation_targets": ("preserve records P0R02570-P0R02579",),
        "null_controls": ("4_the_attractor_l15_l8 must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class Section2TheCentralVoidTheSourceSpec:
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
class Section2TheCentralVoidTheSourceSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section2TheCentralVoidTheSourceSpec, ...]
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


def build_section_2_the_central_void_the_source_specs(
    source_records: list[dict[str, Any]],
) -> Section2TheCentralVoidTheSourceSpecBundle:
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

    specs: list[Section2TheCentralVoidTheSourceSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section2TheCentralVoidTheSourceSpec(
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
        "title": "Paper 0 " + "2. The Central Void (The Source):" + " Specs",
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
        "next_source_boundary": "P0R02580",
    }
    return Section2TheCentralVoidTheSourceSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section2TheCentralVoidTheSourceSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_2_the_central_void_the_source_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section2TheCentralVoidTheSourceSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "2. The Central Void (The Source):" + " Specs",
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
    bundle: Section2TheCentralVoidTheSourceSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_2_the_central_void_the_source_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_section_2_the_central_void_the_source_validation_specs_{date_tag}.md"
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
