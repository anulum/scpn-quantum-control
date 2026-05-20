#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 1. The Bioelectric Code in Neurogenesis and Regeneration: spec builder
"""Promote Paper 0 1. The Bioelectric Code in Neurogenesis and Regeneration: records."""

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
    "P0R04657",
    "P0R04658",
    "P0R04659",
    "P0R04660",
    "P0R04661",
    "P0R04662",
    "P0R04663",
    "P0R04664",
    "P0R04665",
)
CLAIM_BOUNDARY = "source-bounded section 1 the bioelectric code in neurogenesis and regeneration source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_1_the_bioelectric_code_in_neurogenesis_and_regeneration.1_the_bioelectric_code_in_neurogenesis_and_regeneration": {
        "context_id": "1_the_bioelectric_code_in_neurogenesis_and_regeneration",
        "validation_protocol": "paper0.section_1_the_bioelectric_code_in_neurogenesis_and_regeneration.1_the_bioelectric_code_in_neurogenesis_and_regeneration",
        "canonical_statement": "The source-bounded component '1. The Bioelectric Code in Neurogenesis and Regeneration:' preserves Paper 0 records P0R04657-P0R04659 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04657:1_the_bioelectric_code_in_neurogenesis_and_regeneration",
            "P0R04658:1_the_bioelectric_code_in_neurogenesis_and_regeneration",
            "P0R04659:1_the_bioelectric_code_in_neurogenesis_and_regeneration",
        ),
        "source_formulae": (
            "P0R04657: 1. The Bioelectric Code in Neurogenesis and Regeneration:",
            "P0R04658: The endogenous Bioelectric Field (Vmem) provides the template for brain morphogenesis.",
            "P0R04659: Pattern Formation: Specific patterns of Vmem across neural progenitor cells guide differentiation, migration, and connectivity. This code acts as a morphogenetic attractor, guided by the Psi-field. | Plasticity and Regeneration: Manipulating the Bioelectric Code can induce large-scale changes in brain structure, demonstrating the top-down influence of the field over the molecular substrate.",
        ),
        "test_protocols": (
            "preserve 1. The Bioelectric Code in Neurogenesis and Regeneration: source-accounting boundary",
        ),
        "null_results": (
            "1. The Bioelectric Code in Neurogenesis and Regeneration: is not empirical validation evidence",
        ),
        "variables": ("1_the_bioelectric_code_in_neurogenesis_and_regeneration",),
        "validation_targets": ("preserve records P0R04657-P0R04659",),
        "null_controls": (
            "1_the_bioelectric_code_in_neurogenesis_and_regeneration must remain source-bounded accounting",
        ),
    },
    "section_1_the_bioelectric_code_in_neurogenesis_and_regeneration.2_the_optimised_connectome": {
        "context_id": "2_the_optimised_connectome",
        "validation_protocol": "paper0.section_1_the_bioelectric_code_in_neurogenesis_and_regeneration.2_the_optimised_connectome",
        "canonical_statement": "The source-bounded component '2. The Optimised Connectome:' preserves Paper 0 records P0R04660-P0R04662 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04660:2_the_optimised_connectome",
            "P0R04661:2_the_optimised_connectome",
            "P0R04662:2_the_optimised_connectome",
        ),
        "source_formulae": (
            "P0R04660: 2. The Optimised Connectome:",
            "P0R04661: The brain's structural connectivity (The Connectome) is optimised by evolution (L8/L15) to support complex dynamics (high ) at minimal metabolic cost.",
            "P0R04662: Small-World Topology: High clustering (local efficiency) and short path length (global integration). | Hierarchical Modularity: Encapsulates specialised functions while allowing for integration. | The Rich Club: A dense network of highly connected hubs (e.g., Precuneus, PFC, Thalamus). The Rich Club forms the structural backbone for the Global Neuronal Workspace (L5) and is critical for maintaining global coherence (UPDE synchronisation).",
        ),
        "test_protocols": ("preserve 2. The Optimised Connectome: source-accounting boundary",),
        "null_results": ("2. The Optimised Connectome: is not empirical validation evidence",),
        "variables": ("2_the_optimised_connectome",),
        "validation_targets": ("preserve records P0R04660-P0R04662",),
        "null_controls": ("2_the_optimised_connectome must remain source-bounded accounting",),
    },
    "section_1_the_bioelectric_code_in_neurogenesis_and_regeneration.3_the_active_role_of_glia_the_tripartite_synapse": {
        "context_id": "3_the_active_role_of_glia_the_tripartite_synapse",
        "validation_protocol": "paper0.section_1_the_bioelectric_code_in_neurogenesis_and_regeneration.3_the_active_role_of_glia_the_tripartite_synapse",
        "canonical_statement": "The source-bounded component '3. The Active Role of Glia (The Tripartite Synapse):' preserves Paper 0 records P0R04663-P0R04665 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04663:3_the_active_role_of_glia_the_tripartite_synapse",
            "P0R04664:3_the_active_role_of_glia_the_tripartite_synapse",
            "P0R04665:3_the_active_role_of_glia_the_tripartite_synapse",
        ),
        "source_formulae": (
            "P0R04663: 3. The Active Role of Glia (The Tripartite Synapse):",
            'P0R04664: Glia are active participants in information processing, forming the "support structure" for L4 dynamics.',
            "P0R04665: Astrocytes: Form large-scale networks coupled by gap junctions. They integrate neuronal activity via calcium waves and modulate synaptic transmission (The Tripartite Synapse). Astrocytes are crucial for maintaining E/I balance and SOC. | Oligodendrocytes and Myelination: Myelination tunes the conduction velocity of axons, adjusting the phase delays in the UPDE. Activity-dependent myelination is a key mechanism for learning and optimising network synchronisation. | Microglia: The brain's immune cells. They monitor the synaptic environment and prune synapses. Microglial activity is modulated by the Psi-field via the PNI axis, linking stress (L5) to structural changes (L3).",
        ),
        "test_protocols": (
            "preserve 3. The Active Role of Glia (The Tripartite Synapse): source-accounting boundary",
        ),
        "null_results": (
            "3. The Active Role of Glia (The Tripartite Synapse): is not empirical validation evidence",
        ),
        "variables": ("3_the_active_role_of_glia_the_tripartite_synapse",),
        "validation_targets": ("preserve records P0R04663-P0R04665",),
        "null_controls": (
            "3_the_active_role_of_glia_the_tripartite_synapse must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section1TheBioelectricCodeInNeurogenesisAndRegenerationSpec:
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
class Section1TheBioelectricCodeInNeurogenesisAndRegenerationSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section1TheBioelectricCodeInNeurogenesisAndRegenerationSpec, ...]
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


def build_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_specs(
    source_records: list[dict[str, Any]],
) -> Section1TheBioelectricCodeInNeurogenesisAndRegenerationSpecBundle:
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

    specs: list[Section1TheBioelectricCodeInNeurogenesisAndRegenerationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section1TheBioelectricCodeInNeurogenesisAndRegenerationSpec(
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
        + "1. The Bioelectric Code in Neurogenesis and Regeneration:"
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
        "next_source_boundary": "P0R04666",
    }
    return Section1TheBioelectricCodeInNeurogenesisAndRegenerationSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section1TheBioelectricCodeInNeurogenesisAndRegenerationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_specs(
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
    bundle: Section1TheBioelectricCodeInNeurogenesisAndRegenerationSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "1. The Bioelectric Code in Neurogenesis and Regeneration:" + " Specs",
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
    bundle: Section1TheBioelectricCodeInNeurogenesisAndRegenerationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_1_the_bioelectric_code_in_neurogenesis_and_regeneration_validation_specs_{date_tag}.md"
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
