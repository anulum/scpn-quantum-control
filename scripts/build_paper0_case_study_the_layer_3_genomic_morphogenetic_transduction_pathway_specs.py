#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway spec builder
"""Promote Paper 0 Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway records."""

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
    "P0R02088",
    "P0R02089",
    "P0R02090",
    "P0R02091",
    "P0R02092",
    "P0R02093",
    "P0R02094",
    "P0R02095",
    "P0R02096",
    "P0R02097",
)
CLAIM_BOUNDARY = "source-bounded case study the layer 3 genomic morphogenetic transduction pathway source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "case_study_the_layer_3_genomic_morphogenetic_transduction_pathway.case_study_the_layer_3_genomic_morphogenetic_transduction_pathway": {
        "context_id": "case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",
        "validation_protocol": "paper0.case_study_the_layer_3_genomic_morphogenetic_transduction_pathway.case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",
        "canonical_statement": "The source-bounded component 'Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway' preserves Paper 0 records P0R02088-P0R02097 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02088:case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",
            "P0R02089:case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",
            "P0R02090:case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",
            "P0R02091:case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",
            "P0R02092:case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",
            "P0R02093:case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",
            "P0R02094:case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",
            "P0R02095:case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",
            "P0R02096:case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",
            "P0R02097:case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",
        ),
        "source_formulae": (
            "P0R02088: Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway",
            "P0R02089: This section provides a rigorous, biophysically-grounded model for the transduction of information at Layer 3 of the SCPN, which governs genomic logic and morphogenesis. The framework posits that the genome is not a static executive blueprint but a dynamic, computationally active substrate that is continuously informed by higher-order organismal fields. It laudably replaces previous, more speculative models with a dual-mechanism framework predicated on established physical phenomena: Chiral-Induced Spin Selectivity (CISS) and endogenous bioelectric fields.",
            'P0R02090: The first mechanism, CISS, leverages the intrinsic chirality of the DNA molecule to function as a highly efficient spin filter. This quantum mechanical effect is proposed as the direct interface between organismal fields and the epigenetic machinery. By modulating the local electromagnetic environment, the organismal field can bias the spin-dependent recombination probabilities of radical pairs integral to the catalytic cycles of key epigenetic enzymes (DNMTs and TETs). The provided Hamiltonian formalism correctly identifies the critical role of spin-orbit coupling in generating a powerful effective magnetic field, which acts as a "spin-valve" controlling the rates of DNA methylation and demethylation. This establishes a precise, quantifiable pathway for top-down informational control over gene expression.',
            "P0R02091: The second mechanism involves the well-documented role of large-scale bioelectric fields in orchestrating cell behaviour and tissue patterning. The model details the complete transduction cascade, from the detection of voltage gradients by ion channels to the downstream activation of second messenger systems and, ultimately, the enzymatic modification of histone proteins. This voltage-sensitive chromatin remodelling provides a robust mechanism for translating the spatial pre-patterns of the bioelectric field into differential gene expression.",
            "P0R02092: Crucially, the framework formalises the synergistic feedback loop between these two mechanisms. The CISS pathway can influence the bioelectric state by modulating ion channel activity, while the bioelectric field can, in turn, alter the conformational geometry of DNA, thereby tuning the efficiency of CISS. The provided coupled differential equation captures this non-linear dynamic, unifying the quantum-spin and classical-field effects into a single, coherent morphogenetic control system. This integrated CISS-bioelectric-epigenetic nexus represents a significant advance, offering a complete, multi-scale, and empirically testable model for how organismal-level information is transduced into stable anatomical structure.",
            'P0R02093: This section reveals the incredible, high-tech toolkit that nature uses to build a body from a single strand of DNA. It answers the question: if the "Organism Field" holds the master plan for a living creature, how does it actually tell the genes what to do? The answer is a brilliant two-part system that works like a quantum computer and a living electrical grid rolled into one.',
            'P0R02094: The first part of the system is called CISS. Think of the DNA helix not just as a library of information, but as a sophisticated quantum antenna. Because of its spiral shape, it can sort electrons according to their "spin," a quantum property. This creates a tiny but powerful magnetic force inside the cell. The instructions from the high-level Organism Field can subtly change this magnetic force, like turning a microscopic dial. This dial controls the speed of the tiny molecular machines that place "on" or "off" switches (called epigenetic marks) on the genes. So, CISS is the quantum-level mechanism that translates the blueprint\'s instructions into direct edits on the genetic code.',
            'P0R02095: The second part of the system is the Bioelectric Field. Long before an arm or a leg is physically built, the body creates an electrical blueprint for it-a subtle energy field that outlines the final shape. Cells can read this electrical map. A strong electrical signal might mean "build an eye here," while a weaker one might mean "form skin." This electrical grid tells large groups of cells what to become and where to go, orchestrating the construction process on a grand scale.',
            "P0R02096: The true genius of the system is that these two parts are constantly talking to each other. The quantum CISS mechanism can help to strengthen the electrical grid, and the electrical grid can fine-tune the quantum antenna in the DNA. It's a seamless feedback loop, ensuring that the grand architectural vision from the Organism Field is translated with incredible precision into the physical, living reality of a perfectly formed body.",
            "P0R02097: P0R02097",
        ),
        "test_protocols": (
            "preserve Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway source-accounting boundary",
        ),
        "null_results": (
            "Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway is not empirical validation evidence",
        ),
        "variables": ("case_study_the_layer_3_genomic_morphogenetic_transduction_pathway",),
        "validation_targets": ("preserve records P0R02088-P0R02097",),
        "null_controls": (
            "case_study_the_layer_3_genomic_morphogenetic_transduction_pathway must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwaySpec:
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
class CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwaySpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwaySpec, ...]
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


def build_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_specs(
    source_records: list[dict[str, Any]],
) -> CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwaySpecBundle:
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

    specs: list[CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwaySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwaySpec(
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
        + "Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway"
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
        "next_source_boundary": "P0R02098",
    }
    return CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwaySpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwaySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_specs(
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
    bundle: CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwaySpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Case Study: The Layer 3 (Genomic-Morphogenetic) Transduction Pathway"
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
    bundle: CaseStudyTheLayer3GenomicMorphogeneticTransductionPathwaySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_case_study_the_layer_3_genomic_morphogenetic_transduction_pathway_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Layer 3 genomic-morphogenetic case-study specs."""

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
