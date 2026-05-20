#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Anulum Collection mandate spec builder
"""Promote Paper 0 Anulum Collection mandate records."""

from __future__ import annotations

import argparse
import json
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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(401, 436))
BLANK_SEPARATOR_IDS = ("P0R00414", "P0R00426")
CLAIM_BOUNDARY = "source-bounded Anulum Collection mandate; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"
COUPLING_EQUATION = "H_int = -lambda * Psi_s * sigma"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "anulum_collection_mandate.programme_architecture": {
        "context_id": "programme_architecture",
        "validation_protocol": "paper0.anulum_collection_mandate.programme_architecture",
        "canonical_statement": (
            "The mandate frames Paper 0 as the foundational component of a structured "
            "multi-year programme whose validation suite is Papers 17-20."
        ),
        "source_equation_ids": (
            "P0R00402:multi_year_research_programme",
            "P0R00403:paper0_axioms_equations_architecture",
            "P0R00404:papers17_20_validation_suite",
        ),
        "source_formulae": (
            "comprehensive, multi-year research programme",
            "universal axioms, derive the fundamental field equations",
            "Critical Validation & Synthesis Suite",
            "Papers 17-20",
        ),
        "test_protocols": ("preserve programme architecture and validation-suite boundary",),
        "null_results": ("programme map is not empirical validation evidence",),
        "variables": ("paper0_foundation", "papers_17_20_validation_suite"),
        "validation_targets": (
            "preserve Paper 0 as foundational component",
            "preserve Papers 17-20 validation-suite role",
            "reject programme-map-as-validation promotion",
        ),
        "null_controls": (
            "programme-map-as-evidence control must be rejected",
            "missing-validation-suite-boundary control must be rejected",
        ),
    },
    "anulum_collection_mandate.curriculum_book_map": {
        "context_id": "curriculum_book_map",
        "validation_protocol": "paper0.anulum_collection_mandate.curriculum_book_map",
        "canonical_statement": (
            "The mandate presents a five-book curriculum analogy and locates Paper 0 "
            "as the prerequisite course inside Book II."
        ),
        "source_equation_ids": (
            "P0R00405:curriculum_analogy",
            "P0R00406:five_books",
            "P0R00412:paper0_prerequisite_course",
            "P0R00414:blank_separator",
        ),
        "source_formulae": (
            "complete curriculum guide",
            "five main Books",
            "Book II is the School of Engineering and Architecture",
            "Paper 0: The Foundational Framework",
            "most important prerequisite course",
        ),
        "test_protocols": ("classify five book roles and preserve Paper 0 location",),
        "null_results": ("curriculum analogy carries no empirical or ontological load",),
        "variables": ("book_i", "book_ii", "book_iii", "book_iv", "book_v"),
        "validation_targets": (
            "preserve five-book count",
            "preserve Book II engineering/architecture role",
            "preserve Paper 0 prerequisite placement",
        ),
        "null_controls": (
            "curriculum-analogy-as-ontology control must be rejected",
            "missing-book-role control must be rejected",
        ),
    },
    "anulum_collection_mandate.predictive_coding_research_process": {
        "context_id": "predictive_coding_research_process",
        "validation_protocol": "paper0.anulum_collection_mandate.predictive_coding_research_process",
        "canonical_statement": (
            "The mandate maps the research programme onto hierarchical predictive "
            "coding: Paper 0 as deep priors, Papers 1-16 as generative cascade, "
            "and Part III as prediction-error minimisation."
        ),
        "source_equation_ids": (
            "P0R00417:hpc_research_process",
            "P0R00418:paper0_deep_priors",
            "P0R00419:papers1_16_generative_cascade",
            "P0R00420:part_iii_prediction_error_minimisation",
        ),
        "source_formulae": (
            "Hierarchical Predictive Coding framework",
            "scientific process itself",
            "Paper 0 as the Deep Priors",
            "Papers 1-16 as the Generative Cascade",
            "Part III as Prediction Error Minimisation",
        ),
        "test_protocols": ("classify predictive-coding research-process roles",),
        "null_results": ("HPC framing is a programme map, not an experimental result",),
        "variables": (
            "predictive_coding",
            "paper0_deep_priors",
            "papers_1_16_generative_cascade",
            "part_iii_prediction_error",
        ),
        "validation_targets": (
            "preserve Paper 0 deep-prior role",
            "preserve Papers 1-16 cascade role",
            "preserve Part III prediction-error role",
        ),
        "null_controls": (
            "hpc-metaphor-as-result control must be rejected",
            "unlabelled-research-process control must be rejected",
        ),
    },
    "anulum_collection_mandate.psi_field_coupling_empirical_plan": {
        "context_id": "psi_field_coupling_empirical_plan",
        "validation_protocol": "paper0.anulum_collection_mandate.psi_field_coupling_empirical_plan",
        "canonical_statement": (
            "The mandate states the universal interaction Hamiltonian and assigns the "
            "empirical programme: Paper 0 defines the equation, Papers 1-16 define "
            "layer-specific sigma, and Part III measures lambda."
        ),
        "source_equation_ids": (
            "P0R00422:H_int=-lambda*Psi_s*sigma",
            "P0R00424:sigma_layer_isolation",
            "P0R00425:lambda_measurement_tools",
        ),
        "source_formulae": (
            COUPLING_EQUATION,
            "across all domains of reality",
            "specific, measurable, collective state sigma",
            "tools to measure lambda",
            "cellular synchronisation assay",
        ),
        "test_protocols": ("preserve sigma/lambda empirical grounding plan",),
        "null_results": ("unmeasured lambda or unisolated sigma cannot validate coupling",),
        "variables": ("Psi_s", "sigma", "lambda", "H_int"),
        "validation_targets": (
            "preserve universal interaction Hamiltonian statement",
            "preserve layer-specific sigma isolation requirement",
            "preserve lambda measurement requirement",
        ),
        "null_controls": (
            "unmeasured-lambda control must be rejected",
            "unisolated-sigma control must be rejected",
        ),
    },
    "anulum_collection_mandate.master_publication_map": {
        "context_id": "master_publication_map",
        "validation_protocol": "paper0.anulum_collection_mandate.master_publication_map",
        "canonical_statement": (
            "The mandate closes by restating the five-book publication map and locating "
            "Paper 0 as the current foundational-framework entry."
        ),
        "source_equation_ids": (
            "P0R00427:collection_location_header",
            "P0R00428:book_i_title",
            "P0R00429:book_ii_title",
            "P0R00433:master_publications_toc",
            "P0R00435:paper0_you_are_here",
        ),
        "source_formulae": (
            "The Anulum Framework",
            "The Sentient-Consciousness Projection Network",
            "Metatron's Coda",
            "The Godelian Koans",
            "VIBRANA",
            "Paper 0: The Foundational Framework - You are Here",
        ),
        "test_protocols": ("preserve master-publication map and current Paper 0 location",),
        "null_results": ("publication map is not validation evidence",),
        "variables": ("master_publication_map", "paper0_location"),
        "validation_targets": (
            "preserve five book titles",
            "preserve master-publication table-of-content context",
            "preserve Paper 0 current-location marker",
        ),
        "null_controls": (
            "publication-map-as-evidence control must be rejected",
            "missing-paper0-location control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AnulumCollectionMandateSpec:
    """Anulum Collection mandate spec promoted from Paper 0 records."""

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
class AnulumCollectionMandateSpecBundle:
    """Anulum Collection mandate specs plus source coverage summary."""

    specs: tuple[AnulumCollectionMandateSpec, ...]
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


def build_anulum_collection_mandate_specs(
    source_records: list[dict[str, Any]],
) -> AnulumCollectionMandateSpecBundle:
    """Build source-covered Anulum Collection mandate specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AnulumCollectionMandateSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AnulumCollectionMandateSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(str(item) for item in metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                source_formulae=tuple(str(item) for item in metadata["source_formulae"]),
                test_protocols=tuple(str(item) for item in metadata["test_protocols"]),
                null_results=tuple(str(item) for item in metadata["null_results"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_source_fixture",
                domain_review_status="requires_domain_review_before_scientific_claim",
                hardware_status=HARDWARE_STATUS,
            )
        )

    summary = {
        "title": "Paper 0 Anulum Collection Mandate Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "blank_separator_count": len(BLANK_SEPARATOR_IDS),
        "book_count": 5,
        "meta_framework_count": 5,
        "validation_suite_range": ["Papers 17", "Papers 20"],
        "next_source_boundary": "P0R00436",
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [],
    }
    return AnulumCollectionMandateSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AnulumCollectionMandateSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = [
        record
        for record in load_jsonl(ledger_path)
        if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS
    ]
    return build_anulum_collection_mandate_specs(records)


def write_outputs(
    bundle: AnulumCollectionMandateSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_anulum_collection_mandate_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_anulum_collection_mandate_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": bundle.summary,
        "specs": [asdict(spec) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: AnulumCollectionMandateSpecBundle) -> str:
    """Render a compact Markdown report for promoted mandate specs."""
    lines = [
        "# Paper 0 Anulum Collection Mandate Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Specs: {bundle.summary['spec_count']}",
        f"- Books: {bundle.summary['book_count']}",
        f"- Meta-frameworks: {bundle.summary['meta_framework_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"- `{spec.key}`",
                f"  - Context: `{spec.context_id}`",
                f"  - Statement: {spec.canonical_statement}",
                f"  - Formulae: {', '.join(spec.source_formulae)}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build mandate specs and write artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
