#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Axiom II opening spec builder
"""Promote Paper 0 Axiom II opening and interaction-axiom records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(761, 770))
HEADING_RECORD_IDS = ("P0R00761", "P0R00762", "P0R00763", "P0R00764")
CLAIM_BOUNDARY = "source-bounded Axiom II opening map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiom_ii_opening.section_headings": {
        "context_id": "section_headings",
        "validation_protocol": "paper0.axiom_ii_opening.section_headings",
        "canonical_statement": (
            "The source opens Axiom II and records the three immediate navigation "
            "headings: infoton geometry, the Fisher Information Metric solution, "
            "and the informational Lagrangian consequence."
        ),
        "source_equation_ids": (
            "P0R00761:axiom_ii_language_of_information_geometry",
            "P0R00762:infoton_geometry_problem_heading",
            "P0R00763:fim_solution_heading",
            "P0R00764:informational_lagrangian_heading",
        ),
        "source_formulae": (
            "Axiom II: The Language of Information Geometry",
            "The Central Problem: The Geometry of the Infoton",
            "The Fisher Information Metric (FIM) as the Solution",
            "Formal Consequence: The Informational Lagrangian",
        ),
        "test_protocols": ("preserve Axiom II opening navigation headings",),
        "null_results": ("headings are source navigation, not validation results",),
        "variables": ("Axiom_II", "infoton", "FIM", "L_gauge"),
        "validation_targets": (
            "preserve Axiom II title",
            "preserve infoton geometry heading",
            "preserve FIM solution heading",
            "preserve informational Lagrangian heading",
        ),
        "null_controls": (
            "heading-as-equation control must be rejected",
            "missing-FIM-heading control must be rejected",
        ),
    },
    "axiom_ii_opening.source_material": {
        "context_id": "source_material",
        "validation_protocol": "paper0.axiom_ii_opening.source_material",
        "canonical_statement": (
            "The source states that all fundamental interactions are "
            "informational transactions and that their natural descriptive "
            "language is geometry, specifically information geometry."
        ),
        "source_equation_ids": ("P0R00765:source_material_information_geometry",),
        "source_formulae": (
            "all fundamental interactions are informational transactions",
            "natural language of interactions is geometry",
            "specifically information geometry",
        ),
        "test_protocols": ("preserve Axiom II source-material claim boundary",),
        "null_results": ("source-material statement is not an empirical interaction test",),
        "variables": ("interaction", "information_geometry", "geometry"),
        "validation_targets": (
            "preserve informational-transaction statement",
            "preserve information-geometry language statement",
        ),
        "null_controls": (
            "source-material-as-evidence control must be rejected",
            "geometry-without-information control must be rejected",
        ),
    },
    "axiom_ii_opening.ontology_to_dynamics": {
        "context_id": "ontology_to_dynamics",
        "validation_protocol": "paper0.axiom_ii_opening.ontology_to_dynamics",
        "canonical_statement": (
            "The source positions Axiom II as the transition from Axiom I's "
            "Psi-field ontology to interaction dynamics through a universal "
            "grammar for fundamental forces."
        ),
        "source_equation_ids": ("P0R00766:ontology_to_dynamics_transition",),
        "source_formulae": (
            "Axiom I establishes the substance of reality as the Psi-field",
            "Axiom II defines the language of interaction",
            "ontology to dynamics",
            "single universal grammar for all fundamental forces",
        ),
        "test_protocols": ("preserve ontology-to-dynamics transition boundary",),
        "null_results": ("transition language is not a solved dynamics derivation",),
        "variables": ("Psi_field", "ontology", "dynamics", "fundamental_forces"),
        "validation_targets": (
            "preserve Psi-field substance boundary",
            "preserve interaction-language role",
            "preserve universal-grammar role",
        ),
        "null_controls": (
            "ontology-as-dynamics control must be rejected",
            "force-specific-grammar control must be rejected",
        ),
    },
    "axiom_ii_opening.interaction_axiom": {
        "context_id": "interaction_axiom",
        "validation_protocol": "paper0.axiom_ii_opening.interaction_axiom",
        "canonical_statement": (
            "The source states Axiom II as the axiom of interaction: all "
            "fundamental interactions are informational and geometric, and the "
            "claim is marked as a falsifiable physical hypothesis requiring "
            "system informational structure before coupling analysis."
        ),
        "source_equation_ids": (
            "P0R00767:axiom_of_interaction_heading",
            "P0R00768:informational_geometric_interaction_axiom",
            "P0R00769:falsifiable_physical_hypothesis_boundary",
        ),
        "source_formulae": (
            "Axiom II: The Axiom of Interaction (Information Geometry)",
            "All fundamental interactions are informational and geometric",
            "coupling propagation and exchange of conscious information",
            "language of information geometry",
            "falsifiable physical hypothesis",
            "system informational structure precedes coupling analysis",
        ),
        "test_protocols": ("preserve Axiom II interaction statement and falsifiability boundary",),
        "null_results": ("interaction axiom is not empirical validation by registration",),
        "variables": ("coupling", "propagation", "conscious_information", "system_structure"),
        "validation_targets": (
            "preserve informational-geometric interaction statement",
            "preserve coupling-propagation-exchange role",
            "preserve falsifiability boundary",
            "preserve system-structure prerequisite",
        ),
        "null_controls": (
            "axiom-as-validation control must be rejected",
            "coupling-without-system-structure control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomIIOpeningSpec:
    """Axiom II opening spec promoted from Paper 0 records."""

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
class AxiomIIOpeningSpecBundle:
    """Axiom II opening specs plus source coverage summary."""

    specs: tuple[AxiomIIOpeningSpec, ...]
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


def build_axiom_ii_opening_specs(
    source_records: list[dict[str, Any]],
) -> AxiomIIOpeningSpecBundle:
    """Build source-covered Axiom II opening specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomIIOpeningSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomIIOpeningSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0].get("section_path", "")),
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
                implementation_status="implemented_source_accounting_fixture",
                domain_review_status="requires_domain_review_before_public_claim",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 Axiom II Opening Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "heading_record_count": len(HEADING_RECORD_IDS),
        "axiom_statement_count": 1,
        "falsifiability_boundary_count": 1,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00770",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomIIOpeningSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AxiomIIOpeningSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiom_ii_opening_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomIIOpeningSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiom II Opening Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Heading records: {bundle.summary['heading_record_count']}",
        f"- Axiom statements: {bundle.summary['axiom_statement_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Promoted Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### {spec.key}",
                "",
                spec.canonical_statement,
                "",
                "Formulae / source labels:",
            ]
        )
        for formula in spec.source_formulae:
            lines.append(f"- {formula}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    bundle: AxiomIIOpeningSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_axiom_ii_opening_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_axiom_ii_opening_validation_specs_report_{date_tag}.md"
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Axiom II opening specs from the ledger."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    outputs = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
