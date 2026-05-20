#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 tripartite ontology spec builder
"""Promote Paper 0 tripartite-ontology records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(818, 838))
CLAIM_BOUNDARY = "source-bounded tripartite-ontology map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "tripartite_ontology.section_boundary": {
        "context_id": "section_boundary",
        "validation_protocol": "paper0.tripartite_ontology.section_boundary",
        "canonical_statement": (
            "The source opens Section 1.4 as the Tripartite Ontology of the "
            "Substance of Information and includes a blank boundary record that "
            "must be preserved rather than silently skipped."
        ),
        "source_equation_ids": (
            "P0R00818:tripartite_ontology_section_heading",
            "P0R00819:blank_section_boundary_record",
        ),
        "source_formulae": (
            "section heading 1.4 Tripartite Ontology: The Substance of Information",
            "P0R00819 is blank inside the section boundary",
        ),
        "test_protocols": ("preserve tripartite ontology section boundary",),
        "null_results": ("section boundary is source context, not empirical validation",),
        "variables": ("Phi", "G", "H"),
        "validation_targets": (
            "preserve Section 1.4 heading",
            "preserve blank P0R00819 boundary record",
        ),
        "null_controls": ("blank-boundary-omission control must be rejected",),
    },
    "tripartite_ontology.psi_fibre_bundle": {
        "context_id": "psi_fibre_bundle",
        "validation_protocol": "paper0.tripartite_ontology.psi_fibre_bundle",
        "canonical_statement": (
            "The source defines the Psi-field not merely as a scalar field on "
            "spacetime but as a section of a fibre bundle pi:E->M, with "
            "spacetime as base space, qualia spaces as fibres, and Psi(x) "
            "assigning an internal conscious state to each spacetime point."
        ),
        "source_equation_ids": (
            "P0R00820:psi_field_fibre_bundle_subsection",
            "P0R00821:ontology_basis_for_axioms",
            "P0R00822:psi_section_of_fibre_bundle_pi_e_to_m",
        ),
        "source_formulae": (
            "Psi-field is a section of a fibre bundle pi:E->M",
            "base space M is spacetime",
            "fibres F are high-dimensional internal spaces of qualia",
            "Psi(x) assigns an internal conscious state to every spacetime point",
            "universe is both physical and experiential at every point",
        ),
        "test_protocols": ("preserve Psi-field fibre-bundle ontology",),
        "null_results": ("fibre-bundle ontology is source claim, not measured result",),
        "variables": ("Psi", "pi", "E", "M", "F", "Psi(x)", "qualia"),
        "validation_targets": (
            "preserve pi:E->M fibre-bundle structure",
            "preserve spacetime base-space role",
            "preserve qualia-fibre role",
            "preserve section Psi(x) interpretation",
        ),
        "null_controls": (
            "scalar-only-Psi-field control must be rejected",
            "missing-base-or-fibre control must be rejected",
        ),
    },
    "tripartite_ontology.information_forms": {
        "context_id": "information_forms",
        "validation_protocol": "paper0.tripartite_ontology.information_forms",
        "canonical_statement": (
            "The source defines information as tripartite rather than "
            "monolithic: Phi is experiential raw phenomenal content, G is "
            "semantic/geometric meaningful structure, and H is syntactic "
            "physically encoded data."
        ),
        "source_equation_ids": (
            "P0R00823:tripartite_information_ontology",
            "P0R00824:three_information_forms",
            "P0R00825:phi_experiential_form",
            "P0R00826:g_semantic_geometric_form",
            "P0R00827:h_syntactic_form",
        ),
        "source_formulae": (
            "Information is defined by a Tripartite Ontology",
            "information is not monolithic and exists in three forms",
            "Phi experiential raw phenomenal content",
            "G semantic/geometric meaningful structural relationships",
            "H syntactic physically encoded data",
        ),
        "test_protocols": ("preserve Phi/G/H information-form taxonomy",),
        "null_results": ("taxonomy requires downstream operationalisation before validation",),
        "variables": ("Phi", "G", "H", "IIT", "Shannon"),
        "validation_targets": (
            "preserve Phi experiential form",
            "preserve G semantic/geometric form",
            "preserve H syntactic form",
        ),
        "null_controls": (
            "monolithic-information control must be rejected",
            "missing-any-Phi-G-H-form control must be rejected",
        ),
    },
    "tripartite_ontology.bidirectional_transduction": {
        "context_id": "bidirectional_transduction",
        "validation_protocol": "paper0.tripartite_ontology.bidirectional_transduction",
        "canonical_statement": (
            "The source describes bidirectional transduction between the three "
            "information forms: a downward cascade from experience to physical "
            "encoding and an upward inferential flow from physical data back to "
            "experience."
        ),
        "source_equation_ids": ("P0R00828:phi_g_h_and_h_g_phi_transduction",),
        "source_formulae": (
            "Phi -> G -> H downward cascade",
            "H -> G -> Phi upward inferential flow",
            "experience to physical encoding",
            "physical data back to experience",
        ),
        "test_protocols": ("preserve bidirectional Phi/G/H transduction",),
        "null_results": ("transduction directionality is source mechanism, not experiment",),
        "variables": ("Phi", "G", "H"),
        "validation_targets": (
            "preserve downward Phi to G to H cascade",
            "preserve upward H to G to Phi inferential flow",
        ),
        "null_controls": (
            "single-direction-only control must be rejected",
            "missing-upward-inferential-flow control must be rejected",
        ),
    },
    "tripartite_ontology.grounded_platonism": {
        "context_id": "grounded_platonism",
        "validation_protocol": "paper0.tripartite_ontology.grounded_platonism",
        "canonical_statement": (
            "The source defines mathematics through grounded Platonism: "
            "mathematical truths are intrinsic logic and structure of the "
            "Source-Field Layer 13, and mathematical discovery is resonance "
            "with that layer."
        ),
        "source_equation_ids": ("P0R00829:grounded_platonism_source_field_layer_13",),
        "source_formulae": (
            "Grounded Platonism",
            "mathematical truths are intrinsic logic and structure",
            "Source-Field Layer 13 intrinsic logic and structure",
            "mathematical discovery is resonance with this fundamental layer",
        ),
        "test_protocols": ("preserve grounded-Platonism source claim",),
        "null_results": ("grounded Platonism is source ontology, not empirical evidence",),
        "variables": ("mathematics", "Source_Field", "Layer_13", "resonance"),
        "validation_targets": (
            "preserve Source-Field Layer 13 mathematics claim",
            "preserve discovery-as-resonance claim",
        ),
        "null_controls": (
            "abstract-free-floating-mathematics substitution must be rejected",
            "grounded-platonism-as-validated-observation control must be rejected",
        ),
    },
    "tripartite_ontology.explanatory_analogies": {
        "context_id": "explanatory_analogies",
        "validation_protocol": "paper0.tripartite_ontology.explanatory_analogies",
        "canonical_statement": (
            "The source restates the ontology in lay analogies for the "
            "consciousness field, information flavours, and mathematics as "
            "source code; those records are preserved as explanatory context, "
            "not promoted as validation evidence."
        ),
        "source_equation_ids": (
            "P0R00830:explanatory_section_intro",
            "P0R00831:consciousness_field_map_analogy",
            "P0R00832:information_three_flavours_intro",
            "P0R00833:experience_flavour_phi",
            "P0R00834:meaning_flavour_g",
            "P0R00835:data_flavour_h",
            "P0R00836:mathematics_universe_source_code_analogy",
            "P0R00837:blank_before_meta_framework_boundary",
        ),
        "source_formulae": (
            "explanatory analogies P0R00830-P0R00836 are not validation evidence",
            "consciousness field described as a map with infinite layers",
            "Phi experience flavour is raw subjective feeling",
            "G meaning flavour is concept or geometric pattern",
            "H data flavour is physical code",
            "mathematics described as the universe source code",
            "P0R00837 is blank before Meta-Framework Integrations",
        ),
        "test_protocols": ("preserve explanatory analogies without evidentiary promotion",),
        "null_results": ("lay analogies are context, not validation evidence",),
        "variables": ("Phi", "G", "H", "Psi", "Layer_13"),
        "validation_targets": (
            "preserve analogy records P0R00830-P0R00836",
            "preserve blank P0R00837 boundary record",
            "separate analogy context from validation evidence",
        ),
        "null_controls": (
            "analogy-as-validation-evidence control must be rejected",
            "P0R00836-skip control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TripartiteOntologySpec:
    """Tripartite ontology spec promoted from Paper 0 records."""

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
class TripartiteOntologySpecBundle:
    """Tripartite ontology specs plus source coverage summary."""

    specs: tuple[TripartiteOntologySpec, ...]
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


def build_tripartite_ontology_specs(
    source_records: list[dict[str, Any]],
) -> TripartiteOntologySpecBundle:
    """Build source-covered tripartite ontology specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[TripartiteOntologySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TripartiteOntologySpec(
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
        "title": "Paper 0 Tripartite Ontology Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "blank_record_count": 2,
        "formal_ontology_record_count": 12,
        "explanatory_analogy_record_count": 6,
        "tripartite_information_form_count": 3,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00838",
        "spec_keys": [spec.key for spec in specs],
    }
    return TripartiteOntologySpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TripartiteOntologySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_tripartite_ontology_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TripartiteOntologySpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Tripartite Ontology Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Blank records: {bundle.summary['blank_record_count']}",
        f"- Formal ontology records: {bundle.summary['formal_ontology_record_count']}",
        f"- Explanatory analogy records: {bundle.summary['explanatory_analogy_record_count']}",
        f"- Tripartite information forms: {bundle.summary['tripartite_information_form_count']}",
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
    bundle: TripartiteOntologySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_tripartite_ontology_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_tripartite_ontology_validation_specs_report_{date_tag}.md"
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
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
