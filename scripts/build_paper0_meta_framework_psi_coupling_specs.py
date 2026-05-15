#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 meta-framework Psi coupling spec builder
"""Promote Paper 0 meta-framework/Psi-coupling records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(838, 905))
CLAIM_BOUNDARY = "source-bounded meta-framework/Psi-coupling map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "meta_framework_psi_coupling.meta_framework_boundary": {
        "context_id": "meta_framework_boundary",
        "validation_protocol": "paper0.meta_framework_psi_coupling.meta_framework_boundary",
        "canonical_statement": (
            "The source opens a Meta-Framework Integrations block within the "
            "tripartite ontology material and then repeats the same integration "
            "surface under the later Ontology of the Psi-Field and Information block."
        ),
        "source_equation_ids": (
            "P0R00838:meta_framework_integrations_heading",
            "P0R00876:repeated_meta_framework_integrations_heading",
        ),
        "source_formulae": (
            "Meta-Framework Integrations",
            "Predictive Coding Integration",
            "repeated Meta-Framework Integrations block is preserved",
        ),
        "test_protocols": ("preserve meta-framework boundary and repeated block",),
        "null_results": ("meta-framework heading is source context, not validation evidence",),
        "variables": ("Psi", "Phi", "G", "H"),
        "validation_targets": (
            "preserve initial meta-framework heading",
            "preserve repeated meta-framework heading",
        ),
        "null_controls": ("repeated-meta-framework-block omission control must be rejected",),
    },
    "meta_framework_psi_coupling.predictive_coding_loop": {
        "context_id": "predictive_coding_loop",
        "validation_protocol": "paper0.meta_framework_psi_coupling.predictive_coding_loop",
        "canonical_statement": (
            "The source maps the fibre bundle to the state space of a cosmic "
            "generative model and maps tripartite information to an active "
            "inference loop from phenomenal belief through geometric prediction "
            "to sensory data, with prediction error flowing upward."
        ),
        "source_equation_ids": (
            "P0R00839:predictive_coding_heading",
            "P0R00840:cosmic_generative_model",
            "P0R00841:fibre_bundle_state_space_heading",
            "P0R00842:fibre_bundle_state_space_of_beliefs",
            "P0R00843:tripartite_inference_loop_heading",
            "P0R00844:tripartite_active_inference_loop",
            "P0R00877:repeated_predictive_coding_heading",
            "P0R00878:repeated_cosmic_generative_model",
            "P0R00879:repeated_fibre_bundle_state_space_heading",
            "P0R00880:repeated_fibre_bundle_state_space_of_beliefs",
            "P0R00881:repeated_tripartite_inference_loop_heading",
            "P0R00882:repeated_tripartite_active_inference_loop",
        ),
        "source_formulae": (
            "fibre bundle is state space of beliefs",
            "tripartite ontology is active inference loop",
            "Phi phenomenal belief translates into geometric prediction G",
            "G generates predicted sensory data H",
            "prediction error flows upward to update G and Phi",
        ),
        "test_protocols": ("preserve predictive-coding integration mechanism",),
        "null_results": ("predictive-coding mapping is source claim, not experiment",),
        "variables": ("Psi", "Phi", "G", "H", "prediction_error"),
        "validation_targets": (
            "preserve fibre-bundle belief-state mapping",
            "preserve tripartite active-inference loop",
            "preserve upward prediction-error update path",
        ),
        "null_controls": (
            "belief-state mapping omission control must be rejected",
            "prediction-error update omission control must be rejected",
        ),
    },
    "meta_framework_psi_coupling.psi_interaction_hamiltonian": {
        "context_id": "psi_interaction_hamiltonian",
        "validation_protocol": ("paper0.meta_framework_psi_coupling.psi_interaction_hamiltonian"),
        "canonical_statement": (
            "The source states the interaction Hamiltonian H_int = -lambda * "
            "Psis * sigma and defines Psis as a fibre-bundle section while sigma "
            "is the physical system at a point in the base space."
        ),
        "source_equation_ids": (
            "P0R00845:psis_field_coupling_heading",
            "P0R00846:interaction_hamiltonian_terms_context",
            "P0R00847:h_int_minus_lambda_psis_sigma",
            "P0R00848:psis_section_heading",
            "P0R00849:psis_section_of_universal_fibre_bundle",
            "P0R00850:sigma_physical_base_heading",
            "P0R00851:sigma_physical_system_at_base_point",
            "P0R00883:repeated_psis_field_coupling_heading",
            "P0R00884:repeated_interaction_hamiltonian_terms_context",
            "P0R00885:repeated_h_int_minus_lambda_psis_sigma",
            "P0R00886:repeated_psis_section_heading",
            "P0R00887:repeated_psis_section_of_universal_fibre_bundle",
            "P0R00888:repeated_sigma_physical_base_heading",
            "P0R00889:repeated_sigma_physical_system_at_base_point",
        ),
        "source_formulae": (
            "H_int = -lambda * Psis * sigma",
            "Psis is a section of the universal fibre bundle",
            "sigma is the physical system at a point in base space M",
            "Psis imparts a high-dimensional geometric state",
        ),
        "test_protocols": ("preserve source interaction-Hamiltonian statement",),
        "null_results": ("Hamiltonian statement is source accounting, not validated dynamics",),
        "variables": ("H_int", "lambda", "Psis", "sigma", "M"),
        "validation_targets": (
            "preserve H_int source equation",
            "preserve Psis section interpretation",
            "preserve sigma base-space interpretation",
        ),
        "null_controls": (
            "missing-H_int-equation control must be rejected",
            "scalar-only-Psis control must be rejected",
        ),
    },
    "meta_framework_psi_coupling.coupling_projection": {
        "context_id": "coupling_projection",
        "validation_protocol": "paper0.meta_framework_psi_coupling.coupling_projection",
        "canonical_statement": (
            "The source defines the H_int interaction as projection from the "
            "total space of the bundle onto a specific fibre, with lambda "
            "setting projection strength and geometric content G transduced "
            "into syntactic state H of sigma."
        ),
        "source_equation_ids": (
            "P0R00852:coupling_projection_heading",
            "P0R00853:h_int_projection_and_g_to_h_transduction",
            "P0R00890:repeated_coupling_projection_heading",
            "P0R00891:repeated_h_int_projection_and_g_to_h_transduction",
        ),
        "source_formulae": (
            "H_int realises projection from total space onto a specific fibre",
            "lambda determines projection strength",
            "universal Psi-field selects an internal state in qualia fibre F",
            "geometric content G is transduced into syntactic state H of sigma",
        ),
        "test_protocols": ("preserve coupling-as-projection mechanism",),
        "null_results": ("projection mechanism requires downstream formal validation",),
        "variables": ("H_int", "lambda", "Psi", "F", "G", "H", "sigma"),
        "validation_targets": (
            "preserve total-space to fibre projection claim",
            "preserve lambda projection-strength role",
            "preserve G to H transduction claim",
        ),
        "null_controls": (
            "projection-to-fibre omission control must be rejected",
            "G-to-H-transduction omission control must be rejected",
        ),
    },
    "meta_framework_psi_coupling.formal_ontology_restatement": {
        "context_id": "formal_ontology_restatement",
        "validation_protocol": ("paper0.meta_framework_psi_coupling.formal_ontology_restatement"),
        "canonical_statement": (
            "The source repeats formal ontology equations for Psi(x) in E and "
            "pi:E->M, plus base-space, fibre, self-interaction, tripartite "
            "ontology, and grounded-Platonism summaries."
        ),
        "source_equation_ids": (
            "P0R00854:formal_ontology_restatement_heading",
            "P0R00855:psi_section_over_spacetime",
            "P0R00856:psi_x_in_e",
            "P0R00857:pi_e_to_m_m_spacetime",
            "P0R00858:base_fibre_self_interaction",
            "P0R00859:l13_fibre_bundle_summary",
            "P0R00860:tripartite_ontology_summary",
            "P0R00861:grounded_platonism_summary",
        ),
        "source_formulae": (
            "Psi(x) in E",
            "pi:E->M with M=Spacetime",
            "base space M relates to spacetime geometry",
            "fibres F represent internal degrees of freedom",
            "strong self-interaction essential for localised Selves L5 solitons",
            "G <-> H downward and H->G->Phi upward transduction summary",
            "mathematics is intrinsic structure and logic of Source-Field L13",
        ),
        "test_protocols": ("preserve formal ontology restatement equations",),
        "null_results": ("formal restatement is source accounting, not validation evidence",),
        "variables": ("Psi(x)", "E", "pi", "M", "F", "L13", "L5", "Phi", "G", "H"),
        "validation_targets": (
            "preserve Psi(x) in E equation",
            "preserve pi:E->M equation",
            "preserve restated tripartite ontology",
        ),
        "null_controls": (
            "missing-Psi-x-in-E control must be rejected",
            "missing-pi-E-to-M control must be rejected",
        ),
    },
    "meta_framework_psi_coupling.figure_and_image_records": {
        "context_id": "figure_and_image_records",
        "validation_protocol": "paper0.meta_framework_psi_coupling.figure_and_image_records",
        "canonical_statement": (
            "The source includes image placeholders, figure captions, and a "
            "diagram description for fibre-bundle, tripartite-information, and "
            "grounded-Platonism material; these are preserved as source records "
            "and not promoted as empirical evidence."
        ),
        "source_equation_ids": (
            "P0R00862:image_placeholder",
            "P0R00863:figure_caption",
            "P0R00864:diagram_description",
            "P0R00895:source_field_image_placeholder",
            "P0R00896:fibre_bundle_figure_caption",
            "P0R00899:tripartite_information_image_placeholder",
            "P0R00903:information_hierarchy_image_placeholder",
            "P0R00904:information_hierarchy_figure_caption",
        ),
        "source_formulae": (
            "image placeholders are source records, not evidence",
            "Psi-field fibre bundle tripartite information and grounded Platonism figure",
            "diagram describes Phi downward transduction into G and H",
            "upward flow carries constraints from H to G and updates Phi",
            "figure captions are preserved without evidentiary promotion",
        ),
        "test_protocols": ("preserve image and figure-caption source records",),
        "null_results": ("image placeholders do not validate diagram claims",),
        "variables": ("Psi", "Phi", "G", "H", "pi", "M", "E"),
        "validation_targets": (
            "preserve image placeholder records",
            "preserve figure-caption records",
            "separate diagram context from validation evidence",
        ),
        "null_controls": (
            "image-as-evidence control must be rejected",
            "figure-caption omission control must be rejected",
        ),
    },
    "meta_framework_psi_coupling.repeated_ontology_block": {
        "context_id": "repeated_ontology_block",
        "validation_protocol": "paper0.meta_framework_psi_coupling.repeated_ontology_block",
        "canonical_statement": (
            "The source repeats the ontology of the Psi-field and information, "
            "including formal, explanatory, meta-framework, coupling, and figure "
            "subblocks, and terminates before Section 1.5 at P0R00905."
        ),
        "source_equation_ids": (
            "P0R00865:ontology_of_psi_field_and_information_heading",
            "P0R00866:ontology_basis_for_axioms",
            "P0R00867:psi_fibre_bundle_restatement",
            "P0R00868:tripartite_information_axiom_2_restatement",
            "P0R00869:explanatory_two_key_ideas_intro",
            "P0R00870:consciousness_field_map_analogy",
            "P0R00871:information_three_flavours_intro",
            "P0R00872:experience_flavour_phi",
            "P0R00873:meaning_flavour_g",
            "P0R00874:data_flavour_h",
            "P0R00875:blank_record",
            "P0R00892:second_ontology_heading",
            "P0R00893:l13_fibre_bundle_heading",
            "P0R00894:l13_fibre_bundle_master_lagrangian_coupling",
            "P0R00897:blank_record",
            "P0R00898:tripartite_information_heading",
            "P0R00900:tripartite_ontology_axiom_2",
            "P0R00901:phi_g_h_extended_definitions",
            "P0R00902:information_hierarchy_transduction",
        ),
        "source_formulae": (
            "P0R00875 and P0R00897 are blank records",
            "Axiom 2 mandates tripartite information ontology",
            "downward generative cascade Phi -> G -> H",
            "upward inferential flow H -> G -> Phi closes causal-epistemic loop",
            "Fisher Information Metric g_mu_nu Curvature R_mu_nu and Topology H_k",
            "next boundary is P0R00905 Section 1.5 Universal Grammar",
        ),
        "test_protocols": ("preserve repeated ontology block and Section 1.5 boundary",),
        "null_results": ("repeated source block is accounting context, not duplicate evidence",),
        "variables": ("Psi", "Phi", "G", "H", "g_mu_nu", "R_mu_nu", "H_k"),
        "validation_targets": (
            "preserve repeated ontology block",
            "preserve blank records P0R00875 and P0R00897",
            "preserve Section 1.5 boundary at P0R00905",
        ),
        "null_controls": (
            "blank-record omission control must be rejected",
            "section-boundary drift control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class MetaFrameworkPsiCouplingSpec:
    """Meta-framework/Psi-coupling spec promoted from Paper 0 records."""

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
class MetaFrameworkPsiCouplingSpecBundle:
    """Meta-framework/Psi-coupling specs plus source coverage summary."""

    specs: tuple[MetaFrameworkPsiCouplingSpec, ...]
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


def build_meta_framework_psi_coupling_specs(
    source_records: list[dict[str, Any]],
) -> MetaFrameworkPsiCouplingSpecBundle:
    """Build source-covered meta-framework/Psi-coupling specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[MetaFrameworkPsiCouplingSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            MetaFrameworkPsiCouplingSpec(
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
        "title": "Paper 0 Meta-Framework Psi Coupling Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "predictive_coding_record_count": 14,
        "psi_coupling_record_count": 25,
        "formal_restatement_record_count": 22,
        "image_or_figure_record_count": 6,
        "blank_record_count": 2,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00905",
        "spec_keys": [spec.key for spec in specs],
    }
    return MetaFrameworkPsiCouplingSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> MetaFrameworkPsiCouplingSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_meta_framework_psi_coupling_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: MetaFrameworkPsiCouplingSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Meta-Framework Psi Coupling Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Predictive-coding records: {bundle.summary['predictive_coding_record_count']}",
        f"- Psi-coupling records: {bundle.summary['psi_coupling_record_count']}",
        f"- Formal-restatement records: {bundle.summary['formal_restatement_record_count']}",
        f"- Image / figure records: {bundle.summary['image_or_figure_record_count']}",
        f"- Blank records: {bundle.summary['blank_record_count']}",
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
    bundle: MetaFrameworkPsiCouplingSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_meta_framework_psi_coupling_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_meta_framework_psi_coupling_validation_specs_report_{date_tag}.md"
    )
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
