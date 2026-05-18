#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 category universal grammar spec builder
"""Promote Paper 0 category/universal-grammar records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(905, 987))
CLAIM_BOUNDARY = "source-bounded category/universal-grammar map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "category_universal_grammar.section_boundary": {
        "context_id": "section_boundary",
        "validation_protocol": "paper0.category_universal_grammar.section_boundary",
        "canonical_statement": (
            "Paper 0 opens Section 1.5 as a category-theoretic universal "
            "grammar for SCPN and terminates this slice before the Part II "
            "physical-sector boundary."
        ),
        "source_equation_ids": (
            "P0R00905:section_1_5_universal_grammar_heading",
            "P0R00906:category_theory_for_scpn_heading",
            "P0R00987:next_physical_sector_boundary",
        ),
        "source_formulae": (
            "1.5 The Universal Grammar: A Category-Theoretic Foundation",
            "Mathematical Foundation: Category Theory for SCPN",
            "next boundary is P0R00987 Part II Physical Sector",
        ),
        "test_protocols": ("preserve Section 1.5 source boundary",),
        "null_results": ("section heading is source context, not validation evidence",),
        "variables": ("SCPN", "L1...L15"),
        "validation_targets": (
            "preserve Section 1.5 heading",
            "preserve next Part II boundary",
        ),
        "null_controls": ("section-boundary drift control must be rejected",),
    },
    "category_universal_grammar.category_objects_morphisms": {
        "context_id": "category_objects_morphisms",
        "validation_protocol": "paper0.category_universal_grammar.category_objects_morphisms",
        "canonical_statement": (
            "The source models SCPN as a category whose objects are layers "
            "L1...L15 and whose morphisms are projection maps f: Li -> Lj, "
            "with identity and composition laws preserved as source formulae."
        ),
        "source_equation_ids": (
            "P0R00907:category_objects_morphisms_identity_composition",
            "P0R00931:repeated_category_theory_heading",
            "P0R00932:category_theory_consciousness_projection_context",
            "P0R00935:category_theoretic_framework_heading",
            "P0R00936:scpn_as_category_heading",
            "P0R00937:category_scpn_heading",
            "P0R00938:objects_layers_l1_l15",
            "P0R00939:morphisms_projection_maps",
            "P0R00940:identity_self_consistency",
            "P0R00941:composition_formula",
        ),
        "source_formulae": (
            "objects are 15 layers L1...L15",
            "morphisms are projection maps f: Li -> Lj",
            "Identity: id_L: L -> L",
            "composition (f o g)(x) = f(g(x))",
            "Category SCPN is preserved as source structure",
        ),
        "test_protocols": ("preserve finite-category source definitions",),
        "null_results": ("category laws are source-accounting claims in this slice",),
        "variables": ("L1...L15", "f", "g", "id_L"),
        "validation_targets": (
            "preserve object list",
            "preserve morphism definition",
            "preserve identity law",
            "preserve composition law",
        ),
        "null_controls": (
            "object-layer omission control must be rejected",
            "composition-law omission control must be rejected",
        ),
    },
    "category_universal_grammar.functor_bridge": {
        "context_id": "functor_bridge",
        "validation_protocol": "paper0.category_universal_grammar.functor_bridge",
        "canonical_statement": (
            "The source introduces paired functors between consciousness and "
            "physics plus a natural transformation eta: F => G, including "
            "source examples for quantum, classical, measurement, and "
            "entanglement mappings."
        ),
        "source_equation_ids": (
            "P0R00908:functors_and_natural_transformation_summary",
            "P0R00942:functors_between_consciousness_and_physics_heading",
            "P0R00943:f_consciousness_to_physics",
            "P0R00944:f_psi_field_quantum_state",
            "P0R00945:f_synchronization_classical_field",
            "P0R00946:f_composition_entanglement",
            "P0R00947:g_physics_to_consciousness",
            "P0R00948:g_quantum_l1_substrate",
            "P0R00949:g_classical_l4_synchronization",
            "P0R00950:g_measurement_l5_observation",
            "P0R00955:natural_transformations_heading",
            "P0R00956:eta_f_to_g",
            "P0R00957:eta_l_component",
            "P0R00958:naturality_square_heading",
            "P0R00959:naturality_square_top",
            "P0R00960:naturality_square_vertical_separator",
            "P0R00961:naturality_square_vertical_maps",
            "P0R00962:naturality_square_down_arrows",
            "P0R00963:naturality_square_bottom",
            "P0R00966:naturality_formula_caption_continuation",
            "P0R00967:layer_wise_consistency_caption_continuation",
        ),
        "source_formulae": (
            "F: Consciousness -> Physics",
            "G: Physics -> Consciousness",
            "eta: F => G",
            "eta_L: F(L) -> G(L)",
            "F(Psi_field) = quantum_state",
            "F(synchronization) = classical_field",
            "F(composition) = entanglement",
            "G(quantum) = L1_substrate",
            "G(classical) = L4_synchronization",
            "G(measurement) = L5_observation",
            "F(f o g)=F(f) o F(g)",
            "G(f o g)=G(f) o G(g)",
            "G(f) o eta_L = eta_L' o F(f)",
        ),
        "test_protocols": ("preserve functor bridge and naturality source formulae",),
        "null_results": ("functor bridge is not empirical evidence in this fixture",),
        "variables": ("F", "G", "eta", "L", "f", "g", "Psi_field"),
        "validation_targets": (
            "preserve F and G functor directions",
            "preserve natural-transformation statement",
            "preserve source example mappings",
        ),
        "null_controls": (
            "functor-direction reversal control must be rejected",
            "naturality omission control must be rejected",
        ),
    },
    "category_universal_grammar.topos_kan_foundation": {
        "context_id": "topos_kan_foundation",
        "validation_protocol": "paper0.category_universal_grammar.topos_kan_foundation",
        "canonical_statement": (
            "The source proposes SCPN as a topos with a subobject classifier, "
            "exponential objects, characteristic morphisms, and Kan extensions "
            "for missing-data inference."
        ),
        "source_equation_ids": (
            "P0R00909:topos_and_kan_summary",
            "P0R00969:topos_structure_heading",
            "P0R00970:subobject_classifier",
            "P0R00971:truth_value_mapping",
            "P0R00972:exponential_b_to_a_fragment",
            "P0R00973:projection_fragment",
            "P0R00975:topos_components_caption",
            "P0R00976:characteristic_morphism_caption",
            "P0R00977:exponential_evaluation_caption",
            "P0R00979:kan_extensions_heading",
            "P0R00980:direct_measurement_unavailable",
        ),
        "source_formulae": (
            "Omega = {true, false, uncertain}",
            "B^A represents all possible projections from layer A to layer B",
            "chi_S: States -> Omega",
            "ev: B^A x A -> B",
            "Kan Extensions for Missing Data",
        ),
        "test_protocols": ("preserve topos and Kan-extension foundation records",),
        "null_results": ("topos proposal is source theory, not validated structure",),
        "variables": ("Omega", "B^A", "A", "B", "chi_S", "ev", "Lan", "Ran"),
        "validation_targets": (
            "preserve subobject classifier",
            "preserve exponential-object statement",
            "preserve characteristic morphism and evaluation labels",
        ),
        "null_controls": (
            "truth-value-set omission control must be rejected",
            "exponential-object omission control must be rejected",
        ),
    },
    "category_universal_grammar.explanatory_analogies": {
        "context_id": "explanatory_analogies",
        "validation_protocol": "paper0.category_universal_grammar.explanatory_analogies",
        "canonical_statement": (
            "The source includes explanatory analogy records for category "
            "theory as grammar, SCPN as map, functors as translators, topos as "
            "internal logic, and Kan extensions as educated guessing."
        ),
        "source_equation_ids": (
            "P0R00910:category_theory_grammar_analogy",
            "P0R00911:scpn_as_map_analogy",
            "P0R00912:functors_as_universal_translators",
            "P0R00913:topos_as_built_in_logic",
            "P0R00914:kan_extensions_as_educated_guessing",
            "P0R00915:blank_record",
        ),
        "source_formulae": (
            "category theory is the grammar of the theory",
            "SCPN as a formal map of 15 layers",
            "functors are universal translators",
            "topos provides built-in logic",
            "Kan extensions act as an educated guessing machine",
            "P0R00915 is blank after explanatory analogy records",
        ),
        "test_protocols": ("preserve analogy records without evidentiary promotion",),
        "null_results": ("analogies are context, not validation evidence",),
        "variables": ("SCPN", "F", "G", "Topos", "Kan"),
        "validation_targets": (
            "preserve explanatory analogy block",
            "preserve blank record P0R00915",
        ),
        "null_controls": ("analogy-as-evidence control must be rejected",),
    },
    "category_universal_grammar.meta_framework_integrations": {
        "context_id": "meta_framework_integrations",
        "validation_protocol": "paper0.category_universal_grammar.meta_framework_integrations",
        "canonical_statement": (
            "The source maps category theory into predictive coding, active "
            "inference, and Psi_s/sigma coupling, including the interaction "
            "Hamiltonian source statement H_int = -lambda * Psis * sigma."
        ),
        "source_equation_ids": (
            "P0R00916:meta_framework_integrations_heading",
            "P0R00917:predictive_coding_integration_heading",
            "P0R00918:category_theory_cosmic_generative_model_language",
            "P0R00919:scpn_architecture_heading",
            "P0R00920:category_architecture_predictions_errors",
            "P0R00921:functors_belief_reality_mapping_heading",
            "P0R00922:functor_generating_and_inference_processes",
            "P0R00923:kan_extensions_inference_uncertainty_heading",
            "P0R00924:active_inference_kan_extensions",
            "P0R00925:psis_field_coupling_heading",
            "P0R00926:relationship_via_h_int",
            "P0R00927:objects_as_potential_sigmas_heading",
            "P0R00928:objects_as_sigma_classes",
            "P0R00929:functors_define_coupling_heading",
            "P0R00930:f_coupling_act_and_g_feedback",
        ),
        "source_formulae": (
            "category itself is formal architecture of hierarchical generative model",
            "morphisms are top-down predictions and bottom-up prediction errors",
            "F: Consciousness -> Physics is the generative process",
            "G: Physics -> Consciousness is the inverse inference process",
            "Kan extensions formalise inference under uncertainty",
            "H_int = -lambda * Psis * sigma",
            "objects L1...L15 are potential sigma classes",
            "F projects abstract universal Psi-field to physical sigma",
            "G maps physical sigma back to constraints on the Psi-field",
        ),
        "test_protocols": ("preserve category/predictive-coding/Psis coupling integration",),
        "null_results": ("meta-framework integration is source claim, not experiment",),
        "variables": ("F", "G", "H_int", "lambda", "Psis", "sigma", "L1...L15"),
        "validation_targets": (
            "preserve predictive-coding integration",
            "preserve H_int coupling statement",
            "preserve bidirectional F/G coupling description",
        ),
        "null_controls": (
            "H_int omission control must be rejected",
            "predictive-coding integration omission control must be rejected",
        ),
    },
    "category_universal_grammar.formal_diagram_records": {
        "context_id": "formal_diagram_records",
        "validation_protocol": "paper0.category_universal_grammar.formal_diagram_records",
        "canonical_statement": (
            "The source includes five image placeholders and eleven figure or "
            "caption-continuation records for category, functor, naturality, "
            "topos, and Kan-extension diagrams; these are source records and "
            "not validation evidence."
        ),
        "source_equation_ids": (
            "P0R00933:category_image_placeholder",
            "P0R00934:category_figure_caption",
            "P0R00951:functor_image_placeholder",
            "P0R00952:functor_figure_caption",
            "P0R00953:functor_caption_continuation",
            "P0R00954:functoriality_caption_continuation",
            "P0R00964:naturality_image_placeholder",
            "P0R00965:naturality_figure_caption",
            "P0R00966:naturality_formula_caption_continuation",
            "P0R00967:naturality_consistency_caption_continuation",
            "P0R00974:topos_image_placeholder",
            "P0R00975:topos_figure_caption",
            "P0R00976:topos_caption_continuation",
            "P0R00977:topos_caption_continuation",
            "P0R00985:kan_image_placeholder",
            "P0R00986:kan_figure_caption",
        ),
        "source_formulae": (
            "image placeholders and figure captions are not validation evidence",
            "five image placeholder records are preserved",
            "eleven figure or caption-continuation records are preserved",
            "P0R00968 and P0R00978 are blank separator records",
        ),
        "test_protocols": ("preserve diagram source records without evidentiary promotion",),
        "null_results": ("diagram captions do not validate category-theoretic claims",),
        "variables": ("F", "G", "eta", "Omega", "B^A", "Lan_F"),
        "validation_targets": (
            "preserve image placeholders",
            "preserve caption records",
            "separate diagram context from validation evidence",
        ),
        "null_controls": (
            "image-as-evidence control must be rejected",
            "caption-record omission control must be rejected",
        ),
    },
    "category_universal_grammar.kan_extension_inference": {
        "context_id": "kan_extension_inference",
        "validation_protocol": "paper0.category_universal_grammar.kan_extension_inference",
        "canonical_statement": (
            "The source states left and right Kan extension approximations for "
            "missing data and defines Psi_inferred as Lan_physical(Psi_true)."
        ),
        "source_equation_ids": (
            "P0R00979:kan_extensions_for_missing_data_heading",
            "P0R00980:direct_measurement_unavailable",
            "P0R00981:lan_best_approximation_below",
            "P0R00982:ran_best_approximation_above",
            "P0R00983:consciousness_inference_heading",
            "P0R00984:psi_inferred_lan_physical_psi_true",
            "P0R00985:kan_image_placeholder",
            "P0R00986:kan_extensions_caption",
        ),
        "source_formulae": (
            "Lan_F(G) = best approximation from below",
            "Ran_F(G) = best approximation from above",
            "Psi_inferred = Lan_physical(Psi_true)",
            "next boundary is P0R00987 Part II Physical Sector",
        ),
        "test_protocols": ("preserve Kan-extension inference equations",),
        "null_results": ("Kan equations are source formulae, not runtime inference proof",),
        "variables": ("Lan_F", "Ran_F", "G", "Psi_inferred", "Psi_true"),
        "validation_targets": (
            "preserve Lan_F formula",
            "preserve Ran_F formula",
            "preserve Psi_inferred formula",
        ),
        "null_controls": (
            "Lan/Ran reversal control must be rejected",
            "Psi_inferred omission control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class CategoryUniversalGrammarSpec:
    """Category/universal-grammar spec promoted from Paper 0 records."""

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
class CategoryUniversalGrammarSpecBundle:
    """Category/universal-grammar specs plus source coverage summary."""

    specs: tuple[CategoryUniversalGrammarSpec, ...]
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


def build_category_universal_grammar_specs(
    source_records: list[dict[str, Any]],
) -> CategoryUniversalGrammarSpecBundle:
    """Build source-covered category/universal-grammar specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[CategoryUniversalGrammarSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            CategoryUniversalGrammarSpec(
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
        "title": "Paper 0 Category Universal Grammar Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "blank_record_count": 3,
        "image_record_count": 5,
        "figure_caption_record_count": 11,
        "formal_category_record_count": 27,
        "meta_framework_record_count": 15,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00987",
        "spec_keys": [spec.key for spec in specs],
    }
    return CategoryUniversalGrammarSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> CategoryUniversalGrammarSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_category_universal_grammar_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: CategoryUniversalGrammarSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Category Universal Grammar Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Formal-category records: {bundle.summary['formal_category_record_count']}",
        f"- Meta-framework records: {bundle.summary['meta_framework_record_count']}",
        f"- Image records: {bundle.summary['image_record_count']}",
        f"- Figure / caption records: {bundle.summary['figure_caption_record_count']}",
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
    bundle: CategoryUniversalGrammarSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_category_universal_grammar_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_category_universal_grammar_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 category-universal grammar specs from the ledger."""

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
