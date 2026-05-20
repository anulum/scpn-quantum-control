#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 category grammar spec builder
"""Promote Paper 0 integration-synthesis category grammar records into specs."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6815, 6878))
CLAIM_BOUNDARY = "source-bounded category-theory formal grammar fixture; not empirical evidence"
HARDWARE_STATUS = "formal_consistency_fixture_no_execution"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "integration_synthesis.category_grammar.block_boundary": {
        "validation_protocol": "paper0.integration_synthesis.category_grammar.boundary",
        "canonical_statement": (
            "Paper 0 inserts a category-theory integration synthesis before the Grand Synthesis."
        ),
        "source_equation_ids": ("P0R06816:location_boundary",),
        "source_formulae": ('Location: before "Grand Synthesis" section',),
        "source_mechanisms": (
            "integration synthesis insertion",
            "category theory as SCPN formalism",
        ),
        "variables": ("integration_synthesis", "category_grammar"),
        "validation_targets": (
            "preserve insertion boundary",
            "separate category grammar from later Appendix C Hamiltonian index",
            "reject empirical execution claims",
        ),
        "null_controls": (
            "missing-insertion-boundary control must be rejected",
            "appendix-hamiltonian-index bleed-through control must be rejected",
            "unsupported-empirical-claim control must be rejected",
        ),
    },
    "integration_synthesis.category_grammar.scpn_category": {
        "validation_protocol": "paper0.integration_synthesis.category_grammar.scpn_category",
        "canonical_statement": (
            "SCPN is framed as a category whose objects are layers and whose morphisms are projections."
        ),
        "source_equation_ids": (
            "P0R06822:projection_morphism",
            "P0R06824:identity",
            "P0R06825:composition",
        ),
        "source_formulae": (
            "Objects: Layers {L1, L2, ..., L15, L16}",
            "Morphisms: projection maps f: L_i -> L_j",
            "Identity: id_L: L -> L",
            "Composition: f o g well-defined and associative",
        ),
        "source_mechanisms": (
            "layers act as category objects",
            "projection maps act as morphisms",
            "identity and associativity are formal proof obligations",
        ),
        "variables": ("L_i", "L_j", "f", "id_L"),
        "validation_targets": (
            "validate identity laws on a finite layer fixture",
            "validate associativity on composable projection maps",
            "reject noncomposable morphisms",
        ),
        "null_controls": (
            "missing-identity control must be rejected",
            "nonassociative-composition control must be rejected",
            "noncomposable-morphism control must be rejected",
        ),
    },
    "integration_synthesis.category_grammar.functorial_mappings": {
        "validation_protocol": "paper0.integration_synthesis.category_grammar.functorial_mappings",
        "canonical_statement": (
            "Downward and upward functorial mappings are stated between consciousness and physics descriptions."
        ),
        "source_equation_ids": (
            "P0R06827:downward_projection_functor",
            "P0R06830:upward_integration_functor",
            "P0R06833:natural_transformation",
            "P0R06834:natural_transformation_component",
        ),
        "source_formulae": (
            "F: Consciousness -> Physics",
            "G: Physics -> Consciousness",
            "eta: F => G",
            "eta_L: F(L) -> G(L)",
        ),
        "source_mechanisms": (
            "Psi-field states map to quantum states",
            "synchronization patterns map to classical fields",
            "quantum events map to L1 substrate effects",
            "classical dynamics map to L4 synchronization",
        ),
        "variables": ("F", "G", "eta", "eta_L"),
        "validation_targets": (
            "validate endpoint coverage for finite functor object maps",
            "record natural transformation as square-commutation obligation",
            "reject object maps missing morphism endpoints",
        ),
        "null_controls": (
            "missing-functor-endpoint control must be rejected",
            "missing-natural-transformation-boundary control must be rejected",
            "direction-erasure control must be rejected",
        ),
    },
    "integration_synthesis.category_grammar.topos_internal_logic": {
        "validation_protocol": "paper0.integration_synthesis.category_grammar.topos_logic",
        "canonical_statement": (
            "The topos/internal-logic paragraph introduces a three-valued truth classifier and exponential objects."
        ),
        "source_equation_ids": ("P0R06837:subobject_classifier", "P0R06839:exponential_object"),
        "source_formulae": (
            "Subobject classifier Omega = {true, false, uncertain}",
            "Exponential objects: B^A",
        ),
        "source_mechanisms": (
            "truth values classify propositions within SCPN",
            "exponential objects represent spaces of transformations A -> B",
            "function spaces are reserved for layer dynamics",
        ),
        "variables": ("Omega", "A", "B", "B^A"),
        "validation_targets": (
            "preserve exactly three truth values",
            "preserve exponential-object notation as transformation-space claim",
            "reject binary-only truth classifier downgrade",
        ),
        "null_controls": (
            "missing-uncertain-truth-value control must be rejected",
            "missing-exponential-object control must be rejected",
            "truth-classifier-as-empirical-result control must be rejected",
        ),
    },
    "integration_synthesis.category_grammar.kan_inference_mechanism": {
        "validation_protocol": "paper0.integration_synthesis.category_grammar.kan_inference",
        "canonical_statement": (
            "Kan extensions are framed as inference mechanisms with below and above approximation roles."
        ),
        "source_equation_ids": (
            "P0R06843:left_kan_extension",
            "P0R06846:right_kan_extension",
            "P0R06849:psi_inferred_estimate",
        ),
        "source_formulae": (
            "Lan_F(G)",
            "Ran_F(G)",
            "Psi_inferred = Lan_Physical(Psi_true)",
        ),
        "source_mechanisms": (
            "left Kan extension supplies best approximation from below",
            "right Kan extension supplies best approximation from above",
            "physical inference about Psi-field from observations is a validation target",
        ),
        "variables": ("Lan_F", "Ran_F", "Psi_inferred", "Psi_true"),
        "validation_targets": (
            "preserve Lan and Ran as distinct formal roles",
            "map Psi-field prediction about physical outcomes to validation target",
            "reject treating Kan notation as executed inference",
        ),
        "null_controls": (
            "left-right-kan-collapse control must be rejected",
            "missing-psi-inference-estimate control must be rejected",
            "executed-inference-overclaim control must be rejected",
        ),
    },
    "integration_synthesis.category_grammar.string_diagram_calculus": {
        "validation_protocol": "paper0.integration_synthesis.category_grammar.string_diagrams",
        "canonical_statement": (
            "String diagrams are introduced as visual calculus for identity, composition, and tensoring."
        ),
        "source_equation_ids": (
            "P0R06852:composition_diagram",
            "P0R06854:identity_diagram",
            "P0R06855:tensor_diagram",
        ),
        "source_formulae": (
            "A -> B -> C = A -> C",
            "A -> A = A",
            "f tensor g: A tensor B -> C tensor D",
        ),
        "source_mechanisms": (
            "composition diagrams encode path contraction",
            "identity diagrams encode no-op morphisms",
            "tensor diagrams encode parallel transformations",
        ),
        "variables": ("A", "B", "C", "D", "f", "g"),
        "validation_targets": (
            "validate finite composition path contraction",
            "validate identity no-op behaviour",
            "preserve tensor as parallel-composition notation",
        ),
        "null_controls": (
            "broken-composition-path control must be rejected",
            "identity-as-state-change control must be rejected",
            "missing-tensor-arity control must be rejected",
        ),
    },
    "integration_synthesis.category_grammar.upde_category_application": {
        "validation_protocol": "paper0.integration_synthesis.category_grammar.upde_application",
        "canonical_statement": (
            "The UPDE, layer boundaries, and MS-QEC are recast as category-theory obligations."
        ),
        "source_equation_ids": (
            "P0R06858:upde_kan_eta",
            "P0R06860:layer_boundary_natural_transformations",
            "P0R06862:ms_qec_composition_coherence",
        ),
        "source_formulae": (
            "dtheta/dt = Lan_eta(Minimize Free Energy Functor)",
            "Natural transformations eta_i: L_i -> L_{i+1}",
            "MS-QEC: preservation of coherence under F o G composition",
        ),
        "source_mechanisms": (
            "UPDE emergence is expressed through Kan extension over eta",
            "layer boundaries are natural transformations",
            "MS-QEC is preservation under round-trip composition",
        ),
        "variables": ("theta", "Lan_eta", "eta_i", "F_o_G", "MS_QEC"),
        "validation_targets": (
            "preserve UPDE category-theory expression",
            "preserve layer-boundary natural-transformation role",
            "preserve MS-QEC as coherence-preservation obligation",
        ),
        "null_controls": (
            "missing-UPDE-category-expression control must be rejected",
            "missing-layer-boundary-natural-transformation control must be rejected",
            "missing-MS-QEC-composition control must be rejected",
        ),
    },
    "integration_synthesis.category_grammar.theorem_obligation_boundary": {
        "validation_protocol": "paper0.integration_synthesis.category_grammar.theorem_boundary",
        "canonical_statement": (
            "The universal-law prediction is bounded to proof obligations, not accepted as established theorem."
        ),
        "source_equation_ids": (
            "P0R06865:proof_obligations",
            "P0R06868:universal_category_law_prediction",
            "P0R06869:yoneda_and_adjoint_examples",
        ),
        "source_formulae": (
            "proof obligations for consistency",
            "SCPN dynamics obey universal laws of category theory",
            "examples include Yoneda lemma and adjoint functor theorem",
        ),
        "source_mechanisms": (
            "category grammar provides compositional semantics",
            "framework-independent validation remains a target",
            "automated theorem proving is a future path",
        ),
        "variables": ("Yoneda", "adjoint_functor_theorem", "proof_obligation"),
        "validation_targets": (
            "record Yoneda lemma as proof-obligation target",
            "record adjoint functor theorem as proof-obligation target",
            "reject theorem-complete claim without mechanised proof artefact",
        ),
        "null_controls": (
            "missing-proof-obligation-boundary control must be rejected",
            "theorem-complete-without-proof control must be rejected",
            "framework-independent-validation-as-completed control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class CategoryGrammarSpec:
    """Category grammar spec promoted from Paper 0 integration-synthesis records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    source_formulae: tuple[str, ...]
    source_mechanisms: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class CategoryGrammarSpecBundle:
    """Category grammar specs plus source coverage summary."""

    specs: tuple[CategoryGrammarSpec, ...]
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


def build_category_grammar_specs(
    source_records: list[dict[str, Any]],
) -> CategoryGrammarSpecBundle:
    """Build source-covered category grammar specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[CategoryGrammarSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            CategoryGrammarSpec(
                key=key,
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
                source_mechanisms=tuple(str(item) for item in metadata["source_mechanisms"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_executable_fixture",
                domain_review_status="promoted_to_validation_spec",
                hardware_status=HARDWARE_STATUS,
            )
        )

    summary = {
        "title": "Paper 0 Category Grammar Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "unconsumed_source_ledger_ids": [],
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_have_executable_targets": all(
            bool(spec.executable_validation_targets) for spec in specs
        ),
        "all_specs_are_source_anchored": all(
            spec.source_ledger_ids == SOURCE_LEDGER_IDS for spec in specs
        ),
    }
    return CategoryGrammarSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: CategoryGrammarSpecBundle) -> str:
    """Render a compact Markdown report for internal review."""
    lines = [
        "# Paper 0 Category Grammar Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records consumed: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                "",
                f"### {spec.key}",
                f"- Protocol: {spec.validation_protocol}",
                f"- Statement: {spec.canonical_statement}",
                f"- Source equations: {', '.join(spec.source_equation_ids) or 'none'}",
                f"- Validation targets: {len(spec.validation_targets)}",
                f"- Null controls: {len(spec.null_controls)}",
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    bundle: CategoryGrammarSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the category grammar specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_category_grammar_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_category_grammar_validation_specs_report_{date_tag}.md"
    payload = {
        "specs": [asdict(spec) for spec in bundle.specs],
        "summary": bundle.summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> int:
    """Build category grammar validation specs from the canonical review ledger."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_category_grammar_specs(load_jsonl(args.ledger))
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
