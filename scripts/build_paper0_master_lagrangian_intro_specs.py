#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 master Lagrangian intro spec builder
"""Promote Paper 0 master-interaction-Lagrangian introduction records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(987, 1018))
CLAIM_BOUNDARY = "source-bounded master-Lagrangian introduction; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "master_lagrangian_intro.part_ii_boundary": {
        "context_id": "part_ii_boundary",
        "validation_protocol": "paper0.master_lagrangian_intro.part_ii_boundary",
        "canonical_statement": (
            "The source opens Part II and Section 2.1, establishing the "
            "physical-sector and master-interaction-Lagrangian derivation "
            "boundary before the detailed gauge-principle subsection."
        ),
        "source_equation_ids": (
            "P0R00987:part_ii_physical_sector_heading",
            "P0R00988:blank_record",
            "P0R00989:section_2_1_master_interaction_lagrangian_heading",
            "P0R00990:introduction_heading",
            "P0R01018:next_gauge_principle_derivation_boundary",
        ),
        "source_formulae": (
            "Part II: The Physical Sector (Field Theory & Quantization)",
            "2.1 Master Interaction Lagrangian: Derivation from First Principles",
            "Introduction",
            "next boundary is P0R01018 gauge-principle derivation",
        ),
        "test_protocols": ("preserve Part II and Section 2.1 source boundary",),
        "null_results": ("section boundary is source context, not validation evidence",),
        "variables": ("L_Int", "Psi", "A_mu"),
        "validation_targets": (
            "preserve Part II boundary",
            "preserve Section 2.1 boundary",
            "preserve next gauge-principle boundary",
        ),
        "null_controls": ("physical-sector boundary drift control must be rejected",),
    },
    "master_lagrangian_intro.first_principles_framing": {
        "context_id": "first_principles_framing",
        "validation_protocol": "paper0.master_lagrangian_intro.first_principles_framing",
        "canonical_statement": (
            "The source frames the Master Interaction Lagrangian as moving "
            "from phenomenological postulate to first-principles derivation, "
            "with the gauge principle named as the methodological constraint."
        ),
        "source_equation_ids": (
            "P0R00991:phenomenological_to_first_principles",
            "P0R00994:derived_l_int_prime_summary",
            "P0R00997:symmetry_method_context",
            "P0R01000:master_interaction_equation_justification_context",
        ),
        "source_formulae": (
            "phenomenological postulate to direct consequence of first principles",
            "gauge principle as the only sound basis for a fundamental interaction",
            "L_Int' is presented as predictive constrained explanatory foundation",
            "every piece justified derived and locked by modern physics principles",
        ),
        "test_protocols": ("preserve first-principles framing without proof inflation",),
        "null_results": ("first-principles language is not itself a proof fixture",),
        "variables": ("L_Int", "L_Int_prime", "gauge_principle"),
        "validation_targets": (
            "preserve phenomenology-to-derivation claim",
            "preserve L_Int_prime source boundary",
        ),
        "null_controls": ("first-principles-wording-as-proof control must be rejected",),
    },
    "master_lagrangian_intro.two_stream_derivation": {
        "context_id": "two_stream_derivation",
        "validation_protocol": "paper0.master_lagrangian_intro.two_stream_derivation",
        "canonical_statement": (
            "The source introduces two derivation streams: local U(1) gauge "
            "invariance for the informational Lagrangian and curved-spacetime "
            "consistency for the geometric Lagrangian."
        ),
        "source_equation_ids": (
            "P0R00992:u1_gauge_informational_stream",
            "P0R00993:curved_spacetime_geometric_stream",
            "P0R00998:informational_force_analogy",
            "P0R00999:gravity_interface_analogy",
        ),
        "source_formulae": (
            "local U(1) gauge invariance of a free complex scalar Psi-field",
            "covariant derivative",
            "spin-1 gauge field infoton A_mu",
            "FIM kinetic term for the infoton",
            "-xi R Psi*Psi non-minimal coupling",
            "conformal invariance in the massless limit",
            "informational force and infoton required to preserve symmetry",
            "gravity interface required for consistency at high energies",
        ),
        "test_protocols": ("preserve two-stream derivation claims",),
        "null_results": ("two-stream derivation is source claim pending detailed fixtures",),
        "variables": ("Psi", "A_mu", "FIM", "xi", "R", "g"),
        "validation_targets": (
            "preserve U(1) informational stream",
            "preserve FIM kinetic-term claim",
            "preserve non-minimal curvature-coupling claim",
        ),
        "null_controls": (
            "U1-stream omission control must be rejected",
            "curvature-stream omission control must be rejected",
        ),
    },
    "master_lagrangian_intro.explanatory_analogies": {
        "context_id": "explanatory_analogies",
        "validation_protocol": "paper0.master_lagrangian_intro.explanatory_analogies",
        "canonical_statement": (
            "The source includes lay framing and analogy records explaining "
            "the movement from concept sketch to derivation; these are "
            "preserved as context and not treated as evidence."
        ),
        "source_equation_ids": (
            "P0R00995:main_event_context",
            "P0R00996:car_engine_analogy",
            "P0R00997:symmetry_toolbox_context",
            "P0R00998:informational_force_explanation",
            "P0R00999:gravity_interface_explanation",
            "P0R01000:master_interaction_equation_explanation",
            "P0R00988:blank_record",
            "P0R01001:blank_record",
        ),
        "source_formulae": (
            "concept sketch to mathematical necessity is source framing",
            "symmetry is the physicist toolbox framing",
            "informational force analogy is context",
            "gravity interface analogy is context",
            "P0R00988 and P0R01001 are blank records",
        ),
        "test_protocols": ("preserve analogy records without evidentiary promotion",),
        "null_results": ("analogy records are context, not validation evidence",),
        "variables": ("symmetry", "informational_force", "gravity_interface"),
        "validation_targets": (
            "preserve explanatory context records",
            "preserve blank records P0R00988 and P0R01001",
        ),
        "null_controls": ("analogy-as-proof control must be rejected",),
    },
    "master_lagrangian_intro.gauge_inference_integration": {
        "context_id": "gauge_inference_integration",
        "validation_protocol": "paper0.master_lagrangian_intro.gauge_inference_integration",
        "canonical_statement": (
            "The source maps gauge invariance to inference coherence and maps "
            "the infoton gauge field to a prediction-error signal."
        ),
        "source_equation_ids": (
            "P0R01002:meta_framework_integrations_heading",
            "P0R01003:predictive_coding_integration_heading",
            "P0R01004:gauge_principle_reason_for_inference_structure",
            "P0R01005:gauge_invariance_prerequisite_heading",
            "P0R01006:gauge_invariance_prerequisite_for_inference",
            "P0R01007:infoton_prediction_error_heading",
            "P0R01008:infoton_prediction_error_signal",
        ),
        "source_formulae": (
            "gauge principle derivation provides physical reason for inference engine structure",
            "gauge invariance as prerequisite for inference",
            "phase of the Psi-field is belief",
            "infoton as prediction-error signal",
            "derived Lagrangian governs generation propagation and reception of inference signals",
        ),
        "test_protocols": ("preserve gauge/inference integration source claims",),
        "null_results": ("gauge-inference mapping is source claim, not measured inference",),
        "variables": ("Psi", "phase", "A_mu", "infoton", "prediction_error"),
        "validation_targets": (
            "preserve gauge invariance as inference prerequisite",
            "preserve infoton prediction-error mapping",
        ),
        "null_controls": (
            "belief-phase omission control must be rejected",
            "infoton-prediction-error omission control must be rejected",
        ),
    },
    "master_lagrangian_intro.psis_coupling_gauge_interpretation": {
        "context_id": "psis_coupling_gauge_interpretation",
        "validation_protocol": (
            "paper0.master_lagrangian_intro.psis_coupling_gauge_interpretation"
        ),
        "canonical_statement": (
            "The source reinterprets H_int = -lambda * Psis * sigma as a U(1) "
            "gauge interaction with Psis as complex scalar Psi, sigma as the "
            "Noether current, lambda as gauge coupling g, and A_mu as mediator."
        ),
        "source_equation_ids": (
            "P0R01009:psis_field_coupling_heading",
            "P0R01010:h_int_first_principles_derivation_context",
            "P0R01011:h_int_u1_gauge_interaction",
            "P0R01012:psis_complex_scalar_psi",
            "P0R01013:sigma_noether_current",
            "P0R01014:lambda_gauge_coupling_g",
            "P0R01015:infoton_mediator_a_mu",
            "P0R01016:two_faces_of_coupling_heading",
            "P0R01017:informational_and_geometric_coupling_unification",
        ),
        "source_formulae": (
            "H_int = -lambda * Psis * sigma",
            "H_int is a U(1) Gauge Interaction",
            "Psis is the complex scalar field Psi",
            "J_mu = i(Psi* partial_mu Psi - Psi partial_mu Psi*)",
            "lambda is the fundamental gauge coupling constant g",
            "A_mu mediates the interaction as infoton gauge boson",
            "Informational Coupling is the direct gauge interaction",
            "Geometric Coupling is a curved-spacetime consequence",
            "next boundary is P0R01018 gauge-principle derivation",
        ),
        "test_protocols": ("preserve H_int gauge reinterpretation source claims",),
        "null_results": (
            "H_int gauge reinterpretation still requires detailed derivation checks",
        ),
        "variables": ("H_int", "lambda", "Psis", "sigma", "Psi", "J_mu", "g", "A_mu"),
        "validation_targets": (
            "preserve H_int source equation",
            "preserve Noether-current sigma mapping",
            "preserve lambda equals g mapping",
            "preserve dual-coupling unification claim",
        ),
        "null_controls": (
            "Noether-current omission control must be rejected",
            "lambda-g mapping omission control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class MasterLagrangianIntroSpec:
    """Master-Lagrangian-intro spec promoted from Paper 0 records."""

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
class MasterLagrangianIntroSpecBundle:
    """Master-Lagrangian-intro specs plus source coverage summary."""

    specs: tuple[MasterLagrangianIntroSpec, ...]
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


def build_master_lagrangian_intro_specs(
    source_records: list[dict[str, Any]],
) -> MasterLagrangianIntroSpecBundle:
    """Build source-covered master-Lagrangian-intro specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[MasterLagrangianIntroSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            MasterLagrangianIntroSpec(
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
        "title": "Paper 0 Master Lagrangian Intro Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "blank_record_count": 2,
        "introduction_record_count": 13,
        "meta_framework_record_count": 16,
        "gauge_inference_record_count": 6,
        "psis_coupling_record_count": 9,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R01018",
        "spec_keys": [spec.key for spec in specs],
    }
    return MasterLagrangianIntroSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> MasterLagrangianIntroSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_master_lagrangian_intro_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: MasterLagrangianIntroSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Master Lagrangian Intro Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Introduction records: {bundle.summary['introduction_record_count']}",
        f"- Meta-framework records: {bundle.summary['meta_framework_record_count']}",
        f"- Gauge-inference records: {bundle.summary['gauge_inference_record_count']}",
        f"- Psis-coupling records: {bundle.summary['psis_coupling_record_count']}",
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
    bundle: MasterLagrangianIntroSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_master_lagrangian_intro_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_master_lagrangian_intro_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
