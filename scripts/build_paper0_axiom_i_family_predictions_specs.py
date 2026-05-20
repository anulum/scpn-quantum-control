#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Axiom I family predictions spec builder
"""Promote Paper 0 Axiom I family-satisfaction and prediction records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(747, 757))
BLANK_SEPARATOR_IDS = ("P0R00755",)
CLAIM_BOUNDARY = "source-bounded Axiom I family-predictions map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiom_i_family_predictions.criteria_satisfaction": {
        "context_id": "criteria_satisfaction",
        "validation_protocol": "paper0.axiom_i_family_predictions.criteria_satisfaction",
        "canonical_statement": (
            "The source argues that the complex scalar local-U(1) family satisfies "
            "the Axiom I requirements because it is spin-0, phase-carrying, "
            "charge-conserving, and compatible with finite-energy organism-scale "
            "condensates."
        ),
        "source_equation_ids": (
            "P0R00747:family_satisfies_criteria_header",
            "P0R00748:spin0_complex_scalar",
            "P0R00748:amplitude_phase_decomposition",
            "P0R00748:local_u1_noether_current",
            "P0R00748:q_ball_like_condensates",
            "P0R00748:curvature_term_retained",
        ),
        "source_formulae": (
            "complex scalar minimal spin-0",
            "amplitude-phase decomposition Psi = rho exp(i theta)",
            "global U(1) makes theta a state variable",
            "local U(1) upgrades phase to a dynamical degree of freedom",
            "conserved Noether current couples to A_mu",
            "Mexican-hat potential and U(1) charge support Q-ball-like condensates",
            "organism-scale field configurations Phi_O",
            "optional non-minimal curvature term -xi R |Psi|^2",
        ),
        "test_protocols": ("preserve family-satisfaction source accounting",),
        "null_results": ("family satisfaction is a model-class rationale, not observation",),
        "variables": ("Psi", "rho", "theta", "A_mu", "Phi_O", "xi", "R"),
        "validation_targets": (
            "preserve spin-0 minimality rationale",
            "preserve phase-as-state-variable rationale",
            "preserve local-U1 Noether-current role",
            "preserve Q-ball-like condensate mechanism",
        ),
        "null_controls": (
            "criteria-as-observed-result control must be rejected",
            "missing-local-U1-control must be rejected",
        ),
    },
    "axiom_i_family_predictions.rejected_model_classes": {
        "context_id": "rejected_model_classes",
        "validation_protocol": "paper0.axiom_i_family_predictions.rejected_model_classes",
        "canonical_statement": (
            "The source records nearby model classes that were considered and "
            "rejected at the minimal tier because they lack the required phase, "
            "charge, locality, scalar-order-parameter, or parsimony properties."
        ),
        "source_equation_ids": (
            "P0R00749:nearby_model_classes_rejected_header",
            "P0R00750:real_scalar_rejection",
            "P0R00750:global_u1_only_rejection",
            "P0R00750:vector_tensor_primary_rejection",
            "P0R00750:spinor_primary_rejection",
            "P0R00750:non_abelian_minimal_tier_rejection",
        ),
        "source_formulae": (
            "real scalar lacks internal phase",
            "real scalar cannot carry conserved charge for intentional control",
            "real scalar cannot support charge-stabilised solitons",
            "global U(1) only leaves a massless Goldstone mode",
            "global U(1) only fails phase-mediated locality",
            "vector/tensor primaries add unnecessary Lorentz structure",
            "spinor phases are tied to spin representation",
            "spinor primaries lack a universal intentional fibre",
            "non-Abelian base rejected at the minimal tier for now",
            "non-Abelian base increases field content and self-interactions",
        ),
        "test_protocols": ("classify rejected minimal-tier alternatives",),
        "null_results": (
            "rejection catalogue is source-bound model selection, not exclusion data",
        ),
        "variables": ("real_scalar", "global_U1", "vector", "tensor", "spinor", "SU_N"),
        "validation_targets": (
            "preserve real-scalar rejection",
            "preserve global-U1-only rejection",
            "preserve vector/tensor rejection",
            "preserve spinor rejection",
            "preserve non-Abelian minimal-tier deferral",
        ),
        "null_controls": (
            "rejected-class-as-empirical-exclusion control must be rejected",
            "non-Abelian-selected-at-minimal-tier control must be rejected",
        ),
    },
    "axiom_i_family_predictions.conditional_predictions": {
        "context_id": "conditional_predictions",
        "validation_protocol": "paper0.axiom_i_family_predictions.conditional_predictions",
        "canonical_statement": (
            "The source marks three conditional predictions owed by the model "
            "class: a conserved Psi charge, a massive infoton after symmetry "
            "breaking, and a radial Psi-Higgs amplitude excitation."
        ),
        "source_equation_ids": (
            "P0R00751:conditional_predictions_header",
            "P0R00752:model_class_predictions_calibrate_risk",
            "P0R00753:conserved_psi_charge",
            "P0R00753:massive_infoton",
            "P0R00753:psi_higgs_radial_excitation",
        ),
        "source_formulae": (
            "conditional predictions are owed by the model-class",
            "predictions calibrate risk",
            "conserved Psi-charge Noether current",
            "Q_Psi stabilises Phi_O condensates",
            "massive infoton m_A = g v",
            "short-ranged and organism-centred influence",
            "Psi-Higgs amplitude mode eta above v",
            "massive spin-0 excitation with couplings fixed by lambda and v",
            "non-observation prunes parameter volume",
        ),
        "test_protocols": ("classify conditional model-class predictions",),
        "null_results": ("conditional predictions are not observed hardware results",),
        "variables": ("Q_Psi", "m_A", "g", "v", "eta", "lambda", "Phi_O"),
        "validation_targets": (
            "preserve conserved Psi-charge prediction",
            "preserve massive-infoton prediction",
            "preserve radial-excitation prediction",
        ),
        "null_controls": (
            "prediction-as-observation control must be rejected",
            "missing-parameter-volume-risk control must be rejected",
        ),
    },
    "axiom_i_family_predictions.decision_rule": {
        "context_id": "decision_rule",
        "validation_protocol": "paper0.axiom_i_family_predictions.decision_rule",
        "canonical_statement": (
            "The source sets a decision rule: remain with the least-assumptive, "
            "phase-competent, soliton-supporting complex-scalar local-U(1) family "
            "until contrary data require model-class escalation or replacement."
        ),
        "source_equation_ids": (
            "P0R00754:decision_rule",
            "P0R00755:blank_separator",
        ),
        "source_formulae": (
            "least-assumptive phase-competent soliton-supporting realiser",
            "evidence against predictions triggers model-class escalation or replacement",
            "contrary evidence requires recorded justification",
            "blank separator before SU(N) extension boundary",
        ),
        "test_protocols": ("preserve source decision rule and blank separator accounting",),
        "null_results": ("decision rule is governance of validation, not validation itself",),
        "variables": ("model_class", "evidence", "escalation", "replacement"),
        "validation_targets": (
            "preserve least-assumptive-family default",
            "preserve contrary-evidence escalation rule",
            "preserve blank separator accounting",
        ),
        "null_controls": (
            "decision-rule-as-model-proof control must be rejected",
            "unrecorded-escalation control must be rejected",
        ),
    },
    "axiom_i_family_predictions.su_n_qualia_confinement_boundary": {
        "context_id": "su_n_qualia_confinement_boundary",
        "validation_protocol": "paper0.axiom_i_family_predictions.su_n_boundary",
        "canonical_statement": (
            "The source opens the SU(N) Qualia Confinement extension at revision "
            "11.50; this slice records only the boundary marker and does not "
            "promote SU(N) as the selected minimal model."
        ),
        "source_equation_ids": ("P0R00756:su_n_qualia_confinement_header",),
        "source_formulae": (
            "Extension to SU(N) Qualia Confinement",
            "Revision 11.50",
            "SU(N) extension header only",
            "not promoted as selected minimal model in this slice",
        ),
        "test_protocols": ("preserve next-section boundary without overclaiming",),
        "null_results": ("SU(N) header is not SU(N) validation evidence",),
        "variables": ("SU_N", "qualia_confinement", "revision_11_50"),
        "validation_targets": (
            "preserve SU(N) boundary marker",
            "preserve not-selected-minimal-model boundary",
        ),
        "null_controls": (
            "SU-N-selected-by-header control must be rejected",
            "boundary-as-evidence control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomIFamilyPredictionsSpec:
    """Axiom I family-predictions spec promoted from Paper 0 records."""

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
class AxiomIFamilyPredictionsSpecBundle:
    """Axiom I family-predictions specs plus source coverage summary."""

    specs: tuple[AxiomIFamilyPredictionsSpec, ...]
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


def build_axiom_i_family_predictions_specs(
    source_records: list[dict[str, Any]],
) -> AxiomIFamilyPredictionsSpecBundle:
    """Build source-covered Axiom I family-predictions specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomIFamilyPredictionsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomIFamilyPredictionsSpec(
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
        "title": "Paper 0 Axiom I Family Predictions Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "conditional_prediction_count": 3,
        "rejected_model_class_count": 5,
        "blank_separator_count": len(BLANK_SEPARATOR_IDS),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00757",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomIFamilyPredictionsSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AxiomIFamilyPredictionsSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiom_i_family_predictions_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomIFamilyPredictionsSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiom I Family Predictions Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Conditional predictions: {bundle.summary['conditional_prediction_count']}",
        f"- Rejected model classes: {bundle.summary['rejected_model_class_count']}",
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
    lines.extend(
        [
            "## Internal Review",
            "",
            "- Source-accounting reviewer: Noether",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    bundle: AxiomIFamilyPredictionsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_axiom_i_family_predictions_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_axiom_i_family_predictions_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Axiom I family-prediction specs from the ledger."""

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
