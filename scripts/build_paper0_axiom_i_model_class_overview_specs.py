#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Axiom I model-class overview spec builder
"""Promote Paper 0 Axiom I model-class overview records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(703, 717))
BLANK_SEPARATOR_IDS = ("P0R00716",)
CLAIM_BOUNDARY = "source-bounded Axiom I model-class overview; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiom_i_model_class_overview.three_criteria": {
        "context_id": "three_criteria",
        "validation_protocol": "paper0.axiom_i_model_class_overview.three_criteria",
        "canonical_statement": (
            "The source bounds the minimal physical realiser of Axiom I by three "
            "criteria: spin-0 irreducibility, intentional phase, and stable "
            "finite-energy structures."
        ),
        "source_equation_ids": (
            "P0R00703:model_class_justification_header",
            "P0R00704:minimal_physical_realiser_criteria",
            "P0R00705:irreducible_spin0_degree",
            "P0R00706:intentional_phase_variable",
            "P0R00707:stable_finite_energy_solitons",
        ),
        "source_formulae": (
            "minimal physical realiser",
            "irreducible spin-0 degree of freedom",
            "intentional phase variable",
            "stable finite-energy structures",
            "persistent Self Phi_O",
        ),
        "test_protocols": ("classify the three source criteria",),
        "null_results": ("criteria are source requirements, not empirical validation",),
        "variables": ("Psi", "theta", "Phi_O", "soliton"),
        "validation_targets": (
            "preserve spin-0 criterion",
            "preserve intentional-phase criterion",
            "preserve stable-soliton criterion",
        ),
        "null_controls": (
            "two-criterion-collapse control must be rejected",
            "criterion-as-observation control must be rejected",
        ),
    },
    "axiom_i_model_class_overview.complex_scalar_local_u1_ssb": {
        "context_id": "complex_scalar_local_u1_ssb",
        "validation_protocol": "paper0.axiom_i_model_class_overview.complex_scalar_local_u1_ssb",
        "canonical_statement": (
            "The selected model family is a complex scalar field with local U(1) "
            "gauge symmetry and a Mexican-hat SSB potential."
        ),
        "source_equation_ids": (
            "P0R00708:complex_scalar_local_u1_ssb_choice",
            "P0R00708:amplitude_phase_decomposition",
            "P0R00708:local_u1_infoton_agency",
            "P0R00708:ssb_stability_requirement",
        ),
        "source_formulae": (
            "complex scalar field with local U(1) gauge symmetry",
            "Mexican hat potential for SSB",
            "minimal spin-0 carrier with amplitude-phase decomposition",
            "gauge boson infoton endows phase with physical agency",
        ),
        "test_protocols": ("classify selected model-family roles",),
        "null_results": ("model-family selection is not a measured field detection",),
        "variables": ("Psi", "U1", "A_mu", "infoton", "SSB"),
        "validation_targets": (
            "preserve complex-scalar model label",
            "preserve local U(1) gauge-agency role",
            "preserve SSB stability role",
        ),
        "null_controls": (
            "global-only-gauge control must be rejected",
            "SSB-as-measurement control must be rejected",
        ),
    },
    "axiom_i_model_class_overview.rejected_alternatives_and_predictions": {
        "context_id": "rejected_alternatives_and_predictions",
        "validation_protocol": (
            "paper0.axiom_i_model_class_overview.rejected_alternatives_and_predictions"
        ),
        "canonical_statement": (
            "The source rejects nearby model classes and marks the selected "
            "Lagrangian family as constrained and falsifiable rather than ad hoc."
        ),
        "source_equation_ids": (
            "P0R00709:real_scalar_rejected_lacks_phase",
            "P0R00709:global_only_rejected_long_range_forces",
            "P0R00709:vector_spinor_rejected_unnecessary_complexity",
            "P0R00709:falsifiable_predictions_risk_calibration",
        ),
        "source_formulae": (
            "real scalar field lacks phase",
            "global-only symmetry produces unobserved long-range forces",
            "vector or spinor fields are unnecessarily complex",
            "non-negotiable falsifiable predictions calibrate risk",
        ),
        "test_protocols": ("preserve rejected-alternative and falsifiability map",),
        "null_results": ("rejection rationale is not itself falsification data",),
        "variables": ("real_scalar", "global_U1", "vector", "spinor"),
        "validation_targets": (
            "preserve real-scalar rejection reason",
            "preserve global-only rejection reason",
            "preserve vector/spinor rejection reason",
            "preserve falsifiable-prediction boundary",
        ),
        "null_controls": (
            "unrejected-real-scalar control must be rejected",
            "prediction-as-observed-result control must be rejected",
        ),
    },
    "axiom_i_model_class_overview.pedagogical_three_job_restatement": {
        "context_id": "pedagogical_three_job_restatement",
        "validation_protocol": (
            "paper0.axiom_i_model_class_overview.pedagogical_three_job_restatement"
        ),
        "canonical_statement": (
            "The pedagogical restatement mirrors the three criteria as simple, "
            "intentional, and stable, while keeping metaphors outside model terms."
        ),
        "source_equation_ids": (
            "P0R00710:engineers_notes_three_jobs",
            "P0R00711:be_simple",
            "P0R00712:hold_intention_phase",
            "P0R00713:be_stable_knots_self",
            "P0R00714:complex_scalar_energy_landscape_restatement",
            "P0R00715:alternatives_and_lab_predictions_restatement",
            "P0R00716:blank_separator",
        ),
        "source_formulae": (
            "Be Simple",
            "Hold Intention",
            "Be Stable",
            "complex scalar field with specific energy landscape",
            "spontaneous symmetry breaking creates stable form",
            "concrete laboratory predictions",
        ),
        "test_protocols": ("preserve pedagogical restatement without upgrading metaphors",),
        "null_results": ("pedagogical metaphors are not mathematical model terms",),
        "variables": ("simplicity", "phase", "stability", "prediction"),
        "validation_targets": (
            "preserve simple-intentional-stable restatement",
            "preserve alternatives restatement",
            "preserve laboratory-prediction boundary",
            "preserve blank separator",
        ),
        "null_controls": (
            "metaphor-as-field-term control must be rejected",
            "blank-separator-as-content control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomIModelClassOverviewSpec:
    """Axiom I model-class overview spec promoted from Paper 0 records."""

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
class AxiomIModelClassOverviewSpecBundle:
    """Axiom I model-class overview specs plus source coverage summary."""

    specs: tuple[AxiomIModelClassOverviewSpec, ...]
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


def build_axiom_i_model_class_overview_specs(
    source_records: list[dict[str, Any]],
) -> AxiomIModelClassOverviewSpecBundle:
    """Build source-covered Axiom I model-class overview specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomIModelClassOverviewSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomIModelClassOverviewSpec(
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
        "title": "Paper 0 Axiom I Model-Class Overview Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "selection_criterion_count": 3,
        "blank_separator_count": len(BLANK_SEPARATOR_IDS),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00717",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomIModelClassOverviewSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AxiomIModelClassOverviewSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiom_i_model_class_overview_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomIModelClassOverviewSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiom I Model-Class Overview Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
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
    bundle: AxiomIModelClassOverviewSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_axiom_i_model_class_overview_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_axiom_i_model_class_overview_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Axiom I model-class overview specs from the ledger."""

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
