#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Axiom I minimal Lagrangian spec builder
"""Promote Paper 0 Axiom I minimal Psi-field Lagrangian records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(733, 747))
EQUATION_RECORD_IDS = ("P0R00742", "P0R00743", "P0R00744", "P0R00745")
CLAIM_BOUNDARY = "source-bounded Axiom I minimal Lagrangian map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiom_i_minimal_lagrangian.purpose_and_criteria": {
        "context_id": "purpose_and_criteria",
        "validation_protocol": "paper0.axiom_i_minimal_lagrangian.purpose_and_criteria",
        "canonical_statement": (
            "The source restates Axiom I as requiring the least-assumptive physical "
            "realiser that carries spin-0, phase, and stable finite-energy structure."
        ),
        "source_equation_ids": (
            "P0R00733:minimal_lagrangian_header",
            "P0R00734:least_assumptive_physical_realiser",
            "P0R00735:spin0_single_irreducible_dof",
            "P0R00736:intentional_phase_variable",
            "P0R00737:organismal_soliton_phi_o",
        ),
        "source_formulae": (
            "fewest independent assumptions",
            "single irreducible degree of freedom spin-0",
            "intentional phase variable",
            "stable finite-energy structures",
            "organismal soliton Phi_O",
        ),
        "test_protocols": ("classify minimal Lagrangian purpose criteria",),
        "null_results": ("criteria are source requirements, not empirical validation",),
        "variables": ("Psi", "theta", "Phi_O", "soliton"),
        "validation_targets": (
            "preserve spin-0 criterion",
            "preserve intentional-phase criterion",
            "preserve organismal-soliton criterion",
        ),
        "null_controls": (
            "criterion-as-observation control must be rejected",
            "missing-criterion control must be rejected",
        ),
    },
    "axiom_i_minimal_lagrangian.traceable_model_class_boundary": {
        "context_id": "traceable_model_class_boundary",
        "validation_protocol": "paper0.axiom_i_minimal_lagrangian.traceable_boundary",
        "canonical_statement": (
            "The source marks the model class as revision-bounded, traceable, and "
            "explicit about rejected nearby alternatives."
        ),
        "source_equation_ids": (
            "P0R00738:model_class_satisfies_criteria_and_rejections",
            "P0R00739:living_research_project_traceability",
        ),
        "source_formulae": (
            "minimal model-class satisfies criteria (i)-(iii)",
            "nearby alternatives explicitly rejected",
            "manuscript revision 11.00",
            "living research project not ad hoc",
            "traceable thought chain",
        ),
        "test_protocols": ("preserve traceable model-class boundary",),
        "null_results": ("traceability is not model validation",),
        "variables": ("model_class", "revision", "alternatives"),
        "validation_targets": (
            "preserve revision boundary",
            "preserve rejected-alternative marker",
            "preserve traceability not-validation boundary",
        ),
        "null_controls": (
            "traceability-as-validation control must be rejected",
            "unbounded-revision control must be rejected",
        ),
    },
    "axiom_i_minimal_lagrangian.minimal_family_field_content": {
        "context_id": "minimal_family_field_content",
        "validation_protocol": "paper0.axiom_i_minimal_lagrangian.field_content",
        "canonical_statement": (
            "The source selects a complex scalar order parameter on spacetime, "
            "promotes U(1) from global to local, and introduces the infoton "
            "connection for informational coupling."
        ),
        "source_equation_ids": (
            "P0R00740:minimal_family_header",
            "P0R00741:complex_scalar_order_parameter",
            "P0R00741:local_u1_phase_agency",
            "P0R00741:infoton_connection",
            "P0R00741:lowest_dimension_terms",
        ),
        "source_formulae": (
            "complex scalar order parameter Psi on spacetime",
            "global U(1) promoted to local U(1)",
            "phase endowed with physical agency",
            "connection A_mu is the infoton",
            "kinetic potential and curvature are lowest-dimension operators",
        ),
        "test_protocols": ("classify minimal family field content",),
        "null_results": ("field-content statement is not observed field evidence",),
        "variables": ("Psi", "U1", "A_mu", "infoton"),
        "validation_targets": (
            "preserve complex-scalar field content",
            "preserve local-U1 promotion",
            "preserve infoton connection role",
        ),
        "null_controls": (
            "global-only-family control must be rejected",
            "infoton-as-detection control must be rejected",
        ),
    },
    "axiom_i_minimal_lagrangian.l_min_operator_terms": {
        "context_id": "l_min_operator_terms",
        "validation_protocol": "paper0.axiom_i_minimal_lagrangian.l_min_terms",
        "canonical_statement": (
            "The source records the minimal Lagrangian terms, including covariant "
            "Psi kinetic energy, potential, infoton field strength, and curvature "
            "coupling."
        ),
        "source_equation_ids": (
            "P0R00742:L_min_lagrangian_density",
            "P0R00743:covariant_derivative_definition",
            "P0R00743:field_strength_definition",
            "P0R00744:pulled_back_information_metric",
        ),
        "source_formulae": (
            "L_min = |D_mu Psi|^2 - V(|Psi|) - 1/4 g_F F F - xi R |Psi|^2",
            "D_mu = partial_mu - i g A_mu",
            "F_mu_nu = partial_mu A_nu - partial_nu A_mu",
            "pulled-back information metric for infoton dynamics",
            "nonminimal curvature coupling xi R |Psi|^2",
        ),
        "test_protocols": ("preserve minimal Lagrangian operator-term register",),
        "null_results": ("L_min formula is not an empirical fit",),
        "variables": ("L_min", "D_mu", "F_mu_nu", "R", "xi", "g_F"),
        "validation_targets": (
            "preserve covariant kinetic term",
            "preserve field strength definition",
            "preserve information metric role",
            "preserve curvature coupling",
        ),
        "null_controls": (
            "missing-curvature-term control must be rejected",
            "operator-register-as-fit control must be rejected",
        ),
    },
    "axiom_i_minimal_lagrangian.potential_and_ssb_boundary": {
        "context_id": "potential_and_ssb_boundary",
        "validation_protocol": "paper0.axiom_i_minimal_lagrangian.potential_ssb",
        "canonical_statement": (
            "The source records a quartic Mexican-hat potential and marks its "
            "boundedness and SSB role as source claims requiring downstream tests."
        ),
        "source_equation_ids": (
            "P0R00745:quartic_ssb_potential",
            "P0R00746:boundedness_and_ssb",
        ),
        "source_formulae": (
            "V(|Psi|) = -mu^2 |Psi|^2 + lambda |Psi|^4",
            "boundedness below and spontaneous symmetry breaking",
            "Mexican-hat potential",
        ),
        "test_protocols": ("preserve potential and SSB source boundary",),
        "null_results": ("SSB role is not validated by source registration alone",),
        "variables": ("V", "mu", "lambda", "Psi", "SSB"),
        "validation_targets": (
            "preserve quartic potential",
            "preserve boundedness-below role",
            "preserve SSB role",
        ),
        "null_controls": (
            "unbounded-potential control must be rejected",
            "SSB-as-observed-transition control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomIMinimalLagrangianSpec:
    """Axiom I minimal Lagrangian spec promoted from Paper 0 records."""

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
class AxiomIMinimalLagrangianSpecBundle:
    """Axiom I minimal Lagrangian specs plus source coverage summary."""

    specs: tuple[AxiomIMinimalLagrangianSpec, ...]
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


def build_axiom_i_minimal_lagrangian_specs(
    source_records: list[dict[str, Any]],
) -> AxiomIMinimalLagrangianSpecBundle:
    """Build source-covered Axiom I minimal Lagrangian specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomIMinimalLagrangianSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomIMinimalLagrangianSpec(
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
        "title": "Paper 0 Axiom I Minimal Lagrangian Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "minimal_criterion_count": 3,
        "equation_record_count": len(EQUATION_RECORD_IDS),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00747",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomIMinimalLagrangianSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AxiomIMinimalLagrangianSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiom_i_minimal_lagrangian_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomIMinimalLagrangianSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiom I Minimal Lagrangian Specs",
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
    bundle: AxiomIMinimalLagrangianSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_axiom_i_minimal_lagrangian_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_axiom_i_minimal_lagrangian_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Axiom I minimal-Lagrangian specs from the ledger."""

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
