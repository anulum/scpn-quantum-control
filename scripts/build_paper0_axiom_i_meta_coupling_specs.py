#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Axiom I meta-coupling spec builder
"""Promote Paper 0 Axiom I meta-framework and Psi-s coupling records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(717, 733))
CLAIM_BOUNDARY = "source-bounded Axiom I meta-coupling map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiom_i_meta_coupling.predictive_coding_hardware": {
        "context_id": "predictive_coding_hardware",
        "validation_protocol": "paper0.axiom_i_meta_coupling.predictive_coding_hardware",
        "canonical_statement": (
            "The source maps the selected Lagrangian to the minimal hardware for a "
            "cosmic active-inference engine."
        ),
        "source_equation_ids": (
            "P0R00717:meta_framework_integrations",
            "P0R00718:predictive_coding_integration",
            "P0R00719:minimal_algorithmic_complexity",
            "P0R00720:lagrangian_as_simplest_hardware",
            "P0R00721:lagrangian_supports_active_inference",
            "P0R00722:spin0_minimal_processor_components",
            "P0R00723:phase_carrier_beliefs_priors",
            "P0R00724:stable_solitons_deep_prior_self",
        ),
        "source_formulae": (
            "minimal algorithmic complexity",
            "Lagrangian as simplest possible hardware",
            "Spin-0 minimal components",
            "phase theta carries beliefs or priors",
            "local U(1) updates belief by prediction error",
            "stable soliton as persistent deep-prior Self",
        ),
        "test_protocols": ("classify predictive-coding hardware roles",),
        "null_results": ("hardware analogy is not empirical validation",),
        "variables": ("theta", "U1", "prediction_error", "Self"),
        "validation_targets": (
            "preserve minimal-hardware mapping",
            "preserve phase-as-prior-carrier mapping",
            "preserve soliton-as-persistent-self mapping",
        ),
        "null_controls": (
            "hardware-analogy-as-measurement control must be rejected",
            "prediction-error-as-observed-update control must be rejected",
        ),
    },
    "axiom_i_meta_coupling.hint_component_justification": {
        "context_id": "hint_component_justification",
        "validation_protocol": "paper0.axiom_i_meta_coupling.hint_component_justification",
        "canonical_statement": (
            "The source states that the interaction Hamiltonian components require "
            "specific mathematical properties."
        ),
        "source_equation_ids": (
            "P0R00725:psis_field_coupling_integration",
            "P0R00726:H_int_component_justification",
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma",
            "specific mathematical form of interaction Hamiltonian terms",
            "components must have source-defined properties",
        ),
        "test_protocols": ("preserve H_int component-justification boundary",),
        "null_results": ("component justification is not a fitted Hamiltonian result",),
        "variables": ("H_int", "lambda", "Psi_s", "sigma"),
        "validation_targets": (
            "preserve H_int formula label",
            "preserve component-property boundary",
            "preserve no-fit boundary",
        ),
        "null_controls": (
            "H_int-as-fit control must be rejected",
            "missing-component-property control must be rejected",
        ),
    },
    "axiom_i_meta_coupling.psis_complex_scalar_requirement": {
        "context_id": "psis_complex_scalar_requirement",
        "validation_protocol": "paper0.axiom_i_meta_coupling.psis_complex_scalar_requirement",
        "canonical_statement": (
            "The source requires Psi_s to be a complex scalar because stability and "
            "intentionality require phase without unnecessary Lorentz structure."
        ),
        "source_equation_ids": (
            "P0R00727:why_psis_complex_scalar_header",
            "P0R00728:psis_complex_scalar_requirement",
        ),
        "source_formulae": (
            "Psi_s must be a complex scalar",
            "real scalar lacks phase necessary for intentionality",
            "vector or spinor adds unnecessary structure",
            "choice is a requirement, not arbitrary",
        ),
        "test_protocols": ("classify Psi_s complex-scalar requirement",),
        "null_results": ("requirement statement is not direct field detection",),
        "variables": ("Psi_s", "phase", "real_scalar", "spinor"),
        "validation_targets": (
            "preserve real-scalar rejection reason",
            "preserve vector/spinor rejection reason",
            "preserve necessity-language boundary",
        ),
        "null_controls": (
            "real-scalar-suffices control must be rejected",
            "necessity-as-observation control must be rejected",
        ),
    },
    "axiom_i_meta_coupling.gauge_interaction_requirement": {
        "context_id": "gauge_interaction_requirement",
        "validation_protocol": "paper0.axiom_i_meta_coupling.gauge_interaction_requirement",
        "canonical_statement": (
            "The source requires local gauge interaction to avoid unobserved "
            "long-range forces and mediate H_int through the infoton."
        ),
        "source_equation_ids": (
            "P0R00729:why_gauge_interaction_header",
            "P0R00730:gauge_interaction_requirement",
        ),
        "source_formulae": (
            "global U(1) only is rejected",
            "interaction must be local and well behaved",
            "avoid unobserved long-range forces",
            "gauge boson infoton mediates H_int",
        ),
        "test_protocols": ("classify local gauge-interaction requirement",),
        "null_results": ("gauge requirement is not a measured mediator detection",),
        "variables": ("U1", "A_mu", "infoton", "H_int"),
        "validation_targets": (
            "preserve global-only rejection",
            "preserve local-interaction criterion",
            "preserve infoton-mediation label",
        ),
        "null_controls": (
            "global-only-suffices control must be rejected",
            "infoton-as-detected-particle control must be rejected",
        ),
    },
    "axiom_i_meta_coupling.sigma_q_ball_requirement": {
        "context_id": "sigma_q_ball_requirement",
        "validation_protocol": "paper0.axiom_i_meta_coupling.sigma_q_ball_requirement",
        "canonical_statement": (
            "The source requires sigma to be a stable charge-supported soliton, "
            "with SSB and conserved Psi-charge supporting persistence."
        ),
        "source_equation_ids": (
            "P0R00731:why_sigma_soliton_header",
            "P0R00732:sigma_charge_supported_q_ball_requirement",
        ),
        "source_formulae": (
            "sigma must be a stable charge-supported soliton",
            "Q-ball",
            "SSB mechanism and conserved Psi-charge",
            "sigma persists over time",
            "Lagrangian necessity dictated by conscious-universe requirements",
        ),
        "test_protocols": ("classify sigma soliton requirement",),
        "null_results": ("sigma Q-ball requirement is not observed soliton evidence",),
        "variables": ("sigma", "Q_ball", "Psi_charge", "SSB"),
        "validation_targets": (
            "preserve sigma-as-soliton boundary",
            "preserve Q-ball label",
            "preserve conserved-charge persistence role",
        ),
        "null_controls": (
            "unstable-sigma control must be rejected",
            "Q-ball-as-observed-object control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomIMetaCouplingSpec:
    """Axiom I meta-coupling spec promoted from Paper 0 records."""

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
class AxiomIMetaCouplingSpecBundle:
    """Axiom I meta-coupling specs plus source coverage summary."""

    specs: tuple[AxiomIMetaCouplingSpec, ...]
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


def build_axiom_i_meta_coupling_specs(
    source_records: list[dict[str, Any]],
) -> AxiomIMetaCouplingSpecBundle:
    """Build source-covered Axiom I meta-coupling specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomIMetaCouplingSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomIMetaCouplingSpec(
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
        "title": "Paper 0 Axiom I Meta-Coupling Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "interaction_component_count": 3,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00733",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomIMetaCouplingSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AxiomIMetaCouplingSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiom_i_meta_coupling_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomIMetaCouplingSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiom I Meta-Coupling Specs",
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
    bundle: AxiomIMetaCouplingSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_axiom_i_meta_coupling_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_axiom_i_meta_coupling_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Axiom I meta-coupling specs from the ledger."""

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
