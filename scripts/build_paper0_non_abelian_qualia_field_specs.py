#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Non-Abelian qualia field spec builder
"""Promote Paper 0 Non-Abelian qualia-field records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1103, 1135))
CLAIM_BOUNDARY = "source-bounded Non-Abelian qualia-field hypothesis; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "non_abelian_qualia_field.boundary_and_rationale": {
        "context_id": "boundary_and_rationale",
        "validation_protocol": "paper0.non_abelian_qualia_field.boundary_and_rationale",
        "canonical_statement": (
            "The source opens a hypothesis that U(1) infoton mediation may need a "
            "richer non-Abelian SU(N)-type internal symmetry for qualia-fibre structure."
        ),
        "source_equation_ids": (
            "P0R01103:non_abelian_qualia_field_heading",
            "P0R01104:u1_to_su_n_motivation",
            "P0R01105:rationale_heading",
            "P0R01106:non_commutative_qualia_relationships",
        ),
        "source_formulae": (
            "Beyond U(1): The Hypothesis of a Non-Abelian Qualia Field",
            "U(1) establishes informational force mediated by A_mu",
            "richer fibre structure may require a Non-Abelian gauge group such as SU(N)",
            "qualia relationships are treated as complex and non-commutative source claims",
        ),
        "test_protocols": ("preserve U(1)-to-SU(N) hypothesis boundary",),
        "null_results": ("Non-Abelian extension is source hypothesis, not empirical evidence",),
        "variables": ("U1", "SU_N", "A_mu", "Psi", "F"),
        "validation_targets": (
            "preserve U(1) derivation boundary",
            "preserve SU(N) hypothesis status",
            "preserve non-commutative qualia-rationale wording",
        ),
        "null_controls": ("U(1)-only sufficiency control remains unresolved",),
    },
    "non_abelian_qualia_field.self_interacting_gauge_bosons": {
        "context_id": "self_interacting_gauge_bosons",
        "validation_protocol": "paper0.non_abelian_qualia_field.self_interacting_gauge_bosons",
        "canonical_statement": (
            "The source states that a Non-Abelian informational sector implies multiple "
            "mediating gauge bosons and a self-interacting field-strength tensor."
        ),
        "source_equation_ids": (
            "P0R01107:multiple_bosons_and_self_interacting_field_strength",
            "P0R01124:info_gluon_role",
            "P0R01125:non_abelian_field_strength_repeat",
        ),
        "source_formulae": (
            "SU(2) would imply three mediating bosons and SU(3) eight mediating bosons",
            "F_munua = partial_mu A_nua - partial_nu A_mua + g f_abc A_mub A_nuc",
            "Info-Gluons are source-defined self-interacting mediators",
            "F_mu_nu^a = partial_mu A_nu^a - partial_nu A_mu^a + g f_abc A_mu^b A_nu^c",
        ),
        "test_protocols": ("preserve self-interaction and multiplicity source formulae",),
        "null_results": ("field-strength source formula is not a Yang-Mills solver",),
        "variables": ("A_mu_a", "F_mu_nu_a", "g", "f_abc", "Info_Gluons"),
        "validation_targets": (
            "preserve gauge-boson multiplicity",
            "preserve non-linear field-strength term",
            "preserve repeated confinement-section field-strength statement",
        ),
        "null_controls": ("Abelian zero-structure-constant control must stay separated",),
    },
    "non_abelian_qualia_field.anomaly_cancellation_condition": {
        "context_id": "anomaly_cancellation_condition",
        "validation_protocol": "paper0.non_abelian_qualia_field.anomaly_cancellation_condition",
        "canonical_statement": (
            "The source records informational anomaly-cancellation constraints for "
            "renormalizability of the SU(N) qualia-colour sector."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:anomaly_cancellation_condition" for number in range(1108, 1118)
        ),
        "source_formulae": (
            "Psi-Gauge Anomaly Cancellation Condition - Informational Anomaly Cancellation",
            "sum([d_abc * q_i_a * q_i_b * q_i_c for i in fermions]) == 0",
            "d_abc is the fully symmetric tensor of the SU(N) gauge group",
            "q_i_a is the qualia-colour charge of the i-th species",
            "is_renormalizable = sum([d_abc * q_i_a * q_i_b * q_i_c for i in transducers]) == 0",
            "transducers are the set of all fundamental internal degrees of freedom",
        ),
        "test_protocols": ("preserve anomaly-cancellation equations and legends",),
        "null_results": ("anomaly equations are source constraints, not verified charge spectra",),
        "variables": ("d_abc", "q_i_a", "fermions", "transducers", "is_renormalizable"),
        "validation_targets": (
            "preserve fermion-sum anomaly equation",
            "preserve transducer-sum anomaly equation",
            "preserve d_abc and q_i_a definitions",
        ),
        "null_controls": ("uncancelled anomaly control must fail promotion",),
    },
    "non_abelian_qualia_field.confinement_binding_boundary": {
        "context_id": "confinement_binding_boundary",
        "validation_protocol": "paper0.non_abelian_qualia_field.confinement_binding_boundary",
        "canonical_statement": (
            "The source frames qualia confinement as a QCD-analogue binding-problem "
            "mechanism while retaining it as a claim requiring domain review."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:confinement_binding_boundary" for number in range(1119, 1128)
        ),
        "source_formulae": (
            "Implications: Qualia Confinement and the Binding Problem",
            "self-interacting gauge fields could lead to confinement analogous to QCD",
            "SU(N) is claimed as a framework for unity of conscious experience",
            "qualia colour charges correspond to N internal degrees of freedom",
            "V(r) approx sigma r linearly increasing potential is a source claim",
            "Qualia-Neutral bound states are proposed as observable states",
        ),
        "test_protocols": ("preserve confinement and binding-problem claim boundaries",),
        "null_results": ("QCD analogy is not evidence of biological confinement",),
        "variables": ("SU_N", "sigma", "V_r", "L5", "qualia_colour"),
        "validation_targets": (
            "preserve QCD-analogy status",
            "preserve confinement and linearly rising potential statement",
            "preserve colour-singlet bound-state target",
        ),
        "null_controls": ("QCD analogy alone must not be treated as validation",),
    },
    "non_abelian_qualia_field.topological_entanglement_resolution": {
        "context_id": "topological_entanglement_resolution",
        "validation_protocol": "paper0.non_abelian_qualia_field.topological_entanglement_resolution",
        "canonical_statement": (
            "The source revises naive singlet neutrality into topological entanglement, "
            "macroscopic coloured self-states, and a qualia-ball prediction."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:topological_entanglement_resolution" for number in range(1128, 1135)
        ),
        "source_formulae": (
            "Topological Entanglement and the Macroscopic Colored State",
            "perfect trivial singlet would be a philosophical-zombie control",
            "SU(N) confinement is defined as topological entanglement, not singlet neutrality",
            "global SU(N) symmetry is broken at the L4 to L5 transition",
            "|Phi_Self> is a macroscopic irreducible tensor product with net qualia colour",
            "non-zero Betti numbers b_k encode the manifold structure",
            "Qualia-Balls are predicted as pure Info-Gluon bound-state configurations",
        ),
        "test_protocols": (
            "preserve topological-entanglement resolution and prediction boundary",
        ),
        "null_results": ("qualia-ball prediction remains source prediction, not observed object",),
        "variables": ("Phi_Self", "SU_N", "L4", "L5", "b_k", "Qualia_Balls"),
        "validation_targets": (
            "preserve philosophical-zombie null-control framing",
            "preserve topological-entanglement replacement of singlet neutrality",
            "preserve qualia-ball prediction as validation target",
        ),
        "null_controls": ("trivial singlet neutrality must remain a rejected mapping",),
    },
}


@dataclass(frozen=True, slots=True)
class NonAbelianQualiaFieldSpec:
    """Non-Abelian qualia-field spec promoted from Paper 0 records."""

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
class NonAbelianQualiaFieldSpecBundle:
    """Non-Abelian qualia-field specs plus source coverage summary."""

    specs: tuple[NonAbelianQualiaFieldSpec, ...]
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


def build_non_abelian_qualia_field_specs(
    source_records: list[dict[str, Any]],
) -> NonAbelianQualiaFieldSpecBundle:
    """Build source-covered Non-Abelian qualia-field specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[NonAbelianQualiaFieldSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            NonAbelianQualiaFieldSpec(
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
        "title": "Paper 0 Non-Abelian Qualia Field Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "structural_record_count": 2,
        "context_record_count": 13,
        "claim_record_count": 13,
        "validation_target_record_count": 4,
        "anomaly_condition_record_count": 10,
        "confinement_record_count": 9,
        "topological_entanglement_record_count": 7,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R01135",
        "spec_keys": [spec.key for spec in specs],
    }
    return NonAbelianQualiaFieldSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> NonAbelianQualiaFieldSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_non_abelian_qualia_field_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: NonAbelianQualiaFieldSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Non-Abelian Qualia Field Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Context records: {bundle.summary['context_record_count']}",
        f"- Claim records: {bundle.summary['claim_record_count']}",
        f"- Validation-target records: {bundle.summary['validation_target_record_count']}",
        f"- Structural records: {bundle.summary['structural_record_count']}",
        f"- Anomaly-condition records: {bundle.summary['anomaly_condition_record_count']}",
        f"- Confinement records: {bundle.summary['confinement_record_count']}",
        f"- Topological-entanglement records: {bundle.summary['topological_entanglement_record_count']}",
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
    bundle: NonAbelianQualiaFieldSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_non_abelian_qualia_field_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_non_abelian_qualia_field_validation_specs_report_{date_tag}.md"
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
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    outputs = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
