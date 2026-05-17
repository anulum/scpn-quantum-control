#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 derived Lagrangian detail spec builder
"""Promote Paper 0 derived Master Interaction Lagrangian detail records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1422, 1510))
CLAIM_BOUNDARY = "source-bounded derived Master Interaction Lagrangian detail; not experimental validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "derived_lagrangian_detail.derived_lint_split": {
        "context_id": "derived_lint_split",
        "validation_protocol": "paper0.derived_lagrangian_detail.derived_lint_split",
        "canonical_statement": (
            "The source replaces the phenomenological Master Interaction Lagrangian with L_Int' as a derived "
            "sum of informational and geometric components."
        ),
        "source_equation_ids": tuple(f"P0R{n:05d}:derived_lint_split" for n in range(1422, 1428)),
        "source_formulae": (
            "The Master Interaction Lagrangian (Derived from First Principles)",
            "L_Int' = L_Informational' + L_Geometric'",
            "informational component contains U(1) interaction, Psi-A_mu terms, and FIM-governed gauge kinetic term",
            "geometric component contains non-minimal curvature coupling with parameter xi",
        ),
        "test_protocols": ("preserve derived L_Int split",),
        "null_results": ("derived source formula is not empirical validation",),
        "variables": (
            "L_Int_prime",
            "L_Informational_prime",
            "L_Geometric_prime",
            "Psi",
            "A_mu",
            "xi",
        ),
        "validation_targets": (
            "preserve two-component split",
            "preserve derived-vs-phenomenological replacement",
        ),
        "null_controls": (
            "phenomenological L_Int split must not satisfy derived-prime accounting",
        ),
    },
    "derived_lagrangian_detail.informational_lagrangian_fim_kinetics": {
        "context_id": "informational_lagrangian_fim_kinetics",
        "validation_protocol": "paper0.derived_lagrangian_detail.informational_lagrangian_fim_kinetics",
        "canonical_statement": (
            "The source defines L_Informational' with U(1) current coupling, A_mu mass-like term, and gauge kinetic "
            "term contracted with pulled-back Fisher metric factors."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:informational_lagrangian_fim_kinetics" for n in range(1427, 1435)
        ),
        "source_formulae": (
            "L_Informational' = i g A_mu(Psi* partial_mu Psi - Psi partial_mu Psi*) + g^2 A_mu A^mu Psi* Psi - 1/4 g_tilde_F^{mu alpha} g_tilde_F^{nu beta} F_mu_nu F_alpha_beta",
            "pi: (M,g) -> (Theta,g_F) embeds spacetime into the statistical manifold",
            "g_tilde_F^{mu nu} = Lambda_I^-2 (pi* g_F)^{mu nu}",
            "A_mu is the infoton connection on M",
            "FIM anisotropies are communicated to gauge dynamics without index or unit mismatch",
        ),
        "test_protocols": ("preserve informational Lagrangian and FIM kinetic form",),
        "null_results": ("FIM kinetic expression is not a calibrated field measurement",),
        "variables": ("A_mu", "Psi", "g", "g_tilde_F", "F_mu_nu", "Lambda_I", "Theta", "M"),
        "validation_targets": (
            "preserve U1 current coupling",
            "preserve FIM kinetic contraction",
            "preserve dimensional normalisation",
        ),
        "null_controls": ("spacetime-metric contraction must not satisfy FIM kinetic boundary",),
    },
    "derived_lagrangian_detail.operational_pullback_protocol": {
        "context_id": "operational_pullback_protocol",
        "validation_protocol": "paper0.derived_lagrangian_detail.operational_pullback_protocol",
        "canonical_statement": (
            "The source restates the operational pullback protocol: statistical bundle, observable-tied theta(x), "
            "FIM on fibres, spacetime pullback, and gauge kinetic term using only the pulled-back information metric."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:operational_pullback_protocol" for n in range(1435, 1454)
        ),
        "source_formulae": (
            "pi: Theta -> M is a statistical fibre bundle over spacetime M",
            "theta: M -> Theta with theta(x) in Theta_x indexes probability models p(y|x,theta(x))",
            "I_ij(theta) = E_p(y|x,theta)[partial_i log p partial_j log p]",
            "g_F_mu_nu(x) = partial_mu theta^i(x) I_ij(theta(x)) partial_nu theta^j(x)",
            "g_tilde_F^{mu nu}(x) = Lambda_I^-2 (g_F^-1)^{mu nu}(x)",
            "L_gauge = -1/4 g_tilde_F^{mu alpha} g_tilde_F^{nu beta} F_mu_nu F_alpha_beta",
            "gauge kinetic term is never contracted with spacetime metric g_mu_nu in this term",
        ),
        "test_protocols": ("preserve operational pullback protocol equations",),
        "null_results": ("pullback protocol remains source protocol, not experimental evidence",),
        "variables": (
            "pi",
            "Theta",
            "M",
            "theta",
            "I_ij",
            "g_F",
            "g_tilde_F",
            "Lambda_I",
            "F_mu_nu",
        ),
        "validation_targets": (
            "preserve statistical bundle",
            "preserve FIM pullback",
            "preserve gauge kinetic non-mixing rule",
        ),
        "null_controls": (
            "domain-mixed spacetime metric contraction must fail protocol boundary",
        ),
    },
    "derived_lagrangian_detail.observable_l4_l5_prediction": {
        "context_id": "observable_l4_l5_prediction",
        "validation_protocol": "paper0.derived_lagrangian_detail.observable_l4_l5_prediction",
        "canonical_statement": (
            "The source gives observable-tied L5/L11 sections and an L4/L5 neural coding-efficiency case that "
            "predicts stronger Psi-field/infoton coupling where information density is maximised."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:observable_l4_l5_prediction" for n in range(1454, 1465)
        ),
        "source_formulae": (
            "L5 theta parameterises ensemble posteriors for neural latents",
            "L11 theta parameterises population-level phase-density models for interacting agents",
            "coding efficiency is equivalent to maximising det(I(theta))",
            "theta(t) dynamics change the FIM and the pullback metric g_F_mu_nu(x)",
            "infoton dynamics A_mu are governed by the pullback metric",
            "effective coupling is strongest where information density FIM is maximised",
            "NV-centre sensors are prediction targets, not reported evidence",
        ),
        "test_protocols": ("preserve observable section and L4/L5 prediction boundary",),
        "null_results": ("NV-centre sensor prediction is not observed signal evidence",),
        "variables": ("L5", "L11", "theta", "I", "det_I", "g_F", "A_mu", "NV_center"),
        "validation_targets": (
            "preserve layer examples",
            "preserve coding efficiency objective",
            "preserve prediction-only status",
        ),
        "null_controls": ("prediction wording must not be promoted to measurement evidence",),
    },
    "derived_lagrangian_detail.neural_fim_covariance_strategy": {
        "context_id": "neural_fim_covariance_strategy",
        "validation_protocol": "paper0.derived_lagrangian_detail.neural_fim_covariance_strategy",
        "canonical_statement": (
            "The source requires neural FIM estimation with full covariance structure and uses Tr(I(theta)) as a "
            "predicted correlate of infoton coupling strength."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:neural_fim_covariance_strategy" for n in range(1465, 1474)
        ),
        "source_formulae": (
            "FEP and Efficient Coding establish the operational link between FIM and L5 dynamics",
            "I(theta) = -nabla nabla log p(y|theta) approx -nabla nabla F",
            "FIM must use full covariance matrix Sigma(theta)",
            "I(theta) = (nabla_theta mu)^T Sigma^-1 (nabla_theta mu) + 1/2 Tr[(nabla_theta Sigma) Sigma^-1 (nabla_theta Sigma) Sigma^-1]",
            "Tr(I(theta)) is predicted to correlate with infoton coupling strength",
        ),
        "test_protocols": ("preserve neural FIM covariance strategy",),
        "null_results": ("covariance strategy is not a completed neural dataset",),
        "variables": ("I", "theta", "Sigma", "mu", "Tr", "F", "NV_center"),
        "validation_targets": (
            "preserve full covariance FIM",
            "preserve Tr(I) prediction",
            "preserve FEP/efficient-coding bridge",
        ),
        "null_controls": ("mean-only or diagonal FIM shortcut must fail source accounting",),
    },
    "derived_lagrangian_detail.domain_constraints_local_physics": {
        "context_id": "domain_constraints_local_physics",
        "validation_protocol": "paper0.derived_lagrangian_detail.domain_constraints_local_physics",
        "canonical_statement": (
            "The source constrains the pulled-back FIM kinetic term as an EFT medium effect that preserves fundamental "
            "spacetime geometry, Lorentz invariance, locality, and causality."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:domain_constraints_local_physics" for n in range(1474, 1493)
        ),
        "source_formulae": (
            "Psi interaction and mass terms use usual local U(1) structure on M",
            "infoton gauge kinetic term uses g_tilde_F from statistical pullback exclusively",
            "observable changes in theta(x) alter effective propagation of A_mu",
            "Tr g_F(x) gives summary couplings for predicted sensor signatures",
            "FIM dynamics are treated as effective field theory, not spacetime geometry modification",
            "Lorentz invariance is preserved because spacetime M retains g_mu_nu(x)",
            "pullback map pi: M -> Theta must depend only on locally measurable observables",
            "pullback cannot reference non-local or acausal data",
        ),
        "test_protocols": ("preserve EFT, Lorentz, locality, and causality constraints",),
        "null_results": ("constraint statement is not a Lorentz-invariance proof",),
        "variables": ("Psi", "U1", "g_tilde_F", "theta", "A_mu", "Tr_g_F", "pi", "M", "Theta"),
        "validation_targets": (
            "preserve no-domain-mixing rule",
            "preserve EFT medium boundary",
            "preserve local-observable constraint",
        ),
        "null_controls": (
            "nonlocal or acausal pullback dependency must fail constraint boundary",
        ),
    },
    "derived_lagrangian_detail.geometric_constants_predictions": {
        "context_id": "geometric_constants_predictions",
        "validation_protocol": "paper0.derived_lagrangian_detail.geometric_constants_predictions",
        "canonical_statement": (
            "The source defines the geometric Lagrangian, fundamental couplings g and xi, and prediction targets "
            "including short-range force, Psi-Higgs, ALP interface, and infoton signatures."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:geometric_constants_predictions" for n in range(1493, 1510)
        ),
        "source_formulae": (
            "L_Geometric' = -xi R Psi* Psi",
            "g is the U(1) gauge coupling constant for informational force and Psi-charge magnitude",
            "xi is the non-minimal coupling parameter for Ricci scalar R",
            "for a massless field, consistency conditions suggest xi = 1/6",
            "short-range informational force arises when SSB gives infoton mass m_A = g v",
            "Psi-Higgs is a predicted massive spin-0 scalar particle",
            "phase component behaves as ALP-like pseudoscalar with Primakoff-like EM coupling",
            "predictions remain targets, not experimental evidence",
        ),
        "test_protocols": (
            "preserve geometric coupling constants and prediction target boundaries",
        ),
        "null_results": ("prediction maps and figures are not detection evidence",),
        "variables": ("L_Geometric", "xi", "R", "Psi", "g", "m_A", "v", "Psi_Higgs", "ALP"),
        "validation_targets": (
            "preserve geometric coupling",
            "preserve g and xi definitions",
            "preserve prediction-only status",
        ),
        "null_controls": ("Psi-Higgs and ALP targets must not be counted as observed particles",),
    },
}


@dataclass(frozen=True, slots=True)
class DerivedLagrangianDetailSpec:
    """Derived Lagrangian detail spec promoted from Paper 0 records."""

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
class DerivedLagrangianDetailSpecBundle:
    """Derived Lagrangian detail specs plus source coverage summary."""

    specs: tuple[DerivedLagrangianDetailSpec, ...]
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


def build_derived_lagrangian_detail_specs(
    source_records: list[dict[str, Any]],
) -> DerivedLagrangianDetailSpecBundle:
    """Build source-covered derived Lagrangian detail specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[DerivedLagrangianDetailSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            DerivedLagrangianDetailSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 foundational extraction",
                section_path=str(anchors[0]["section_path"]),
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
                implementation_status="promoted_source_accounting_fixture",
                domain_review_status="source_bounded_no_empirical_validation",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 Derived Lagrangian Detail Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": sorted(set(SOURCE_LEDGER_IDS) - set(consumed)),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "category_counts": dict(sorted(category_counts.items())),
        "block_type_counts": dict(sorted(block_counts.items())),
        "math_ids": sorted(
            {math_id for record in anchors for math_id in record.get("math_ids", [])}
        ),
        "image_ids": sorted(
            {image_id for record in anchors for image_id in record.get("image_ids", [])}
        ),
        "table_ids": sorted(
            {str(record["table_id"]) for record in anchors if record.get("table_id") is not None}
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R01510",
    }
    return DerivedLagrangianDetailSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> DerivedLagrangianDetailSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_derived_lagrangian_detail_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: DerivedLagrangianDetailSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 Derived Lagrangian Detail Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Next source boundary: {bundle.summary['next_source_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### `{spec.key}`",
                "",
                spec.canonical_statement,
                "",
                f"- Context: `{spec.context_id}`",
                f"- Protocol: `{spec.validation_protocol}`",
                f"- Source equations: {', '.join(spec.source_equation_ids)}",
                f"- Null controls: {', '.join(spec.null_controls)}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    bundle: DerivedLagrangianDetailSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artifacts for promoted specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_derived_lagrangian_detail_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_derived_lagrangian_detail_validation_specs_report_{date_tag}.md"
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
