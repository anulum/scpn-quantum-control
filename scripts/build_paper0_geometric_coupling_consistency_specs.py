#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 geometric coupling consistency spec builder
"""Promote Paper 0 geometric-coupling consistency records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1135, 1189))
CLAIM_BOUNDARY = (
    "source-bounded geometric-coupling consistency derivation; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "geometric_coupling_consistency.coupling_problem_boundary": {
        "context_id": "coupling_problem_boundary",
        "validation_protocol": "paper0.geometric_coupling_consistency.coupling_problem_boundary",
        "canonical_statement": (
            "The source marks that internal U(1) gauge covariance does not itself "
            "generate direct coupling to external spacetime curvature."
        ),
        "source_equation_ids": (
            "P0R01135:geometric_coupling_consistency_heading",
            "P0R01136:u1_internal_symmetry_cannot_generate_curvature_coupling",
            "P0R01137:curvature_coupling_challenge_heading",
            "P0R01138:gauge_covariant_vs_gr_covariant_derivative_distinction",
        ),
        "source_formulae": (
            "Consistency Conditions and the Origin of Geometric Coupling",
            "U(1) acts on the internal phase space of the Psi field",
            "direct non-minimal coupling to the Ricci scalar R needs a separate principle",
            "D_mu = partial_mu - i g A_mu is distinct from the general-relativistic covariant derivative",
        ),
        "test_protocols": ("preserve internal-gauge versus spacetime-covariance distinction",),
        "null_results": ("U(1) gauge covariance alone is not a derivation of f(Psi)R",),
        "variables": ("D_mu", "A_mu", "R", "Psi", "nabla_mu"),
        "validation_targets": (
            "preserve direct-curvature-coupling gap",
            "preserve derivative-domain distinction",
            "preserve non-minimal-coupling boundary",
        ),
        "null_controls": (
            "internal gauge covariance must not be treated as spacetime curvature coupling",
        ),
    },
    "geometric_coupling_consistency.minimal_curved_spacetime_coupling": {
        "context_id": "minimal_curved_spacetime_coupling",
        "validation_protocol": "paper0.geometric_coupling_consistency.minimal_curved_spacetime_coupling",
        "canonical_statement": (
            "The source records the minimal curved-spacetime scalar-field coupling and "
            "its limitation: stress-energy sources curvature but no direct R-amplitude term appears."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:minimal_curved_spacetime_coupling" for number in range(1139, 1143)
        ),
        "source_formulae": (
            "ordinary derivatives are replaced by covariant derivatives",
            "eta_mu_nu is replaced by g_mu_nu",
            "L_Psi_curved = g^{mu nu}(nabla_mu Psi)^*(nabla_nu Psi) - V(|Psi|)",
            "Einstein-Hilbert variation yields stress-energy of the Psi field",
            "minimal coupling does not include direct interaction between field amplitude and R",
        ),
        "test_protocols": ("preserve minimal-coupling equation and limitation",),
        "null_results": ("minimal coupling is not the final geometric interaction",),
        "variables": ("g_mu_nu", "eta_mu_nu", "nabla_mu", "L_Psi_curved", "S_EH", "R"),
        "validation_targets": (
            "preserve colon-to-semicolon replacement rule",
            "preserve curved scalar Lagrangian",
            "preserve missing direct R coupling statement",
        ),
        "null_controls": ("minimal-only control must not satisfy geometric interaction claim",),
    },
    "geometric_coupling_consistency.non_minimal_consistency_condition": {
        "context_id": "non_minimal_consistency_condition",
        "validation_protocol": "paper0.geometric_coupling_consistency.non_minimal_consistency_condition",
        "canonical_statement": (
            "The source motivates non-minimal scalar-curvature coupling from conformal "
            "invariance and renormalizability requirements in curved spacetime."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:non_minimal_consistency_condition" for number in range(1143, 1150)
        ),
        "source_formulae": (
            "The Consistency Condition: Conformal Invariance and Renormalizability",
            "L_non_minimal = - xi R Psi^* Psi",
            "xi is a dimensionless coupling constant",
            "conformal rescaling g_mu_nu -> Omega^2(x) g_mu_nu preserves angles",
            "massless scalar conformal invariance selects xi = 1/6",
            "quantum loop corrections generate non-minimal coupling even if xi is set to zero classically",
        ),
        "test_protocols": ("preserve conformal and renormalizability source arguments",),
        "null_results": ("source consistency arguments are not measured curvature coupling",),
        "variables": ("xi", "R", "Psi", "Omega", "g_mu_nu"),
        "validation_targets": (
            "preserve xi R Psi-star Psi term",
            "preserve xi equals one-sixth conformal value",
            "preserve loop-correction renormalizability argument",
        ),
        "null_controls": ("xi-zero classical control remains incomplete in curved-spacetime QFT",),
    },
    "geometric_coupling_consistency.derived_geometric_lagrangian": {
        "context_id": "derived_geometric_lagrangian",
        "validation_protocol": "paper0.geometric_coupling_consistency.derived_geometric_lagrangian",
        "canonical_statement": (
            "The source maps the non-minimal consistency term into a derived geometric "
            "interaction Lagrangian and records it as an equation-bearing source claim."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:derived_geometric_lagrangian" for number in range(1150, 1159)
        ),
        "source_formulae": (
            "Derivation of the Geometric Lagrangian",
            "the geometric interaction is presented as consistency-required rather than phenomenological",
            "the simplest scalar quantity from the complex Psi field is Psi^* Psi",
            "L_Geometric_prime = - g_PsiG R Psi^* Psi",
            "the source aligns the geometric coefficient with the non-minimal scalar-curvature coupling",
            "the unified interaction construction follows this geometric term",
        ),
        "test_protocols": (
            "preserve derived geometric Lagrangian equation and coefficient boundary",
        ),
        "null_results": (
            "derived source equation is not a numerical curvature-coupling measurement",
        ),
        "variables": ("L_Geometric_prime", "g_PsiG", "R", "Psi"),
        "validation_targets": (
            "preserve Psi-star-Psi scalar choice",
            "preserve R coupling term",
            "preserve consistency-derived status",
        ),
        "null_controls": ("arbitrary phenomenological f(Psi) control must remain distinct",),
    },
    "geometric_coupling_consistency.complete_covariant_action": {
        "context_id": "complete_covariant_action",
        "validation_protocol": "paper0.geometric_coupling_consistency.complete_covariant_action",
        "canonical_statement": (
            "The source assembles the total generally covariant, gauge-invariant action "
            "and isolates the derived informational plus geometric interaction terms."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:complete_covariant_action" for number in range(1159, 1174)
        ),
        "source_formulae": (
            "replace partial_mu by nabla_mu, eta_mu_nu by g_mu_nu, and d4x by d4x sqrt(-g)",
            "promote nabla_mu to tilde_D_mu = nabla_mu - i g A_mu",
            "S_Total = integral d^4x sqrt(-g) ...",
            "L_Int_prime = L_Informational_prime + L_Geometric_prime",
            "L_Informational_prime includes Psi-current coupling and gauge-field kinetic terms",
            "L_Geometric_prime = - xi R Psi^* Psi",
            "IMG0020 is retained as source media, not independent evidence",
        ),
        "test_protocols": ("preserve total action and interaction decomposition",),
        "null_results": ("complete action source accounting is not a solved field simulation",),
        "variables": (
            "S_Total",
            "L_Int_prime",
            "L_Informational_prime",
            "L_Geometric_prime",
            "A_mu",
        ),
        "validation_targets": (
            "preserve general covariance replacements",
            "preserve gauge-covariant derivative promotion",
            "preserve informational/geometric decomposition",
        ),
        "null_controls": ("action assembly without both gauge and geometric terms must fail",),
    },
    "geometric_coupling_consistency.interpretation_prediction_comments": {
        "context_id": "interpretation_prediction_comments",
        "validation_protocol": "paper0.geometric_coupling_consistency.interpretation_prediction_comments",
        "canonical_statement": (
            "The source records comparative interpretation, infoton prediction targets, "
            "and derivation comments while leaving them as source claims requiring review."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:interpretation_prediction_comments" for number in range(1174, 1189)
        ),
        "source_formulae": (
            "Comparative Analysis and Interpretation",
            "TBL002 is retained as a source comparison table",
            "informational coupling is described as mediated by a gauge field",
            "geometric coupling is described as required for curved-spacetime consistency",
            "New Physical Predictions",
            "the infoton is source-described as a massless spin-1 gauge boson",
            "j_Psi^mu = i g (Psi^* nabla_mu Psi - Psi nabla_mu Psi^*)",
            "infoton dynamics are source-linked to the Fisher Information Metric",
            "the derivation comments recap U(1) gauging plus non-minimal curvature coupling",
        ),
        "test_protocols": (
            "preserve interpretation, prediction, and comments as source-bounded outputs",
        ),
        "null_results": ("infoton prediction is not detector evidence",),
        "variables": ("TBL002", "A_mu", "j_Psi_mu", "FIM", "infoton"),
        "validation_targets": (
            "preserve comparison table source role",
            "preserve infoton properties as prediction targets",
            "preserve Fisher-information dynamics claim boundary",
        ),
        "null_controls": ("prediction wording must not be promoted to observation",),
    },
}


@dataclass(frozen=True, slots=True)
class GeometricCouplingConsistencySpec:
    """Geometric-coupling consistency spec promoted from Paper 0 records."""

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
class GeometricCouplingConsistencySpecBundle:
    """Geometric-coupling consistency specs plus source coverage summary."""

    specs: tuple[GeometricCouplingConsistencySpec, ...]
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


def build_geometric_coupling_consistency_specs(
    source_records: list[dict[str, Any]],
) -> GeometricCouplingConsistencySpecBundle:
    """Build source-covered geometric-coupling consistency specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)
    math_ids = sorted(
        {str(math_id) for record in anchors for math_id in record.get("math_ids", [])}
    )
    image_ids = sorted(
        {str(image_id) for record in anchors for image_id in record.get("image_ids", [])}
    )
    table_ids = sorted(
        {str(record["table_id"]) for record in anchors if record.get("table_id") is not None}
    )

    specs: list[GeometricCouplingConsistencySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            GeometricCouplingConsistencySpec(
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
        "title": "Paper 0 Geometric Coupling Consistency Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "category_counts": dict(sorted(category_counts.items())),
        "block_type_counts": dict(sorted(block_counts.items())),
        "math_ids": math_ids,
        "image_ids": image_ids,
        "table_ids": table_ids,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R01189",
        "spec_keys": [spec.key for spec in specs],
    }
    return GeometricCouplingConsistencySpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> GeometricCouplingConsistencySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_geometric_coupling_consistency_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: GeometricCouplingConsistencySpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Geometric Coupling Consistency Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Category counts: {bundle.summary['category_counts']}",
        f"- Block-type counts: {bundle.summary['block_type_counts']}",
        f"- Math IDs: {bundle.summary['math_ids']}",
        f"- Image IDs: {bundle.summary['image_ids']}",
        f"- Table IDs: {bundle.summary['table_ids']}",
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
    bundle: GeometricCouplingConsistencySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_geometric_coupling_consistency_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_geometric_coupling_consistency_validation_specs_report_{date_tag}.md"
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
