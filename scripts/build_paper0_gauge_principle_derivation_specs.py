#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 gauge-principle derivation spec builder
"""Promote Paper 0 gauge-principle derivation records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1018, 1078))
CLAIM_BOUNDARY = "source-bounded gauge-principle derivation; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "gauge_principle_derivation.derivation_boundary": {
        "context_id": "derivation_boundary",
        "validation_protocol": "paper0.gauge_principle_derivation.derivation_boundary",
        "canonical_statement": (
            "The source opens the detailed gauge-principle derivation of the "
            "Psi-field interaction Lagrangian and terminates this slice before "
            "the Lorentz-covariance/EFT resolution."
        ),
        "source_equation_ids": (
            "P0R01018:gauge_principle_derivation_heading",
            "P0R01019:phenomenology_to_first_principles_heading",
            "P0R01078:next_lorentz_covariance_boundary",
        ),
        "source_formulae": (
            "A Gauge-Principle Derivation of the Psi-Field Interaction Lagrangian",
            "Introduction: From Phenomenology to First Principles",
            "next boundary is P0R01078 Lorentz covariance EFT resolution",
        ),
        "test_protocols": ("preserve gauge-principle derivation boundary",),
        "null_results": ("section heading is source context, not validation evidence",),
        "variables": ("Psi", "L_Int", "A_mu"),
        "validation_targets": (
            "preserve detailed derivation heading",
            "preserve Lorentz-covariance next boundary",
        ),
        "null_controls": ("derivation-boundary drift control must be rejected",),
    },
    "gauge_principle_derivation.phenomenology_symmetry_roadmap": {
        "context_id": "phenomenology_symmetry_roadmap",
        "validation_protocol": (
            "paper0.gauge_principle_derivation.phenomenology_symmetry_roadmap"
        ),
        "canonical_statement": (
            "The source critiques the phenomenological dual-coupling Lagrangian "
            "and frames local gauge invariance and theoretical consistency as "
            "the roadmap for deriving a constrained L_Int prime."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:phenomenology_symmetry_roadmap" for number in range(1020, 1035)
        ),
        "source_formulae": (
            "L_Int = L_Geometric + L_Informational",
            "L_Geometric = g_PsiG f(Psi) R",
            "L_Informational = g_PsiI Psi det(g_mu_nu(x))",
            "phenomenological rather than fundamental",
            "local gauge invariance fixes force mediators",
            "renormalizability and conformal invariance constrain interaction terms",
            "L_Int' grounded in gauge symmetry and theoretical consistency",
            "informational coupling derived from U(1) gauge invariance",
            "unified interaction Lagrangian is source-framed as derived constrained predictive",
        ),
        "test_protocols": ("preserve phenomenology critique and symmetry roadmap",),
        "null_results": ("roadmap records are source claims, not derivation proof",),
        "variables": ("L_Int", "L_Geometric", "L_Informational", "g_PsiG", "g_PsiI"),
        "validation_targets": (
            "preserve old phenomenological formulas",
            "preserve gauge-principle roadmap",
            "preserve theoretical-consistency roadmap",
        ),
        "null_controls": (
            "phenomenological-formula omission control must be rejected",
            "roadmap-as-proof control must be rejected",
        ),
    },
    "gauge_principle_derivation.free_scalar_global_u1": {
        "context_id": "free_scalar_global_u1",
        "validation_protocol": "paper0.gauge_principle_derivation.free_scalar_global_u1",
        "canonical_statement": (
            "The source defines the free complex scalar Psi-field Lagrangian, "
            "its global U(1) phase transformation, kinetic/potential invariance, "
            "and Noether-current context."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:free_scalar_global_u1" for number in range(1035, 1047)
        ),
        "source_formulae": (
            "The Gauge Principle I: U(1) Symmetry and the Origin of Informational Coupling",
            "L_Psi = (partial_mu Psi)* (partial^mu Psi) - V(|Psi|)",
            "Psi(x) -> Psi'(x) = exp(i alpha) Psi(x)",
            "|Psi'| = |Psi|",
            "partial_mu Psi' = exp(i alpha) partial_mu Psi",
            "kinetic term is invariant under global U(1)",
            "Noether theorem implies conserved current and conserved charge",
            "P0R01046 is blank after Noether-current context",
        ),
        "test_protocols": ("preserve free scalar and global U1 records",),
        "null_results": ("free-scalar records are source equations, not runtime validation",),
        "variables": ("Psi", "alpha", "L_Psi", "V", "partial_mu"),
        "validation_targets": (
            "preserve free Lagrangian",
            "preserve global U1 transformation",
            "preserve Noether-current context",
        ),
        "null_controls": (
            "free-Lagrangian omission control must be rejected",
            "blank-record omission control must be rejected",
        ),
    },
    "gauge_principle_derivation.local_u1_derivative_failure": {
        "context_id": "local_u1_derivative_failure",
        "validation_protocol": ("paper0.gauge_principle_derivation.local_u1_derivative_failure"),
        "canonical_statement": (
            "The source promotes the global phase to alpha(x), records the "
            "ordinary derivative product-rule term, and identifies derivative "
            "failure as requiring new gauge structure."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:local_u1_derivative_failure" for number in range(1047, 1054)
        ),
        "source_formulae": (
            "Promoting Global to Local Invariance",
            "Psi(x) -> Psi'(x) = exp(i alpha(x)) Psi(x)",
            "partial_mu Psi' = exp(i alpha(x)) partial_mu Psi + i(partial_mu alpha(x)) exp(i alpha(x)) Psi",
            "ordinary derivative failure introduces i(partial_mu alpha(x)) term",
            "kinetic term is no longer invariant under local phase transformation",
            "new structure must be introduced",
        ),
        "test_protocols": ("preserve local U1 derivative-failure source records",),
        "null_results": ("derivative-failure statement awaits detailed algebra fixture",),
        "variables": ("Psi", "alpha(x)", "partial_mu"),
        "validation_targets": (
            "preserve local phase transformation",
            "preserve product-rule failure term",
            "preserve need for new gauge structure",
        ),
        "null_controls": ("local-phase derivative omission control must be rejected",),
    },
    "gauge_principle_derivation.covariant_derivative_minimal_coupling": {
        "context_id": "covariant_derivative_minimal_coupling",
        "validation_protocol": (
            "paper0.gauge_principle_derivation.covariant_derivative_minimal_coupling"
        ),
        "canonical_statement": (
            "The source introduces the gauge covariant derivative D_mu, the "
            "gauge-field transformation law, the locally invariant Lagrangian, "
            "and the minimal-coupling interaction expansion."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:covariant_derivative_minimal_coupling"
            for number in range(1054, 1068)
        ),
        "source_formulae": (
            "The Covariant Derivative and the Emergence of the Gauge Field",
            "(D_mu Psi)' = exp(i alpha(x)) (D_mu Psi)",
            "D_mu = partial_mu - i g A_mu",
            "A_mu' = A_mu + (1/g) partial_mu alpha(x)",
            "L_Psi,int = (D_mu Psi)* (D^mu Psi) - V(|Psi|)",
            "L_Psi,int = L_Free + L_Interaction",
            "i g A_mu(Psi* partial_mu Psi - Psi partial_mu Psi*)",
            "g^2 A_mu A^mu Psi* Psi",
            "minimal coupling is an unavoidable consequence of local phase invariance",
        ),
        "test_protocols": ("preserve covariant derivative and minimal-coupling records",),
        "null_results": ("minimal-coupling expansion is source algebra pending fixture",),
        "variables": ("D_mu", "A_mu", "g", "Psi", "alpha(x)", "L_Psi_int"),
        "validation_targets": (
            "preserve D_mu definition",
            "preserve A_mu transformation",
            "preserve interaction expansion",
            "preserve minimal-coupling claim",
        ),
        "null_controls": (
            "D_mu omission control must be rejected",
            "A_mu transformation omission control must be rejected",
        ),
    },
    "gauge_principle_derivation.fim_gauge_dynamics": {
        "context_id": "fim_gauge_dynamics",
        "validation_protocol": "paper0.gauge_principle_derivation.fim_gauge_dynamics",
        "canonical_statement": (
            "The source proposes informational gauge-field dynamics based on "
            "the Fisher Information Metric and preserves the image placeholder "
            "without treating it as evidence."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:fim_gauge_dynamics" for number in range(1068, 1078)
        ),
        "source_formulae": (
            "A Novel Identification: The Gauge Dynamics of Information Geometry",
            "F_mu_nu = partial_mu A_nu - partial_nu A_mu",
            "L_gauge = -1/4 F_mu_nu F^mu_nu",
            "field strength tensor built from A_mu",
            "Fisher Information Metric governs informational gauge-field dynamics",
            "L_Informational' = L_Interaction + L_gauge",
            "L_Informational' includes -1/4 g_FIM^{mu alpha} g_FIM^{nu beta} F_mu_nu F_alpha_beta",
            "P0R01076 is image placeholder not validation evidence",
            "FIM proposal replaces phenomenological g_PsiI Psi det(g_mu_nu(x))",
            "next boundary is P0R01078 Lorentz covariance EFT resolution",
        ),
        "test_protocols": ("preserve FIM gauge-dynamics source proposal",),
        "null_results": (
            "FIM metric replacement is not Lorentz-safe until the next EFT-resolution slice"
        ),
        "variables": ("A_mu", "F_mu_nu", "g_FIM", "L_Informational_prime"),
        "validation_targets": (
            "preserve field-strength tensor",
            "preserve FIM kinetic-term proposal",
            "preserve image placeholder as non-evidence",
            "preserve Lorentz-covariance next boundary",
        ),
        "null_controls": (
            "F_mu_nu omission control must be rejected",
            "image-as-evidence control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class GaugePrincipleDerivationSpec:
    """Gauge-principle derivation spec promoted from Paper 0 records."""

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
class GaugePrincipleDerivationSpecBundle:
    """Gauge-principle derivation specs plus source coverage summary."""

    specs: tuple[GaugePrincipleDerivationSpec, ...]
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


def build_gauge_principle_derivation_specs(
    source_records: list[dict[str, Any]],
) -> GaugePrincipleDerivationSpecBundle:
    """Build source-covered gauge-principle derivation specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[GaugePrincipleDerivationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            GaugePrincipleDerivationSpec(
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
        "title": "Paper 0 Gauge Principle Derivation Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "blank_record_count": 1,
        "image_record_count": 1,
        "phenomenology_symmetry_record_count": 17,
        "free_scalar_record_count": 12,
        "local_u1_record_count": 7,
        "covariant_derivative_record_count": 14,
        "fim_dynamics_record_count": 10,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R01078",
        "spec_keys": [spec.key for spec in specs],
    }
    return GaugePrincipleDerivationSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> GaugePrincipleDerivationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_gauge_principle_derivation_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: GaugePrincipleDerivationSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Gauge Principle Derivation Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Phenomenology/symmetry records: {bundle.summary['phenomenology_symmetry_record_count']}",
        f"- Free-scalar records: {bundle.summary['free_scalar_record_count']}",
        f"- Local-U1 records: {bundle.summary['local_u1_record_count']}",
        f"- Covariant-derivative records: {bundle.summary['covariant_derivative_record_count']}",
        f"- FIM-dynamics records: {bundle.summary['fim_dynamics_record_count']}",
        f"- Blank records: {bundle.summary['blank_record_count']}",
        f"- Image records: {bundle.summary['image_record_count']}",
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
    bundle: GaugePrincipleDerivationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_gauge_principle_derivation_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_gauge_principle_derivation_validation_specs_report_{date_tag}.md"
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
