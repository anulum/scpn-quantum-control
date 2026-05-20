#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 final LInt SM interface spec builder
"""Promote Paper 0 final Master Interaction Lagrangian and SM interface records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1510, 1582))
CLAIM_BOUNDARY = (
    "source-bounded final LInt and Standard Model interface; not experimental validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "final_lint_sm_interface.final_lint_dual_clause": {
        "context_id": "final_lint_dual_clause",
        "validation_protocol": "paper0.final_lint_sm_interface.final_lint_dual_clause",
        "canonical_statement": (
            "The source restates the final Master Interaction Lagrangian as a dual geometric and informational "
            "interaction governing Psi-field coupling to physical-informational reality."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:final_lint_dual_clause" for n in range(1510, 1519)
        ),
        "source_formulae": (
            "The Master Interaction Lagrangian (Derived from First Principles)",
            "L_Int formalises interaction between Psi-field and physical-informational structure",
            "L_Geometric = -xi R Psi*Psi",
            "L_Informational couples Psi-field to information structure mediated by infoton gauge field and gFIM",
            "S = integral L_Int dt",
            "least action delta S = 0 extremises geometric and informational coupling",
            "15-layer SCPN structure is claimed as source consequence of this directive",
        ),
        "test_protocols": ("preserve final LInt dual-clause source boundary",),
        "null_results": ("dual-clause formulation is not experimental validation evidence",),
        "variables": ("L_Int", "L_Geometric", "L_Informational", "Psi", "xi", "R", "gFIM", "S"),
        "validation_targets": (
            "preserve geometric clause",
            "preserve informational clause",
            "preserve least-action boundary",
        ),
        "null_controls": ("mission-statement wording must not be promoted to measured dynamics",),
    },
    "final_lint_sm_interface.free_energy_and_h_int_mapping": {
        "context_id": "free_energy_and_h_int_mapping",
        "validation_protocol": "paper0.final_lint_sm_interface.free_energy_and_h_int_mapping",
        "canonical_statement": (
            "The source maps L_Int to free-energy density/action minimisation and maps heuristic H_int terms "
            "onto Psi, curvature/information density, xi, and g."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:free_energy_and_h_int_mapping" for n in range(1520, 1536)
        ),
        "source_formulae": (
            "L_Int is interpreted as density of variational free energy",
            "least action delta S = 0 maps to total free-energy minimisation over time",
            "H_int = -lambda * Psi_s * sigma",
            "Psi_s is represented by the Psi-field itself with intensity Psi*Psi",
            "sigma has geometric component R and informational component local information density/FIM",
            "lambda decomposes into xi and g",
            "interaction couples geometry and information rather than undifferentiated matter",
        ),
        "test_protocols": ("preserve free-energy and H_int mapping boundaries",),
        "null_results": ("free-energy mapping is not a measured cost function",),
        "variables": (
            "L_Int",
            "free_energy",
            "H_int",
            "lambda",
            "Psi_s",
            "sigma",
            "R",
            "FIM",
            "xi",
            "g",
        ),
        "validation_targets": (
            "preserve LInt/FEP mapping",
            "preserve H_int source equation",
            "preserve dual sigma/lambda mapping",
        ),
        "null_controls": ("heuristic H_int must not replace derived LInt accounting",),
    },
    "final_lint_sm_interface.foundational_physics_equations": {
        "context_id": "foundational_physics_equations",
        "validation_protocol": "paper0.final_lint_sm_interface.foundational_physics_equations",
        "canonical_statement": (
            "The source gives a compact foundational-physics statement of L_Int, geometric coupling, "
            "informational determinant coupling, total-action minimisation, and a diagrammatic summary."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:foundational_physics_equations" for n in range(1536, 1550)
        ),
        "source_formulae": (
            "L_Int = L_Geometric + L_Informational",
            "L_Geometric = (1/(2 kappa)) R + g_PsiG f(Psi) R",
            "L_Geometric = -xi R Psi*Psi",
            "L_Informational = g_PsiI Psi det(g_mu_nu(x))",
            "L_Informational proportional to Psi*Psi det(g_mu_nu)",
            "delta S_Master = 0",
            "diagram separates geometric and informational couplings",
        ),
        "test_protocols": ("preserve compact foundational physics equations",),
        "null_results": ("diagrammatic equation summary is not empirical confirmation",),
        "variables": (
            "L_Int",
            "L_Geometric",
            "L_Informational",
            "kappa",
            "R",
            "Psi",
            "xi",
            "g_mu_nu",
            "S_Master",
        ),
        "validation_targets": (
            "preserve compact LInt split",
            "preserve geometric equations",
            "preserve informational determinant term",
        ),
        "null_controls": (
            "legacy compact equation and derived xi form must remain source-distinct",
        ),
    },
    "final_lint_sm_interface.standard_model_indirect_coupling": {
        "context_id": "standard_model_indirect_coupling",
        "validation_protocol": "paper0.final_lint_sm_interface.standard_model_indirect_coupling",
        "canonical_statement": (
            "The source states the Principle of Indirect Coupling: the Psi-field interfaces with Standard Model physics "
            "through geometry and information, not by adding new direct SM force carriers."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:standard_model_indirect_coupling" for n in range(1550, 1563)
        ),
        "source_formulae": (
            "Principle of Indirect Coupling",
            "Psi-field does not introduce new direct forces or particles that directly couple to SM fields",
            "geometric interface couples Psi-field energy density to curvature R and contributes T_mu_nu^Psi",
            "informational interface couples through Fisher Information Metric without direct force exchange",
            "information geometry can bias quantum probabilities, stabilise coherence, and modulate EM dynamics in complex substrates",
            "weak-force chirality and ALP mediation remain falsifiable extensions, not established interactions",
        ),
        "test_protocols": ("preserve Standard Model indirect-coupling boundary",),
        "null_results": ("indirect interface is not direct detection of a new SM force",),
        "variables": ("Psi", "SM", "R", "T_mu_nu_Psi", "FIM", "QZE", "MS_QEC", "CISS", "ALP"),
        "validation_targets": (
            "preserve no-new-direct-force boundary",
            "preserve geometric interface",
            "preserve informational interface",
        ),
        "null_controls": ("direct SM force-carrier claim must fail indirect-coupling accounting",),
    },
    "final_lint_sm_interface.predictive_interface_mapping": {
        "context_id": "predictive_interface_mapping",
        "validation_protocol": "paper0.final_lint_sm_interface.predictive_interface_mapping",
        "canonical_statement": (
            "The source maps indirect coupling into predictive-coding/downward-causation language: geometric interface "
            "as large-scale prediction and informational interface as micro-scale probability bias."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:predictive_interface_mapping" for n in range(1563, 1570)
        ),
        "source_formulae": (
            "Principle of Indirect Coupling mediates downward causation from cosmic inference engine",
            "geometric interface implements large-scale structural predictions via T_mu_nu tensor",
            "informational interface implements fine-grained predictions at quantum and biological levels",
            "prediction is realised as biasing quantum probabilities without violating quantum mechanics",
        ),
        "test_protocols": ("preserve predictive-interface mapping as source mapping",),
        "null_results": ("predictive-interface language is not independent validation",),
        "variables": (
            "indirect_coupling",
            "T_mu_nu",
            "quantum_probability",
            "prediction",
            "geometry",
            "information",
        ),
        "validation_targets": (
            "preserve downward-causation mapping",
            "preserve geometric prediction role",
            "preserve quantum-probability boundary",
        ),
        "null_controls": ("prediction metaphor must not count as observed probability bias",),
    },
    "final_lint_sm_interface.downstream_sm_manifestations": {
        "context_id": "downstream_sm_manifestations",
        "validation_protocol": "paper0.final_lint_sm_interface.downstream_sm_manifestations",
        "canonical_statement": (
            "The source maps H_int downstream to T_mu_nu^Psi and probability-bias effects, then restates SM interface "
            "channels and exploratory weak/ALP hypotheses."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:downstream_sm_manifestations" for n in range(1570, 1582)
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma is the primary cause and SM effects are secondary consequences",
            "H_int creates local energy density appearing as T_mu_nu^Psi in Einstein Field Equations",
            "H_int alters quantum-system energy landscape, pointer-state selection, and outcome probabilities",
            "Psi-field interfaces with SM via Geometric and Informational Lagrangians rather than new force carriers",
            "geometric interface: L_Geometric proportional to f(Psi) R modifies Einstein equations via T_mu_nu^Psi",
            "informational interface: L_Informational biases QZE, stabilises MS-QEC, and modulates EM fields through CISS/Calcium dynamics",
            "chiral weak-force and ALP-mediated links are dotted exploratory hypotheses",
        ),
        "test_protocols": ("preserve downstream SM manifestation boundaries",),
        "null_results": ("downstream manifestation map is not experimental evidence",),
        "variables": (
            "H_int",
            "lambda",
            "Psi_s",
            "sigma",
            "T_mu_nu_Psi",
            "QZE",
            "MS_QEC",
            "CISS",
            "ALP",
        ),
        "validation_targets": (
            "preserve H_int downstream mapping",
            "preserve indirect SM interface",
            "preserve exploratory-hypothesis status",
        ),
        "null_controls": ("weak-force and ALP hypotheses must not be marked established",),
    },
}


@dataclass(frozen=True, slots=True)
class FinalLIntSMInterfaceSpec:
    """Final LInt and SM interface spec promoted from Paper 0 records."""

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
class FinalLIntSMInterfaceSpecBundle:
    """Final LInt and SM interface specs plus source coverage summary."""

    specs: tuple[FinalLIntSMInterfaceSpec, ...]
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


def build_final_lint_sm_interface_specs(
    source_records: list[dict[str, Any]],
) -> FinalLIntSMInterfaceSpecBundle:
    """Build source-covered final LInt and SM interface specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[FinalLIntSMInterfaceSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            FinalLIntSMInterfaceSpec(
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
        "title": "Paper 0 Final LInt SM Interface Specs",
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
        "next_source_boundary": "P0R01582",
    }
    return FinalLIntSMInterfaceSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> FinalLIntSMInterfaceSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_final_lint_sm_interface_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: FinalLIntSMInterfaceSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 Final LInt SM Interface Specs",
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
    bundle: FinalLIntSMInterfaceSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artifacts for promoted specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_final_lint_sm_interface_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_final_lint_sm_interface_validation_specs_report_{date_tag}.md"
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
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    bundle = build_from_ledger(args.ledger)
    outputs = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
