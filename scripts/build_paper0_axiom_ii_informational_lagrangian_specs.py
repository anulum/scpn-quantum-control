#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Axiom II informational Lagrangian spec builder
"""Promote Paper 0 Axiom II informational-Lagrangian records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(782, 791))
CLAIM_BOUNDARY = "source-bounded Axiom II informational-Lagrangian map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiom_ii_informational_lagrangian.kinetic_term_modification": {
        "context_id": "kinetic_term_modification",
        "validation_protocol": (
            "paper0.axiom_ii_informational_lagrangian.kinetic_term_modification"
        ),
        "canonical_statement": (
            "The source states that Axiom II modifies the infoton gauge-field "
            "kinetic term by replacing the spacetime metric with the pulled-back "
            "information metric."
        ),
        "source_equation_ids": (
            "P0R00782:informational_lagrangian_heading",
            "P0R00783:infoton_kinetic_term_metric_replacement",
        ),
        "source_formulae": (
            "Formal Consequence: The Informational Lagrangian",
            "kinetic term of the infoton gauge field A_mu",
            "replace spacetime metric with pulled-back information metric",
        ),
        "test_protocols": ("preserve informational-Lagrangian kinetic-term claim",),
        "null_results": ("informational Lagrangian is source formula, not empirical evidence",),
        "variables": ("A_mu", "g_F", "g_mu_nu", "F_mu_nu"),
        "validation_targets": (
            "preserve informational-Lagrangian heading",
            "preserve infoton gauge-field kinetic-term target",
            "preserve spacetime-metric replacement claim",
        ),
        "null_controls": (
            "kinetic-term-formula-as-validated-physics control must be rejected",
            "metric-replacement-without-pullback control must be rejected",
        ),
    },
    "axiom_ii_informational_lagrangian.standard_gauge_baseline": {
        "context_id": "standard_gauge_baseline",
        "validation_protocol": (
            "paper0.axiom_ii_informational_lagrangian.standard_gauge_baseline"
        ),
        "canonical_statement": (
            "The source records the standard gauge-theory baseline Lagrangian "
            "and labels it as dynamics governed by spacetime geometry."
        ),
        "source_equation_ids": (
            "P0R00784:standard_gauge_theory_baseline",
            "P0R00785:standard_gauge_lagrangian",
            "P0R00786:spacetime_geometry_dynamics",
        ),
        "source_formulae": (
            "Standard Gauge Theory e.g. Electromagnetism",
            "L_gauge = -1/4 g^{mu alpha} g^{nu beta} F_{mu nu} F_{alpha beta}",
            "dynamics governed by spacetime geometry",
        ),
        "test_protocols": ("preserve standard gauge-theory baseline equation",),
        "null_results": ("standard gauge equation is baseline context, not SCPN evidence",),
        "variables": ("L_gauge", "g_mu_alpha", "g_nu_beta", "F_mu_nu", "F_alpha_beta"),
        "validation_targets": (
            "preserve standard gauge baseline label",
            "preserve spacetime-metric Lagrangian",
            "preserve spacetime-dynamics label",
        ),
        "null_controls": (
            "baseline-as-SCPN-result control must be rejected",
            "missing-spacetime-geometry-label control must be rejected",
        ),
    },
    "axiom_ii_informational_lagrangian.scpn_gauge_lagrangian": {
        "context_id": "scpn_gauge_lagrangian",
        "validation_protocol": ("paper0.axiom_ii_informational_lagrangian.scpn_gauge_lagrangian"),
        "canonical_statement": (
            "The source records the SCPN Axiom II gauge Lagrangian obtained by "
            "substituting the pulled-back Fisher information metric for the "
            "spacetime metric, with dynamics governed by informational geometry."
        ),
        "source_equation_ids": (
            "P0R00787:scpn_gauge_theory_axiom_ii",
            "P0R00788:scpn_gauge_lagrangian_information_metric",
            "P0R00789:informational_geometry_dynamics",
        ),
        "source_formulae": (
            "SCPN Gauge Theory Axiom II",
            "L_gauge = -1/4 tilde_g_F^{mu alpha} tilde_g_F^{nu beta} F_{mu nu} F_{alpha beta}",
            "dynamics governed by informational geometry",
        ),
        "test_protocols": ("preserve SCPN gauge-Lagrangian source equation",),
        "null_results": ("SCPN gauge Lagrangian requires downstream bridge validation",),
        "variables": (
            "L_gauge",
            "tilde_g_F_mu_alpha",
            "tilde_g_F_nu_beta",
            "F_mu_nu",
            "F_alpha_beta",
        ),
        "validation_targets": (
            "preserve SCPN Axiom II gauge label",
            "preserve pulled-back-FIM Lagrangian",
            "preserve informational-geometry dynamics label",
        ),
        "null_controls": (
            "informational-metric-equation-as-measured-result control must be rejected",
            "spacetime-metric-retained control must be rejected",
        ),
    },
    "axiom_ii_informational_lagrangian.operational_pullback_protocol": {
        "context_id": "operational_pullback_protocol",
        "validation_protocol": (
            "paper0.axiom_ii_informational_lagrangian.operational_pullback_protocol"
        ),
        "canonical_statement": (
            "The source defines tilde-g_F as the FIM pulled back from the abstract "
            "statistical manifold to physical spacetime, points the concrete "
            "mechanism to the Chapter 6 Operational Pullback Protocol, and states "
            "that this makes Axiom II falsifiable and predictive."
        ),
        "source_equation_ids": (
            "P0R00790:pulled_back_fim_definition",
            "P0R00790:operational_pullback_protocol_chapter6",
            "P0R00790:falsifiable_predictive_axiom_ii_bridge",
        ),
        "source_formulae": (
            "tilde_g_F is the Fisher Information Metric pulled back from the abstract statistical manifold",
            "physical spacetime manifold",
            "Operational Pullback Protocol is detailed in Chapter 6",
            "concrete testable bridge between mathematics of information and physics of the brain",
            "Axiom II a falsifiable predictive hypothesis",
        ),
        "test_protocols": ("preserve Chapter 6 pullback-protocol dependency",),
        "null_results": ("falsifiability claim requires downstream bridge tests",),
        "variables": ("tilde_g_F", "statistical_manifold", "spacetime", "brain_physics"),
        "validation_targets": (
            "preserve pulled-back-FIM definition",
            "preserve Chapter 6 operational-protocol pointer",
            "preserve falsifiable predictive bridge claim",
        ),
        "null_controls": (
            "pullback-protocol-as-implemented-here control must be rejected",
            "falsifiability-without-bridge-test control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomIIInformationalLagrangianSpec:
    """Axiom II informational-Lagrangian spec promoted from Paper 0 records."""

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
class AxiomIIInformationalLagrangianSpecBundle:
    """Axiom II informational-Lagrangian specs plus source coverage summary."""

    specs: tuple[AxiomIIInformationalLagrangianSpec, ...]
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


def build_axiom_ii_informational_lagrangian_specs(
    source_records: list[dict[str, Any]],
) -> AxiomIIInformationalLagrangianSpecBundle:
    """Build source-covered Axiom II informational-Lagrangian specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomIIInformationalLagrangianSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomIIInformationalLagrangianSpec(
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
        "title": "Paper 0 Axiom II Informational Lagrangian Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "lagrangian_heading_count": 1,
        "standard_gauge_baseline_count": 3,
        "scpn_gauge_count": 3,
        "pullback_protocol_count": 1,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00791",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomIIInformationalLagrangianSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AxiomIIInformationalLagrangianSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiom_ii_informational_lagrangian_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomIIInformationalLagrangianSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiom II Informational Lagrangian Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Lagrangian-heading records: {bundle.summary['lagrangian_heading_count']}",
        f"- Standard gauge baseline records: {bundle.summary['standard_gauge_baseline_count']}",
        f"- SCPN gauge records: {bundle.summary['scpn_gauge_count']}",
        f"- Pullback protocol records: {bundle.summary['pullback_protocol_count']}",
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
    bundle: AxiomIIInformationalLagrangianSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_axiom_ii_informational_lagrangian_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_axiom_ii_informational_lagrangian_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Axiom II informational-Lagrangian specs from the ledger."""

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
