#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 terminology bridge spec builder
"""Promote Paper 0 terminology-bridge records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(610, 635))
CLAIM_BOUNDARY = "source-bounded terminology bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "terminology_bridge.mainstream_anchor_map": {
        "context_id": "mainstream_anchor_map",
        "validation_protocol": "paper0.terminology_bridge.mainstream_anchor_map",
        "canonical_statement": (
            "The terminology bridge maps core Anulum terms to mainstream mathematical "
            "anchors while preserving that the mapping is a translation layer."
        ),
        "source_equation_ids": (
            "P0R00616:psi_field_fibre_bundle_anchor",
            "P0R00617:upde_coupled_oscillator_anchor",
            "P0R00618:geometric_qualia_tda_anchor",
            "P0R00619:pela_yang_mills_tool_not_literal_claim",
        ),
        "source_formulae": (
            "Psi-field -> fibre bundle",
            "UPDE -> coupled oscillator model",
            "Geometric Qualia -> Topological Data Analysis",
            "PELA/Yang-Mills similarity is a tool, not a literal claim",
        ),
        "test_protocols": ("classify mainstream terminology anchors",),
        "null_results": ("anchor mapping is not validation evidence",),
        "variables": ("Psi-field", "UPDE", "Geometric Qualia", "PELA"),
        "validation_targets": (
            "preserve Psi-field fibre-bundle anchor",
            "preserve UPDE coupled-oscillator anchor",
            "preserve Geometric Qualia TDA anchor",
            "preserve PELA analogy boundary",
        ),
        "null_controls": (
            "anchor-map-as-proof control must be rejected",
            "literal-ethics-is-physics control must be rejected",
        ),
    },
    "terminology_bridge.predictive_coding_precision": {
        "context_id": "predictive_coding_precision",
        "validation_protocol": "paper0.terminology_bridge.predictive_coding_precision",
        "canonical_statement": (
            "The bridge is framed as precision-weighting the deepest priors of the "
            "generative model by linking terms to formal anchors."
        ),
        "source_equation_ids": (
            "P0R00621:meta_framework_integrations",
            "P0R00622:predictive_coding_integration",
            "P0R00623:precision_of_deepest_priors",
            "P0R00624:mainstream_anchor_precision_weighting",
        ),
        "source_formulae": (
            "precision of deepest priors",
            "confidence or inverse variance",
            "mainstream anchor",
            "precision-weighting",
            "strong falsifiable predictions",
        ),
        "test_protocols": ("preserve predictive-coding precision mapping",),
        "null_results": ("precision wording is not empirical confirmation",),
        "variables": ("priors", "precision", "prediction_error"),
        "validation_targets": (
            "preserve inverse-variance precision framing",
            "preserve falsifiability-through-anchor claim boundary",
        ),
        "null_controls": (
            "precision-language-as-data control must be rejected",
            "formal-anchor-as-confirmation control must be rejected",
        ),
    },
    "terminology_bridge.psi_field_coupling_context": {
        "context_id": "psi_field_coupling_context",
        "validation_protocol": "paper0.terminology_bridge.psi_field_coupling_context",
        "canonical_statement": (
            "The bridge gives the interaction Hamiltonian context by assigning Psi_s "
            "the mathematical character of a field-theory section of a fibre bundle."
        ),
        "source_equation_ids": (
            "P0R00625:psis_field_coupling_integration",
            "P0R00626:H_int=-lambda*Psi_s*sigma",
            "P0R00627:defines_mathematical_character_of_Psi_s",
            "P0R00628:Psi_field_fibre_bundle_section",
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma",
            "Psi_s as field theory section of a fibre bundle",
            "geometric language for interaction terms",
        ),
        "test_protocols": ("preserve H_int context and Psi_s object type",),
        "null_results": ("H_int context is not a fitted interaction result",),
        "variables": ("H_int", "lambda", "Psi_s", "sigma"),
        "validation_targets": (
            "preserve interaction-Hamiltonian formula label",
            "preserve Psi_s fibre-bundle-section anchor",
        ),
        "null_controls": (
            "h_int-context-as-fit control must be rejected",
            "psi_s-untyped-field control must be rejected",
        ),
    },
    "terminology_bridge.sigma_topology_target": {
        "context_id": "sigma_topology_target",
        "validation_protocol": "paper0.terminology_bridge.sigma_topology_target",
        "canonical_statement": (
            "The bridge constrains sigma by linking Geometric Qualia to topology or "
            "metric structure of a state manifold."
        ),
        "source_equation_ids": (
            "P0R00629:constrains_search_for_sigma",
            "P0R00630:sigma_topology_metric_state_manifold",
        ),
        "source_formulae": (
            "sigma search constrained by topology/metric of a state manifold",
            "Betti numbers or Ricci curvature as sigma candidate properties",
            "topological invariants or geometric properties",
        ),
        "test_protocols": ("preserve sigma topological/geometric target class",),
        "null_results": ("candidate target class is not a selected observable",),
        "variables": ("sigma", "Betti_numbers", "Ricci_curvature"),
        "validation_targets": (
            "preserve topological-invariant sigma target",
            "preserve geometric-property sigma target",
        ),
        "null_controls": (
            "candidate-class-as-observed-sigma control must be rejected",
            "missing-manifold-metric-anchor control must be rejected",
        ),
    },
    "terminology_bridge.pela_yang_mills_analogy_boundary": {
        "context_id": "pela_yang_mills_analogy_boundary",
        "validation_protocol": "paper0.terminology_bridge.pela_yang_mills_analogy_boundary",
        "canonical_statement": (
            "The PELA/Yang-Mills comparison is explicitly bounded as an analogy and "
            "regulariser, not a deductive derivation of ethics from gauge theory."
        ),
        "source_equation_ids": (
            "P0R00612:pela_yang_mills_heuristic_regulariser",
            "P0R00631:global_constraints_context",
            "P0R00632:pela_not_force_term_boundary",
            "P0R00633:terminology_table",
            "P0R00634:ethics_gauge_correspondence_analogy",
        ),
        "source_formulae": (
            "computationally convenient regulariser",
            "PELA does not add a new force term to H_int",
            "sets boundary conditions or tunes parameters",
            "no deductive derivation of ethics from gauge theory",
            "stress-tested in simulations and Pareto fronts",
        ),
        "test_protocols": ("preserve PELA analogy and supervisory-control boundary",),
        "null_results": ("Yang-Mills similarity is not deductive equivalence",),
        "variables": ("PELA", "H_int", "boundary_conditions", "Pareto_fronts"),
        "validation_targets": (
            "preserve not-a-force-term boundary",
            "preserve analogy-not-equivalence boundary",
            "preserve stress-testable optimisation-prior framing",
        ),
        "null_controls": (
            "pela-as-gauge-force control must be rejected",
            "ethics-from-gauge-theory control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TerminologyBridgeSpec:
    """Terminology bridge spec promoted from Paper 0 records."""

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
class TerminologyBridgeSpecBundle:
    """Terminology bridge specs plus source coverage summary."""

    specs: tuple[TerminologyBridgeSpec, ...]
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


def build_terminology_bridge_specs(
    source_records: list[dict[str, Any]],
) -> TerminologyBridgeSpecBundle:
    """Build source-covered terminology-bridge specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[TerminologyBridgeSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TerminologyBridgeSpec(
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
        "title": "Paper 0 Terminology Bridge Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "mainstream_anchor_count": 4,
        "analogy_boundary_count": 2,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00635",
        "spec_keys": [spec.key for spec in specs],
    }
    return TerminologyBridgeSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> TerminologyBridgeSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_terminology_bridge_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TerminologyBridgeSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Terminology Bridge Specs",
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
    bundle: TerminologyBridgeSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_terminology_bridge_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_terminology_bridge_validation_specs_report_{date_tag}.md"
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
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    outputs = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
