#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Axiom II infoton geometry spec builder
"""Promote Paper 0 Axiom II infoton-geometry records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(770, 775))
CLAIM_BOUNDARY = "source-bounded Axiom II infoton-geometry map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiom_ii_infoton_geometry.problem_heading": {
        "context_id": "problem_heading",
        "validation_protocol": "paper0.axiom_ii_infoton_geometry.problem_heading",
        "canonical_statement": (
            "The source opens the infoton-geometry problem and asks what "
            "geometric space the infoton propagates through."
        ),
        "source_equation_ids": (
            "P0R00770:infoton_geometry_problem_heading",
            "P0R00772:infoton_space_question",
        ),
        "source_formulae": (
            'The Central Problem: The Geometry of the "Infoton"',
            "What space does the infoton travel through?",
        ),
        "test_protocols": ("preserve infoton-geometry problem statement",),
        "null_results": ("problem heading is not an empirical geometry result",),
        "variables": ("infoton", "geometry", "propagation_space"),
        "validation_targets": (
            "preserve infoton geometry heading",
            "preserve propagation-space question",
        ),
        "null_controls": (
            "heading-as-solution control must be rejected",
            "missing-space-question control must be rejected",
        ),
    },
    "axiom_ii_infoton_geometry.gauge_necessity": {
        "context_id": "gauge_necessity",
        "validation_protocol": "paper0.axiom_ii_infoton_geometry.gauge_necessity",
        "canonical_statement": (
            "The source points to Chapter 4 for the derivation that applying "
            "the U(1) gauge principle to the local complex Psi field is a "
            "mathematical necessity, requiring a mediating spin-1 infoton."
        ),
        "source_equation_ids": (
            "P0R00771:u1_gauge_principle_necessity",
            "P0R00771:mediating_gauge_field",
            "P0R00771:spin1_infoton",
        ),
        "source_formulae": (
            "U(1) gauge principle is a mathematical necessity",
            "local complex consciousness field requires a mediating gauge field",
            "new fundamental force",
            "spin-1 vector boson infoton",
            "fundamental carrier of the informational force",
        ),
        "test_protocols": ("preserve source-bounded gauge-necessity pointer",),
        "null_results": ("Chapter 4 derivation is referenced, not rederived in this slice",),
        "variables": ("U1", "Psi", "A_mu", "infoton", "spin1"),
        "validation_targets": (
            "preserve U1 mathematical-necessity claim",
            "preserve mediating-gauge-field requirement",
            "preserve spin-1 infoton role",
        ),
        "null_controls": (
            "derivation-as-completed-here control must be rejected",
            "nonlocal-field control must be rejected",
        ),
    },
    "axiom_ii_infoton_geometry.spacetime_metric_baseline": {
        "context_id": "spacetime_metric_baseline",
        "validation_protocol": "paper0.axiom_ii_infoton_geometry.spacetime_metric_baseline",
        "canonical_statement": (
            "The source uses electromagnetism as the baseline: photon dynamics "
            "are governed by the spacetime metric and the standard gauge kinetic "
            "term."
        ),
        "source_equation_ids": (
            "P0R00773:standard_gauge_theory_baseline",
            "P0R00773:electromagnetic_kinetic_term",
        ),
        "source_formulae": (
            "standard gauge theories like electromagnetism",
            "photon dynamics governed by spacetime metric g_mu_nu",
            "force propagates through physical spacetime",
            "L_EM = -1/4 F_mu_nu F^mu_nu",
        ),
        "test_protocols": ("preserve standard gauge baseline",),
        "null_results": ("EM kinetic term is baseline context, not an SCPN result",),
        "variables": ("g_mu_nu", "F_mu_nu", "L_EM", "photon"),
        "validation_targets": (
            "preserve spacetime-metric baseline",
            "preserve EM kinetic-term baseline",
        ),
        "null_controls": (
            "baseline-as-SCPN-result control must be rejected",
            "missing-spacetime-metric control must be rejected",
        ),
    },
    "axiom_ii_infoton_geometry.fim_dynamics_claim": {
        "context_id": "fim_dynamics_claim",
        "validation_protocol": "paper0.axiom_ii_infoton_geometry.fim_dynamics_claim",
        "canonical_statement": (
            "The source states the Axiom II claim that infoton dynamics are "
            "governed by the Fisher Information Metric of the system's "
            "statistical manifold rather than by spacetime geometry."
        ),
        "source_equation_ids": (
            "P0R00774:fim_governed_infoton_dynamics",
            "P0R00774:statistical_manifold_metric",
        ),
        "source_formulae": (
            "infoton dynamics not governed by spacetime geometry",
            "Fisher Information Metric g_FIM",
            "system statistical manifold",
        ),
        "test_protocols": ("preserve FIM-governed dynamics source claim",),
        "null_results": ("FIM dynamics claim requires downstream validation",),
        "variables": ("infoton", "g_FIM", "statistical_manifold"),
        "validation_targets": (
            "preserve not-spacetime-geometry contrast",
            "preserve FIM dynamics claim",
            "preserve statistical-manifold target",
        ),
        "null_controls": (
            "FIM-claim-as-measurement control must be rejected",
            "spacetime-metric-infoton control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomIIInfotonGeometrySpec:
    """Axiom II infoton-geometry spec promoted from Paper 0 records."""

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
class AxiomIIInfotonGeometrySpecBundle:
    """Axiom II infoton-geometry specs plus source coverage summary."""

    specs: tuple[AxiomIIInfotonGeometrySpec, ...]
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


def build_axiom_ii_infoton_geometry_specs(
    source_records: list[dict[str, Any]],
) -> AxiomIIInfotonGeometrySpecBundle:
    """Build source-covered Axiom II infoton-geometry specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomIIInfotonGeometrySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomIIInfotonGeometrySpec(
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
        "title": "Paper 0 Axiom II Infoton Geometry Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "gauge_necessity_count": 1,
        "baseline_lagrangian_count": 1,
        "fim_claim_count": 1,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00775",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomIIInfotonGeometrySpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AxiomIIInfotonGeometrySpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiom_ii_infoton_geometry_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomIIInfotonGeometrySpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiom II Infoton Geometry Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Gauge-necessity records: {bundle.summary['gauge_necessity_count']}",
        f"- Baseline Lagrangian records: {bundle.summary['baseline_lagrangian_count']}",
        f"- FIM claim records: {bundle.summary['fim_claim_count']}",
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
    bundle: AxiomIIInfotonGeometrySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_axiom_ii_infoton_geometry_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_axiom_ii_infoton_geometry_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Axiom II infoton-geometry specs from the ledger."""

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
