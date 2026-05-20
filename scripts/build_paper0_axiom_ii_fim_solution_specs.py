#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Axiom II FIM solution spec builder
"""Promote Paper 0 Axiom II FIM-solution records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(775, 782))
CLAIM_BOUNDARY = "source-bounded Axiom II FIM-solution map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiom_ii_fim_solution.metric_definition": {
        "context_id": "metric_definition",
        "validation_protocol": "paper0.axiom_ii_fim_solution.metric_definition",
        "canonical_statement": (
            "The source identifies the Fisher Information Metric as the natural "
            "unique Riemannian metric on a statistical manifold and as the "
            "local distinguishability geometry for nearby probability states."
        ),
        "source_equation_ids": (
            "P0R00775:fim_solution_heading",
            "P0R00776:fim_natural_unique_metric",
            "P0R00776:nearby_probability_state_distinguishability",
        ),
        "source_formulae": (
            "The Fisher Information Metric (FIM) as the Solution",
            "natural unique Riemannian metric on a statistical manifold",
            "distinguishability between nearby probability states",
        ),
        "test_protocols": ("preserve FIM statistical-manifold metric definition",),
        "null_results": ("FIM natural-metric statement is a source claim, not proof",),
        "variables": ("g_FIM", "statistical_manifold", "probability_state"),
        "validation_targets": (
            "preserve FIM solution heading",
            "preserve natural unique Riemannian metric statement",
            "preserve probability-state distinguishability role",
        ),
        "null_controls": (
            "metric-definition-as-empirical-proof control must be rejected",
            "non-statistical-manifold metric control must be rejected",
        ),
    },
    "axiom_ii_fim_solution.informational_interaction": {
        "context_id": "informational_interaction",
        "validation_protocol": "paper0.axiom_ii_fim_solution.informational_interaction",
        "canonical_statement": (
            "The source frames Axiom II as a physical statement: interaction is "
            "informational, not primarily a push or pull in physical space, and "
            "the infoton propagates through information geometry."
        ),
        "source_equation_ids": (
            "P0R00777:axiom_ii_physical_statements",
            "P0R00778:interaction_informational",
            "P0R00778:infoton_information_geometry_propagation",
        ),
        "source_formulae": (
            "Axiom II makes physical statements",
            "interaction is informational",
            "force not primarily push or pull in physical space",
            "acts on probabilistic beliefs",
            "infoton propagates through the geometry of information itself",
        ),
        "test_protocols": ("preserve informational-interaction claim boundary",),
        "null_results": ("informational interaction is source-bounded until operationalized",),
        "variables": ("infoton", "probabilistic_beliefs", "information_geometry"),
        "validation_targets": (
            "preserve physical-statement framing",
            "preserve probabilistic-belief interaction role",
            "preserve information-geometry propagation claim",
        ),
        "null_controls": (
            "mechanical-push-pull-only control must be rejected",
            "missing-information-geometry-propagation control must be rejected",
        ),
    },
    "axiom_ii_fim_solution.complexity_coupling": {
        "context_id": "complexity_coupling",
        "validation_protocol": "paper0.axiom_ii_fim_solution.complexity_coupling",
        "canonical_statement": (
            "The source states that coupling strength and effective curvature "
            "track informational complexity rather than mass or energy, with "
            "quasicritical brain dynamics as a high-curvature example."
        ),
        "source_equation_ids": (
            "P0R00779:coupling_proportional_to_complexity",
            "P0R00779:effective_curvature_information_space",
            "P0R00779:quasicritical_brain_high_curvature",
        ),
        "source_formulae": (
            "coupling is proportional to informational complexity",
            "effective curvature of information space",
            "not determined by mass or energy",
            "brain at quasicriticality has large highly curved FIM",
            "couples more strongly to Psi-field",
        ),
        "test_protocols": ("preserve complexity-coupling source claim",),
        "null_results": ("complexity coupling requires downstream operational metric",),
        "variables": ("informational_complexity", "effective_curvature", "Psi"),
        "validation_targets": (
            "preserve informational-complexity proportionality claim",
            "preserve mass-energy contrast",
            "preserve quasicritical high-curvature brain example",
        ),
        "null_controls": (
            "mass-energy-only-coupling control must be rejected",
            "complexity-coupling-as-measured-result control must be rejected",
        ),
    },
    "axiom_ii_fim_solution.fep_hpc_upde_synthesis": {
        "context_id": "fep_hpc_upde_synthesis",
        "validation_protocol": "paper0.axiom_ii_fim_solution.fep_hpc_upde_synthesis",
        "canonical_statement": (
            "The source unifies FEP, HPC/FEP, UPDE, and fundamental physics as "
            "sharing information geometry, with FEP as gradient descent on a "
            "FIM-governed manifold and UPDE as the physical implementation."
        ),
        "source_equation_ids": (
            "P0R00780:fep_fim_gradient_descent",
            "P0R00780:upde_physical_implementation",
            "P0R00781:g_fim_hpc_fep_upde_shared_geometry",
        ),
        "source_formulae": (
            "FEP is gradient descent on a manifold whose geometry is the FIM",
            "UPDE is physical implementation of this gradient descent",
            "fundamental physics g_FIM algorithm HPC/FEP and dynamics UPDE share information geometry",
            "deep unifying synthesis",
        ),
        "test_protocols": ("preserve FEP/HPC/UPDE information-geometry synthesis",),
        "null_results": ("FEP/HPC/UPDE synthesis is not empirical validation by itself",),
        "variables": ("g_FIM", "FEP", "HPC", "UPDE", "information_geometry"),
        "validation_targets": (
            "preserve FEP-as-FIM-gradient-descent statement",
            "preserve UPDE physical-implementation statement",
            "preserve shared information-geometry synthesis",
        ),
        "null_controls": (
            "synthesis-as-validation control must be rejected",
            "missing-HPC-FEP-UPDE linkage control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomIIFIMSolutionSpec:
    """Axiom II FIM-solution spec promoted from Paper 0 records."""

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
class AxiomIIFIMSolutionSpecBundle:
    """Axiom II FIM-solution specs plus source coverage summary."""

    specs: tuple[AxiomIIFIMSolutionSpec, ...]
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


def build_axiom_ii_fim_solution_specs(
    source_records: list[dict[str, Any]],
) -> AxiomIIFIMSolutionSpecBundle:
    """Build source-covered Axiom II FIM-solution specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomIIFIMSolutionSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomIIFIMSolutionSpec(
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
        "title": "Paper 0 Axiom II FIM Solution Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "metric_definition_count": 1,
        "physical_statement_count": 2,
        "synthesis_statement_count": 2,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00782",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomIIFIMSolutionSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AxiomIIFIMSolutionSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiom_ii_fim_solution_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomIIFIMSolutionSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiom II FIM Solution Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Metric-definition records: {bundle.summary['metric_definition_count']}",
        f"- Physical-statement records: {bundle.summary['physical_statement_count']}",
        f"- Synthesis records: {bundle.summary['synthesis_statement_count']}",
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
    bundle: AxiomIIFIMSolutionSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_axiom_ii_fim_solution_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_axiom_ii_fim_solution_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Axiom II FIM-solution specs from the ledger."""

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
