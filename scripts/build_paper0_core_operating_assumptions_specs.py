#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 core operating assumptions spec builder
"""Promote Paper 0 core-operating-assumptions records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(635, 670))
BLANK_SEPARATOR_IDS = ("P0R00646", "P0R00661", "P0R00669")
CLAIM_BOUNDARY = "source-bounded core operating assumptions; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "core_operating_assumptions.five_assumption_bedrock": {
        "context_id": "five_assumption_bedrock",
        "validation_protocol": "paper0.core_operating_assumptions.five_assumption_bedrock",
        "canonical_statement": (
            "The SCPN core-operating-assumptions section compresses the framework "
            "into five source-bounded assumptions."
        ),
        "source_equation_ids": (
            "P0R00641:consciousness_fundamental",
            "P0R00642:bidirectional_causality",
            "P0R00643:field_realism",
            "P0R00644:unified_phase_dynamics",
            "P0R00645:ethical_functionals",
        ),
        "source_formulae": (
            "Consciousness Fundamentality",
            "Bidirectional Causality",
            "Field Realism",
            "Unified Phase Dynamics",
            "Ethical Functionals",
        ),
        "test_protocols": ("classify five operating assumptions",),
        "null_results": ("operating assumptions are not empirical results",),
        "variables": ("Psi", "causality", "field", "UPDE", "ethical_functional"),
        "validation_targets": (
            "preserve five-assumption count",
            "preserve assumption role labels",
            "preserve source-methodology boundary",
        ),
        "null_controls": (
            "assumptions-as-validation control must be rejected",
            "missing-assumption control must be rejected",
        ),
    },
    "core_operating_assumptions.predictive_coding_mapping": {
        "context_id": "predictive_coding_mapping",
        "validation_protocol": "paper0.core_operating_assumptions.predictive_coding_mapping",
        "canonical_statement": (
            "The five assumptions are mapped to active inference and hierarchical "
            "predictive coding roles."
        ),
        "source_equation_ids": (
            "P0R00649:active_inference_specification",
            "P0R00650:consciousness_as_inference_engine",
            "P0R00651:top_down_bottom_up_hpc",
            "P0R00652:physical_priors_as_field",
            "P0R00653:phase_synchrony_inference_mechanism",
            "P0R00654:ethical_functional_deep_prior",
        ),
        "source_formulae": (
            "active inference",
            "consciousness as inference engine",
            "top-down predictions and bottom-up prediction errors",
            "physical instantiation of priors as field",
            "phase synchrony and desynchrony implement inference",
            "Ethical Functional as deep prior",
        ),
        "test_protocols": ("preserve predictive-coding mapping for five assumptions",),
        "null_results": ("predictive-coding mapping is not empirical confirmation",),
        "variables": ("priors", "predictions", "prediction_error", "phase_synchrony"),
        "validation_targets": (
            "preserve top-down/bottom-up HPC mapping",
            "preserve phase-synchrony inference mapping",
            "preserve ethical-functional deep-prior boundary",
        ),
        "null_controls": (
            "active-inference-map-as-data control must be rejected",
            "ethical-prior-as-observed-objective control must be rejected",
        ),
    },
    "core_operating_assumptions.hint_assumption_roles": {
        "context_id": "hint_assumption_roles",
        "validation_protocol": "paper0.core_operating_assumptions.hint_assumption_roles",
        "canonical_statement": (
            "The assumptions provide source context for H_int by assigning roles to "
            "Psi_s, sigma, reciprocal causality, and lambda."
        ),
        "source_equation_ids": (
            "P0R00655:psis_field_coupling_integration",
            "P0R00656:H_int=-lambda*Psi_s*sigma",
            "P0R00657:Psi_s_real_field_context",
            "P0R00658:reciprocal_causality_context",
            "P0R00659:sigma_phase_coherence_candidate",
            "P0R00660:lambda_ethical_tuning_boundary",
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma",
            "Psi_s as real physical field",
            "reciprocal top-down and bottom-up interaction",
            "sigma as phase coherence or synchrony candidate",
            "lambda as ethical-functional tuning parameter",
        ),
        "test_protocols": ("classify H_int assumption roles",),
        "null_results": ("H_int context is not fitted coupling evidence",),
        "variables": ("H_int", "lambda", "Psi_s", "sigma"),
        "validation_targets": (
            "preserve Psi_s field-context role",
            "preserve reciprocal-causality role",
            "preserve sigma phase-coherence candidate role",
            "preserve lambda tuning role",
        ),
        "null_controls": (
            "h_int-context-as-fit control must be rejected",
            "lambda-as-new-force control must be rejected",
        ),
    },
    "core_operating_assumptions.lambda_ethical_tuning_boundary": {
        "context_id": "lambda_ethical_tuning_boundary",
        "validation_protocol": "paper0.core_operating_assumptions.lambda_ethical_tuning_boundary",
        "canonical_statement": (
            "Ethical Functionals are explicitly bounded as selective pressure over "
            "lambda rather than a new force term in H_int."
        ),
        "source_equation_ids": (
            "P0R00660:ethical_functionals_tune_lambda_not_force",
            "P0R00668:layer15_objective_functions",
        ),
        "source_formulae": (
            "Ethical Functionals tune lambda and do not add a force term",
            "selective pressure over evolutionary timescales",
            "Layer 15 objective functions",
            "increasing coherence, complexity, and experiential depth",
        ),
        "test_protocols": ("preserve lambda-tuning not-force boundary",),
        "null_results": ("teleological tuning is not a measured force term",),
        "variables": ("lambda", "H_int", "Layer_15", "SEC"),
        "validation_targets": (
            "preserve not-a-force-term boundary",
            "preserve higher-level tuning boundary",
            "preserve Layer 15 teleological-objective boundary",
        ),
        "null_controls": (
            "ethical-functional-force-term control must be rejected",
            "lambda-tuning-as-observed-dynamics control must be rejected",
        ),
    },
    "core_operating_assumptions.revised_programme_boundary": {
        "context_id": "revised_programme_boundary",
        "validation_protocol": "paper0.core_operating_assumptions.revised_programme_boundary",
        "canonical_statement": (
            "The v8.6 programme claim references QEC, CISS, and quasicriticality as "
            "integrated refinements, but remains an internal-audit programme boundary."
        ),
        "source_equation_ids": (
            "P0R00636:v86_research_programme",
            "P0R00662:multiscale_v86_integrations",
            "P0R00663:five_foundational_assumptions_restatement",
            "P0R00664:consciousness_restatement",
            "P0R00665:bidirectional_causality_restatement",
            "P0R00666:field_realism_restatement",
            "P0R00667:upde_spine_restatement",
            "P0R00668:teleology_restatement",
            "P0R00669:blank_separator",
        ),
        "source_formulae": (
            "v8.6 research programme",
            "MS-QEC",
            "CISS",
            "Quasicriticality",
            "five foundational assumptions",
            "internal auditing",
        ),
        "test_protocols": ("preserve revised-programme and restatement boundary",),
        "null_results": ("internal auditing reference is not external validation",),
        "variables": ("MS_QEC", "CISS", "quasicriticality", "UPDE"),
        "validation_targets": (
            "preserve integrated-refinement labels",
            "preserve restated five-assumption boundary",
            "preserve internal-audit not-validation boundary",
        ),
        "null_controls": (
            "internal-audit-as-external-validation control must be rejected",
            "v86-as-final-version control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class CoreOperatingAssumptionsSpec:
    """Core operating assumptions spec promoted from Paper 0 records."""

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
class CoreOperatingAssumptionsSpecBundle:
    """Core operating assumptions specs plus source coverage summary."""

    specs: tuple[CoreOperatingAssumptionsSpec, ...]
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


def build_core_operating_assumptions_specs(
    source_records: list[dict[str, Any]],
) -> CoreOperatingAssumptionsSpecBundle:
    """Build source-covered core-operating-assumptions specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[CoreOperatingAssumptionsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            CoreOperatingAssumptionsSpec(
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
        "title": "Paper 0 Core Operating Assumptions Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "core_assumption_count": 5,
        "blank_separator_count": len(BLANK_SEPARATOR_IDS),
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00670",
        "spec_keys": [spec.key for spec in specs],
    }
    return CoreOperatingAssumptionsSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> CoreOperatingAssumptionsSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_core_operating_assumptions_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: CoreOperatingAssumptionsSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Core Operating Assumptions Specs",
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
    bundle: CoreOperatingAssumptionsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_core_operating_assumptions_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_core_operating_assumptions_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 core operating-assumption specs from the ledger."""

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
