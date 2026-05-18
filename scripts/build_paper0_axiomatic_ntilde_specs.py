#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 axiomatic Ntilde spec builder
"""Promote Paper 0 formal Logos/Ntilde records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(578, 610))
CLAIM_BOUNDARY = "source-bounded formal Logos and Ntilde invariant; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiomatic_ntilde.formal_axiom_system_boundary": {
        "context_id": "formal_axiom_system_boundary",
        "validation_protocol": "paper0.axiomatic_ntilde.formal_axiom_system_boundary",
        "canonical_statement": (
            "The formal Logos section states that the SCPN framework is grounded in "
            "a minimal axiom set that provides causal closure."
        ),
        "source_equation_ids": (
            "P0R00578:axiomatic_system_header",
            "P0R00579:logos_causal_closure",
        ),
        "source_formulae": (
            "The Axiomatic System (The Logos)",
            "minimal set of axioms",
            "causal closure (L13)",
        ),
        "test_protocols": ("preserve formal Logos source boundary",),
        "null_results": ("formal axiom framing is not empirical validation",),
        "variables": ("Logos", "axioms", "L13"),
        "validation_targets": (
            "preserve causal-closure framing",
            "preserve minimal-axiom-set boundary",
        ),
        "null_controls": (
            "axiom-system-as-observed-result control must be rejected",
            "missing-causal-closure-label control must be rejected",
        ),
    },
    "axiomatic_ntilde.axiom_three_status_transition": {
        "context_id": "axiom_three_status_transition",
        "validation_protocol": "paper0.axiomatic_ntilde.axiom_three_status_transition",
        "canonical_statement": (
            "The section restates Axiom 3 as teleological/normative, then introduces "
            "a proposed falsifiable Ntilde physical invariant; the transition must remain explicit."
        ),
        "source_equation_ids": (
            "P0R00581:axiom1_existence",
            "P0R00582:axiom1_generative_hypothesis",
            "P0R00584:axiom2_information_geometry",
            "P0R00586:axiom3_teleological_normative",
            "P0R00591:axiom3_falsifiable_ntilde_law",
        ),
        "source_formulae": (
            "Axiom 1: Consciousness is the irreducible ontological primitive",
            "generative hypothesis",
            "Axiom 2: interactions are informational and geometric",
            "Axiom 3: teleological/normative postulate",
            "Axiom III status tension: normative teleology plus proposed falsifiable invariant",
        ),
        "test_protocols": ("preserve Axiom 3 status-transition boundary",),
        "null_results": ("proposed physical invariant is not empirical confirmation",),
        "variables": ("axiom_1", "axiom_2", "axiom_3", "Ntilde"),
        "validation_targets": (
            "preserve Axiom 1 generative-metaphysical status",
            "preserve Axiom 2 falsifiable physical-hypothesis status",
            "preserve Axiom 3 normative-to-physical claim transition",
        ),
        "null_controls": (
            "status-transition-erasure control must be rejected",
            "axiom3-confirmed-physical-law control must be rejected",
        ),
    },
    "axiomatic_ntilde.ntilde_invariant_definition": {
        "context_id": "ntilde_invariant_definition",
        "validation_protocol": "paper0.axiomatic_ntilde.ntilde_invariant_definition",
        "canonical_statement": (
            "Ntilde is introduced as a dimensionless measurable ratio linking power, "
            "reversible free-energy cost per bit, and reliable information rate."
        ),
        "source_equation_ids": (
            "P0R00592:dimensionless_measurable_invariant",
            "P0R00593:Ntilde=P/(epsilon_b*I_dot)",
            "P0R00595:P=E/t",
            "P0R00596:I_dot_information_rate",
            "P0R00597:epsilon_b_reversible_cost_per_bit",
        ),
        "source_formulae": (
            "dimensionless measurable invariant",
            "Ntilde = P / (epsilon_b * I_dot)",
            "P = E / t",
            "I_dot = reliably processed information rate",
            "epsilon_b = Delta F_rev / Delta I",
        ),
        "test_protocols": ("compute guarded dimensionless Ntilde ratio",),
        "null_results": ("invalid nonpositive inputs reject ratio claims",),
        "variables": ("Ntilde", "P", "epsilon_b", "I_dot"),
        "validation_targets": (
            "preserve source ratio definition",
            "preserve positive finite input constraints",
        ),
        "null_controls": (
            "zero-power ratio control must be rejected",
            "negative-information-rate control must be rejected",
        ),
    },
    "axiomatic_ntilde.unity_irreversibility_target": {
        "context_id": "unity_irreversibility_target",
        "validation_protocol": "paper0.axiomatic_ntilde.unity_irreversibility_target",
        "canonical_statement": (
            "The source identifies the teleological target as Ntilde approaching one, "
            "with the tight formulation Ntilde = 1 + delta_irr."
        ),
        "source_equation_ids": (
            "P0R00598:teleological_drive_to_unity",
            "P0R00599:Ntilde_to_1",
            "P0R00600:Ntilde=1+delta_irr",
            "P0R00601:SEC_as_macroscopic_unity_state",
        ),
        "source_formulae": (
            "Ntilde -> 1",
            "Ntilde = 1 + delta_irr",
            "delta_irr is irreversibility or entropy production",
            "SEC is macroscopic state of Ntilde = 1",
        ),
        "test_protocols": ("classify Ntilde unity target and irreversibility delta",),
        "null_results": ("unity target is not an observed hardware result",),
        "variables": ("Ntilde", "delta_irr", "SEC"),
        "validation_targets": (
            "preserve reversible-threshold target",
            "preserve irreversibility-delta relation",
            "preserve SEC-as-target boundary",
        ),
        "null_controls": (
            "unity-as-measured-result control must be rejected",
            "negative-delta-as-efficiency-proof control must be rejected",
        ),
    },
    "axiomatic_ntilde.quasicritical_efficiency_target": {
        "context_id": "quasicritical_efficiency_target",
        "validation_protocol": "paper0.axiomatic_ntilde.quasicritical_efficiency_target",
        "canonical_statement": (
            "The repeated coherence-invariant framing maps Ntilde toward one to "
            "quasicriticality and maximum thermodynamic efficiency."
        ),
        "source_equation_ids": (
            "P0R00603:universal_coherence_invariant_validation_claim",
            "P0R00604:energy_information_time_ratio",
            "P0R00605:Ntilde=(E/t)/(epsilon_b*I_dot)",
            "P0R00606:self_organized_transition_threshold",
            "P0R00608:quasicriticality_definition",
            "P0R00609:efficiency_definition",
        ),
        "source_formulae": (
            "Universal Coherence Invariant",
            "Ntilde = (E/t) / (epsilon_b * I_dot)",
            "Ntilde = (E/t) / ((k_B T ln 2) * I_dot)",
            "self-organized coherent states where Ntilde -> 1",
            "E_dot_actual / E_dot_rev = 1 + delta_irr",
            "J_SEC",
        ),
        "test_protocols": ("preserve quasicritical-efficiency target mapping",),
        "null_results": ("internal repetition is not independent validation",),
        "variables": ("Ntilde", "E_dot_actual", "E_dot_rev", "delta_irr", "J_SEC"),
        "validation_targets": (
            "preserve quasicritical-controller target",
            "preserve thermodynamic-efficiency mapping",
            "preserve repeated-source-claim boundary",
        ),
        "null_controls": (
            "self-citation-as-independent-validation control must be rejected",
            "quasicritical-target-as-measured-transition control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomaticNtildeSpec:
    """Formal Logos/Ntilde spec promoted from Paper 0 records."""

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
class AxiomaticNtildeSpecBundle:
    """Formal Logos/Ntilde specs plus source coverage summary."""

    specs: tuple[AxiomaticNtildeSpec, ...]
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


def build_axiomatic_ntilde_specs(
    source_records: list[dict[str, Any]],
) -> AxiomaticNtildeSpecBundle:
    """Build source-covered formal Logos/Ntilde specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomaticNtildeSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomaticNtildeSpec(
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
        "title": "Paper 0 Axiomatic Ntilde Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "axiom_count": 3,
        "ntilde_formula_count": 5,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00610",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomaticNtildeSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> AxiomaticNtildeSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiomatic_ntilde_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomaticNtildeSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiomatic Ntilde Specs",
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
    bundle: AxiomaticNtildeSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_axiomatic_ntilde_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_axiomatic_ntilde_validation_specs_report_{date_tag}.md"
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 axiomatic n-tilde specs from the ledger."""

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
