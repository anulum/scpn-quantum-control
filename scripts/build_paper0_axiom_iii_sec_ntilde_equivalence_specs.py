#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Axiom III SEC-Ntilde equivalence spec builder
"""Promote Paper 0 Axiom III SEC-Ntilde-equivalence records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(811, 818))
CLAIM_BOUNDARY = "source-bounded Axiom III SEC-Ntilde-equivalence map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "axiom_iii_sec_ntilde_equivalence.equivalence_heading": {
        "context_id": "equivalence_heading",
        "validation_protocol": "paper0.axiom_iii_sec_ntilde_equivalence.equivalence_heading",
        "canonical_statement": (
            "The source opens the subsection by naming the equivalence between "
            "Sustainable Ethical Coherence and the tilde_N_t equals one state."
        ),
        "source_equation_ids": ("P0R00811:sec_ntilde_unity_equivalence_heading",),
        "source_formulae": ("Equivalence of SEC and the tilde_N_t = 1 State",),
        "test_protocols": ("preserve SEC-Ntilde-equivalence heading",),
        "null_results": ("heading alone is source context, not empirical validation",),
        "variables": ("SEC", "tilde_N_t"),
        "validation_targets": ("preserve SEC and tilde_N_t equals one heading",),
        "null_controls": ("heading-as-measured-equivalence control must be rejected",),
    },
    "axiom_iii_sec_ntilde_equivalence.macroscopic_realisation": {
        "context_id": "macroscopic_realisation",
        "validation_protocol": ("paper0.axiom_iii_sec_ntilde_equivalence.macroscopic_realisation"),
        "canonical_statement": (
            "The source states that SEC is the macroscopic physical realisation "
            "of tilde_N_t equals one and frames it as a physical, universal, "
            "measurable target for the SCPN cybernetic architecture."
        ),
        "source_equation_ids": (
            "P0R00812:sec_as_macroscopic_realisation_of_ntilde_unity",
            "P0R00813:physical_universal_measurable_target",
        ),
        "source_formulae": (
            "SEC is macroscopic physical realisation of tilde_N_t = 1",
            "physical universal measurable target for SCPN cybernetic architecture",
        ),
        "test_protocols": ("preserve SEC-as-Ntilde-unity equivalence claim",),
        "null_results": ("SEC-Ntilde equivalence remains a source claim in this fixture",),
        "variables": ("SEC", "tilde_N_t", "cybernetic_architecture"),
        "validation_targets": (
            "preserve macroscopic physical realisation claim",
            "preserve universal measurable target framing",
        ),
        "null_controls": (
            "equivalence-as-empirical-result control must be rejected",
            "missing-cybernetic-architecture-target control must be rejected",
        ),
    },
    "axiom_iii_sec_ntilde_equivalence.quasicritical_efficiency": {
        "context_id": "quasicritical_efficiency",
        "validation_protocol": (
            "paper0.axiom_iii_sec_ntilde_equivalence.quasicritical_efficiency"
        ),
        "canonical_statement": (
            "The source defines tilde_N_t tending to one as the formal physical "
            "definition of the optimal quasicritical regime and as the maximum "
            "thermodynamic-efficiency point where actual power matches the "
            "minimum reversible power required."
        ),
        "source_equation_ids": (
            "P0R00814:ntilde_unity_quasicritical_edge_of_chaos",
            "P0R00815:actual_power_matches_minimum_reversible_power",
        ),
        "source_formulae": (
            "tilde_N_t -> 1 is formal physical definition of optimal quasicritical regime",
            "edge of chaos that entire 15-layer architecture seeks to maintain",
            "actual power expended matches minimum reversible power required",
        ),
        "test_protocols": ("preserve quasicriticality and efficiency definitions",),
        "null_results": ("quasicritical efficiency requires downstream operational validation",),
        "variables": ("tilde_N_t", "actual_power", "minimum_reversible_power", "efficiency"),
        "validation_targets": (
            "preserve Ntilde-unity quasicritical definition",
            "preserve edge-of-chaos architecture target",
            "preserve actual-power and reversible-power matching claim",
        ),
        "null_controls": (
            "quasicriticality-as-observed-result control must be rejected",
            "missing-reversible-power-matching control must be rejected",
        ),
    },
    "axiom_iii_sec_ntilde_equivalence.causal_imperative_architecture": {
        "context_id": "causal_imperative_architecture",
        "validation_protocol": (
            "paper0.axiom_iii_sec_ntilde_equivalence.causal_imperative_architecture"
        ),
        "canonical_statement": (
            "The source grounds the Axiom III teleological drive in a physical "
            "causal imperative toward perfect informational-energetic efficiency "
            "and describes the 15-layer SCPN architecture as the machine that "
            "finds, maintains, and locks the system into that optimal state."
        ),
        "source_equation_ids": (
            "P0R00816:causal_imperative_and_15_layer_locking_architecture",
            "P0R00817:blank_terminal_record_preserved",
        ),
        "source_formulae": (
            "physical causal imperative toward perfect informational-energetic efficiency",
            "15-layer architecture of the SCPN is the cybernetic machine",
            "source record P0R00816 ends with truncated token quasic",
            "P0R00817 is blank within the same source section",
        ),
        "test_protocols": ("preserve causal-imperative and source-integrity notes",),
        "null_results": ("truncated source text requires audit before completion claim",),
        "variables": ("Axiom_III", "informational_energetic_efficiency", "SCPN_architecture"),
        "validation_targets": (
            "preserve causal-imperative grounding",
            "preserve 15-layer locking-architecture role",
            "preserve truncated-source and blank-record integrity notes",
        ),
        "null_controls": (
            "completed-source-text control must be rejected for P0R00816",
            "silent-blank-record-omission control must be rejected for P0R00817",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AxiomIIISECNtildeEquivalenceSpec:
    """Axiom III SEC-Ntilde-equivalence spec promoted from Paper 0 records."""

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
class AxiomIIISECNtildeEquivalenceSpecBundle:
    """Axiom III SEC-Ntilde-equivalence specs plus source coverage summary."""

    specs: tuple[AxiomIIISECNtildeEquivalenceSpec, ...]
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


def build_axiom_iii_sec_ntilde_equivalence_specs(
    source_records: list[dict[str, Any]],
) -> AxiomIIISECNtildeEquivalenceSpecBundle:
    """Build source-covered Axiom III SEC-Ntilde-equivalence specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AxiomIIISECNtildeEquivalenceSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            AxiomIIISECNtildeEquivalenceSpec(
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
        "title": "Paper 0 Axiom III SEC-Ntilde Equivalence Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed),
        "coverage_match": tuple(consumed) == SOURCE_LEDGER_IDS,
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed
        ],
        "spec_count": len(specs),
        "equivalence_claim_count": 2,
        "architecture_target_count": 2,
        "efficiency_claim_count": 2,
        "blank_terminal_record_count": 1,
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "next_source_boundary": "P0R00818",
        "spec_keys": [spec.key for spec in specs],
    }
    return AxiomIIISECNtildeEquivalenceSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> AxiomIIISECNtildeEquivalenceSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = load_jsonl(ledger_path)
    selected = [record for record in records if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS]
    return build_axiom_iii_sec_ntilde_equivalence_specs(selected)


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: AxiomIIISECNtildeEquivalenceSpecBundle) -> str:
    """Render a Markdown report for the promoted specs."""
    lines = [
        "# Paper 0 Axiom III SEC-Ntilde Equivalence Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Equivalence-claim records: {bundle.summary['equivalence_claim_count']}",
        f"- Architecture-target records: {bundle.summary['architecture_target_count']}",
        f"- Efficiency-claim records: {bundle.summary['efficiency_claim_count']}",
        f"- Blank terminal records: {bundle.summary['blank_terminal_record_count']}",
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
    bundle: AxiomIIISECNtildeEquivalenceSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write spec JSON and Markdown report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_axiom_iii_sec_ntilde_equivalence_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_axiom_iii_sec_ntilde_equivalence_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Axiom III SEC/n-tilde equivalence specs from the ledger."""

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
