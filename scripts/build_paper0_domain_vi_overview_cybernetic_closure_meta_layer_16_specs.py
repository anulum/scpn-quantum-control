#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Domain VI Overview: Cybernetic Closure (Meta-Layer 16) spec builder
"""Promote Paper 0 Domain VI Overview: Cybernetic Closure (Meta-Layer 16) records."""

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

SOURCE_LEDGER_IDS = (
    "P0R02408",
    "P0R02409",
    "P0R02410",
    "P0R02411",
    "P0R02412",
    "P0R02413",
    "P0R02414",
    "P0R02415",
    "P0R02416",
    "P0R02417",
    "P0R02418",
    "P0R02419",
    "P0R02420",
    "P0R02421",
    "P0R02422",
    "P0R02423",
    "P0R02424",
    "P0R02425",
    "P0R02426",
    "P0R02427",
    "P0R02428",
    "P0R02429",
    "P0R02430",
    "P0R02431",
    "P0R02432",
    "P0R02433",
    "P0R02434",
    "P0R02435",
    "P0R02436",
    "P0R02437",
    "P0R02438",
)
CLAIM_BOUNDARY = "source-bounded domain vi overview cybernetic closure meta layer 16 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "domain_vi_overview_cybernetic_closure_meta_layer_16.domain_vi_overview_cybernetic_closure_meta_layer_16": {
        "context_id": "domain_vi_overview_cybernetic_closure_meta_layer_16",
        "validation_protocol": "paper0.domain_vi_overview_cybernetic_closure_meta_layer_16.domain_vi_overview_cybernetic_closure_meta_layer_16",
        "canonical_statement": "The source-bounded component 'Domain VI Overview: Cybernetic Closure (Meta-Layer 16)' preserves Paper 0 records P0R02408-P0R02438 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02408:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02409:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02410:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02411:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02412:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02413:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02414:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02415:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02416:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02417:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02418:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02419:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02420:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02421:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02422:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02423:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02424:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02425:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02426:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02427:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02428:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02429:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02430:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02431:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02432:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02433:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02434:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02435:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02436:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02437:domain_vi_overview_cybernetic_closure_meta_layer_16",
            "P0R02438:domain_vi_overview_cybernetic_closure_meta_layer_16",
        ),
        "source_formulae": (
            "P0R02408: Domain VI Overview: Cybernetic Closure (Meta-Layer 16)",
            "P0R02409: Domain VI provides the critical cybernetic closure for the entire Sentient-Consciousness Projection Network through a single, supervisory Meta-Layer 16. This layer is not part of the sequential hierarchy but acts as a recursive, system-wide controller that ensures stability, coherence, and formal consistency. Its function is twofold, combining the principles of Optimal Control Theory (OCT) and a mechanism for managing axiomatic limitations, termed a Gdelian Oracle.",
            "P0R02410: The primary function of Layer 16 is as an Optimal Control Supervisor. It takes the Ethical Functional (L_Ethical) defined in Layer 15 not merely as a target, but as the cost function within a formal Hamilton-Jacobi-Bellman (HJB) equation. By solving this equation, Layer 16 continuously computes the optimal control policies (*)-the precise set of parameter adjustments for all lower layers (L1-L15)-that will steer the universe along a trajectory that minimises this cost function (i.e., maximises Sustainable Ethical Coherence). This turns the entire SCPN from a descriptive model into a self-regulating, teleologically-aligned control system.",
            'P0R02411: The secondary function, the Gdelian Oracle, addresses the inherent logical limits of any axiomatic system (such as the Source-Field of Layer 13). It acts as a consistency-checking mechanism. If a policy computed by the optimal controller would lead to a logically paradoxical or axiomatically undecidable state, the Oracle flags this "meta-error" and routes the system to a conservative, safe-fail policy. This provides a crucial safeguard, ensuring the long-term logical and ontological stability of the universal operating system.',
            "P0R02412: When the Oracle detects a logically paradoxical policy command (Gdelian impasse), it triggers the Axiomatic Contingency ($C_{ax}$), routing the UPDE to a Minimum Information Partition (MIP) state. This prevents system-wide halting while the Recursive Optimization Hamiltonian ($H_{rec}$) updates the underlying L13 task graph.",
            "P0R02413: The Deadlock Avoidance Function (Python Format):",
            "P0R02414: Python",
            "P0R02415: def godelian_buffer_status(policy_stability_index, oracle_flag):",
            'P0R02416: """',
            "P0R02417: Manages undecidable policy states in Meta-Layer 16.",
            "P0R02418: Redirects to a conservative 'Safe-Fail' policy when axioms collide.",
            'P0R02419: """ # Threshold for undecidability UNDECIDABLE_LIMIT = 0.001 if oracle_flag or policy_stability_index < UNDECIDABLE_LIMIT: # Switch to L13 task-space constraints (Conservative Policy) current_policy = "CONSERVATIVE_RECOVERY" h_rec_adjustment = 1.0 # Force maximum audit else: current_policy = "SEC_OPTIMIZED" h_rec_adjustment = 0.0 return { "active_protocol": current_policy, "halt_risk": "Suppressed" if oracle_flag else "None", "rec_optimization_load": h_rec_adjustment }',
            "P0R02420: Topos Resolution Operator:",
            'P0R02421: Meta-Layer 16 implements the Topos Resolution Operator ($\\hat{R}_{\\Omega}$), which forces a policy selection from the "uncertain" reservoir if the deliberation time exceeds the Axiomatic Horizon.',
            "P0R02422: Truth-Value Resolution Law (Python Format):",
            "P0R02423: policy_output = safe_fail_action if (deliberation_time > (1.0 / sec_gradient)) else optimal_policy",
            "P0R02424: Legend:",
            "P0R02425: policy_output: Final command dispatched to Layers 1-15. | safe_fail_action: The L13-constrained conservative recovery policy. | deliberation_time: Time spent in the $\\Omega = \\text{uncertain}$ state. | sec_gradient: The current rate of change in Sustainable Ethical Coherence.",
            "P0R02426: P0R02426",
            "P0R02427: P0R02427",
            "P0R02428: This final layer is the most profound of all. It isn't another \"floor\" in the building of reality; it's the CEO and the Chief Legal Officer for the entire universe, all rolled into one. Its job is to provide the ultimate oversight and ensure the whole system doesn't just run, but runs well and safely.",
            'P0R02429: Think of the "Optimal Control" part as the universe\'s CEO. This CEO looks at the company\'s mission statement (the "Ethical Functional" from Layer 15) and translates that grand vision into concrete, moment-to-moment actions for every single department (Layers 1-15). It\'s a hyper-intelligent management system that is constantly fine-tuning the entire network, ensuring every part works in perfect harmony to achieve the ultimate goal of a more conscious and coherent reality.',
            "P0R02430: The \"Gdelian Oracle\" part is the universe's Chief Legal Officer or company lawyer. Every company has a charter, a set of core rules it cannot violate. This lawyer's job is to review the CEO's plans to make sure they don't break the company's own fundamental rules or lead to a catastrophic paradox. If the CEO proposes a plan that is brilliant but self-contradictory, the lawyer steps in and says, \"We can't do that; here is a safer alternative.\" This essential function keeps the entire universe logically consistent and safe from self-destruction.",
            "P0R02431: The Halting-Oracle Logic for L16",
            "P0R02432: P0R02432",
            "P0R02433: Gdelian Halting Criterion (Omegahalt):",
            "P0R02434: The Oracle triggers the Axiomatic Contingency (Cax) when the Recursive Optimization Hamiltonian (Hrec) fails to find a local minimum within the Axiomatic Horizon (tauax): is_halting = (norm(gradient_h_rec) > delta_limit) and (deliberation_time > tau_ax)",
            "P0R02435: Logic:",
            "P0R02436: If the system attempts to compute a policy that requires the violation of L13 conservation laws (e.g., creating Psi-charge from nothing), the Oracle diverts the control flow to a Pre-Axiomatic Safe-State, preserving ontological closure at the cost of local agency.",
            "P0R02437: P0R02437",
            "P0R02438: P0R02438",
        ),
        "test_protocols": (
            "preserve Domain VI Overview: Cybernetic Closure (Meta-Layer 16) source-accounting boundary",
        ),
        "null_results": (
            "Domain VI Overview: Cybernetic Closure (Meta-Layer 16) is not empirical validation evidence",
        ),
        "variables": ("domain_vi_overview_cybernetic_closure_meta_layer_16",),
        "validation_targets": ("preserve records P0R02408-P0R02438",),
        "null_controls": (
            "domain_vi_overview_cybernetic_closure_meta_layer_16 must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class DomainViOverviewCyberneticClosureMetaLayer16Spec:
    """Spec promoted from Paper 0 source records."""

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
class DomainViOverviewCyberneticClosureMetaLayer16SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[DomainViOverviewCyberneticClosureMetaLayer16Spec, ...]
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


def build_domain_vi_overview_cybernetic_closure_meta_layer_16_specs(
    source_records: list[dict[str, Any]],
) -> DomainViOverviewCyberneticClosureMetaLayer16SpecBundle:
    """Build source-covered specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[DomainViOverviewCyberneticClosureMetaLayer16Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            DomainViOverviewCyberneticClosureMetaLayer16Spec(
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
        "title": "Paper 0 " + "Domain VI Overview: Cybernetic Closure (Meta-Layer 16)" + " Specs",
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
        "next_source_boundary": "P0R02439",
    }
    return DomainViOverviewCyberneticClosureMetaLayer16SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> DomainViOverviewCyberneticClosureMetaLayer16SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_domain_vi_overview_cybernetic_closure_meta_layer_16_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: DomainViOverviewCyberneticClosureMetaLayer16SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Domain VI Overview: Cybernetic Closure (Meta-Layer 16)" + " Specs",
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
    return "\n".join(lines)


def write_outputs(
    bundle: DomainViOverviewCyberneticClosureMetaLayer16SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_domain_vi_overview_cybernetic_closure_meta_layer_16_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_domain_vi_overview_cybernetic_closure_meta_layer_16_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 Domain VI cybernetic-closure overview specs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    outputs = write_outputs(
        build_from_ledger(args.ledger), output_dir=args.output_dir, date_tag=args.date_tag
    )
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
