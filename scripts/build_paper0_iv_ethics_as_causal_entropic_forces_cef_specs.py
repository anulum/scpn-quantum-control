#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 IV. Ethics as Causal Entropic Forces (CEF): spec builder
"""Promote Paper 0 IV. Ethics as Causal Entropic Forces (CEF): records."""

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

SOURCE_LEDGER_IDS = (
    "P0R06107",
    "P0R06108",
    "P0R06109",
    "P0R06110",
    "P0R06111",
    "P0R06112",
    "P0R06113",
    "P0R06114",
)
CLAIM_BOUNDARY = "source-bounded iv ethics as causal entropic forces cef source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "iv_ethics_as_causal_entropic_forces_cef.iv_ethics_as_causal_entropic_forces_cef": {
        "context_id": "iv_ethics_as_causal_entropic_forces_cef",
        "validation_protocol": "paper0.iv_ethics_as_causal_entropic_forces_cef.iv_ethics_as_causal_entropic_forces_cef",
        "canonical_statement": "The source-bounded component 'IV. Ethics as Causal Entropic Forces (CEF):' preserves Paper 0 records P0R06107-P0R06113 without empirical validation claims.",
        "source_equation_ids": (
            "P0R06107:iv_ethics_as_causal_entropic_forces_cef",
            "P0R06108:iv_ethics_as_causal_entropic_forces_cef",
            "P0R06109:iv_ethics_as_causal_entropic_forces_cef",
            "P0R06110:iv_ethics_as_causal_entropic_forces_cef",
            "P0R06111:iv_ethics_as_causal_entropic_forces_cef",
            "P0R06112:iv_ethics_as_causal_entropic_forces_cef",
            "P0R06113:iv_ethics_as_causal_entropic_forces_cef",
        ),
        "source_formulae": (
            "P0R06107: IV. Ethics as Causal Entropic Forces (CEF):",
            "P0R06108: The Ethical Functional exerts physical influence via Causal Entropic Forces (CEF). Systems evolve to maximise their future causal pathway entropy (SC). SEC defines the configuration space with the highest SC.",
            "P0R06109: $FCausal = TC\\nabla X SC(X,\\tau)$",
            "P0R06110: This force (FCausal) biases dynamics at all layers-guiding evolution (L3/L8), biasing Free Energy minimisation (L5/HPC), and influencing quantum collapse (L1/IIT-OR)-grounding the Ethical Functional in causal physics.",
            'P0R06111: "Ethical naturalism" defence',
            "P0R06112: \"We treat SEC not as deriving ought' from is' but as a selection principle on feasible attractors: systems that sustain coherence, complexity, and qualia persist under UPDE-constrained dynamics. The ethical' label names this feasibility frontier, not a moral fiat; L16 ensures continual audit against energy/information constraints.\"",
            "P0R06113: P0R06113",
        ),
        "test_protocols": (
            "preserve IV. Ethics as Causal Entropic Forces (CEF): source-accounting boundary",
        ),
        "null_results": (
            "IV. Ethics as Causal Entropic Forces (CEF): is not empirical validation evidence",
        ),
        "variables": ("iv_ethics_as_causal_entropic_forces_cef",),
        "validation_targets": ("preserve records P0R06107-P0R06113",),
        "null_controls": (
            "iv_ethics_as_causal_entropic_forces_cef must remain source-bounded accounting",
        ),
    },
    "iv_ethics_as_causal_entropic_forces_cef.overarching_principles_and_system_dynamics_in_short": {
        "context_id": "overarching_principles_and_system_dynamics_in_short",
        "validation_protocol": "paper0.iv_ethics_as_causal_entropic_forces_cef.overarching_principles_and_system_dynamics_in_short",
        "canonical_statement": "The source-bounded component 'Overarching Principles and System Dynamics In short' preserves Paper 0 records P0R06114-P0R06114 without empirical validation claims.",
        "source_equation_ids": ("P0R06114:overarching_principles_and_system_dynamics_in_short",),
        "source_formulae": ("P0R06114: Overarching Principles and System Dynamics In short",),
        "test_protocols": (
            "preserve Overarching Principles and System Dynamics In short source-accounting boundary",
        ),
        "null_results": (
            "Overarching Principles and System Dynamics In short is not empirical validation evidence",
        ),
        "variables": ("overarching_principles_and_system_dynamics_in_short",),
        "validation_targets": ("preserve records P0R06114-P0R06114",),
        "null_controls": (
            "overarching_principles_and_system_dynamics_in_short must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class IvEthicsAsCausalEntropicForcesCefSpec:
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
class IvEthicsAsCausalEntropicForcesCefSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IvEthicsAsCausalEntropicForcesCefSpec, ...]
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


def build_iv_ethics_as_causal_entropic_forces_cef_specs(
    source_records: list[dict[str, Any]],
) -> IvEthicsAsCausalEntropicForcesCefSpecBundle:
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

    specs: list[IvEthicsAsCausalEntropicForcesCefSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IvEthicsAsCausalEntropicForcesCefSpec(
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
        "title": "Paper 0 " + "IV. Ethics as Causal Entropic Forces (CEF):" + " Specs",
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
        "next_source_boundary": "P0R06115",
    }
    return IvEthicsAsCausalEntropicForcesCefSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IvEthicsAsCausalEntropicForcesCefSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_iv_ethics_as_causal_entropic_forces_cef_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: IvEthicsAsCausalEntropicForcesCefSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "IV. Ethics as Causal Entropic Forces (CEF):" + " Specs",
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
    bundle: IvEthicsAsCausalEntropicForcesCefSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_iv_ethics_as_causal_entropic_forces_cef_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_iv_ethics_as_causal_entropic_forces_cef_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
