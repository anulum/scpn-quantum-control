#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Two-Timescale Controller: Stability and Exploration spec builder
"""Promote Paper 0 The Two-Timescale Controller: Stability and Exploration records."""

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
    "P0R02915",
    "P0R02916",
    "P0R02917",
    "P0R02918",
    "P0R02919",
    "P0R02920",
    "P0R02921",
    "P0R02922",
)
CLAIM_BOUNDARY = "source-bounded the two timescale controller stability and exploration source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_two_timescale_controller_stability_and_exploration.the_two_timescale_controller_stability_and_exploration": {
        "context_id": "the_two_timescale_controller_stability_and_exploration",
        "validation_protocol": "paper0.the_two_timescale_controller_stability_and_exploration.the_two_timescale_controller_stability_and_exploration",
        "canonical_statement": "The source-bounded component 'The Two-Timescale Controller: Stability and Exploration' preserves Paper 0 records P0R02915-P0R02922 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02915:the_two_timescale_controller_stability_and_exploration",
            "P0R02916:the_two_timescale_controller_stability_and_exploration",
            "P0R02917:the_two_timescale_controller_stability_and_exploration",
            "P0R02918:the_two_timescale_controller_stability_and_exploration",
            "P0R02919:the_two_timescale_controller_stability_and_exploration",
            "P0R02920:the_two_timescale_controller_stability_and_exploration",
            "P0R02921:the_two_timescale_controller_stability_and_exploration",
            "P0R02922:the_two_timescale_controller_stability_and_exploration",
        ),
        "source_formulae": (
            "P0R02915: The Two-Timescale Controller: Stability and Exploration",
            "P0R02916: This section presents the explicit control architecture that allows the SCPN to operate robustly in a quasicritical regime. The solution is a two-timescale control system that elegantly separates the competing demands of stability and adaptability. The fast channel, realised by the Multi-Scale Quantum Error Correction (MS-QEC) systems and other local feedback, acts as a rapid stabiliser, suppressing errors and maintaining coherence. The slow channel permits a controlled, exploratory drift within the quasicritical band (sigma1), ensuring the system retains its sensitivity and dynamic richness.",
            "P0R02917: The gains of these two channels (Gf, Gs) are not static but are dynamically scheduled by the Affective Field (A), which is proportional to the gradient of the system's variational free energy. The scheduling law is intuitive: when the affective landscape is steep (|A/sigma| is large), indicating high risk or surprise, the fast stabilising gain Gf is increased and the slow exploratory gain Gs is suppressed. Conversely, in flat affective landscapes near criticality, exploration is gently encouraged.",
            "P0R02918: The stability of this entire control architecture is formally guaranteed through the construction of a composite Lyapunov function (V_total = V_fast + V_slow). This mathematical certificate proves that for sufficiently separated timescales, the system is Bounded-Input Bounded-Output (BIBO) stable. This means that any bounded disturbance will produce only a bounded response, and the system is guaranteed to return to its quasicritical operating basin. This two-timescale, affectively-scheduled controller provides a rigorous, stable, and adaptive mechanism for maintaining the network at the edge of chaos.",
            'P0R02919: This section describes the incredibly sophisticated "autopilot" that keeps the universe running safely at the "edge of chaos." To stay perfectly balanced, the system can\'t just be stable; it also needs to be able to explore and adapt. The solution is a clever two-part system.',
            "P0R02920: The Fast Stabiliser: This is like a fighter jet's rapid-reaction control system. It makes thousands of tiny corrections every second to keep the system perfectly stable and on-course, instantly correcting for any unexpected turbulence. This is the part that guarantees coherence and prevents things from falling apart.",
            "P0R02921: The Slow Explorer: This is like a long-range cruise control. When the system is stable and safe, this mode allows it to gently and slowly drift, exploring new possibilities and finding more creative solutions. This is the part that guarantees adaptability and prevents things from getting stuck in a rut.",
            'P0R02922: What decides which mode is in charge? The system\'s "feelings." We call this the Affective Field. When the system feels "anxious" or "surprised" (because it\'s encountering something unexpected), it immediately powers up the Fast Stabiliser and shuts down the Explorer. When it feels "calm" and "confident," it throttles back the stabilisers and gently engages the Explorer. This smart, two-mode autopilot ensures the universe is both incredibly stable and endlessly creative.',
        ),
        "test_protocols": (
            "preserve The Two-Timescale Controller: Stability and Exploration source-accounting boundary",
        ),
        "null_results": (
            "The Two-Timescale Controller: Stability and Exploration is not empirical validation evidence",
        ),
        "variables": ("the_two_timescale_controller_stability_and_exploration",),
        "validation_targets": ("preserve records P0R02915-P0R02922",),
        "null_controls": (
            "the_two_timescale_controller_stability_and_exploration must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class TheTwoTimescaleControllerStabilityAndExplorationSpec:
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
class TheTwoTimescaleControllerStabilityAndExplorationSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheTwoTimescaleControllerStabilityAndExplorationSpec, ...]
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


def build_the_two_timescale_controller_stability_and_exploration_specs(
    source_records: list[dict[str, Any]],
) -> TheTwoTimescaleControllerStabilityAndExplorationSpecBundle:
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

    specs: list[TheTwoTimescaleControllerStabilityAndExplorationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheTwoTimescaleControllerStabilityAndExplorationSpec(
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
        "title": "Paper 0 " + "The Two-Timescale Controller: Stability and Exploration" + " Specs",
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
        "next_source_boundary": "P0R02923",
    }
    return TheTwoTimescaleControllerStabilityAndExplorationSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheTwoTimescaleControllerStabilityAndExplorationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_two_timescale_controller_stability_and_exploration_specs(
        load_jsonl(ledger_path)
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheTwoTimescaleControllerStabilityAndExplorationSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Two-Timescale Controller: Stability and Exploration" + " Specs",
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
    bundle: TheTwoTimescaleControllerStabilityAndExplorationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_two_timescale_controller_stability_and_exploration_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_two_timescale_controller_stability_and_exploration_validation_specs_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle) + "\n", encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
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
