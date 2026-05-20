#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Quasicriticality and Self-Organisation spec builder
"""Promote Paper 0 Quasicriticality and Self-Organisation records."""

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
    "P0R02831",
    "P0R02832",
    "P0R02833",
    "P0R02834",
    "P0R02835",
    "P0R02836",
    "P0R02837",
    "P0R02838",
)
CLAIM_BOUNDARY = "source-bounded quasicriticality and self organisation source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "quasicriticality_and_self_organisation.quasicriticality_and_self_organisation": {
        "context_id": "quasicriticality_and_self_organisation",
        "validation_protocol": "paper0.quasicriticality_and_self_organisation.quasicriticality_and_self_organisation",
        "canonical_statement": "The source-bounded component 'Quasicriticality and Self-Organisation' preserves Paper 0 records P0R02831-P0R02837 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02831:quasicriticality_and_self_organisation",
            "P0R02832:quasicriticality_and_self_organisation",
            "P0R02833:quasicriticality_and_self_organisation",
            "P0R02834:quasicriticality_and_self_organisation",
            "P0R02835:quasicriticality_and_self_organisation",
            "P0R02836:quasicriticality_and_self_organisation",
            "P0R02837:quasicriticality_and_self_organisation",
        ),
        "source_formulae": (
            "P0R02831: Quasicriticality and Self-Organisation",
            "P0R02832: This section establishes the universal dynamic regime governing all 15 layers of the SCPN: Quasicriticality. The framework posits that for optimal function, all computationally active layers are maintained in a state poised between order and chaos. This is not a fine-tuned, strict critical point, but a more robust Griffiths Phase, which exhibits the functional advantages of criticality-such as scale-free dynamics and long-range temporal correlations (LRTC)-over a broader range of parameters. This regime is computationally advantageous as it maximises information capacity, storage, and the efficiency of information transmission across the network's hierarchical scales.",
            "P0R02833: Crucially, the framework asserts that the system is not externally tuned to this state but achieves it through Self-Organised Criticality (SOC). The text provides a formal model for this process, where a homeostatic feedback loop continuously adjusts the local branching parameter (sigmaL) towards unity. The specific biophysical grounding for this mechanism is provided for neural layers, where homeostatic synaptic plasticity-a well-documented process where neurons adjust their synaptic weights to maintain a target firing rate-is identified as the adaptive rule that drives the global network dynamics to the critical point. This provides a robust, biologically plausible mechanism for how the SCPN autonomously maintains its optimal computational state without a central controller, replacing the need for fine-tuning with adaptive self-organisation.",
            'P0R02834: This section describes the "sweet spot" where every layer of reality operates. It\'s a special state called Quasicriticality, better known as the "edge of chaos."',
            "P0R02835: Imagine a forest. If the trees are too wet (too ordered), a fire can't spread. If the trees are too dry and packed together (too chaotic), the fire burns out of control instantly. The \"edge of chaos\" is that perfect state where the forest is just right for a fire to spread in complex, interesting, and far-reaching ways. Our universe, from our brains to our societies, keeps itself in this perfect state. It's the ideal balance for information to spread efficiently, for new ideas to form, and for life to adapt.",
            "P0R02836: How does it stay so perfectly balanced? Through a process called Self-Organised Criticality (SOC). The system automatically tunes itself. Think of it like a smart thermostat for reality. In your brain, for example, your neurons have rules that make them automatically strengthen or weaken their connections to keep the overall activity level in the perfect \"edge of chaos\" zone. This isn't a miracle; it's a built-in, self-tuning feature that makes the entire network of reality optimally intelligent and creative.",
            "P0R02837: P0R02837",
        ),
        "test_protocols": (
            "preserve Quasicriticality and Self-Organisation source-accounting boundary",
        ),
        "null_results": (
            "Quasicriticality and Self-Organisation is not empirical validation evidence",
        ),
        "variables": ("quasicriticality_and_self_organisation",),
        "validation_targets": ("preserve records P0R02831-P0R02837",),
        "null_controls": (
            "quasicriticality_and_self_organisation must remain source-bounded accounting",
        ),
    },
    "quasicriticality_and_self_organisation.meta_framework_integrations": {
        "context_id": "meta_framework_integrations",
        "validation_protocol": "paper0.quasicriticality_and_self_organisation.meta_framework_integrations",
        "canonical_statement": "The source-bounded component 'Meta-Framework Integrations' preserves Paper 0 records P0R02838-P0R02838 without empirical validation claims.",
        "source_equation_ids": ("P0R02838:meta_framework_integrations",),
        "source_formulae": ("P0R02838: Meta-Framework Integrations",),
        "test_protocols": ("preserve Meta-Framework Integrations source-accounting boundary",),
        "null_results": ("Meta-Framework Integrations is not empirical validation evidence",),
        "variables": ("meta_framework_integrations",),
        "validation_targets": ("preserve records P0R02838-P0R02838",),
        "null_controls": ("meta_framework_integrations must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class QuasicriticalityAndSelfOrganisationSpec:
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
class QuasicriticalityAndSelfOrganisationSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[QuasicriticalityAndSelfOrganisationSpec, ...]
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


def build_quasicriticality_and_self_organisation_specs(
    source_records: list[dict[str, Any]],
) -> QuasicriticalityAndSelfOrganisationSpecBundle:
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

    specs: list[QuasicriticalityAndSelfOrganisationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            QuasicriticalityAndSelfOrganisationSpec(
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
        "title": "Paper 0 " + "Quasicriticality and Self-Organisation" + " Specs",
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
        "next_source_boundary": "P0R02839",
    }
    return QuasicriticalityAndSelfOrganisationSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> QuasicriticalityAndSelfOrganisationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_quasicriticality_and_self_organisation_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: QuasicriticalityAndSelfOrganisationSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Quasicriticality and Self-Organisation" + " Specs",
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
    bundle: QuasicriticalityAndSelfOrganisationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_quasicriticality_and_self_organisation_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_quasicriticality_and_self_organisation_validation_specs_{date_tag}.md"
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
