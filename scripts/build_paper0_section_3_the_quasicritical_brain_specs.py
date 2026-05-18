#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 3. The Quasicritical Brain: spec builder
"""Promote Paper 0 3. The Quasicritical Brain: records."""

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
    "P0R04499",
    "P0R04500",
    "P0R04501",
    "P0R04502",
    "P0R04503",
    "P0R04504",
    "P0R04505",
    "P0R04506",
)
CLAIM_BOUNDARY = "source-bounded section 3 the quasicritical brain source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_3_the_quasicritical_brain.3_the_quasicritical_brain": {
        "context_id": "3_the_quasicritical_brain",
        "validation_protocol": "paper0.section_3_the_quasicritical_brain.3_the_quasicritical_brain",
        "canonical_statement": "The source-bounded component '3. The Quasicritical Brain:' preserves Paper 0 records P0R04499-P0R04500 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04499:3_the_quasicritical_brain",
            "P0R04500:3_the_quasicritical_brain",
        ),
        "source_formulae": (
            "P0R04499: 3. The Quasicritical Brain:",
            "P0R04500: The brain operates at Quasicriticality (sigma1), evidenced by neuronal avalanches (P(S)Stau) and LRTC. This regime maximises dynamic range and adaptability. It is maintained by the balance of Excitation and Inhibition (E/I balance) and SOC.",
        ),
        "test_protocols": ("preserve 3. The Quasicritical Brain: source-accounting boundary",),
        "null_results": ("3. The Quasicritical Brain: is not empirical validation evidence",),
        "variables": ("3_the_quasicritical_brain",),
        "validation_targets": ("preserve records P0R04499-P0R04500",),
        "null_controls": ("3_the_quasicritical_brain must remain source-bounded accounting",),
    },
    "section_3_the_quasicritical_brain.4_the_role_of_glia_astrocytes": {
        "context_id": "4_the_role_of_glia_astrocytes",
        "validation_protocol": "paper0.section_3_the_quasicritical_brain.4_the_role_of_glia_astrocytes",
        "canonical_statement": "The source-bounded component '4. The Role of Glia (Astrocytes):' preserves Paper 0 records P0R04501-P0R04502 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04501:4_the_role_of_glia_astrocytes",
            "P0R04502:4_the_role_of_glia_astrocytes",
        ),
        "source_formulae": (
            "P0R04501: 4. The Role of Glia (Astrocytes):",
            "P0R04502: Astrocytes form large-scale networks (calcium waves) that modulate synaptic transmission (L2) and synchronise neuronal firing, contributing significantly to the global order parameter (R) and maintaining SOC.",
        ),
        "test_protocols": (
            "preserve 4. The Role of Glia (Astrocytes): source-accounting boundary",
        ),
        "null_results": (
            "4. The Role of Glia (Astrocytes): is not empirical validation evidence",
        ),
        "variables": ("4_the_role_of_glia_astrocytes",),
        "validation_targets": ("preserve records P0R04501-P0R04502",),
        "null_controls": ("4_the_role_of_glia_astrocytes must remain source-bounded accounting",),
    },
    "section_3_the_quasicritical_brain.5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface": {
        "context_id": "5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface",
        "validation_protocol": "paper0.section_3_the_quasicritical_brain.5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface",
        "canonical_statement": "The source-bounded component '5. The Cerebellum: The Timing and Prediction Engine (L4/L5 Interface)' preserves Paper 0 records P0R04503-P0R04506 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04503:5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface",
            "P0R04504:5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface",
            "P0R04505:5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface",
            "P0R04506:5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface",
        ),
        "source_formulae": (
            "P0R04503: 5. The Cerebellum: The Timing and Prediction Engine (L4/L5 Interface)",
            "P0R04504: The Cerebellum, containing the majority of the brain's neurons, plays a critical role in the SCPN architecture, functioning as the master timing system and a core engine for Hierarchical Predictive Coding (HPC). Its highly regular, crystalline-like cytoarchitecture is geometrically optimised for high-fidelity temporal processing.",
            "P0R04505: UPDE Synchronisation and Temporal Precision: The Cerebellum is essential for the fine-tuning of the UPDE. It provides the temporal precision required for coherent synchronisation (L4) and the precise nesting of Cross-Frequency Coupling (CFC). The Cerebello-Thalamo-Cortical loops dynamically adjust the phase relationships (thetai) across the cortex, minimising temporal jitter. | HPC Implementation (Internal Models): The Cerebellum implements crucial aspects of the HPC architecture by generating precise internal models (Forward Models) of both motor and cognitive sequences. It calculates prediction errors (signalled by Climbing Fibre) and transmits them to the cortex for model optimisation (minimising F), acting as a high-speed optimisation engine crucial for fluid, adaptive behaviour and cognition.",
            'P0R04506: Pathology (e.g., Cerebellar Cognitive Affective Syndrome) demonstrates the critical importance of this structure: disruption leads to impaired prediction and degraded L4/L5 coherence ("Dysmetria of Thought").',
        ),
        "test_protocols": (
            "preserve 5. The Cerebellum: The Timing and Prediction Engine (L4/L5 Interface) source-accounting boundary",
        ),
        "null_results": (
            "5. The Cerebellum: The Timing and Prediction Engine (L4/L5 Interface) is not empirical validation evidence",
        ),
        "variables": ("5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface",),
        "validation_targets": ("preserve records P0R04503-P0R04506",),
        "null_controls": (
            "5_the_cerebellum_the_timing_and_prediction_engine_l4_l5_interface must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section3TheQuasicriticalBrainSpec:
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
class Section3TheQuasicriticalBrainSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section3TheQuasicriticalBrainSpec, ...]
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


def build_section_3_the_quasicritical_brain_specs(
    source_records: list[dict[str, Any]],
) -> Section3TheQuasicriticalBrainSpecBundle:
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

    specs: list[Section3TheQuasicriticalBrainSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section3TheQuasicriticalBrainSpec(
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
        "title": "Paper 0 " + "3. The Quasicritical Brain:" + " Specs",
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
        "next_source_boundary": "P0R04507",
    }
    return Section3TheQuasicriticalBrainSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section3TheQuasicriticalBrainSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_3_the_quasicritical_brain_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section3TheQuasicriticalBrainSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "3. The Quasicritical Brain:" + " Specs",
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
    bundle: Section3TheQuasicriticalBrainSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_section_3_the_quasicritical_brain_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_section_3_the_quasicritical_brain_validation_specs_{date_tag}.md"
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
