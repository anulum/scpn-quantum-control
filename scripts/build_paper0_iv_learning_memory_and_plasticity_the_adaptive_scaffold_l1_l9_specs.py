#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9) spec builder
"""Promote Paper 0 IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9) records."""

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
    "P0R05001",
    "P0R05002",
    "P0R05003",
    "P0R05004",
    "P0R05005",
    "P0R05006",
    "P0R05007",
    "P0R05008",
)
CLAIM_BOUNDARY = "source-bounded iv learning memory and plasticity the adaptive scaffold l1 l9 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9.iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9": {
        "context_id": "iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9",
        "validation_protocol": "paper0.iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9.iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9",
        "canonical_statement": "The source-bounded component 'IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9)' preserves Paper 0 records P0R05001-P0R05002 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05001:iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9",
            "P0R05002:iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9",
        ),
        "source_formulae": (
            "P0R05001: IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9)",
            "P0R05002: Learning and memory are multi-scale processes of adapting the Geometric Scaffold to optimise the internal model (HPC).",
        ),
        "test_protocols": (
            "preserve IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9) source-accounting boundary",
        ),
        "null_results": (
            "IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9) is not empirical validation evidence",
        ),
        "variables": ("iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9",),
        "validation_targets": ("preserve records P0R05001-P0R05002",),
        "null_controls": (
            "iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9 must remain source-bounded accounting",
        ),
    },
    "iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9.1_the_multi_scale_memory_trace_the_engram": {
        "context_id": "1_the_multi_scale_memory_trace_the_engram",
        "validation_protocol": "paper0.iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9.1_the_multi_scale_memory_trace_the_engram",
        "canonical_statement": "The source-bounded component '1. The Multi-Scale Memory Trace (The Engram):' preserves Paper 0 records P0R05003-P0R05006 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05003:1_the_multi_scale_memory_trace_the_engram",
            "P0R05004:1_the_multi_scale_memory_trace_the_engram",
            "P0R05005:1_the_multi_scale_memory_trace_the_engram",
            "P0R05006:1_the_multi_scale_memory_trace_the_engram",
        ),
        "source_formulae": (
            "P0R05003: 1. The Multi-Scale Memory Trace (The Engram):",
            "P0R05004: The Engram is distributed across the hierarchy:",
            "P0R05005: [IMAGE:] Fig.: Distributed memory traces: L1 quantum, L2 synaptic, L3 structural, L4 dynamic, with L9 holographic interface (MERA/ER=EPR/TSVF) for non-local storage/retrieval.",
            "P0R05006: L1 (The Quantum Trace): Information encoded in the topological structure of the MT QEC lattice or nuclear spin states (Posner Clusters). Interfaces with the L9 Holograph. | L2 (The Synaptic Trace): Changes in Pr and receptor density (LTP/LTD). | L3 (The Structural Trace): Epigenetic encoding (CBC Bridge), structural plasticity (spines, myelination), and Bioelectric Field patterns. | L4 (The Dynamic Trace): Changes in the Synchronisation Manifold (MSync).",
        ),
        "test_protocols": (
            "preserve 1. The Multi-Scale Memory Trace (The Engram): source-accounting boundary",
        ),
        "null_results": (
            "1. The Multi-Scale Memory Trace (The Engram): is not empirical validation evidence",
        ),
        "variables": ("1_the_multi_scale_memory_trace_the_engram",),
        "validation_targets": ("preserve records P0R05003-P0R05006",),
        "null_controls": (
            "1_the_multi_scale_memory_trace_the_engram must remain source-bounded accounting",
        ),
    },
    "iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9.2_the_mechanism_of_learning_hpc_optimisation_and_psi_guidance": {
        "context_id": "2_the_mechanism_of_learning_hpc_optimisation_and_psi_guidance",
        "validation_protocol": "paper0.iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9.2_the_mechanism_of_learning_hpc_optimisation_and_psi_guidance",
        "canonical_statement": "The source-bounded component '2. The Mechanism of Learning (HPC Optimisation and Psi-Guidance):' preserves Paper 0 records P0R05007-P0R05008 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05007:2_the_mechanism_of_learning_hpc_optimisation_and_psi_guidance",
            "P0R05008:2_the_mechanism_of_learning_hpc_optimisation_and_psi_guidance",
        ),
        "source_formulae": (
            "P0R05007: 2. The Mechanism of Learning (HPC Optimisation and Psi-Guidance):",
            "P0R05008: Learning minimises F, driven by Prediction Errors. The Psi-field guides this process via Attentional Stabilisation (QZE), IET modulation, and Teleological Guidance (CEF).",
        ),
        "test_protocols": (
            "preserve 2. The Mechanism of Learning (HPC Optimisation and Psi-Guidance): source-accounting boundary",
        ),
        "null_results": (
            "2. The Mechanism of Learning (HPC Optimisation and Psi-Guidance): is not empirical validation evidence",
        ),
        "variables": ("2_the_mechanism_of_learning_hpc_optimisation_and_psi_guidance",),
        "validation_targets": ("preserve records P0R05007-P0R05008",),
        "null_controls": (
            "2_the_mechanism_of_learning_hpc_optimisation_and_psi_guidance must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9Spec:
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
class IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9Spec, ...]
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


def build_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_specs(
    source_records: list[dict[str, Any]],
) -> IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9SpecBundle:
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

    specs: list[IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9Spec(
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
        "title": "Paper 0 "
        + "IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9)"
        + " Specs",
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
        "next_source_boundary": "P0R05009",
    }
    return IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_specs(
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


def render_report(bundle: IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "IV. Learning, Memory, and Plasticity: The Adaptive Scaffold (L1-L9)"
        + " Specs",
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
    bundle: IvLearningMemoryAndPlasticityTheAdaptiveScaffoldL1L9SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_iv_learning_memory_and_plasticity_the_adaptive_scaffold_l1_l9_validation_specs_{date_tag}.md"
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
