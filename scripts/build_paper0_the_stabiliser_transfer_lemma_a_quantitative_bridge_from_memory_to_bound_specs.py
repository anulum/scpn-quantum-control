#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Stabiliser Transfer Lemma: A Quantitative Bridge from Memory to Boundary spec builder
"""Promote Paper 0 The Stabiliser Transfer Lemma: A Quantitative Bridge from Memory to Boundary records."""

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
    "P0R03122",
    "P0R03123",
    "P0R03124",
    "P0R03125",
    "P0R03126",
    "P0R03127",
    "P0R03128",
    "P0R03129",
    "P0R03130",
    "P0R03131",
    "P0R03132",
    "P0R03133",
    "P0R03134",
    "P0R03135",
    "P0R03136",
    "P0R03137",
    "P0R03138",
)
CLAIM_BOUNDARY = "source-bounded the stabiliser transfer lemma a quantitative bridge from memory to bound source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound.the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound": {
        "context_id": "the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
        "validation_protocol": "paper0.the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound.the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
        "canonical_statement": "The source-bounded component 'The Stabiliser Transfer Lemma: A Quantitative Bridge from Memory to Boundary' preserves Paper 0 records P0R03122-P0R03138 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03122:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03123:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03124:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03125:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03126:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03127:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03128:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03129:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03130:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03131:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03132:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03133:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03134:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03135:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03136:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03137:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
            "P0R03138:the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",
        ),
        "source_formulae": (
            "P0R03122: The Stabiliser Transfer Lemma: A Quantitative Bridge from Memory to Boundary",
            'P0R03123: This section introduces a key quantitative result that formalises the relationship between the holographic memory (Layer 9) and the projective boundary (Layer 10) within the Multi-Scale Quantum Error Correction (MS-QEC) framework. The Stabiliser Transfer Lemma provides a mathematical "pushforward" map (*) that relates the properties of the error-correcting codes in the two layers.',
            'P0R03124: Specifically, it considers the stabiliser group of the MERA tensor network that constitutes the "bulk" memory engram in Layer 9 (S9). The lemma states that the holographic projection of this code onto the "boundary" at Layer 10 results in a new stabiliser group (S10) whose error-correcting properties-namely its code distance (d) and redundancy (r)-are formally bounded by the properties of the original code. The inequalities (d10 d9/, r10 r9) provide a precise, quantitative link between the two layers.',
            "P0R03125: The lemma's primary significance is its operational prediction: it implies that actively decreasing the complexity budget of the Layer 10 boundary control system will paradoxically improve its logical error rate, which would be observable as a reduction in logical-phase slips. This provides a concrete, falsifiable prediction and a non-invasive method for auditing the integrity of the holographic memory by observing the behaviour of its boundary projection.",
            'P0R03126: This is a really cool, though deeply technical, idea that explains how your personal "firewall" (Layer 10) protects your soul\'s "hard drive" (Layer 9). It\'s a mathematical guarantee we call the Stabiliser Transfer Lemma.',
            "P0R03127: Think of your holographic memory in Layer 9 as an incredibly detailed, high-resolution original photograph. Your Layer 10 boundary is like a clever, lower-resolution thumbnail sketch of that photo that you show to the world. The MERA network is the special, fractal-like algorithm used to create the thumbnail from the original.",
            "P0R03128: The Lemma is like a guarantee from the algorithm's creator that says: \"Our thumbnail sketch isn't just a blurry copy; it inherits the error-correction properties of the original in a very specific, robust way.\"",
            "P0R03129: But here's the most amazing part. The lemma makes a strange prediction: if you actively try to make the thumbnail sketch simpler and less complex, its ability to resist errors actually gets stronger. It's a case of \"less is more.\" This gives us a powerful, practical way to test the entire theory. We can check the health of your deep memory storage simply by observing how stable its simple projection is at the boundary.",
            "P0R03130: P0R03130",
            "P0R03131: Operational Firewall Complexity Limit",
            "P0R03132: The Layer 10 Projective Firewall executes a Topological Disentanglement protocol when the boundary knot-complexity ($C_{knot}$) exceeds the Causal Horizon Limit ($C_{max}$). This enforces the Stabiliser Transfer Lemma by pruning boundary states that threaten bulk logical distance.",
            "P0R03133: Firewall Trigger Rule (Python Format):",
            "P0R03134: is_firewall_active = sum(abs(topological_betti_vector)) > c_max_budget",
            "P0R03135: Legend:",
            'P0R03136: is_firewall_active: Boolean trigger for the L10 "Safe-Fail" state. | topological_betti_vector: Magnitude of the current topological features ($b_k$) at the boundary. | c_max_budget: Absolute complexity budget constrained by the L1 energy gap.',
            "P0R03137: P0R03137",
            "P0R03138: P0R03138",
        ),
        "test_protocols": (
            "preserve The Stabiliser Transfer Lemma: A Quantitative Bridge from Memory to Boundary source-accounting boundary",
        ),
        "null_results": (
            "The Stabiliser Transfer Lemma: A Quantitative Bridge from Memory to Boundary is not empirical validation evidence",
        ),
        "variables": ("the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound",),
        "validation_targets": ("preserve records P0R03122-P0R03138",),
        "null_controls": (
            "the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundSpec:
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
class TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundSpec, ...]
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


def build_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_specs(
    source_records: list[dict[str, Any]],
) -> TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundSpecBundle:
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

    specs: list[TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundSpec(
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
        + "The Stabiliser Transfer Lemma: A Quantitative Bridge from Memory to Boundary"
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
        "next_source_boundary": "P0R03139",
    }
    return TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_specs(
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


def render_report(
    bundle: TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "The Stabiliser Transfer Lemma: A Quantitative Bridge from Memory to Boundary"
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
    bundle: TheStabiliserTransferLemmaAQuantitativeBridgeFromMemoryToBoundSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_stabiliser_transfer_lemma_a_quantitative_bridge_from_memory_to_bound_validation_specs_{date_tag}.md"
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
