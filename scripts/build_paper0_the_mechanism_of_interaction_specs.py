#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Mechanism of Interaction: spec builder
"""Promote Paper 0 The Mechanism of Interaction: records."""

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
    "P0R03148",
    "P0R03149",
    "P0R03150",
    "P0R03151",
    "P0R03152",
    "P0R03153",
    "P0R03154",
    "P0R03155",
    "P0R03156",
    "P0R03157",
    "P0R03158",
    "P0R03159",
    "P0R03160",
    "P0R03161",
    "P0R03162",
    "P0R03163",
    "P0R03164",
    "P0R03165",
    "P0R03166",
    "P0R03167",
    "P0R03168",
    "P0R03169",
    "P0R03170",
    "P0R03171",
    "P0R03172",
    "P0R03173",
)
CLAIM_BOUNDARY = (
    "source-bounded the mechanism of interaction source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_mechanism_of_interaction.the_mechanism_of_interaction": {
        "context_id": "the_mechanism_of_interaction",
        "validation_protocol": "paper0.the_mechanism_of_interaction.the_mechanism_of_interaction",
        "canonical_statement": "The source-bounded component 'The Mechanism of Interaction:' preserves Paper 0 records P0R03148-P0R03149 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03148:the_mechanism_of_interaction",
            "P0R03149:the_mechanism_of_interaction",
        ),
        "source_formulae": (
            "P0R03148: The Mechanism of Interaction:",
            "P0R03149: The H_int = -lambda * Psis * sigma interaction is therefore not with a noisy or degraded copy of the Self's information, but with a robust, logically sound representation. The operational prediction-that reducing boundary complexity improves its logical stability-implies that an agent can achieve a clearer, more coherent coupling with the universal Psi-field by simplifying its interface with the world, a principle found in many contemplative traditions.",
        ),
        "test_protocols": ("preserve The Mechanism of Interaction: source-accounting boundary",),
        "null_results": ("The Mechanism of Interaction: is not empirical validation evidence",),
        "variables": ("the_mechanism_of_interaction",),
        "validation_targets": ("preserve records P0R03148-P0R03149",),
        "null_controls": ("the_mechanism_of_interaction must remain source-bounded accounting",),
    },
    "the_mechanism_of_interaction.stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros": {
        "context_id": "stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
        "validation_protocol": "paper0.the_mechanism_of_interaction.stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
        "canonical_statement": "The source-bounded component 'Stabiliser Transfer Lemma (Sketch) - MS-QEC Bridge: Stabiliser Transfer across L9->L10' preserves Paper 0 records P0R03150-P0R03173 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03150:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03151:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03152:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03153:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03154:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03155:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03156:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03157:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03158:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03159:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03160:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03161:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03162:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03163:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03164:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03165:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03166:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03167:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03168:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03169:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03170:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03171:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03172:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
            "P0R03173:stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",
        ),
        "source_formulae": (
            "P0R03150: Stabiliser Transfer Lemma (Sketch) - MS-QEC Bridge: Stabiliser Transfer across L9->L10",
            "P0R03151: Let",
            "P0R03152: S9PnS_9\\subset\\mathcal{P}^{\\otimes n}S9Pn",
            "P0R03153: be the MERA bulk stabiliser group (L9).",
            "P0R03154: Let",
            "P0R03155: :bulk->boundary\\Pi:\\text{bulk}\\to\\text{boundary}:bulk->boundary",
            "P0R03156: be the holographic rendering map (RT/MERA contraction). If L10 enforces complexity minimisation (CA dual) with topological censorship,",
            "P0R03157: then there exists a pushforward",
            "P0R03158: \\*:S9->S10\\Pi_\\* : S_9 \\to S_{10}\\*:S9->S10",
            "P0R03159: such that",
            "P0R03160: d10 d9,r10 r9,d_{10}\\;\\ge\\; \\frac{d_9}{\\chi},\\qquad r_{10}\\;\\ge\\; r_9,d10d9,r10r9,",
            "P0R03161: where ddd is code distance, rrr redundancy, and \\chi the MERA branching factor.",
            "P0R03162: Operational prediction. Boundary logical error rates drop when L10's complexity budget decreases-visible as reduced logical-phase slips during firewall-induced disentangling. Use. Provides a quantitative handle for L10 audit without peeking behind the boundary.",
            'P0R03163: The "Warm, Wet, and Noisy" Problem',
            "P0R03164: Source Material: The sections that directly confront the challenge of maintaining quantum coherence in biological and other complex systems, setting the stage for the necessity of a robust error correction architecture.",
            "P0R03165: P0R03165",
            "P0R03166: The Nested Hierarchy of Error Protection",
            "P0R03167: Source Material: The master overview of the multi-scale QEC system, presenting it as a nested set of codes, from the local to the universal.",
            "P0R03168: P0R03168",
            "P0R03169: Specific Mechanisms: From Biological to Cosmological QEC",
            "P0R03170: Source Material: This section will collate the specific examples of QEC mechanisms mentioned: Biological QEC (e.g., microtubule energy gaps), Network QEC (redundancy in neural codes), Holographic QEC (MERA tensor networks in memory), and Cosmological QEC (the global constraint of the Ethical Functional).",
            "P0R03171: P0R03171",
            "P0R03172: P0R03172",
            "P0R03173: P0R03173",
        ),
        "test_protocols": (
            "preserve Stabiliser Transfer Lemma (Sketch) - MS-QEC Bridge: Stabiliser Transfer across L9->L10 source-accounting boundary",
        ),
        "null_results": (
            "Stabiliser Transfer Lemma (Sketch) - MS-QEC Bridge: Stabiliser Transfer across L9->L10 is not empirical validation evidence",
        ),
        "variables": ("stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros",),
        "validation_targets": ("preserve records P0R03150-P0R03173",),
        "null_controls": (
            "stabiliser_transfer_lemma_sketch_ms_qec_bridge_stabiliser_transfer_acros must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheMechanismOfInteractionSpec:
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
class TheMechanismOfInteractionSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheMechanismOfInteractionSpec, ...]
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


def build_the_mechanism_of_interaction_specs(
    source_records: list[dict[str, Any]],
) -> TheMechanismOfInteractionSpecBundle:
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

    specs: list[TheMechanismOfInteractionSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheMechanismOfInteractionSpec(
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
        "title": "Paper 0 " + "The Mechanism of Interaction:" + " Specs",
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
        "next_source_boundary": "P0R03174",
    }
    return TheMechanismOfInteractionSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheMechanismOfInteractionSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_mechanism_of_interaction_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheMechanismOfInteractionSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Mechanism of Interaction:" + " Specs",
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
    bundle: TheMechanismOfInteractionSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_the_mechanism_of_interaction_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_the_mechanism_of_interaction_validation_specs_{date_tag}.md"
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
