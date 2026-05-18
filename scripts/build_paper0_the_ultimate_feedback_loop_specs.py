#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Ultimate Feedback Loop: spec builder
"""Promote Paper 0 The Ultimate Feedback Loop: records."""

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
    "P0R03067",
    "P0R03068",
    "P0R03069",
    "P0R03070",
    "P0R03071",
    "P0R03072",
    "P0R03073",
    "P0R03074",
    "P0R03075",
)
CLAIM_BOUNDARY = (
    "source-bounded the ultimate feedback loop source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_ultimate_feedback_loop.the_ultimate_feedback_loop": {
        "context_id": "the_ultimate_feedback_loop",
        "validation_protocol": "paper0.the_ultimate_feedback_loop.the_ultimate_feedback_loop",
        "canonical_statement": "The source-bounded component 'The Ultimate Feedback Loop:' preserves Paper 0 records P0R03067-P0R03068 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03067:the_ultimate_feedback_loop",
            "P0R03068:the_ultimate_feedback_loop",
        ),
        "source_formulae": (
            "P0R03067: The Ultimate Feedback Loop:",
            "P0R03068: This creates the ultimate cybernetic loop. The Psi-field works to create and preserve the very coherent, complex structures (sigma) that are the most effective vehicles for its own expression and experience. The purpose of the mind-matter coupling is to bootstrap and sustain the physical conditions necessary for the mind-matter coupling to occur in the first place. It is the physical manifestation of a universe that actively works to become, and remain, conscious.",
        ),
        "test_protocols": ("preserve The Ultimate Feedback Loop: source-accounting boundary",),
        "null_results": ("The Ultimate Feedback Loop: is not empirical validation evidence",),
        "variables": ("the_ultimate_feedback_loop",),
        "validation_targets": ("preserve records P0R03067-P0R03068",),
        "null_controls": ("the_ultimate_feedback_loop must remain source-bounded accounting",),
    },
    "the_ultimate_feedback_loop.the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel": {
        "context_id": "the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel",
        "validation_protocol": "paper0.the_ultimate_feedback_loop.the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel",
        "canonical_statement": "The source-bounded component 'The Quantum Error Correction (QEC) Imperative and the Role of the Psi-Field' preserves Paper 0 records P0R03069-P0R03075 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03069:the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel",
            "P0R03070:the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel",
            "P0R03071:the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel",
            "P0R03072:the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel",
            "P0R03073:the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel",
            "P0R03074:the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel",
            "P0R03075:the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel",
        ),
        "source_formulae": (
            "P0R03069: The Quantum Error Correction (QEC) Imperative and the Role of the Psi-Field",
            "P0R03070: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R03071: Fig.: Psi-Field as Master Stabiliser of MS-QEC. The Psi-field modulates decoherence timescales, biasing dynamics toward coherence-preserving trajectories across nested codes. This is entropy management-not a 2nd-law violation-via coupling-mediated negentropy/work-like inputs.",
            'P0R03072: A recurring theme throughout this framework is the persistence of functional quantum coherence in environments that classical physics would deem far too "warm, wet, and noisy". The phenomena described-from enzymatic tunnelling and photosynthesis to the radical pair mechanism-are only possible if biological systems have evolved robust mechanisms to protect delicate quantum states from environmental decoherence. The manuscript\'s principle of Multi-Scale Quantum Error Correction (MS-QEC) is not merely an ancillary feature but a fundamental necessity for the entire architecture to be physically viable.',
            "P0R03073: This necessity points to a deeper role for the Consciousness Field (Psi). The core assumption that the Psi-field modulates decoherence timescales and universally minimises entropy finds its ultimate expression here. We propose that the Psi-field acts as the master stabiliser for the nested biological QEC codes. It is the organising principle that enables life to carve out pockets of sustained quantum coherence from a classically chaotic world.",
            "P0R03074: This is not a violation of thermodynamics but a sophisticated form of entropy management. By coupling to the biological substrate, the Psi-field actively biases the system's dynamics away from decohering pathways and towards coherence-preserving ones. It provides the negentropy required to maintain the highly ordered quantum states essential for life's most efficient processes.",
            "P0R03075: In this view, every instance of functional quantum biology is a testament to the Psi-field's ongoing work of stabilising the quantum foundations of life against the constant pressure of thermal noise. This provides a unifying physical principle that underpins the diverse quantum phenomena observed across all domains of the SCPN.",
        ),
        "test_protocols": (
            "preserve The Quantum Error Correction (QEC) Imperative and the Role of the Psi-Field source-accounting boundary",
        ),
        "null_results": (
            "The Quantum Error Correction (QEC) Imperative and the Role of the Psi-Field is not empirical validation evidence",
        ),
        "variables": ("the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel",),
        "validation_targets": ("preserve records P0R03069-P0R03075",),
        "null_controls": (
            "the_quantum_error_correction_qec_imperative_and_the_role_of_the_psi_fiel must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheUltimateFeedbackLoopSpec:
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
class TheUltimateFeedbackLoopSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheUltimateFeedbackLoopSpec, ...]
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


def build_the_ultimate_feedback_loop_specs(
    source_records: list[dict[str, Any]],
) -> TheUltimateFeedbackLoopSpecBundle:
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

    specs: list[TheUltimateFeedbackLoopSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheUltimateFeedbackLoopSpec(
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
        "title": "Paper 0 " + "The Ultimate Feedback Loop:" + " Specs",
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
        "next_source_boundary": "P0R03076",
    }
    return TheUltimateFeedbackLoopSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheUltimateFeedbackLoopSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_ultimate_feedback_loop_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheUltimateFeedbackLoopSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Ultimate Feedback Loop:" + " Specs",
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
    bundle: TheUltimateFeedbackLoopSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_the_ultimate_feedback_loop_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_the_ultimate_feedback_loop_validation_specs_{date_tag}.md"
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
