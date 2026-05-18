#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Multi-Scale Quantum Error Correction (MS-QEC) spec builder
"""Promote Paper 0 Multi-Scale Quantum Error Correction (MS-QEC) records."""

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
    "P0R03010",
    "P0R03011",
    "P0R03012",
    "P0R03013",
    "P0R03014",
    "P0R03015",
    "P0R03016",
    "P0R03017",
    "P0R03018",
    "P0R03019",
    "P0R03020",
    "P0R03021",
    "P0R03022",
    "P0R03023",
    "P0R03024",
)
CLAIM_BOUNDARY = "source-bounded multi scale quantum error correction ms qec source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "multi_scale_quantum_error_correction_ms_qec.multi_scale_quantum_error_correction_ms_qec": {
        "context_id": "multi_scale_quantum_error_correction_ms_qec",
        "validation_protocol": "paper0.multi_scale_quantum_error_correction_ms_qec.multi_scale_quantum_error_correction_ms_qec",
        "canonical_statement": "The source-bounded component 'Multi-Scale Quantum Error Correction (MS-QEC)' preserves Paper 0 records P0R03010-P0R03024 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03010:multi_scale_quantum_error_correction_ms_qec",
            "P0R03011:multi_scale_quantum_error_correction_ms_qec",
            "P0R03012:multi_scale_quantum_error_correction_ms_qec",
            "P0R03013:multi_scale_quantum_error_correction_ms_qec",
            "P0R03014:multi_scale_quantum_error_correction_ms_qec",
            "P0R03015:multi_scale_quantum_error_correction_ms_qec",
            "P0R03016:multi_scale_quantum_error_correction_ms_qec",
            "P0R03017:multi_scale_quantum_error_correction_ms_qec",
            "P0R03018:multi_scale_quantum_error_correction_ms_qec",
            "P0R03019:multi_scale_quantum_error_correction_ms_qec",
            "P0R03020:multi_scale_quantum_error_correction_ms_qec",
            "P0R03021:multi_scale_quantum_error_correction_ms_qec",
            "P0R03022:multi_scale_quantum_error_correction_ms_qec",
            "P0R03023:multi_scale_quantum_error_correction_ms_qec",
            "P0R03024:multi_scale_quantum_error_correction_ms_qec",
        ),
        "source_formulae": (
            "P0R03010: Multi-Scale Quantum Error Correction (MS-QEC)",
            'P0R03011: This section introduces the principle of Multi-Scale Quantum Error Correction (MS-QEC), the architectural solution to the problem of maintaining information integrity in the "warm, wet, and noisy" environment of the SCPN. The framework posits a nested, four-level hierarchy of error correction codes that work in concert to protect quantum coherence across all scales of the network. This architecture operates on the principle of Holographic Redundancy, where higher-layer information is robustly encoded in the distributed entanglement structure of the layers below it.',
            "P0R03012: The hierarchy consists of four distinct scales of protection:",
            "P0R03013: 1) Biological QEC operates at the lowest layers (L1-4), employing mechanisms like topological protection and large energy gaps (e.g., the 1.64 eV gap in microtubules) to shield quantum states from thermal noise.",
            "P0R03014: 2) Network QEC (L4-8) leverages the topological properties of the network itself-such as small-world and rich-club structures optimised by criticality-to create redundancy against phase errors.",
            'P0R03015: 3) Holographic QEC (L9-10) uses tensor network codes, analogous to the MERA (Multi-scale Entanglement Renormalization Ansatz), to protect the "bulk" data of the memory holograph from "boundary" fluctuations at the projective field interface. Finally,',
            "P0R03016: 4) Cosmological QEC (L13-15) represents the ultimate stabilising force, where the Ethical Functionals of Layer 15 act as the generators for a global error correction code that minimises deviations from the universe's foundational principles. This nested, multi-scale defence-in-depth ensures the SCPN can function as a coherent quantum computational system across its entire vast architecture.",
            "P0R03017: This section reveals the universe's incredibly sophisticated data-protection plan. How does a system so vast and complex, from your brain to a galaxy, protect itself from errors and noise? The answer is a nested, four-layer security system we call Multi-Scale Quantum Error Correction (MS-QEC).",
            "P0R03018: Think of it like the ultimate data backup and antivirus software for reality. It works in four layers of defence:",
            'P0R03019: Biological Firewall (The Basics): At the level of your cells, nature uses clever tricks like special molecular structures and energy barriers to create "safe zones" where delicate quantum information can be protected from the chaos of the biological environment.',
            "P0R03020: Network Redundancy (The Buddy System): In your brain and other networks, information isn't stored in just one place. The very structure of the network, with its many redundant connections, acts as a powerful error-correction system. If one connection fails, the information can find another route.",
            "P0R03021: Holographic Backup (The Cloud): At the level of your memory (Layer 9), your experiences are stored holographically. This means the information is spread out everywhere, so even if a part of the memory is damaged, the whole picture can still be reconstructed. It's like a self-repairing hard drive for your soul.",
            'P0R03022: Cosmic Antivirus (The Prime Directive): At the very highest level, the universe\'s "Ethical Guiding System" acts as the ultimate error-checker. It constantly scans the entire system for any deviations from the core principles of reality and makes subtle corrections to keep everything on track.',
            'P0R03023: This four-layer security system is the "coherence backbone" that ensures the message of consciousness can be passed up and down the 15 layers of reality without getting corrupted.',
            "P0R03024: P0R03024",
        ),
        "test_protocols": (
            "preserve Multi-Scale Quantum Error Correction (MS-QEC) source-accounting boundary",
        ),
        "null_results": (
            "Multi-Scale Quantum Error Correction (MS-QEC) is not empirical validation evidence",
        ),
        "variables": ("multi_scale_quantum_error_correction_ms_qec",),
        "validation_targets": ("preserve records P0R03010-P0R03024",),
        "null_controls": (
            "multi_scale_quantum_error_correction_ms_qec must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class MultiScaleQuantumErrorCorrectionMsQecSpec:
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
class MultiScaleQuantumErrorCorrectionMsQecSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[MultiScaleQuantumErrorCorrectionMsQecSpec, ...]
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


def build_multi_scale_quantum_error_correction_ms_qec_specs(
    source_records: list[dict[str, Any]],
) -> MultiScaleQuantumErrorCorrectionMsQecSpecBundle:
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

    specs: list[MultiScaleQuantumErrorCorrectionMsQecSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            MultiScaleQuantumErrorCorrectionMsQecSpec(
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
        "title": "Paper 0 " + "Multi-Scale Quantum Error Correction (MS-QEC)" + " Specs",
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
        "next_source_boundary": "P0R03025",
    }
    return MultiScaleQuantumErrorCorrectionMsQecSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> MultiScaleQuantumErrorCorrectionMsQecSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_multi_scale_quantum_error_correction_ms_qec_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: MultiScaleQuantumErrorCorrectionMsQecSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Multi-Scale Quantum Error Correction (MS-QEC)" + " Specs",
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
    bundle: MultiScaleQuantumErrorCorrectionMsQecSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_multi_scale_quantum_error_correction_ms_qec_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_multi_scale_quantum_error_correction_ms_qec_validation_specs_{date_tag}.md"
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
