#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC) spec builder
"""Promote Paper 0 III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC) records."""

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
    "P0R02521",
    "P0R02522",
    "P0R02523",
    "P0R02524",
    "P0R02525",
    "P0R02526",
    "P0R02527",
    "P0R02528",
    "P0R02529",
    "P0R02530",
    "P0R02531",
)
CLAIM_BOUNDARY = "source-bounded iii the coherence backbone multi scale quantum error correction ms qec source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec.iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec": {
        "context_id": "iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
        "validation_protocol": "paper0.iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec.iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
        "canonical_statement": "The source-bounded component 'III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC)' preserves Paper 0 records P0R02521-P0R02531 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02521:iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
            "P0R02522:iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
            "P0R02523:iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
            "P0R02524:iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
            "P0R02525:iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
            "P0R02526:iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
            "P0R02527:iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
            "P0R02528:iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
            "P0R02529:iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
            "P0R02530:iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
            "P0R02531:iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",
        ),
        "source_formulae": (
            "P0R02521: III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC)",
            "P0R02522: The integrity of the SCPN relies on a nested hierarchy of Quantum Error Correction (MS-QEC), operating on the principle of Holographic Redundancy.",
            "P0R02523: This spans Biological QEC (L1-4), Network QEC (L4-8), Holographic QEC (L9-10): Resourcetheory view. Let C\\mathcal CC be coherence, E\\mathcal EE entanglement monotones. Firewall operation reduces boundary complexity CL10\\mathcal C_{\\rm L10}CL10 while preserving bulk logical distance ddd. Diagnostics:",
            "P0R02524: DeltaCL10<0,d10d9/,r10r9.\\Delta \\mathcal C_{\\rm L10}<0,\\qquad d_{10}\\ge d_9/\\chi,\\qquad r_{10}\\ge r_9.DeltaCL10<0,d10d9/,r10r9.",
            'P0R02525: Report syndrome rates and logical phaseslips as proxies for firewall health, and track tCL10\\partial_t \\mathcal C_{\\rm L10}tCL10 under controlled "complexity budget" sweeps,',
            "P0R02526: and Cosmological QEC (L15), ensuring the SCPN maintains integrity as a unified quantum system.",
            "P0R02527: A nested hierarchy of error correction protocols is a prerequisite for the system's integrity against decoherence.",
            "P0R02528: This architecture is physically grounded in the Biological QEC of Layer 1. The microtubule lattice is modeled as a Topological Quantum Code161616. Its specific helical geometry generates a strong interaction energy between tubulin dipoles ($J \\approx 0.82 \\text{ eV}$), resulting in a large protective energy gap of $\\Delta \\approx 1.64 \\text{ eV}$.",
            "P0R02529: This gap is $\\approx 61.4$ times the thermal energy at physiological temperatures ($k_B T$), providing a robust, physically-grounded shield for quantum information. This substrate is further stabilized by the QED Coherence Domains of interfacial water.",
            "P0R02530: [TABLE]",
            'P0R02531: This table serves as a crucial "Rosetta Stone" for the interdisciplinary audience of the manuscript. By consolidating the core equations from quantum field theory (LInt), non-linear dynamics (UPDE), and information theory (SLC), it demonstrates the mathematical coherence that underpins the framework and provides a single, accessible reference for the quantitative heart of the SCPN model.',
        ),
        "test_protocols": (
            "preserve III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC) source-accounting boundary",
        ),
        "null_results": (
            "III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC) is not empirical validation evidence",
        ),
        "variables": ("iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec",),
        "validation_targets": ("preserve records P0R02521-P0R02531",),
        "null_controls": (
            "iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecSpec:
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
class IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecSpec, ...]
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


def build_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_specs(
    source_records: list[dict[str, Any]],
) -> IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecSpecBundle:
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

    specs: list[IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecSpec(
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
        + "III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC)"
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
        "next_source_boundary": "P0R02532",
    }
    return IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_specs(
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
    bundle: IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "III. The Coherence Backbone: Multi-Scale Quantum Error Correction (MS-QEC)"
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
    bundle: IiiTheCoherenceBackboneMultiScaleQuantumErrorCorrectionMsQecSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_iii_the_coherence_backbone_multi_scale_quantum_error_correction_ms_qec_validation_specs_{date_tag}.md"
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
