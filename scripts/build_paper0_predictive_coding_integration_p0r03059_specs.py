#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Predictive Coding Integration spec builder
"""Promote Paper 0 Predictive Coding Integration records."""

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
    "P0R03059",
    "P0R03060",
    "P0R03061",
    "P0R03062",
    "P0R03063",
    "P0R03064",
    "P0R03065",
    "P0R03066",
)
CLAIM_BOUNDARY = "source-bounded predictive coding integration p0r03059 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "predictive_coding_integration_p0r03059.predictive_coding_integration": {
        "context_id": "predictive_coding_integration",
        "validation_protocol": "paper0.predictive_coding_integration_p0r03059.predictive_coding_integration",
        "canonical_statement": "The source-bounded component 'Predictive Coding Integration' preserves Paper 0 records P0R03059-P0R03060 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03059:predictive_coding_integration",
            "P0R03060:predictive_coding_integration",
        ),
        "source_formulae": (
            "P0R03059: Predictive Coding Integration",
            "P0R03060: This principle describes how the cosmic inference engine maintains the integrity of its own computational hardware.",
        ),
        "test_protocols": ("preserve Predictive Coding Integration source-accounting boundary",),
        "null_results": ("Predictive Coding Integration is not empirical validation evidence",),
        "variables": ("predictive_coding_integration",),
        "validation_targets": ("preserve records P0R03059-P0R03060",),
        "null_controls": ("predictive_coding_integration must remain source-bounded accounting",),
    },
    "predictive_coding_integration_p0r03059.the_psi_field_as_a_precision_enhancer": {
        "context_id": "the_psi_field_as_a_precision_enhancer",
        "validation_protocol": "paper0.predictive_coding_integration_p0r03059.the_psi_field_as_a_precision_enhancer",
        "canonical_statement": "The source-bounded component 'The Psi-Field as a Precision-Enhancer:' preserves Paper 0 records P0R03061-P0R03062 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03061:the_psi_field_as_a_precision_enhancer",
            "P0R03062:the_psi_field_as_a_precision_enhancer",
        ),
        "source_formulae": (
            "P0R03061: The Psi-Field as a Precision-Enhancer:",
            "P0R03062: An effective inference engine requires reliable hardware that can carry signals with high fidelity. Noise and decoherence are, in computational terms, a loss of precision. The Psi-field, by acting as the master stabiliser of the QEC codes, is fundamentally a precision-enhancing mechanism. It actively works to increase the signal-to-noise ratio of the biological substrate, ensuring that the predictions and prediction errors that drive the active inference process are transmitted and processed with the highest possible fidelity.",
        ),
        "test_protocols": (
            "preserve The Psi-Field as a Precision-Enhancer: source-accounting boundary",
        ),
        "null_results": (
            "The Psi-Field as a Precision-Enhancer: is not empirical validation evidence",
        ),
        "variables": ("the_psi_field_as_a_precision_enhancer",),
        "validation_targets": ("preserve records P0R03061-P0R03062",),
        "null_controls": (
            "the_psi_field_as_a_precision_enhancer must remain source-bounded accounting",
        ),
    },
    "predictive_coding_integration_p0r03059.psis_field_coupling_integration": {
        "context_id": "psis_field_coupling_integration",
        "validation_protocol": "paper0.predictive_coding_integration_p0r03059.psis_field_coupling_integration",
        "canonical_statement": "The source-bounded component 'Psis Field Coupling Integration' preserves Paper 0 records P0R03063-P0R03064 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03063:psis_field_coupling_integration",
            "P0R03064:psis_field_coupling_integration",
        ),
        "source_formulae": (
            "P0R03063: Psis Field Coupling Integration",
            "P0R03064: This provides the ultimate functional purpose for the interaction Hamiltonian, H_int = -lambda * Psis * sigma. The coupling is not just for communication, but for preservation.",
        ),
        "test_protocols": ("preserve Psis Field Coupling Integration source-accounting boundary",),
        "null_results": ("Psis Field Coupling Integration is not empirical validation evidence",),
        "variables": ("psis_field_coupling_integration",),
        "validation_targets": ("preserve records P0R03063-P0R03064",),
        "null_controls": (
            "psis_field_coupling_integration must remain source-bounded accounting",
        ),
    },
    "predictive_coding_integration_p0r03059.h_int_as_a_stabilising_potential": {
        "context_id": "h_int_as_a_stabilising_potential",
        "validation_protocol": "paper0.predictive_coding_integration_p0r03059.h_int_as_a_stabilising_potential",
        "canonical_statement": "The source-bounded component 'H_int as a Stabilising Potential:' preserves Paper 0 records P0R03065-P0R03066 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03065:h_int_as_a_stabilising_potential",
            "P0R03066:h_int_as_a_stabilising_potential",
        ),
        "source_formulae": (
            "P0R03065: H_int as a Stabilising Potential:",
            "P0R03066: The interaction Hamiltonian can be interpreted as adding a \"coherence-preserving potential\" to the system's dynamics. The coupling of the universal Psi-field (Psis) to the biological substrate (sigma) makes the coherent, quantum states of that substrate energetically favorable. It actively biases the system's evolution away from decohering pathways.",
        ),
        "test_protocols": (
            "preserve H_int as a Stabilising Potential: source-accounting boundary",
        ),
        "null_results": (
            "H_int as a Stabilising Potential: is not empirical validation evidence",
        ),
        "variables": ("h_int_as_a_stabilising_potential",),
        "validation_targets": ("preserve records P0R03065-P0R03066",),
        "null_controls": (
            "h_int_as_a_stabilising_potential must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PredictiveCodingIntegrationP0r03059Spec:
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
class PredictiveCodingIntegrationP0r03059SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[PredictiveCodingIntegrationP0r03059Spec, ...]
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


def build_predictive_coding_integration_p0r03059_specs(
    source_records: list[dict[str, Any]],
) -> PredictiveCodingIntegrationP0r03059SpecBundle:
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

    specs: list[PredictiveCodingIntegrationP0r03059Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PredictiveCodingIntegrationP0r03059Spec(
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
        "title": "Paper 0 " + "Predictive Coding Integration" + " Specs",
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
        "next_source_boundary": "P0R03067",
    }
    return PredictiveCodingIntegrationP0r03059SpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PredictiveCodingIntegrationP0r03059SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_predictive_coding_integration_p0r03059_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: PredictiveCodingIntegrationP0r03059SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Predictive Coding Integration" + " Specs",
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
    bundle: PredictiveCodingIntegrationP0r03059SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_predictive_coding_integration_p0r03059_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_predictive_coding_integration_p0r03059_validation_specs_{date_tag}.md"
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
