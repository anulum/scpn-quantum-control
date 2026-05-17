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
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = (
    "P0R01763",
    "P0R01764",
    "P0R01765",
    "P0R01766",
    "P0R01767",
    "P0R01768",
    "P0R01769",
    "P0R01770",
)
CLAIM_BOUNDARY = "source-bounded predictive coding integration source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "predictive_coding_integration.predictive_coding_integration": {
        "context_id": "predictive_coding_integration",
        "validation_protocol": "paper0.predictive_coding_integration.predictive_coding_integration",
        "canonical_statement": "The source-bounded component 'Predictive Coding Integration' preserves Paper 0 records P0R01763-P0R01764 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01763:predictive_coding_integration",
            "P0R01764:predictive_coding_integration",
        ),
        "source_formulae": (
            "P0R01763: Predictive Coding Integration",
            "P0R01764: The intrinsic Lagrangian LPsi defines the state space and dynamics of the cosmic generative model itself.",
        ),
        "test_protocols": ("preserve Predictive Coding Integration source-accounting boundary",),
        "null_results": ("Predictive Coding Integration is not empirical validation evidence",),
        "variables": ("predictive_coding_integration",),
        "validation_targets": ("preserve records P0R01763-P0R01764",),
        "null_controls": ("predictive_coding_integration must remain source-bounded accounting",),
    },
    "predictive_coding_integration.the_potential_v_psi_as_the_landscape_of_priors": {
        "context_id": "the_potential_v_psi_as_the_landscape_of_priors",
        "validation_protocol": "paper0.predictive_coding_integration.the_potential_v_psi_as_the_landscape_of_priors",
        "canonical_statement": "The source-bounded component 'The Potential V(|Psi|) as the Landscape of Priors:' preserves Paper 0 records P0R01765-P0R01766 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01765:the_potential_v_psi_as_the_landscape_of_priors",
            "P0R01766:the_potential_v_psi_as_the_landscape_of_priors",
        ),
        "source_formulae": (
            "P0R01765: The Potential V(|Psi|) as the Landscape of Priors:",
            'P0R01766: The shape of the potential V(|Psi|) represents the landscape of the model\'s possible "beliefs" or priors. The stable minima of the potential (the bottom of the "moat" in the Mexican hat) are the most stable, lowest-free-energy beliefs the system can hold.',
        ),
        "test_protocols": (
            "preserve The Potential V(|Psi|) as the Landscape of Priors: source-accounting boundary",
        ),
        "null_results": (
            "The Potential V(|Psi|) as the Landscape of Priors: is not empirical validation evidence",
        ),
        "variables": ("the_potential_v_psi_as_the_landscape_of_priors",),
        "validation_targets": ("preserve records P0R01765-P0R01766",),
        "null_controls": (
            "the_potential_v_psi_as_the_landscape_of_priors must remain source-bounded accounting",
        ),
    },
    "predictive_coding_integration.the_sextic_term_as_a_sanity_check": {
        "context_id": "the_sextic_term_as_a_sanity_check",
        "validation_protocol": "paper0.predictive_coding_integration.the_sextic_term_as_a_sanity_check",
        "canonical_statement": "The source-bounded component 'The Sextic Term as a \"Sanity Check\":' preserves Paper 0 records P0R01767-P0R01768 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01767:the_sextic_term_as_a_sanity_check",
            "P0R01768:the_sextic_term_as_a_sanity_check",
        ),
        "source_formulae": (
            'P0R01767: The Sextic Term as a "Sanity Check":',
            'P0R01768: The stabilising ^6 term is crucial. It ensures that the space of possible beliefs is bounded. In computational terms, it prevents the model from developing "runaway," infinitely confident (and therefore delusional) beliefs. It guarantees the integrity and stability of the cosmic inference engine\'s state space.',
        ),
        "test_protocols": (
            'preserve The Sextic Term as a "Sanity Check": source-accounting boundary',
        ),
        "null_results": (
            'The Sextic Term as a "Sanity Check": is not empirical validation evidence',
        ),
        "variables": ("the_sextic_term_as_a_sanity_check",),
        "validation_targets": ("preserve records P0R01767-P0R01768",),
        "null_controls": (
            "the_sextic_term_as_a_sanity_check must remain source-bounded accounting",
        ),
    },
    "predictive_coding_integration.psis_field_coupling_integration": {
        "context_id": "psis_field_coupling_integration",
        "validation_protocol": "paper0.predictive_coding_integration.psis_field_coupling_integration",
        "canonical_statement": "The source-bounded component 'Psis Field Coupling Integration' preserves Paper 0 records P0R01769-P0R01770 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01769:psis_field_coupling_integration",
            "P0R01770:psis_field_coupling_integration",
        ),
        "source_formulae": (
            "P0R01769: Psis Field Coupling Integration",
            "P0R01770: The intrinsic Lagrangian LPsi governs the behavior of the Psis field, which is the first term in the interaction Hamiltonian H_int = -lambda * Psis * sigma.",
        ),
        "test_protocols": ("preserve Psis Field Coupling Integration source-accounting boundary",),
        "null_results": ("Psis Field Coupling Integration is not empirical validation evidence",),
        "variables": ("psis_field_coupling_integration",),
        "validation_targets": ("preserve records P0R01769-P0R01770",),
        "null_controls": (
            "psis_field_coupling_integration must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PredictiveCodingIntegrationSpec:
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
class PredictiveCodingIntegrationSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[PredictiveCodingIntegrationSpec, ...]
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


def build_predictive_coding_integration_specs(
    source_records: list[dict[str, Any]],
) -> PredictiveCodingIntegrationSpecBundle:
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

    specs: list[PredictiveCodingIntegrationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PredictiveCodingIntegrationSpec(
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
        "title": "Paper 0 Predictive Coding Integration Specs",
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
        "next_source_boundary": "P0R01771",
    }
    return PredictiveCodingIntegrationSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PredictiveCodingIntegrationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_predictive_coding_integration_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: PredictiveCodingIntegrationSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 Predictive Coding Integration Specs",
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
    bundle: PredictiveCodingIntegrationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_predictive_coding_integration_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_predictive_coding_integration_validation_specs_{date_tag}.md"
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
