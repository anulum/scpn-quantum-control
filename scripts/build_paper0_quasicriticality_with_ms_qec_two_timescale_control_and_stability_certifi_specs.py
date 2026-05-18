#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07) spec builder
"""Promote Paper 0 Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07) records."""

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
    "P0R02983",
    "P0R02984",
    "P0R02985",
    "P0R02986",
    "P0R02987",
    "P0R02988",
    "P0R02989",
    "P0R02990",
)
CLAIM_BOUNDARY = "source-bounded quasicriticality with ms qec two timescale control and stability certifi source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi.quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi": {
        "context_id": "quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi",
        "validation_protocol": "paper0.quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi.quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi",
        "canonical_statement": "The source-bounded component 'Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07)' preserves Paper 0 records P0R02983-P0R02984 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02983:quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi",
            "P0R02984:quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi",
        ),
        "source_formulae": (
            "P0R02983: Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07)",
            "P0R02984: Purpose. We now make explicit the control architecture that allows the system to operate near quasicriticality while maintaining stability. This is achieved through a two-timescale control strategy, separating fast stabilization from slow exploration, certified by a composite Lyapunov analysis.",
        ),
        "test_protocols": (
            "preserve Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07) source-accounting boundary",
        ),
        "null_results": (
            "Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07) is not empirical validation evidence",
        ),
        "variables": ("quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi",),
        "validation_targets": ("preserve records P0R02983-P0R02984",),
        "null_controls": (
            "quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi must remain source-bounded accounting",
        ),
    },
    "quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi.two_timescale_structure": {
        "context_id": "two_timescale_structure",
        "validation_protocol": "paper0.quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi.two_timescale_structure",
        "canonical_statement": "The source-bounded component 'Two-Timescale Structure:' preserves Paper 0 records P0R02985-P0R02987 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02985:two_timescale_structure",
            "P0R02986:two_timescale_structure",
            "P0R02987:two_timescale_structure",
        ),
        "source_formulae": (
            "P0R02985: Two-Timescale Structure:",
            "P0R02986: The control system is decomposed into two channels operating on distinct timescales (tauftaus):",
            "P0R02987: Fast Channel (Stabilization): Implemented by MS-QEC and local feedback mechanisms (e.g., homeostatic synaptic plasticity) acting with gain Gf on timescale tauf. Its function is rapid error suppression and maintenance of coherence. | Slow Channel (Exploration): Controlled drift within the quasicritical band (sigma1) with gain Gs on timescale taus. Its function is to maintain the sensitivity, diversity, and adaptability of the system's dynamics.",
        ),
        "test_protocols": ("preserve Two-Timescale Structure: source-accounting boundary",),
        "null_results": ("Two-Timescale Structure: is not empirical validation evidence",),
        "variables": ("two_timescale_structure",),
        "validation_targets": ("preserve records P0R02985-P0R02987",),
        "null_controls": ("two_timescale_structure must remain source-bounded accounting",),
    },
    "quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi.gain_scheduling_via_affective_field_sensitivity": {
        "context_id": "gain_scheduling_via_affective_field_sensitivity",
        "validation_protocol": "paper0.quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi.gain_scheduling_via_affective_field_sensitivity",
        "canonical_statement": "The source-bounded component 'Gain Scheduling via Affective Field Sensitivity:' preserves Paper 0 records P0R02988-P0R02990 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02988:gain_scheduling_via_affective_field_sensitivity",
            "P0R02989:gain_scheduling_via_affective_field_sensitivity",
            "P0R02990:gain_scheduling_via_affective_field_sensitivity",
        ),
        "source_formulae": (
            "P0R02988: Gain Scheduling via Affective Field Sensitivity:",
            "P0R02989: The gains are dynamically adjusted based on the local sensitivity of the Affective Field (A) to the criticality coordinate (sigma). The Affective Field, related to the gradient of the Free Energy, reflects the organism's alignment with its generative model.",
            "P0R02990: If A/sigma is large (steep affective landscape), stabilization is prioritized: Gf is increased, Gs decreased. | If A/sigma is small near sigma1, safe exploration is prioritized: Gs is increased.",
        ),
        "test_protocols": (
            "preserve Gain Scheduling via Affective Field Sensitivity: source-accounting boundary",
        ),
        "null_results": (
            "Gain Scheduling via Affective Field Sensitivity: is not empirical validation evidence",
        ),
        "variables": ("gain_scheduling_via_affective_field_sensitivity",),
        "validation_targets": ("preserve records P0R02988-P0R02990",),
        "null_controls": (
            "gain_scheduling_via_affective_field_sensitivity must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiSpec:
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
class QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiSpec, ...]
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


def build_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_specs(
    source_records: list[dict[str, Any]],
) -> QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiSpecBundle:
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

    specs: list[QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiSpec(
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
        + "Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07)"
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
        "next_source_boundary": "P0R02991",
    }
    return QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_specs(
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
    bundle: QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Quasicriticality with MS-QEC: Two-Timescale Control and Stability Certificates (revision 11.07)"
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
    bundle: QuasicriticalityWithMsQecTwoTimescaleControlAndStabilityCertifiSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_quasicriticality_with_ms_qec_two_timescale_control_and_stability_certifi_validation_specs_{date_tag}.md"
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
