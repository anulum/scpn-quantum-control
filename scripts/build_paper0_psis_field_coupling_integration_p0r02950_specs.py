#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Psis Field Coupling Integration spec builder
"""Promote Paper 0 Psis Field Coupling Integration records."""

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
    "P0R02950",
    "P0R02951",
    "P0R02952",
    "P0R02953",
    "P0R02954",
    "P0R02955",
    "P0R02956",
    "P0R02957",
)
CLAIM_BOUNDARY = "source-bounded psis field coupling integration p0r02950 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "psis_field_coupling_integration_p0r02950.psis_field_coupling_integration": {
        "context_id": "psis_field_coupling_integration",
        "validation_protocol": "paper0.psis_field_coupling_integration_p0r02950.psis_field_coupling_integration",
        "canonical_statement": "The source-bounded component 'Psis Field Coupling Integration' preserves Paper 0 records P0R02950-P0R02951 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02950:psis_field_coupling_integration",
            "P0R02951:psis_field_coupling_integration",
        ),
        "source_formulae": (
            "P0R02950: Psis Field Coupling Integration",
            "P0R02951: The two-timescale controller provides a sophisticated interface for the Psis field to guide the system's dynamic state. The H_int = -lambda * Psis * sigma interaction here is not about setting a static state, but about biasing a dynamic balance.",
        ),
        "test_protocols": ("preserve Psis Field Coupling Integration source-accounting boundary",),
        "null_results": ("Psis Field Coupling Integration is not empirical validation evidence",),
        "variables": ("psis_field_coupling_integration",),
        "validation_targets": ("preserve records P0R02950-P0R02951",),
        "null_controls": (
            "psis_field_coupling_integration must remain source-bounded accounting",
        ),
    },
    "psis_field_coupling_integration_p0r02950.the_affective_field_as_the_coupling_interface": {
        "context_id": "the_affective_field_as_the_coupling_interface",
        "validation_protocol": "paper0.psis_field_coupling_integration_p0r02950.the_affective_field_as_the_coupling_interface",
        "canonical_statement": "The source-bounded component 'The Affective Field as the Coupling Interface:' preserves Paper 0 records P0R02952-P0R02953 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02952:the_affective_field_as_the_coupling_interface",
            "P0R02953:the_affective_field_as_the_coupling_interface",
        ),
        "source_formulae": (
            "P0R02952: The Affective Field as the Coupling Interface:",
            "P0R02953: The Affective Field is the direct readout of the system's internal state of surprise and is therefore the ideal collective state variable (sigma) for this level of control. sigma = |A/sigma|.",
        ),
        "test_protocols": (
            "preserve The Affective Field as the Coupling Interface: source-accounting boundary",
        ),
        "null_results": (
            "The Affective Field as the Coupling Interface: is not empirical validation evidence",
        ),
        "variables": ("the_affective_field_as_the_coupling_interface",),
        "validation_targets": ("preserve records P0R02952-P0R02953",),
        "null_controls": (
            "the_affective_field_as_the_coupling_interface must remain source-bounded accounting",
        ),
    },
    "psis_field_coupling_integration_p0r02950.the_psis_field_as_the_scheduler": {
        "context_id": "the_psis_field_as_the_scheduler",
        "validation_protocol": "paper0.psis_field_coupling_integration_p0r02950.the_psis_field_as_the_scheduler",
        "canonical_statement": "The source-bounded component 'The Psis Field as the Scheduler:' preserves Paper 0 records P0R02954-P0R02955 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02954:the_psis_field_as_the_scheduler",
            "P0R02955:the_psis_field_as_the_scheduler",
        ),
        "source_formulae": (
            "P0R02954: The Psis Field as the Scheduler:",
            'P0R02955: The universal Psis field couples to this measure of affective salience. The H_int interaction can be viewed as providing a top-down, goal-directed signal that modulates the affective landscape. By subtly increasing or decreasing the "affective cost" of certain states, the Psis field can encourage the local controller to shift the balance between exploration and exploitation. For example, to promote novelty, the Psi-field could "dampen" the affective response to surprise, encouraging the Gs gain to increase. To promote stability, it could amplify the response, forcing Gf to dominate. This allows the highest levels of the SCPN (e.g., Layer 15) to guide the fundamental learning strategy of all the layers below it.',
        ),
        "test_protocols": (
            "preserve The Psis Field as the Scheduler: source-accounting boundary",
        ),
        "null_results": ("The Psis Field as the Scheduler: is not empirical validation evidence",),
        "variables": ("the_psis_field_as_the_scheduler",),
        "validation_targets": ("preserve records P0R02954-P0R02955",),
        "null_controls": (
            "the_psis_field_as_the_scheduler must remain source-bounded accounting",
        ),
    },
    "psis_field_coupling_integration_p0r02950.quasicriticality_with_ms_qec_two_timescale_control_and_certificates_revi": {
        "context_id": "quasicriticality_with_ms_qec_two_timescale_control_and_certificates_revi",
        "validation_protocol": "paper0.psis_field_coupling_integration_p0r02950.quasicriticality_with_ms_qec_two_timescale_control_and_certificates_revi",
        "canonical_statement": "The source-bounded component 'Quasicriticality with MS-QEC: Two-Timescale Control and Certificates (revision 11.00)' preserves Paper 0 records P0R02956-P0R02957 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02956:quasicriticality_with_ms_qec_two_timescale_control_and_certificates_revi",
            "P0R02957:quasicriticality_with_ms_qec_two_timescale_control_and_certificates_revi",
        ),
        "source_formulae": (
            "P0R02956: Quasicriticality with MS-QEC: Two-Timescale Control and Certificates (revision 11.00)",
            "P0R02957: Purpose. We now make explicit the control split that lets the system live near quasicriticality without falling apart. Fast stabilisers (MS-QEC and local feedback) keep coherence in check; slow exploratory drift lets the phase spine sample rich dynamics in the sigma1 band. The pair is certified for bounded-input bounded-output (BIBO) stability, and the torus surface flow acquires a Lyapunov-style certificate. Gain scheduling is driven by the Affective Field A = F via its sensitivity to sigma.",
        ),
        "test_protocols": (
            "preserve Quasicriticality with MS-QEC: Two-Timescale Control and Certificates (revision 11.00) source-accounting boundary",
        ),
        "null_results": (
            "Quasicriticality with MS-QEC: Two-Timescale Control and Certificates (revision 11.00) is not empirical validation evidence",
        ),
        "variables": ("quasicriticality_with_ms_qec_two_timescale_control_and_certificates_revi",),
        "validation_targets": ("preserve records P0R02956-P0R02957",),
        "null_controls": (
            "quasicriticality_with_ms_qec_two_timescale_control_and_certificates_revi must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PsisFieldCouplingIntegrationP0r02950Spec:
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
class PsisFieldCouplingIntegrationP0r02950SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[PsisFieldCouplingIntegrationP0r02950Spec, ...]
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


def build_psis_field_coupling_integration_p0r02950_specs(
    source_records: list[dict[str, Any]],
) -> PsisFieldCouplingIntegrationP0r02950SpecBundle:
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

    specs: list[PsisFieldCouplingIntegrationP0r02950Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PsisFieldCouplingIntegrationP0r02950Spec(
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
        "title": "Paper 0 " + "Psis Field Coupling Integration" + " Specs",
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
        "next_source_boundary": "P0R02958",
    }
    return PsisFieldCouplingIntegrationP0r02950SpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PsisFieldCouplingIntegrationP0r02950SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_psis_field_coupling_integration_p0r02950_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: PsisFieldCouplingIntegrationP0r02950SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Psis Field Coupling Integration" + " Specs",
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
    bundle: PsisFieldCouplingIntegrationP0r02950SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_psis_field_coupling_integration_p0r02950_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_psis_field_coupling_integration_p0r02950_validation_specs_{date_tag}.md"
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
