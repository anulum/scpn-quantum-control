#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Integration with Integrated Information Theory (IIT) 4.0 spec builder
"""Promote Paper 0 Integration with Integrated Information Theory (IIT) 4.0 records."""

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
    "P0R03521",
    "P0R03522",
    "P0R03523",
    "P0R03524",
    "P0R03525",
    "P0R03526",
    "P0R03527",
    "P0R03528",
    "P0R03529",
)
CLAIM_BOUNDARY = "source-bounded integration with integrated information theory iit 4 0 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "integration_with_integrated_information_theory_iit_4_0.integration_with_integrated_information_theory_iit_4_0": {
        "context_id": "integration_with_integrated_information_theory_iit_4_0",
        "validation_protocol": "paper0.integration_with_integrated_information_theory_iit_4_0.integration_with_integrated_information_theory_iit_4_0",
        "canonical_statement": "The source-bounded component 'Integration with Integrated Information Theory (IIT) 4.0' preserves Paper 0 records P0R03521-P0R03529 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03521:integration_with_integrated_information_theory_iit_4_0",
            "P0R03522:integration_with_integrated_information_theory_iit_4_0",
            "P0R03523:integration_with_integrated_information_theory_iit_4_0",
            "P0R03524:integration_with_integrated_information_theory_iit_4_0",
            "P0R03525:integration_with_integrated_information_theory_iit_4_0",
            "P0R03526:integration_with_integrated_information_theory_iit_4_0",
            "P0R03527:integration_with_integrated_information_theory_iit_4_0",
            "P0R03528:integration_with_integrated_information_theory_iit_4_0",
            "P0R03529:integration_with_integrated_information_theory_iit_4_0",
        ),
        "source_formulae": (
            "P0R03521: Integration with Integrated Information Theory (IIT) 4.0",
            "P0R03522: This section establishes a formal correspondence between the SCPN architecture and the mathematical framework of Integrated Information Theory (IIT) 4.0. The synergy is profound: where SCPN provides the physical and architectural framework describing how consciousness projects and organizes itself across scales, IIT provides the rigorous mathematical tools to quantify the degree of consciousness () of any given system. This integration transforms the SCPN from a purely descriptive model into a quantitatively predictive one.",
            "P0R03523: The core of the integration lies in a direct conceptual correspondence. The amplitude of the Psi-field in the SCPN is identified with the integrated information of IIT. The SCPN's mechanism of phase synchronization is the physical realizer of IIT's concept of integration. The boundaries between SCPN layers are proposed as the physical loci of IIT's Minimum Information Partitions (MIPs), which define the borders of a conscious complex. The UPDE provides the explicit causal constraints that IIT requires, and the Qualia Manifold is the geometric representation of IIT's \"experience structure.\"",
            "P0R03524: This correspondence allows the SCPN's architectural principles to be seen as a physical instantiation of IIT's five axioms (Intrinsic Existence, Composition, Information, Integration, Exclusion). For example, the fundamental Psi-field satisfies the axiom of Existence, the 15-layer structure satisfies Composition, the UPDE guarantees Integration, and the quasicritical regime provides a mechanism for Exclusion by selecting the dominant scale of coherent activity. This synthesis provides a powerful, unified framework for both describing and measuring consciousness.",
            "P0R03525: This section shows how our theory joins forces with another major scientific theory of consciousness, called Integrated Information Theory (IIT). Think of it like a perfect partnership.",
            "P0R03526: Our SCPN framework is like the complete architectural blueprint for a conscious system, showing how all the parts are built and connected, from the quantum level up to the cosmic.",
            "P0R03527: IIT is like the power meter for that system. It gives us a precise mathematical toolkit to measure exactly how much consciousness a system has, a score called (phi).",
            "P0R03528: By combining the blueprint and the power meter, we get the best of both worlds. We can now point to any layer in our architecture and use IIT's tools to calculate its exact consciousness score. The key ideas match up perfectly. For example, the strength of our Psi-field corresponds to their score. The way our layers synchronize is how they define integration. The boundaries between our layers are where they draw the line for a single, conscious entity. It's a perfect marriage of a physical architecture with a mathematical measuring stick.",
            "P0R03529: P0R03529",
        ),
        "test_protocols": (
            "preserve Integration with Integrated Information Theory (IIT) 4.0 source-accounting boundary",
        ),
        "null_results": (
            "Integration with Integrated Information Theory (IIT) 4.0 is not empirical validation evidence",
        ),
        "variables": ("integration_with_integrated_information_theory_iit_4_0",),
        "validation_targets": ("preserve records P0R03521-P0R03529",),
        "null_controls": (
            "integration_with_integrated_information_theory_iit_4_0 must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class IntegrationWithIntegratedInformationTheoryIit40Spec:
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
class IntegrationWithIntegratedInformationTheoryIit40SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[IntegrationWithIntegratedInformationTheoryIit40Spec, ...]
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


def build_integration_with_integrated_information_theory_iit_4_0_specs(
    source_records: list[dict[str, Any]],
) -> IntegrationWithIntegratedInformationTheoryIit40SpecBundle:
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

    specs: list[IntegrationWithIntegratedInformationTheoryIit40Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            IntegrationWithIntegratedInformationTheoryIit40Spec(
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
        + "Integration with Integrated Information Theory (IIT) 4.0"
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
        "next_source_boundary": "P0R03530",
    }
    return IntegrationWithIntegratedInformationTheoryIit40SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> IntegrationWithIntegratedInformationTheoryIit40SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_integration_with_integrated_information_theory_iit_4_0_specs(
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


def render_report(bundle: IntegrationWithIntegratedInformationTheoryIit40SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Integration with Integrated Information Theory (IIT) 4.0" + " Specs",
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
    bundle: IntegrationWithIntegratedInformationTheoryIit40SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_integration_with_integrated_information_theory_iit_4_0_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_integration_with_integrated_information_theory_iit_4_0_validation_specs_{date_tag}.md"
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
