#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Meta-Framework Integrations spec builder
"""Promote Paper 0 Meta-Framework Integrations records."""

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
    "P0R03284",
    "P0R03285",
    "P0R03286",
    "P0R03287",
    "P0R03288",
    "P0R03289",
    "P0R03290",
    "P0R03291",
    "P0R03292",
    "P0R03293",
    "P0R03294",
)
CLAIM_BOUNDARY = "source-bounded meta framework integrations p0r03284 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "meta_framework_integrations_p0r03284.meta_framework_integrations": {
        "context_id": "meta_framework_integrations",
        "validation_protocol": "paper0.meta_framework_integrations_p0r03284.meta_framework_integrations",
        "canonical_statement": "The source-bounded component 'Meta-Framework Integrations' preserves Paper 0 records P0R03284-P0R03284 without empirical validation claims.",
        "source_equation_ids": ("P0R03284:meta_framework_integrations",),
        "source_formulae": ("P0R03284: Meta-Framework Integrations",),
        "test_protocols": ("preserve Meta-Framework Integrations source-accounting boundary",),
        "null_results": ("Meta-Framework Integrations is not empirical validation evidence",),
        "variables": ("meta_framework_integrations",),
        "validation_targets": ("preserve records P0R03284-P0R03284",),
        "null_controls": ("meta_framework_integrations must remain source-bounded accounting",),
    },
    "meta_framework_integrations_p0r03284.predictive_coding_integration": {
        "context_id": "predictive_coding_integration",
        "validation_protocol": "paper0.meta_framework_integrations_p0r03284.predictive_coding_integration",
        "canonical_statement": "The source-bounded component 'Predictive Coding Integration' preserves Paper 0 records P0R03285-P0R03286 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03285:predictive_coding_integration",
            "P0R03286:predictive_coding_integration",
        ),
        "source_formulae": (
            "P0R03285: Predictive Coding Integration",
            "P0R03286: CIGD is the physical mechanism that instantiates the predictions of the generative model at the most fundamental level of reality.",
        ),
        "test_protocols": ("preserve Predictive Coding Integration source-accounting boundary",),
        "null_results": ("Predictive Coding Integration is not empirical validation evidence",),
        "variables": ("predictive_coding_integration",),
        "validation_targets": ("preserve records P0R03285-P0R03286",),
        "null_controls": ("predictive_coding_integration must remain source-bounded accounting",),
    },
    "meta_framework_integrations_p0r03284.collapse_as_inference": {
        "context_id": "collapse_as_inference",
        "validation_protocol": "paper0.meta_framework_integrations_p0r03284.collapse_as_inference",
        "canonical_statement": "The source-bounded component 'Collapse as Inference:' preserves Paper 0 records P0R03287-P0R03288 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03287:collapse_as_inference",
            "P0R03288:collapse_as_inference",
        ),
        "source_formulae": (
            "P0R03287: Collapse as Inference:",
            "P0R03288: In the HPC framework, perception is inference-the process of selecting the most likely cause of sensory data from a landscape of possibilities. The CIGD mechanism is the physical realisation of this selection process. The quantum superposition represents the space of all possible \"hypotheses\" about reality. The generative model's current prediction, embodied by a high- state of the local Psi-field, acts as the selection pressure that collapses this space of hypotheses into a single, definite outcome that is consistent with the model's belief.",
        ),
        "test_protocols": ("preserve Collapse as Inference: source-accounting boundary",),
        "null_results": ("Collapse as Inference: is not empirical validation evidence",),
        "variables": ("collapse_as_inference",),
        "validation_targets": ("preserve records P0R03287-P0R03288",),
        "null_controls": ("collapse_as_inference must remain source-bounded accounting",),
    },
    "meta_framework_integrations_p0r03284.psis_field_coupling_integration": {
        "context_id": "psis_field_coupling_integration",
        "validation_protocol": "paper0.meta_framework_integrations_p0r03284.psis_field_coupling_integration",
        "canonical_statement": "The source-bounded component 'Psis Field Coupling Integration' preserves Paper 0 records P0R03289-P0R03294 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03289:psis_field_coupling_integration",
            "P0R03290:psis_field_coupling_integration",
            "P0R03291:psis_field_coupling_integration",
            "P0R03292:psis_field_coupling_integration",
            "P0R03293:psis_field_coupling_integration",
            "P0R03294:psis_field_coupling_integration",
        ),
        "source_formulae": (
            "P0R03289: Psis Field Coupling Integration",
            "P0R03290: CIGD is a direct, physical consequence of the geometric coupling term in the Master Lagrangian, and it provides a profound interpretation of the H_int = -lambda * Psis * sigma interaction at the quantum level.",
            "P0R03291: The Geometric H_int: CIGD is the primary effect of the geometric part of the interaction.",
            "P0R03292: Psis: This is the integrated consciousness field, whose intensity is measured by .",
            "P0R03293: sigma: This is the quantum system in superposition, which is also a superposition of spacetime geometries.",
            "P0R03294: The Interaction: The coupling between the high- field and the superposition of geometries creates an unstable state that rapidly resolves.",
        ),
        "test_protocols": ("preserve Psis Field Coupling Integration source-accounting boundary",),
        "null_results": ("Psis Field Coupling Integration is not empirical validation evidence",),
        "variables": ("psis_field_coupling_integration",),
        "validation_targets": ("preserve records P0R03289-P0R03294",),
        "null_controls": (
            "psis_field_coupling_integration must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class MetaFrameworkIntegrationsP0r03284Spec:
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
class MetaFrameworkIntegrationsP0r03284SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[MetaFrameworkIntegrationsP0r03284Spec, ...]
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


def build_meta_framework_integrations_p0r03284_specs(
    source_records: list[dict[str, Any]],
) -> MetaFrameworkIntegrationsP0r03284SpecBundle:
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

    specs: list[MetaFrameworkIntegrationsP0r03284Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            MetaFrameworkIntegrationsP0r03284Spec(
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
        "title": "Paper 0 " + "Meta-Framework Integrations" + " Specs",
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
        "next_source_boundary": "P0R03295",
    }
    return MetaFrameworkIntegrationsP0r03284SpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> MetaFrameworkIntegrationsP0r03284SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_meta_framework_integrations_p0r03284_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: MetaFrameworkIntegrationsP0r03284SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Meta-Framework Integrations" + " Specs",
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
    bundle: MetaFrameworkIntegrationsP0r03284SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_meta_framework_integrations_p0r03284_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_meta_framework_integrations_p0r03284_validation_specs_{date_tag}.md"
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
