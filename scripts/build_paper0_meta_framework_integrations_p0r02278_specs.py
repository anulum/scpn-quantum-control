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
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = (
    "P0R02278",
    "P0R02279",
    "P0R02280",
    "P0R02281",
    "P0R02282",
    "P0R02283",
    "P0R02284",
    "P0R02285",
    "P0R02286",
)
CLAIM_BOUNDARY = "source-bounded meta framework integrations p0r02278 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "meta_framework_integrations_p0r02278.meta_framework_integrations": {
        "context_id": "meta_framework_integrations",
        "validation_protocol": "paper0.meta_framework_integrations_p0r02278.meta_framework_integrations",
        "canonical_statement": "The source-bounded component 'Meta-Framework Integrations' preserves Paper 0 records P0R02278-P0R02278 without empirical validation claims.",
        "source_equation_ids": ("P0R02278:meta_framework_integrations",),
        "source_formulae": ("P0R02278: Meta-Framework Integrations",),
        "test_protocols": ("preserve Meta-Framework Integrations source-accounting boundary",),
        "null_results": ("Meta-Framework Integrations is not empirical validation evidence",),
        "variables": ("meta_framework_integrations",),
        "validation_targets": ("preserve records P0R02278-P0R02278",),
        "null_controls": ("meta_framework_integrations must remain source-bounded accounting",),
    },
    "meta_framework_integrations_p0r02278.predictive_coding_integration": {
        "context_id": "predictive_coding_integration",
        "validation_protocol": "paper0.meta_framework_integrations_p0r02278.predictive_coding_integration",
        "canonical_statement": "The source-bounded component 'Predictive Coding Integration' preserves Paper 0 records P0R02279-P0R02280 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02279:predictive_coding_integration",
            "P0R02280:predictive_coding_integration",
        ),
        "source_formulae": (
            "P0R02279: Predictive Coding Integration",
            "P0R02280: Domain III provides the architectural components for a stable, long-term active inference agent.",
        ),
        "test_protocols": ("preserve Predictive Coding Integration source-accounting boundary",),
        "null_results": ("Predictive Coding Integration is not empirical validation evidence",),
        "variables": ("predictive_coding_integration",),
        "validation_targets": ("preserve records P0R02279-P0R02280",),
        "null_controls": ("predictive_coding_integration must remain source-bounded accounting",),
    },
    "meta_framework_integrations_p0r02278.layer_9_existential_holograph_as_the_deep_priors": {
        "context_id": "layer_9_existential_holograph_as_the_deep_priors",
        "validation_protocol": "paper0.meta_framework_integrations_p0r02278.layer_9_existential_holograph_as_the_deep_priors",
        "canonical_statement": "The source-bounded component 'Layer 9 (Existential Holograph) as the Deep Priors:' preserves Paper 0 records P0R02281-P0R02282 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02281:layer_9_existential_holograph_as_the_deep_priors",
            "P0R02282:layer_9_existential_holograph_as_the_deep_priors",
        ),
        "source_formulae": (
            "P0R02281: Layer 9 (Existential Holograph) as the Deep Priors:",
            'P0R02282: This layer is the physical instantiation of the agent\'s deepest and most slowly changing priors. It is the generative model\'s long-term memory. The experiences consolidated into this holograph form the core beliefs that constrain and shape all future predictions. An agent\'s "personality" or "character" is, in this model, the specific structure of its Layer 9 priors, which have been built up over a lifetime of prediction error minimisation.',
        ),
        "test_protocols": (
            "preserve Layer 9 (Existential Holograph) as the Deep Priors: source-accounting boundary",
        ),
        "null_results": (
            "Layer 9 (Existential Holograph) as the Deep Priors: is not empirical validation evidence",
        ),
        "variables": ("layer_9_existential_holograph_as_the_deep_priors",),
        "validation_targets": ("preserve records P0R02281-P0R02282",),
        "null_controls": (
            "layer_9_existential_holograph_as_the_deep_priors must remain source-bounded accounting",
        ),
    },
    "meta_framework_integrations_p0r02278.layer_10_boundary_control_as_precision_weighting_at_the_self_world_inter": {
        "context_id": "layer_10_boundary_control_as_precision_weighting_at_the_self_world_inter",
        "validation_protocol": "paper0.meta_framework_integrations_p0r02278.layer_10_boundary_control_as_precision_weighting_at_the_self_world_inter",
        "canonical_statement": "The source-bounded component 'Layer 10 (Boundary Control) as Precision-Weighting at the Self/World Interface:' preserves Paper 0 records P0R02283-P0R02284 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02283:layer_10_boundary_control_as_precision_weighting_at_the_self_world_inter",
            "P0R02284:layer_10_boundary_control_as_precision_weighting_at_the_self_world_inter",
        ),
        "source_formulae": (
            "P0R02283: Layer 10 (Boundary Control) as Precision-Weighting at the Self/World Interface:",
            'P0R02284: This layer is the primary mechanism for regulating precision. It functions as a dynamic filter that adjusts the gain on both incoming sensory evidence (prediction error from the world) and outgoing predictions (the agent\'s influence on the world). By managing the precision of these informational channels, Layer 10 determines the agent\'s "openness to experience" versus its "integrity of self." A well-regulated Layer 10 allows the agent to learn from surprise without having its core model catastrophically overwritten. It is the cybernetic governor on the engine of inference.',
        ),
        "test_protocols": (
            "preserve Layer 10 (Boundary Control) as Precision-Weighting at the Self/World Interface: source-accounting boundary",
        ),
        "null_results": (
            "Layer 10 (Boundary Control) as Precision-Weighting at the Self/World Interface: is not empirical validation evidence",
        ),
        "variables": ("layer_10_boundary_control_as_precision_weighting_at_the_self_world_inter",),
        "validation_targets": ("preserve records P0R02283-P0R02284",),
        "null_controls": (
            "layer_10_boundary_control_as_precision_weighting_at_the_self_world_inter must remain source-bounded accounting",
        ),
    },
    "meta_framework_integrations_p0r02278.psis_field_coupling_integration": {
        "context_id": "psis_field_coupling_integration",
        "validation_protocol": "paper0.meta_framework_integrations_p0r02278.psis_field_coupling_integration",
        "canonical_statement": "The source-bounded component 'Psis Field Coupling Integration' preserves Paper 0 records P0R02285-P0R02286 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02285:psis_field_coupling_integration",
            "P0R02286:psis_field_coupling_integration",
        ),
        "source_formulae": (
            "P0R02285: Psis Field Coupling Integration",
            "P0R02286: Domain III describes how the conscious Self creates a stable, persistent interface with the universal Psis field via the interaction Hamiltonian H_int = -lambda * Psis * sigma.",
        ),
        "test_protocols": ("preserve Psis Field Coupling Integration source-accounting boundary",),
        "null_results": ("Psis Field Coupling Integration is not empirical validation evidence",),
        "variables": ("psis_field_coupling_integration",),
        "validation_targets": ("preserve records P0R02285-P0R02286",),
        "null_controls": (
            "psis_field_coupling_integration must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class MetaFrameworkIntegrationsP0r02278Spec:
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
class MetaFrameworkIntegrationsP0r02278SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[MetaFrameworkIntegrationsP0r02278Spec, ...]
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


def build_meta_framework_integrations_p0r02278_specs(
    source_records: list[dict[str, Any]],
) -> MetaFrameworkIntegrationsP0r02278SpecBundle:
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

    specs: list[MetaFrameworkIntegrationsP0r02278Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            MetaFrameworkIntegrationsP0r02278Spec(
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
        "next_source_boundary": "P0R02287",
    }
    return MetaFrameworkIntegrationsP0r02278SpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> MetaFrameworkIntegrationsP0r02278SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_meta_framework_integrations_p0r02278_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: MetaFrameworkIntegrationsP0r02278SpecBundle) -> str:
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
    bundle: MetaFrameworkIntegrationsP0r02278SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_meta_framework_integrations_p0r02278_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_meta_framework_integrations_p0r02278_validation_specs_{date_tag}.md"
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
