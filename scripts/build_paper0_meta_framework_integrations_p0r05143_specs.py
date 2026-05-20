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
    "P0R05143",
    "P0R05144",
    "P0R05145",
    "P0R05146",
    "P0R05147",
    "P0R05148",
    "P0R05149",
    "P0R05150",
    "P0R05151",
)
CLAIM_BOUNDARY = "source-bounded meta framework integrations p0r05143 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "meta_framework_integrations_p0r05143.meta_framework_integrations": {
        "context_id": "meta_framework_integrations",
        "validation_protocol": "paper0.meta_framework_integrations_p0r05143.meta_framework_integrations",
        "canonical_statement": "The source-bounded component 'Meta-Framework Integrations' preserves Paper 0 records P0R05143-P0R05143 without empirical validation claims.",
        "source_equation_ids": ("P0R05143:meta_framework_integrations",),
        "source_formulae": ("P0R05143: Meta-Framework Integrations",),
        "test_protocols": ("preserve Meta-Framework Integrations source-accounting boundary",),
        "null_results": ("Meta-Framework Integrations is not empirical validation evidence",),
        "variables": ("meta_framework_integrations",),
        "validation_targets": ("preserve records P0R05143-P0R05143",),
        "null_controls": ("meta_framework_integrations must remain source-bounded accounting",),
    },
    "meta_framework_integrations_p0r05143.predictive_coding_integration": {
        "context_id": "predictive_coding_integration",
        "validation_protocol": "paper0.meta_framework_integrations_p0r05143.predictive_coding_integration",
        "canonical_statement": "The source-bounded component 'Predictive Coding Integration' preserves Paper 0 records P0R05144-P0R05145 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05144:predictive_coding_integration",
            "P0R05145:predictive_coding_integration",
        ),
        "source_formulae": (
            "P0R05144: Predictive Coding Integration",
            "P0R05145: These experiments are direct tests of the physical reality and causal efficacy of the generative model.",
        ),
        "test_protocols": ("preserve Predictive Coding Integration source-accounting boundary",),
        "null_results": ("Predictive Coding Integration is not empirical validation evidence",),
        "variables": ("predictive_coding_integration",),
        "validation_targets": ("preserve records P0R05144-P0R05145",),
        "null_controls": ("predictive_coding_integration must remain source-bounded accounting",),
    },
    "meta_framework_integrations_p0r05143.prediction_i_nv_mea_tests_the_model_s_geometry": {
        "context_id": "prediction_i_nv_mea_tests_the_model_s_geometry",
        "validation_protocol": "paper0.meta_framework_integrations_p0r05143.prediction_i_nv_mea_tests_the_model_s_geometry",
        "canonical_statement": "The source-bounded component 'Prediction I (NV-MEA) Tests the Model's Geometry:' preserves Paper 0 records P0R05146-P0R05147 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05146:prediction_i_nv_mea_tests_the_model_s_geometry",
            "P0R05147:prediction_i_nv_mea_tests_the_model_s_geometry",
        ),
        "source_formulae": (
            "P0R05146: Prediction I (NV-MEA) Tests the Model's Geometry:",
            'P0R05147: The FIM is the geometry of the generative model\'s belief space. This experiment is a direct test of whether this abstract geometry is a physically real object that can exert a measurable force. It asks: does the "shape of belief" itself have a physical presence?',
        ),
        "test_protocols": (
            "preserve Prediction I (NV-MEA) Tests the Model's Geometry: source-accounting boundary",
        ),
        "null_results": (
            "Prediction I (NV-MEA) Tests the Model's Geometry: is not empirical validation evidence",
        ),
        "variables": ("prediction_i_nv_mea_tests_the_model_s_geometry",),
        "validation_targets": ("preserve records P0R05146-P0R05147",),
        "null_controls": (
            "prediction_i_nv_mea_tests_the_model_s_geometry must remain source-bounded accounting",
        ),
    },
    "meta_framework_integrations_p0r05143.prediction_ii_qrng_tests_the_model_s_deepest_prior": {
        "context_id": "prediction_ii_qrng_tests_the_model_s_deepest_prior",
        "validation_protocol": "paper0.meta_framework_integrations_p0r05143.prediction_ii_qrng_tests_the_model_s_deepest_prior",
        "canonical_statement": "The source-bounded component 'Prediction II (QRNG) Tests the Model's Deepest Prior:' preserves Paper 0 records P0R05148-P0R05149 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05148:prediction_ii_qrng_tests_the_model_s_deepest_prior",
            "P0R05149:prediction_ii_qrng_tests_the_model_s_deepest_prior",
        ),
        "source_formulae": (
            "P0R05148: Prediction II (QRNG) Tests the Model's Deepest Prior:",
            "P0R05149: The CEF is the physical manifestation of the model's deepest prior-the teleological drive toward coherence (Axiom 3). This experiment tests whether this fundamental \"belief\" of the cosmic generative model is powerful enough to act as a causal force, biasing the sensory data (the QRNG output) to make it conform to the model's prediction that the universe should be coherent.",
        ),
        "test_protocols": (
            "preserve Prediction II (QRNG) Tests the Model's Deepest Prior: source-accounting boundary",
        ),
        "null_results": (
            "Prediction II (QRNG) Tests the Model's Deepest Prior: is not empirical validation evidence",
        ),
        "variables": ("prediction_ii_qrng_tests_the_model_s_deepest_prior",),
        "validation_targets": ("preserve records P0R05148-P0R05149",),
        "null_controls": (
            "prediction_ii_qrng_tests_the_model_s_deepest_prior must remain source-bounded accounting",
        ),
    },
    "meta_framework_integrations_p0r05143.psis_field_coupling_integration": {
        "context_id": "psis_field_coupling_integration",
        "validation_protocol": "paper0.meta_framework_integrations_p0r05143.psis_field_coupling_integration",
        "canonical_statement": "The source-bounded component 'Psis Field Coupling Integration' preserves Paper 0 records P0R05150-P0R05151 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05150:psis_field_coupling_integration",
            "P0R05151:psis_field_coupling_integration",
        ),
        "source_formulae": (
            "P0R05150: Psis Field Coupling Integration",
            "P0R05151: These experiments provide falsifiable tests for the two distinct coupling channels in the Master Interaction Lagrangian, LInt, which is the full expression of the H_int interaction.",
        ),
        "test_protocols": ("preserve Psis Field Coupling Integration source-accounting boundary",),
        "null_results": ("Psis Field Coupling Integration is not empirical validation evidence",),
        "variables": ("psis_field_coupling_integration",),
        "validation_targets": ("preserve records P0R05150-P0R05151",),
        "null_controls": (
            "psis_field_coupling_integration must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class MetaFrameworkIntegrationsP0r05143Spec:
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
class MetaFrameworkIntegrationsP0r05143SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[MetaFrameworkIntegrationsP0r05143Spec, ...]
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


def build_meta_framework_integrations_p0r05143_specs(
    source_records: list[dict[str, Any]],
) -> MetaFrameworkIntegrationsP0r05143SpecBundle:
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

    specs: list[MetaFrameworkIntegrationsP0r05143Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            MetaFrameworkIntegrationsP0r05143Spec(
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
        "next_source_boundary": "P0R05152",
    }
    return MetaFrameworkIntegrationsP0r05143SpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> MetaFrameworkIntegrationsP0r05143SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_meta_framework_integrations_p0r05143_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: MetaFrameworkIntegrationsP0r05143SpecBundle) -> str:
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
    bundle: MetaFrameworkIntegrationsP0r05143SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_meta_framework_integrations_p0r05143_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_meta_framework_integrations_p0r05143_validation_specs_{date_tag}.md"
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
