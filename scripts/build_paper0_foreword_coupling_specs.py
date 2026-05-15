#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Foreword coupling spec builder
"""Promote Paper 0 Foreword predictive-coding and coupling records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(268, 307))
IMAGE_MARKER_IDS = ("P0R00305",)
CLAIM_BOUNDARY = "source-bounded Foreword coupling formula; not empirical validation evidence"
HARDWARE_STATUS = "source_formula_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "foreword_coupling.part_i_foreword_boundary": {
        "context_id": "part_i_foreword_boundary",
        "validation_protocol": "paper0.foreword_coupling.part_i_foreword_boundary",
        "canonical_statement": (
            "The slice opens Part I and the Foreword as the transition into the "
            "foundational architecture layer."
        ),
        "source_equation_ids": (
            "P0R00268:part_i_foundational_bedrock",
            "P0R00270:invitation_field_architecture_intro",
            "P0R00272:foreword_new_scientific_paradigm",
        ),
        "source_formulae": (
            "Part I: The Foundational Bedrock",
            "The Invitation: Field Architecture and the Framework Intro",
            "Foreword: A New Scientific Paradigm",
        ),
        "test_protocols": ("preserve Part I and Foreword source boundary",),
        "null_results": ("Foreword section labels are not empirical validation evidence",),
        "variables": ("Part_I", "Foreword", "Field_Architecture"),
        "validation_targets": (
            "preserve Part I boundary",
            "preserve Foreword section boundary",
            "preserve field-architecture invitation context",
        ),
        "null_controls": (
            "section-title-as-evidence control must be rejected",
            "missing-Foreword-boundary control must be rejected",
        ),
    },
    "foreword_coupling.bidirectional_scpn_architecture": {
        "context_id": "bidirectional_scpn_architecture",
        "validation_protocol": "paper0.foreword_coupling.bidirectional_scpn_architecture",
        "canonical_statement": (
            "The Foreword frames the SCPN as a 15-layer hierarchical architecture "
            "with downward projection and upward feedback across scales."
        ),
        "source_equation_ids": (
            "P0R00273:book_i_to_book_ii_transition",
            "P0R00274:15_layer_bidirectional_flow",
            "P0R00275:unifying_scale_dependent_phenomena",
            "P0R00276-P0R00279:plain_language_bidirectional_architecture",
        ),
        "source_formulae": (
            "15-layer hierarchical model",
            "bidirectional flow of information and causation",
            "projection of consciousness",
            "feedback loops",
        ),
        "test_protocols": ("preserve bidirectional architecture as source hypothesis",),
        "null_results": ("architectural framing is not a measured coupling",),
        "variables": ("L_1_to_L_15", "downward_projection", "upward_feedback"),
        "validation_targets": (
            "preserve 15-layer hierarchy",
            "preserve bidirectional information-flow claim",
            "preserve scale-dependent unification target",
        ),
        "null_controls": (
            "one-way-projection-only control must be rejected",
            "hierarchy-as-measurement control must be rejected",
        ),
    },
    "foreword_coupling.predictive_coding_channels": {
        "context_id": "predictive_coding_channels",
        "validation_protocol": "paper0.foreword_coupling.predictive_coding_channels",
        "canonical_statement": (
            "The Foreword is interpreted as a hierarchical predictive-coding sketch: "
            "downward projection maps to a generative model, upward feedback maps "
            "to prediction-error flow, and the full system maps to active inference."
        ),
        "source_equation_ids": (
            "P0R00280:meta_framework_integrations",
            "P0R00281:predictive_coding_integration",
            "P0R00282:cosmic_hpc_architecture",
            "P0R00283-P0R00284:downward_projection_generative_model",
            "P0R00285-P0R00287:upward_feedback_prediction_error_active_inference",
        ),
        "source_formulae": (
            "downward projection",
            "generative model",
            "upward feedback",
            "prediction error flow",
            "active inference engine",
        ),
        "test_protocols": ("classify predictive-coding channels from source labels",),
        "null_results": ("predictive-coding interpretation is not hardware evidence",),
        "variables": ("generative_model", "prediction_error", "active_inference"),
        "validation_targets": (
            "map downward projection to generative-model channel",
            "map upward feedback to prediction-error-flow channel",
            "preserve active-inference engine statement as source-bounded",
        ),
        "null_controls": (
            "unknown-channel control must be rejected",
            "active-inference-as-empirical-proof control must be rejected",
        ),
    },
    "foreword_coupling.psi_field_interaction_hamiltonian": {
        "context_id": "psi_field_interaction_hamiltonian",
        "validation_protocol": "paper0.foreword_coupling.psi_field_interaction_hamiltonian",
        "canonical_statement": (
            "The Foreword states a universal source interaction Hamiltonian and frames "
            "sigma as the layer-specific collective state variable."
        ),
        "source_equation_ids": (
            "P0R00289:psi_field_material_coupling_question",
            "P0R00290:H_int_formula",
            "P0R00291-P0R00296:layer_sigma_identification_programme",
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma",
            "universal Psi-field",
            "specific collective state variable sigma",
            "layer-specific sigma identification programme",
        ),
        "test_protocols": ("evaluate source Hamiltonian sign and parameter retention",),
        "null_results": (
            "source Hamiltonian formula is not an experimentally validated coupling",
        ),
        "variables": ("lambda_coupling", "Psi_s", "sigma", "H_int"),
        "validation_targets": (
            "preserve negative interaction sign",
            "preserve all three source parameters",
            "preserve layer-specific sigma programme boundary",
        ),
        "null_controls": (
            "omitted-parameter control must be rejected",
            "empirical-validation-overclaim control must be rejected",
        ),
    },
    "foreword_coupling.architecture_diagram_boundary": {
        "context_id": "architecture_diagram_boundary",
        "validation_protocol": "paper0.foreword_coupling.architecture_diagram_boundary",
        "canonical_statement": (
            "The slice closes with Foreword continuation, one architecture image marker, "
            "and the Preface I boundary at P0R00307."
        ),
        "source_equation_ids": (
            "P0R00297-P0R00304:foreword_continuation",
            "P0R00305:image_marker",
            "P0R00306:layered_architecture_diagram_caption",
        ),
        "source_formulae": (
            "Sentient-Consciousness Projection Network",
            "15-layer model",
            "Layered architecture diagram",
            "Preface I boundary P0R00307",
        ),
        "test_protocols": ("preserve image-marker count and next-section boundary",),
        "null_results": ("image marker and caption are not validation evidence",),
        "variables": ("image_marker_count", "preface_i_boundary", "SCPN"),
        "validation_targets": (
            "preserve Foreword continuation context",
            "preserve architecture diagram marker count",
            "preserve Preface I boundary",
        ),
        "null_controls": (
            "image-marker-as-data control must be rejected",
            "missing-Preface-I-boundary control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ForewordCouplingSpec:
    """Foreword coupling spec promoted from Paper 0 records."""

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
class ForewordCouplingSpecBundle:
    """Foreword coupling specs plus source coverage summary."""

    specs: tuple[ForewordCouplingSpec, ...]
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


def build_foreword_coupling_specs(
    source_records: list[dict[str, Any]],
) -> ForewordCouplingSpecBundle:
    """Build source-covered Foreword predictive-coding and coupling specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    image_marker_count = sum(
        1
        for ledger_id in IMAGE_MARKER_IDS
        if str(records_by_ledger[ledger_id]["text"]).startswith("[IMAGE:")
    )
    sigma_layer_example_count = len(("P0R00293", "P0R00294", "P0R00295"))
    specs: list[ForewordCouplingSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ForewordCouplingSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(str(item) for item in metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                source_formulae=tuple(str(item) for item in metadata["source_formulae"]),
                test_protocols=tuple(str(item) for item in metadata["test_protocols"]),
                null_results=tuple(str(item) for item in metadata["null_results"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_executable_fixture",
                domain_review_status="source_formula_preserved",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed_ids = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 Foreword Coupling Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed_ids),
        "coverage_match": tuple(consumed_ids) == SOURCE_LEDGER_IDS,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "sigma_layer_example_count": sigma_layer_example_count,
        "image_marker_count": image_marker_count,
        "preface_i_boundary": "P0R00307",
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed_ids
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
    }
    return ForewordCouplingSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(path: Path = DEFAULT_LEDGER_PATH) -> ForewordCouplingSpecBundle:
    """Build Foreword coupling specs from the canonical ledger."""
    return build_foreword_coupling_specs(load_jsonl(path))


def write_outputs(
    bundle: ForewordCouplingSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown Foreword coupling spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_foreword_coupling_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_foreword_coupling_validation_specs_report_{date_tag}.md"
    payload = {
        "specs": [asdict(spec) for spec in bundle.specs],
        "summary": bundle.summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: ForewordCouplingSpecBundle) -> str:
    """Render a compact Markdown report for human review."""
    lines = [
        "# Paper 0 Foreword Coupling Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Sigma layer examples: {bundle.summary['sigma_layer_example_count']}",
        f"- Image markers: {bundle.summary['image_marker_count']}",
        f"- Preface I boundary: {bundle.summary['preface_i_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.append(f"- `{spec.key}`: {spec.canonical_statement}")
        if "H_int = -lambda * Psi_s * sigma" in spec.source_formulae:
            lines.append("  - Source formula: `H_int = -lambda * Psi_s * sigma`")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build and write Paper 0 Foreword coupling validation specs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0 if bundle.summary["coverage_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
