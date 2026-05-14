#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Layer 5 TDA/neurophenomenology spec builder
"""Promote Paper 0 Layer 5 TDA/neurophenomenology records into validation specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6504, 6519))
STRUCTURAL_SOURCE_LEDGER_IDS = (
    "P0R06504",
    "P0R06508",
    "P0R06510",
    "P0R06513",
    "P0R06515",
    "P0R06517",
)
CAPTION_SOURCE_LEDGER_IDS = ("P0R06509", "P0R06511", "P0R06514")
PROTOCOL_STEP_LEDGER_IDS = ("P0R06518",)
PROTOCOL_STEPS = (
    "record high-density EEG while eliciting varied subjective experiences",
    "conduct structured neurophenomenological interview immediately after task",
    "score reports for richness, intensity, and structural complexity",
    "compute Betti numbers of neural manifold M with TDA",
    "test systematic correlation between reports and topological features",
)

MECHANISMS_BY_SPEC = {
    "l5_tda_neurophenomenology.geometric_qualia_hypothesis": (
        (),
        (),
        (
            "quality of consciousness is hypothesised to be determined by geometry of the consciousness manifold M",
            "neurophenomenology supplies structured first-person experience vectors",
            "TDA supplies neural-state manifold topology from EEG or fMRI",
        ),
        (),
    ),
    "l5_tda_neurophenomenology.neurophenomenology_protocol": (
        (),
        (),
        (
            "micro-phenomenology moves beyond simple self-report",
            "first-person reports must be structured, reliable, and replicable",
            "reports are paired with third-person neurophysiological data",
        ),
        PROTOCOL_STEPS,
    ),
    "l5_tda_neurophenomenology.persistent_homology_features": (
        ("P0R06509:sum_betti_persistence", "P0R06511:barcode_lifetime"),
        (
            "persistence distance from diagonal contributes to sum_k b_k(M)",
            "persistent bars represent feature lifetimes across filtration",
            "Betti features include b0 connected components, b1 loops, and bk higher-dimensional voids",
        ),
        (
            "persistent homology quantifies topology of reconstructed neural state space",
            "longer persistence indicates more persistent topological structure",
        ),
        (),
    ),
    "l5_tda_neurophenomenology.qualia_richness_regression": (
        ("P0R06512:qualia_richness", "P0R06514:regression_test"),
        (
            "Qualia Richness proportional_to Vol(M) x sum_k b_k(M)",
            "regress richness, intensity, and structure against Vol(M) x sum_k b_k(M)",
        ),
        (
            "geometric qualia hypothesis is tested by correlation or regression",
            "richer reported experiences should correspond to higher summed Betti topology",
        ),
        (),
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "l5_tda_neurophenomenology.geometric_qualia_hypothesis": {
        "validation_protocol": "paper0.l5_tda_neurophenomenology.geometric_qualia_hypothesis",
        "canonical_statement": (
            "The source frames Geometric Qualia as the hypothesis that consciousness "
            "quality is determined by the geometry of a consciousness manifold M."
        ),
        "variables": ("qualia", "M", "neural_data", "first_person_vectors"),
        "validation_targets": (
            "preserve manifold-geometry hypothesis",
            "preserve first-person and third-person bridge",
            "reject hypothesis wording as empirical proof",
        ),
        "null_controls": (
            "missing-manifold-geometry control must be rejected",
            "missing-neurophenomenology control must be rejected",
            "unsupported-empirical-qualia control must be rejected",
        ),
    },
    "l5_tda_neurophenomenology.neurophenomenology_protocol": {
        "validation_protocol": "paper0.l5_tda_neurophenomenology.neurophenomenology_protocol",
        "canonical_statement": (
            "The protocol requires high-density neurophysiology, immediate structured "
            "interview, report scoring, TDA feature extraction, and correlation testing."
        ),
        "variables": ("EEG", "interview", "richness", "intensity", "structure"),
        "validation_targets": (
            "preserve all five protocol steps",
            "require immediate structured report collection",
            "reject incomplete protocol catalogues",
        ),
        "null_controls": (
            "missing-recording control must be rejected",
            "missing-report-scoring control must be rejected",
            "missing-correlation-test control must be rejected",
        ),
    },
    "l5_tda_neurophenomenology.persistent_homology_features": {
        "validation_protocol": "paper0.l5_tda_neurophenomenology.persistent_homology_features",
        "canonical_statement": (
            "The TDA side is bounded to persistence diagrams, barcodes, Betti numbers, "
            "and feature lifetimes extracted from neural state-space reconstructions."
        ),
        "variables": ("persistence_pairs", "birth", "death", "b0", "b1", "bk"),
        "validation_targets": (
            "preserve persistence diagram and barcode roles",
            "preserve Betti-number feature taxonomy",
            "reject invalid birth/death intervals",
        ),
        "null_controls": (
            "invalid-birth-death control must be rejected",
            "missing-Betti-features control must be rejected",
            "missing-persistence-lifetime control must be rejected",
        ),
    },
    "l5_tda_neurophenomenology.qualia_richness_regression": {
        "validation_protocol": "paper0.l5_tda_neurophenomenology.qualia_richness_regression",
        "canonical_statement": (
            "The Geometric Qualia test is promoted as a source-bounded regression or "
            "correlation target between scored reports and Vol(M) times summed Betti numbers."
        ),
        "variables": ("Vol_M", "sum_betti", "richness", "intensity", "structure"),
        "validation_targets": (
            "preserve Qualia Richness proportionality",
            "preserve correlation/regression target",
            "reject simulator correlation as empirical consciousness evidence",
        ),
        "null_controls": (
            "constant-report control must be rejected",
            "zero-volume control must be rejected",
            "unsupported-empirical-qualia control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class L5TDANeurophenomenologyValidationSpec:
    """Validation spec promoted from Paper 0 Layer 5 TDA records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_formulae: tuple[str, ...]
    source_mechanisms: tuple[str, ...]
    source_protocol_steps: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    anchor_math_ids: tuple[str, ...]
    image_ledger_ids: tuple[str, ...]
    caption_ledger_ids: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class L5TDANeurophenomenologyValidationSpecBundle:
    """Layer 5 TDA/neurophenomenology validation specs plus coverage summary."""

    specs: tuple[L5TDANeurophenomenologyValidationSpec, ...]
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


def build_l5_tda_neurophenomenology_specs(
    source_records: list[dict[str, Any]],
) -> L5TDANeurophenomenologyValidationSpecBundle:
    """Build source-covered specs for the Layer 5 TDA/neurophenomenology block."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[L5TDANeurophenomenologyValidationSpec] = []
    for key, (equation_ids, formulae, mechanisms, protocol_steps) in MECHANISMS_BY_SPEC.items():
        metadata = SPEC_METADATA[key]
        specs.append(
            L5TDANeurophenomenologyValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(equation_ids),
                source_formulae=tuple(formulae),
                source_mechanisms=tuple(mechanisms),
                source_protocol_steps=tuple(protocol_steps),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(record["source_record_id"]) for record in anchors),
                source_block_indices=tuple(
                    int(record["source_block_index"]) for record in anchors
                ),
                anchor_math_ids=tuple(
                    math_id for record in anchors for math_id in tuple(record.get("math_ids", ()))
                ),
                image_ledger_ids=tuple(
                    record["ledger_id"]
                    for record in anchors
                    if record["ledger_id"] in STRUCTURAL_SOURCE_LEDGER_IDS
                    and record.get("image_ids")
                ),
                caption_ledger_ids=CAPTION_SOURCE_LEDGER_IDS,
                variables=tuple(metadata["variables"]),
                validation_targets=tuple(metadata["validation_targets"]),
                executable_validation_targets=tuple(metadata["validation_targets"]),
                null_controls=tuple(metadata["null_controls"]),
                claim_boundary=(
                    "source-bounded Layer 5 TDA/neurophenomenology simulator contract; "
                    "not empirical evidence"
                ),
                implementation_status="implemented",
                domain_review_status="source_promoted_requires_empirical_review",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary: dict[str, Any] = {
        "title": "Paper 0 Layer 5 TDA Neurophenomenology Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(anchors),
        "coverage_match": len(anchors) == len(SOURCE_LEDGER_IDS),
        "structural_source_ledger_ids": list(STRUCTURAL_SOURCE_LEDGER_IDS),
        "caption_source_ledger_ids": list(CAPTION_SOURCE_LEDGER_IDS),
        "protocol_step_ledger_ids": list(PROTOCOL_STEP_LEDGER_IDS),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "hardware_status": "simulator_only_no_provider_submission",
        "claim_boundary": "source-bounded Layer 5 TDA/neurophenomenology simulator contract; not empirical evidence",
    }
    return L5TDANeurophenomenologyValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: L5TDANeurophenomenologyValidationSpecBundle) -> str:
    """Render a compact Markdown report for human audit."""
    lines = [
        "# Paper 0 Layer 5 TDA Neurophenomenology Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - {bundle.summary['source_ledger_span'][1]}",
        f"- Source records consumed: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                "",
                f"### {spec.key}",
                "",
                spec.canonical_statement,
                "",
                "Formulae:",
                *[f"- {formula}" for formula in spec.source_formulae],
                "",
                "Protocol:",
                *[f"- {step}" for step in spec.source_protocol_steps],
                "",
                "Null controls:",
                *[f"- {control}" for control in spec.null_controls],
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    bundle: L5TDANeurophenomenologyValidationSpecBundle,
    output_path: Path,
    report_path: Path,
) -> None:
    """Write JSON and Markdown artefacts."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "summary": bundle.summary,
                "specs": [asdict(spec) for spec in bundle.specs],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")


def main() -> int:
    """Build the default Layer 5 TDA/neurophenomenology validation-spec artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_l5_tda_neurophenomenology_validation_specs_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_l5_tda_neurophenomenology_validation_specs_report_2026-05-13.md",
    )
    args = parser.parse_args()

    bundle = build_l5_tda_neurophenomenology_specs(load_jsonl(args.ledger))
    write_outputs(bundle=bundle, output_path=args.output, report_path=args.report)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
