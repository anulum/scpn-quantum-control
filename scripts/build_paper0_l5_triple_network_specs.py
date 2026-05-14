#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Layer 5 Triple Network spec builder
"""Promote Paper 0 Layer 5 Triple Network records into validation specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6485, 6504))
STRUCTURAL_SOURCE_LEDGER_IDS = (
    "P0R06485",
    "P0R06488",
    "P0R06491",
    "P0R06492",
    "P0R06493",
    "P0R06496",
    "P0R06499",
    "P0R06501",
)
CAPTION_SOURCE_LEDGER_IDS = ("P0R06489", "P0R06497", "P0R06500", "P0R06502")

MECHANISMS_BY_SPEC = {
    "l5_triple_network.anatomical_mapping": (
        (),
        (),
        (
            "DMN supports internally focused self-referential processing and narrative self",
            "CEN supports externally focused working memory, planning, and goal-directed control",
            "SN is anchored in anterior insula and dorsal anterior cingulate cortex",
        ),
    ),
    "l5_triple_network.salience_switching": (
        (),
        (),
        (
            "DMN and CEN are typically anti-correlated",
            "SN detects salient prediction errors",
            "SN switches dominance between DMN and CEN when salience demands attention",
        ),
    ),
    "l5_triple_network.interoceptive_inference": (
        ("P0R06502:salience_precision_error",),
        (
            "salience approximates precision x abs(prediction_error)",
            "salience threshold crossing triggers CEN engagement",
            "below salience threshold supports DMN dominance",
        ),
        (
            "anterior insula receives interoceptive inputs and generative predictions",
            "anterior insula registers mismatch between predicted and actual physiological state",
            "interoceptive prediction error drives salience-network switching",
        ),
    ),
    "l5_triple_network.homeostatic_qualia_boundary": (
        (),
        (),
        (
            "emotional qualia are bounded to subjective interoceptive inference in insula",
            "psychosomatic harmonics are insula-body feedback loops via autonomic and neuroendocrine pathways",
            "organismal self-organisation is bounded to homeostasis and low-surprise body regulation",
        ),
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "l5_triple_network.anatomical_mapping": {
        "validation_protocol": "paper0.l5_triple_network.anatomical_mapping",
        "canonical_statement": (
            "The source maps Layer 5 self dynamics onto the DMN, CEN, and SN as "
            "large-scale brain-network substrates."
        ),
        "variables": ("DMN", "CEN", "SN", "mPFC", "PCC", "DLPFC", "PPC", "AI", "dACC"),
        "validation_targets": (
            "preserve all three network roles",
            "preserve anatomical hub labels",
            "reject missing-network mappings",
        ),
        "null_controls": (
            "missing-DMN mapping control must be rejected",
            "missing-CEN mapping control must be rejected",
            "missing-SN mapping control must be rejected",
        ),
    },
    "l5_triple_network.salience_switching": {
        "validation_protocol": "paper0.l5_triple_network.salience_switching",
        "canonical_statement": (
            "The source describes the SN as the switch between internally focused DMN "
            "dominance and externally focused CEN dominance under salient prediction error."
        ),
        "variables": ("DMN_activity", "CEN_activity", "SN_gate", "salience"),
        "validation_targets": (
            "preserve DMN-CEN anti-correlation",
            "preserve SN switching role",
            "preserve salient prediction-error trigger",
        ),
        "null_controls": (
            "missing-anti-correlation control must be rejected",
            "missing-salience-switch control must be rejected",
            "missing-prediction-error trigger control must be rejected",
        ),
    },
    "l5_triple_network.interoceptive_inference": {
        "validation_protocol": "paper0.l5_triple_network.interoceptive_inference",
        "canonical_statement": (
            "The anterior insula is promoted as a precision-weighted interoceptive "
            "inference hub where salience is bounded to precision times prediction error."
        ),
        "variables": ("precision", "prediction_error", "salience", "threshold", "AI"),
        "validation_targets": (
            "preserve salience formula",
            "preserve threshold-triggered mode switching",
            "reject negative precision and shape mismatch",
        ),
        "null_controls": (
            "negative-precision control must be rejected",
            "shape-mismatch control must be rejected",
            "missing-threshold-switch control must be rejected",
        ),
    },
    "l5_triple_network.homeostatic_qualia_boundary": {
        "validation_protocol": "paper0.l5_triple_network.homeostatic_qualia_boundary",
        "canonical_statement": (
            "The source bounds emotional qualia and psychosomatic harmonics to "
            "interoceptive inference and body-regulation feedback loops rather than "
            "empirical confirmation."
        ),
        "variables": ("insula", "body_state", "autonomic_pathways", "homeostasis"),
        "validation_targets": (
            "preserve interoceptive-qualia boundary",
            "preserve autonomic and neuroendocrine feedback channels",
            "reject simulator output as empirical neurophysiology",
        ),
        "null_controls": (
            "unsupported-empirical-mapping control must be rejected",
            "missing-homeostasis control must be rejected",
            "missing-body-feedback control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class L5TripleNetworkValidationSpec:
    """Validation spec promoted from Paper 0 Layer 5 Triple Network records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_formulae: tuple[str, ...]
    source_mechanisms: tuple[str, ...]
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
class L5TripleNetworkValidationSpecBundle:
    """Layer 5 Triple Network validation specs plus coverage summary."""

    specs: tuple[L5TripleNetworkValidationSpec, ...]
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


def build_l5_triple_network_specs(
    source_records: list[dict[str, Any]],
) -> L5TripleNetworkValidationSpecBundle:
    """Build source-covered specs for the Layer 5 Triple Network block."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[L5TripleNetworkValidationSpec] = []
    for key, (equation_ids, formulae, mechanisms) in MECHANISMS_BY_SPEC.items():
        metadata = SPEC_METADATA[key]
        specs.append(
            L5TripleNetworkValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(equation_ids),
                source_formulae=tuple(formulae),
                source_mechanisms=tuple(mechanisms),
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
                    "source-bounded Layer 5 Triple Network simulator contract; "
                    "not empirical evidence"
                ),
                implementation_status="implemented",
                domain_review_status="source_promoted_requires_empirical_review",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary: dict[str, Any] = {
        "title": "Paper 0 Layer 5 Triple Network Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(anchors),
        "coverage_match": len(anchors) == len(SOURCE_LEDGER_IDS),
        "structural_source_ledger_ids": list(STRUCTURAL_SOURCE_LEDGER_IDS),
        "caption_source_ledger_ids": list(CAPTION_SOURCE_LEDGER_IDS),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "hardware_status": "simulator_only_no_provider_submission",
        "claim_boundary": "source-bounded Layer 5 Triple Network simulator contract; not empirical evidence",
    }
    return L5TripleNetworkValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: L5TripleNetworkValidationSpecBundle) -> str:
    """Render a compact Markdown report for human audit."""
    lines = [
        "# Paper 0 Layer 5 Triple Network Specs",
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
                "Mechanisms:",
                *[f"- {mechanism}" for mechanism in spec.source_mechanisms],
                "",
                "Null controls:",
                *[f"- {control}" for control in spec.null_controls],
            ]
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    *,
    bundle: L5TripleNetworkValidationSpecBundle,
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
    """Build the default Layer 5 Triple Network validation-spec artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_l5_triple_network_validation_specs_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_l5_triple_network_validation_specs_report_2026-05-13.md",
    )
    args = parser.parse_args()

    bundle = build_l5_triple_network_specs(load_jsonl(args.ledger))
    write_outputs(bundle=bundle, output_path=args.output, report_path=args.report)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
