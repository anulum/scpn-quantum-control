#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 collective niche construction spec builder
"""Promote Paper 0 collective niche construction records into validation specs."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6519, 6530))
STRUCTURAL_SOURCE_LEDGER_IDS = ("P0R06519", "P0R06521", "P0R06528")
CAPTION_SOURCE_LEDGER_IDS = ("P0R06529",)

MECHANISMS_BY_SPEC = {
    "collective_niche.shared_generative_model": (
        (),
        (),
        (
            "culture is framed as agents converging on a shared generative model",
            "shared beliefs, values, language, and norms make others predictable",
            "communication, imitation, and shared artefacts actively achieve convergence",
        ),
    ),
    "collective_niche.noosphere_entrainment": (
        (),
        (),
        (
            "cultural attractors and memes are components of the shared generative model",
            "institutions, rituals, language, and art entrain individual generative models",
            "collective free-energy reduction is linked to Noosphere emergence",
        ),
    ),
    "collective_niche.biosphere_feedback_loop": (
        (),
        (),
        (
            "collective niche construction modifies the biosphere to fit shared predictions",
            "modified environment supplies training data to the next generation",
            "collective mind and planet form a co-evolutionary feedback loop",
        ),
    ),
    "collective_niche.gaian_synchrony_boundary": (
        ("P0R06527:recursive_loop", "P0R06529:agent_planet_loop"),
        (
            "collective mind shapes planet and earth shapes collective mind",
            "planetary-scale active inference seeks mutual predictability, stability, and coherence",
        ),
        (
            "Gaian Synchrony is framed as emergent planetary-scale active inference",
            "trajectory is guided by FEP toward lower surprise across evolutionary timescales",
            "biosphere-noosphere co-evolution is retained as a source-bounded hypothesis",
        ),
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "collective_niche.shared_generative_model": {
        "validation_protocol": "paper0.collective_niche.shared_generative_model",
        "canonical_statement": (
            "Culture is promoted as agents converging on a shared generative model "
            "through beliefs, values, language, norms, communication, imitation, and artefacts."
        ),
        "variables": (
            "beliefs",
            "values",
            "language",
            "norms",
            "communication",
            "imitation",
            "artefacts",
        ),
        "validation_targets": (
            "preserve all shared-model channels",
            "preserve active convergence mechanisms",
            "reject missing artefact-mediated convergence",
        ),
        "null_controls": (
            "missing-artefacts control must be rejected",
            "missing-language control must be rejected",
            "missing-norms control must be rejected",
        ),
    },
    "collective_niche.noosphere_entrainment": {
        "validation_protocol": "paper0.collective_niche.noosphere_entrainment",
        "canonical_statement": (
            "Noosphere emergence is source-bounded to cultural attractors and memes "
            "entraining individual generative models through social mechanisms."
        ),
        "variables": ("cultural_attractors", "memes", "institutions", "rituals", "art"),
        "validation_targets": (
            "preserve attractor and meme wording",
            "preserve institutions, rituals, language, and art entrainment channels",
            "reject entrainment without collective free-energy boundary",
        ),
        "null_controls": (
            "missing-institutions control must be rejected",
            "missing-rituals control must be rejected",
            "missing-art control must be rejected",
        ),
    },
    "collective_niche.biosphere_feedback_loop": {
        "validation_protocol": "paper0.collective_niche.biosphere_feedback_loop",
        "canonical_statement": (
            "Collective niche construction is promoted as bidirectional coupling where "
            "collectives modify the environment and the modified environment trains later models."
        ),
        "variables": ("collective_action", "environment", "training_data", "next_generation"),
        "validation_targets": (
            "preserve collective-to-environment action",
            "preserve environment-to-next-generation training",
            "reject unidirectional feedback-only accounts",
        ),
        "null_controls": (
            "shape-mismatch control must be rejected",
            "missing-environment-training control must be rejected",
            "missing-collective-action control must be rejected",
        ),
    },
    "collective_niche.gaian_synchrony_boundary": {
        "validation_protocol": "paper0.collective_niche.gaian_synchrony_boundary",
        "canonical_statement": (
            "Gaian Synchrony is retained as a source-bounded planetary active-inference "
            "hypothesis, not empirical evidence for biosphere-noosphere consciousness."
        ),
        "variables": ("biosphere", "noosphere", "predictability", "stability", "coherence"),
        "validation_targets": (
            "preserve recursive planet-mind loop",
            "preserve mutual predictability/stability/coherence target",
            "reject simulator output as planetary empirical evidence",
        ),
        "null_controls": (
            "unsupported-planetary-evidence control must be rejected",
            "missing-mutual-predictability control must be rejected",
            "missing-speculative-boundary control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class CollectiveNicheConstructionValidationSpec:
    """Validation spec promoted from Paper 0 collective niche construction records."""

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
class CollectiveNicheConstructionValidationSpecBundle:
    """Collective niche construction validation specs plus coverage summary."""

    specs: tuple[CollectiveNicheConstructionValidationSpec, ...]
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


def build_collective_niche_construction_specs(
    source_records: list[dict[str, Any]],
) -> CollectiveNicheConstructionValidationSpecBundle:
    """Build source-covered specs for the collective niche construction block."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[CollectiveNicheConstructionValidationSpec] = []
    for key, (equation_ids, formulae, mechanisms) in MECHANISMS_BY_SPEC.items():
        metadata = SPEC_METADATA[key]
        specs.append(
            CollectiveNicheConstructionValidationSpec(
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
                    "source-bounded collective niche construction simulator contract; "
                    "not empirical evidence"
                ),
                implementation_status="implemented",
                domain_review_status="source_promoted_requires_empirical_review",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary: dict[str, Any] = {
        "title": "Paper 0 Collective Niche Construction Specs",
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(anchors),
        "coverage_match": len(anchors) == len(SOURCE_LEDGER_IDS),
        "structural_source_ledger_ids": list(STRUCTURAL_SOURCE_LEDGER_IDS),
        "caption_source_ledger_ids": list(CAPTION_SOURCE_LEDGER_IDS),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "hardware_status": "simulator_only_no_provider_submission",
        "claim_boundary": "source-bounded collective niche construction simulator contract; not empirical evidence",
    }
    return CollectiveNicheConstructionValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: CollectiveNicheConstructionValidationSpecBundle) -> str:
    """Render a compact Markdown report for human audit."""
    lines = [
        "# Paper 0 Collective Niche Construction Specs",
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
    bundle: CollectiveNicheConstructionValidationSpecBundle,
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
    """Build the default collective niche construction validation-spec artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_collective_niche_construction_validation_specs_2026-05-13.json",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR
        / "paper0_collective_niche_construction_validation_specs_report_2026-05-13.md",
    )
    args = parser.parse_args()

    bundle = build_collective_niche_construction_specs(load_jsonl(args.ledger))
    write_outputs(bundle=bundle, output_path=args.output, report_path=args.report)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
