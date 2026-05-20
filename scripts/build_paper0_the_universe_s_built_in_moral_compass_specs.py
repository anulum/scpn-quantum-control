#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Universe's Built-in Moral Compass spec builder
"""Promote Paper 0 The Universe's Built-in Moral Compass records."""

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
    "P0R03715",
    "P0R03716",
    "P0R03717",
    "P0R03718",
    "P0R03719",
    "P0R03720",
    "P0R03721",
    "P0R03722",
)
CLAIM_BOUNDARY = "source-bounded the universe s built in moral compass source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_universe_s_built_in_moral_compass.the_universe_s_built_in_moral_compass": {
        "context_id": "the_universe_s_built_in_moral_compass",
        "validation_protocol": "paper0.the_universe_s_built_in_moral_compass.the_universe_s_built_in_moral_compass",
        "canonical_statement": "The source-bounded component 'The Universe's Built-in Moral Compass' preserves Paper 0 records P0R03715-P0R03717 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03715:the_universe_s_built_in_moral_compass",
            "P0R03716:the_universe_s_built_in_moral_compass",
            "P0R03717:the_universe_s_built_in_moral_compass",
        ),
        "source_formulae": (
            "P0R03715: The Universe's Built-in Moral Compass",
            "P0R03716: Why does the universe seem to evolve toward more complexity, more structure, and more coherence? Is it just a random accident, or is there a deeper principle at work? This framework proposes that the universe has a built-in, goal-seeking directive: to maximize what we call Sustainable Ethical Coherence (SEC). This is not a human moral code, but a fundamental law of physics. SEC is a measure of a system's harmony, complexity, and richness of experience.",
            'P0R03717: Imagine the universe as a vast, self-playing orchestra. Its ultimate goal is to play the most beautiful, complex, and harmonious symphony possible. Every event, from the formation of a star to the evolution of life, is a "choice" made by the universe to improve the music. This goal-seeking process, not a rigid set of rules, is the true origin of what we perceive as purpose and ethics.',
        ),
        "test_protocols": (
            "preserve The Universe's Built-in Moral Compass source-accounting boundary",
        ),
        "null_results": (
            "The Universe's Built-in Moral Compass is not empirical validation evidence",
        ),
        "variables": ("the_universe_s_built_in_moral_compass",),
        "validation_targets": ("preserve records P0R03715-P0R03717",),
        "null_controls": (
            "the_universe_s_built_in_moral_compass must remain source-bounded accounting",
        ),
    },
    "the_universe_s_built_in_moral_compass.the_pull_of_the_future_how_purpose_guides_the_present": {
        "context_id": "the_pull_of_the_future_how_purpose_guides_the_present",
        "validation_protocol": "paper0.the_universe_s_built_in_moral_compass.the_pull_of_the_future_how_purpose_guides_the_present",
        "canonical_statement": "The source-bounded component 'The Pull of the Future: How Purpose Guides the Present' preserves Paper 0 records P0R03718-P0R03721 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03718:the_pull_of_the_future_how_purpose_guides_the_present",
            "P0R03719:the_pull_of_the_future_how_purpose_guides_the_present",
            "P0R03720:the_pull_of_the_future_how_purpose_guides_the_present",
            "P0R03721:the_pull_of_the_future_how_purpose_guides_the_present",
        ),
        "source_formulae": (
            "P0R03718: The Pull of the Future: How Purpose Guides the Present",
            "P0R03719: How does this cosmic goal actually influence events? The mechanism is through Causal Entropic Forces (CEF). This is a subtle but powerful force that arises from the future itself.",
            'P0R03720: Think of the future not as a single path, but as a branching tree of infinite possibilities. Some of these future paths lead to dead ends-disorder, collapse, and silence. Others lead to futures with greater harmony, complexity, and potential-a richer cosmic symphony. These "high-SEC" futures exert a gentle "pull" on the present, like the aroma of a wonderful meal drawing you towards the kitchen.',
            "P0R03721: This pull subtly biases everything that happens, from the collapse of a quantum particle to the grand sweep of evolution. It's a form of retrocausality-the future reaching back to guide the present. Every quantum \"choice\" is a handshake between the possibilities of the past and the pull of the most coherent possible future. This is how the universe's ultimate purpose is woven into the fabric of every moment.",
        ),
        "test_protocols": (
            "preserve The Pull of the Future: How Purpose Guides the Present source-accounting boundary",
        ),
        "null_results": (
            "The Pull of the Future: How Purpose Guides the Present is not empirical validation evidence",
        ),
        "variables": ("the_pull_of_the_future_how_purpose_guides_the_present",),
        "validation_targets": ("preserve records P0R03718-P0R03721",),
        "null_controls": (
            "the_pull_of_the_future_how_purpose_guides_the_present must remain source-bounded accounting",
        ),
    },
    "the_universe_s_built_in_moral_compass.formalisation_of_the_causal_entropic_principle": {
        "context_id": "formalisation_of_the_causal_entropic_principle",
        "validation_protocol": "paper0.the_universe_s_built_in_moral_compass.formalisation_of_the_causal_entropic_principle",
        "canonical_statement": "The source-bounded component 'Formalisation of the Causal Entropic Principle:' preserves Paper 0 records P0R03722-P0R03722 without empirical validation claims.",
        "source_equation_ids": ("P0R03722:formalisation_of_the_causal_entropic_principle",),
        "source_formulae": ("P0R03722: Formalisation of the Causal Entropic Principle:",),
        "test_protocols": (
            "preserve Formalisation of the Causal Entropic Principle: source-accounting boundary",
        ),
        "null_results": (
            "Formalisation of the Causal Entropic Principle: is not empirical validation evidence",
        ),
        "variables": ("formalisation_of_the_causal_entropic_principle",),
        "validation_targets": ("preserve records P0R03722-P0R03722",),
        "null_controls": (
            "formalisation_of_the_causal_entropic_principle must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheUniverseSBuiltInMoralCompassSpec:
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
class TheUniverseSBuiltInMoralCompassSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheUniverseSBuiltInMoralCompassSpec, ...]
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


def build_the_universe_s_built_in_moral_compass_specs(
    source_records: list[dict[str, Any]],
) -> TheUniverseSBuiltInMoralCompassSpecBundle:
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

    specs: list[TheUniverseSBuiltInMoralCompassSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheUniverseSBuiltInMoralCompassSpec(
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
        "title": "Paper 0 " + "The Universe's Built-in Moral Compass" + " Specs",
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
        "next_source_boundary": "P0R03723",
    }
    return TheUniverseSBuiltInMoralCompassSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheUniverseSBuiltInMoralCompassSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_universe_s_built_in_moral_compass_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheUniverseSBuiltInMoralCompassSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Universe's Built-in Moral Compass" + " Specs",
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
    bundle: TheUniverseSBuiltInMoralCompassSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_universe_s_built_in_moral_compass_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_the_universe_s_built_in_moral_compass_validation_specs_{date_tag}.md"
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
