#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 sigma is the Q-ball Soliton: spec builder
"""Promote Paper 0 sigma is the Q-ball Soliton: records."""

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
    "P0R01812",
    "P0R01813",
    "P0R01814",
    "P0R01815",
    "P0R01816",
    "P0R01817",
    "P0R01818",
    "P0R01819",
)
CLAIM_BOUNDARY = (
    "source-bounded sigma is the q ball soliton source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "sigma_is_the_q_ball_soliton.sigma_is_the_q_ball_soliton": {
        "context_id": "sigma_is_the_q_ball_soliton",
        "validation_protocol": "paper0.sigma_is_the_q_ball_soliton.sigma_is_the_q_ball_soliton",
        "canonical_statement": "The source-bounded component 'sigma is the Q-ball Soliton:' preserves Paper 0 records P0R01812-P0R01813 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01812:sigma_is_the_q_ball_soliton",
            "P0R01813:sigma_is_the_q_ball_soliton",
        ),
        "source_formulae": (
            "P0R01812: sigma is the Q-ball Soliton:",
            "P0R01813: The collective state variable sigma for a Layer 5 Self is explicitly the coherent field configuration of the Q-ball, O.",
        ),
        "test_protocols": ("preserve sigma is the Q-ball Soliton: source-accounting boundary",),
        "null_results": ("sigma is the Q-ball Soliton: is not empirical validation evidence",),
        "variables": ("sigma_is_the_q_ball_soliton",),
        "validation_targets": ("preserve records P0R01812-P0R01813",),
        "null_controls": ("sigma_is_the_q_ball_soliton must remain source-bounded accounting",),
    },
    "sigma_is_the_q_ball_soliton.why_the_q_ball_is_the_perfect_sigma": {
        "context_id": "why_the_q_ball_is_the_perfect_sigma",
        "validation_protocol": "paper0.sigma_is_the_q_ball_soliton.why_the_q_ball_is_the_perfect_sigma",
        "canonical_statement": "The source-bounded component 'Why the Q-ball is the Perfect sigma:' preserves Paper 0 records P0R01814-P0R01815 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01814:why_the_q_ball_is_the_perfect_sigma",
            "P0R01815:why_the_q_ball_is_the_perfect_sigma",
        ),
        "source_formulae": (
            "P0R01814: Why the Q-ball is the Perfect sigma:",
            "P0R01815: The entire derivation explains why this is the correct choice. A viable sigma must be:",
        ),
        "test_protocols": (
            "preserve Why the Q-ball is the Perfect sigma: source-accounting boundary",
        ),
        "null_results": (
            "Why the Q-ball is the Perfect sigma: is not empirical validation evidence",
        ),
        "variables": ("why_the_q_ball_is_the_perfect_sigma",),
        "validation_targets": ("preserve records P0R01814-P0R01815",),
        "null_controls": (
            "why_the_q_ball_is_the_perfect_sigma must remain source-bounded accounting",
        ),
    },
    "sigma_is_the_q_ball_soliton.collective": {
        "context_id": "collective",
        "validation_protocol": "paper0.sigma_is_the_q_ball_soliton.collective",
        "canonical_statement": "The source-bounded component 'Collective:' preserves Paper 0 records P0R01816-P0R01817 without empirical validation claims.",
        "source_equation_ids": ("P0R01816:collective", "P0R01817:collective"),
        "source_formulae": (
            "P0R01816: Collective:",
            "P0R01817: The soliton is a coherent state of countless Psi-field quanta.",
        ),
        "test_protocols": ("preserve Collective: source-accounting boundary",),
        "null_results": ("Collective: is not empirical validation evidence",),
        "variables": ("collective",),
        "validation_targets": ("preserve records P0R01816-P0R01817",),
        "null_controls": ("collective must remain source-bounded accounting",),
    },
    "sigma_is_the_q_ball_soliton.stable": {
        "context_id": "stable",
        "validation_protocol": "paper0.sigma_is_the_q_ball_soliton.stable",
        "canonical_statement": "The source-bounded component 'Stable:' preserves Paper 0 records P0R01818-P0R01819 without empirical validation claims.",
        "source_equation_ids": ("P0R01818:stable", "P0R01819:stable"),
        "source_formulae": (
            "P0R01818: Stable:",
            "P0R01819: Its existence is guaranteed by charge conservation.",
        ),
        "test_protocols": ("preserve Stable: source-accounting boundary",),
        "null_results": ("Stable: is not empirical validation evidence",),
        "variables": ("stable",),
        "validation_targets": ("preserve records P0R01818-P0R01819",),
        "null_controls": ("stable must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class SigmaIsTheQBallSolitonSpec:
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
class SigmaIsTheQBallSolitonSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[SigmaIsTheQBallSolitonSpec, ...]
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


def build_sigma_is_the_q_ball_soliton_specs(
    source_records: list[dict[str, Any]],
) -> SigmaIsTheQBallSolitonSpecBundle:
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

    specs: list[SigmaIsTheQBallSolitonSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            SigmaIsTheQBallSolitonSpec(
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
        "title": "Paper 0 " + "sigma is the Q-ball Soliton:" + " Specs",
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
        "next_source_boundary": "P0R01820",
    }
    return SigmaIsTheQBallSolitonSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> SigmaIsTheQBallSolitonSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_sigma_is_the_q_ball_soliton_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: SigmaIsTheQBallSolitonSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "sigma is the Q-ball Soliton:" + " Specs",
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
    bundle: SigmaIsTheQBallSolitonSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_sigma_is_the_q_ball_soliton_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_sigma_is_the_q_ball_soliton_validation_specs_{date_tag}.md"
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
