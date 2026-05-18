#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 experimental signatures spec builder
"""Promote Paper 0 experimental-signatures search-strategy records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1647, 1655))
CLAIM_BOUNDARY = (
    "source-bounded experimental-signatures search-strategy bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "experimental_signatures_search_strategies.falsifiability_frame": {
        "context_id": "falsifiability_frame",
        "validation_protocol": "paper0.experimental_signatures_search_strategies.falsifiability_frame",
        "canonical_statement": (
            "The source frames the massive infoton and Psi-Higgs predictions as falsifiable through collider and cosmological searches."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:falsifiability_frame" for n in range(1647, 1649)
        ),
        "source_formulae": (
            "Experimental Signatures and Search Strategies",
            "two new particles are the massive vector infoton and massive scalar Psi-Higgs",
            "direct detection may depend on masses and coupling strengths",
            "search strategies are proposed for particle colliders and cosmological observations",
        ),
        "test_protocols": ("preserve falsifiability framing boundary",),
        "null_results": ("testability framing is not completed detection evidence",),
        "variables": ("infoton", "Psi_Higgs", "mass", "coupling_strength", "search_strategy"),
        "validation_targets": (
            "preserve two-particle search framing",
            "preserve direct-detection caveat",
        ),
        "null_controls": (
            "falsifiable-search framing must not be reported as current validation",
        ),
    },
    "experimental_signatures_search_strategies.collider_channel": {
        "context_id": "collider_channel",
        "validation_protocol": "paper0.experimental_signatures_search_strategies.collider_channel",
        "canonical_statement": (
            "The source identifies exotic Standard Model Higgs decays at the LHC as a proposed Psi-Higgs search channel."
        ),
        "source_equation_ids": tuple(f"P0R{n:05d}:collider_channel" for n in range(1649, 1652)),
        "source_formulae": (
            "Particle Collider Searches",
            "Standard Model Higgs could decay into a pair of Psi-Higgs bosons",
            "h_SM -> h_Psi h_Psi",
            "CMS and ATLAS could search for excess events with invariant-mass signatures",
            "null results can constrain Psi-Higgs mass and coupling parameter space",
        ),
        "test_protocols": ("preserve LHC exotic-Higgs-decay search boundary",),
        "null_results": ("LHC search channel is not an observed excess",),
        "variables": ("h_SM", "h_Psi", "CMS", "ATLAS", "invariant_mass", "coupling"),
        "validation_targets": (
            "preserve exotic-decay channel",
            "preserve null-result constraint role",
        ),
        "null_controls": ("collider-channel roadmap must not imply observed CMS or ATLAS signal",),
    },
    "experimental_signatures_search_strategies.cosmological_channel": {
        "context_id": "cosmological_channel",
        "validation_protocol": "paper0.experimental_signatures_search_strategies.cosmological_channel",
        "canonical_statement": (
            "The source identifies ultralight-field black-hole superradiance and continuous gravitational waves as an independent search strategy."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:cosmological_channel" for n in range(1652, 1654)
        ),
        "source_formulae": (
            "Cosmological and Astrophysical Signatures",
            "ultralight Psi-Higgs or Psi-field can be framed as dark-matter candidates",
            "scalar boson clouds could form around spinning black holes by superradiance",
            "annihilation could produce nearly monochromatic continuous gravitational waves",
            "LISA, Einstein Telescope, and Cosmic Explorer are future or next-generation search instruments",
        ),
        "test_protocols": ("preserve cosmological continuous-wave search boundary",),
        "null_results": ("gravitational-wave search channel is not a detected boson cloud",),
        "variables": (
            "Psi_field",
            "Psi_Higgs",
            "superradiance",
            "black_hole",
            "LISA",
            "Einstein_Telescope",
        ),
        "validation_targets": (
            "preserve boson-cloud search channel",
            "preserve detector-target framing",
        ),
        "null_controls": (
            "future detector sensitivity must not be stated as an existing detection",
        ),
    },
    "experimental_signatures_search_strategies.complementary_test_boundary": {
        "context_id": "complementary_test_boundary",
        "validation_protocol": "paper0.experimental_signatures_search_strategies.complementary_test_boundary",
        "canonical_statement": (
            "The source summarises collider and cosmological channels as complementary routes for converting abstract predictions into falsifiable hypotheses."
        ),
        "source_equation_ids": ("P0R01654:complementary_test_boundary",),
        "source_formulae": (
            "collider and cosmological strategies are independent and complementary",
            "strategies test fundamental physical predictions of SCPN",
            "abstract claims are transformed into concrete falsifiable hypotheses",
        ),
        "test_protocols": ("preserve complementary-test boundary",),
        "null_results": ("complementary search plan is not experimental confirmation",),
        "variables": ("collider_search", "cosmological_search", "falsifiable_hypothesis"),
        "validation_targets": (
            "preserve complementary-channel summary",
            "preserve hypothesis-only boundary",
        ),
        "null_controls": ("falsifiable hypothesis language must remain non-evidential",),
    },
}


@dataclass(frozen=True, slots=True)
class ExperimentalSignaturesSearchStrategiesSpec:
    """Experimental-signatures search-strategy spec promoted from Paper 0 records."""

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
class ExperimentalSignaturesSearchStrategiesSpecBundle:
    """Experimental-signatures search-strategy specs plus source coverage summary."""

    specs: tuple[ExperimentalSignaturesSearchStrategiesSpec, ...]
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


def build_experimental_signatures_search_strategies_specs(
    source_records: list[dict[str, Any]],
) -> ExperimentalSignaturesSearchStrategiesSpecBundle:
    """Build source-covered experimental-signatures search-strategy specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[ExperimentalSignaturesSearchStrategiesSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ExperimentalSignaturesSearchStrategiesSpec(
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
        "title": "Paper 0 Experimental Signatures Search Strategies Specs",
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
        "next_source_boundary": "P0R01655",
    }
    return ExperimentalSignaturesSearchStrategiesSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ExperimentalSignaturesSearchStrategiesSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_experimental_signatures_search_strategies_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ExperimentalSignaturesSearchStrategiesSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 Experimental Signatures Search Strategies Specs",
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
    bundle: ExperimentalSignaturesSearchStrategiesSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_experimental_signatures_search_strategies_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_experimental_signatures_search_strategies_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    bundle = build_from_ledger(args.ledger)
    outputs = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
