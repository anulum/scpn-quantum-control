#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 3.1 Complexity (K) and the Cardinality of the State Space spec builder
"""Promote Paper 0 3.1 Complexity (K) and the Cardinality of the State Space records."""

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
    "P0R03781",
    "P0R03782",
    "P0R03783",
    "P0R03784",
    "P0R03785",
    "P0R03786",
    "P0R03787",
    "P0R03788",
)
CLAIM_BOUNDARY = "source-bounded section 3 1 complexity k and the cardinality of the state space source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_3_1_complexity_k_and_the_cardinality_of_the_state_space.3_1_complexity_k_and_the_cardinality_of_the_state_space": {
        "context_id": "3_1_complexity_k_and_the_cardinality_of_the_state_space",
        "validation_protocol": "paper0.section_3_1_complexity_k_and_the_cardinality_of_the_state_space.3_1_complexity_k_and_the_cardinality_of_the_state_space",
        "canonical_statement": "The source-bounded component '3.1 Complexity (K) and the Cardinality of the State Space' preserves Paper 0 records P0R03781-P0R03788 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03781:3_1_complexity_k_and_the_cardinality_of_the_state_space",
            "P0R03782:3_1_complexity_k_and_the_cardinality_of_the_state_space",
            "P0R03783:3_1_complexity_k_and_the_cardinality_of_the_state_space",
            "P0R03784:3_1_complexity_k_and_the_cardinality_of_the_state_space",
            "P0R03785:3_1_complexity_k_and_the_cardinality_of_the_state_space",
            "P0R03786:3_1_complexity_k_and_the_cardinality_of_the_state_space",
            "P0R03787:3_1_complexity_k_and_the_cardinality_of_the_state_space",
            "P0R03788:3_1_complexity_k_and_the_cardinality_of_the_state_space",
        ),
        "source_formulae": (
            "P0R03781: 3.1 Complexity (K) and the Cardinality of the State Space",
            "P0R03782: The most fundamental constraint on the number of possible future paths is the sheer number of distinct states the system can occupy. A system with more available states has, all else being equal, a combinatorially larger set of possible trajectories it can trace through time. The SCPN framework identifies the measure of a system's Complexity (K) with the principles of Integrated Information Theory (IIT), specifically the quantity . There exists a direct and rigorously proven mathematical relationship between a system's integrated information and the size and differentiation of its state space.",
            "P0R03783: According to IIT, a system's capacity for integrated information is fundamentally bounded by the number of its constituent elements and the richness of their potential interactions. For a system composed of n binary elements, the total state space OmegaS has a cardinality of OmegaS=2n. Theoretical work has established a formal upper bound on the maximum possible integrated information, max, which scales with both n and the size of this state space :",
            "P0R03784: $\\Phi max(S) \\leq (2n - 1)43n2$",
            "P0R03785: This theorem demonstrates that a high value of is impossible without a large number of elements and, consequently, an exponentially large state space. Furthermore, to achieve a high average value of , a system must not only possess a large state space but must also have the capacity actually to visit many of these states and for these visited states to be maximally different from one another-a property known as state differentiation.",
            "P0R03786: This provides the first crucial link in our proof. The number of possible paths of a given length T that a system can trace through its state space scales exponentially with the number of accessible states. For a simplified discrete system, the number of distinct paths is approximately OmegaaccT, where Omegaacc is the set of accessible states. The causal path entropy, being logarithmic in this number, therefore scales directly with the logarithm of the state space size:",
            "P0R03787: $SC \\propto \\ln\\left( \\mid\\Omega acc\\mid T \\right) = Tln \\mid \\Omega acc \\mid$",
            "P0R03788: Since maximising Complexity (K), as measured by , requires maximising the size and differentiation of the state space, it follows that maximising K is a necessary prerequisite for maximising the raw number of potential future histories. It is the system's complexity that provides the vast canvas of possibilities upon which its future can be drawn.",
        ),
        "test_protocols": (
            "preserve 3.1 Complexity (K) and the Cardinality of the State Space source-accounting boundary",
        ),
        "null_results": (
            "3.1 Complexity (K) and the Cardinality of the State Space is not empirical validation evidence",
        ),
        "variables": ("3_1_complexity_k_and_the_cardinality_of_the_state_space",),
        "validation_targets": ("preserve records P0R03781-P0R03788",),
        "null_controls": (
            "3_1_complexity_k_and_the_cardinality_of_the_state_space must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class Section31ComplexityKAndTheCardinalityOfTheStateSpaceSpec:
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
class Section31ComplexityKAndTheCardinalityOfTheStateSpaceSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section31ComplexityKAndTheCardinalityOfTheStateSpaceSpec, ...]
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


def build_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_specs(
    source_records: list[dict[str, Any]],
) -> Section31ComplexityKAndTheCardinalityOfTheStateSpaceSpecBundle:
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

    specs: list[Section31ComplexityKAndTheCardinalityOfTheStateSpaceSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section31ComplexityKAndTheCardinalityOfTheStateSpaceSpec(
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
        "title": "Paper 0 "
        + "3.1 Complexity (K) and the Cardinality of the State Space"
        + " Specs",
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
        "next_source_boundary": "P0R03789",
    }
    return Section31ComplexityKAndTheCardinalityOfTheStateSpaceSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section31ComplexityKAndTheCardinalityOfTheStateSpaceSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_specs(
        load_jsonl(ledger_path)
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section31ComplexityKAndTheCardinalityOfTheStateSpaceSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "3.1 Complexity (K) and the Cardinality of the State Space" + " Specs",
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
    bundle: Section31ComplexityKAndTheCardinalityOfTheStateSpaceSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_3_1_complexity_k_and_the_cardinality_of_the_state_space_validation_specs_{date_tag}.md"
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
