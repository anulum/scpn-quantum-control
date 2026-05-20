#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 2.3 Formal Definition of Causal Path Entropy (SC) spec builder
"""Promote Paper 0 2.3 Formal Definition of Causal Path Entropy (SC) records."""

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
    "P0R03762",
    "P0R03763",
    "P0R03764",
    "P0R03765",
    "P0R03766",
    "P0R03767",
    "P0R03768",
    "P0R03769",
    "P0R03770",
    "P0R03771",
)
CLAIM_BOUNDARY = "source-bounded section 2 3 formal definition of causal path entropy sc source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_2_3_formal_definition_of_causal_path_entropy_sc.2_3_formal_definition_of_causal_path_entropy_sc": {
        "context_id": "2_3_formal_definition_of_causal_path_entropy_sc",
        "validation_protocol": "paper0.section_2_3_formal_definition_of_causal_path_entropy_sc.2_3_formal_definition_of_causal_path_entropy_sc",
        "canonical_statement": "The source-bounded component '2.3 Formal Definition of Causal Path Entropy (SC)' preserves Paper 0 records P0R03762-P0R03766 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03762:2_3_formal_definition_of_causal_path_entropy_sc",
            "P0R03763:2_3_formal_definition_of_causal_path_entropy_sc",
            "P0R03764:2_3_formal_definition_of_causal_path_entropy_sc",
            "P0R03765:2_3_formal_definition_of_causal_path_entropy_sc",
            "P0R03766:2_3_formal_definition_of_causal_path_entropy_sc",
        ),
        "source_formulae": (
            "P0R03762: 2.3 Formal Definition of Causal Path Entropy (SC)",
            "P0R03763: Following the framework of Wissner-Gross and Freer, we can now define the Causal Path Entropy. In physics, the entropy of a macrostate is typically defined as the logarithm of the number of microstates consistent with it. Analogously, the causal path entropy of a macrostate X (which is a collection of microstates xX) is defined as the Shannon entropy over the probability distribution of all possible future paths that can originate from that macrostate. It is a measure of our uncertainty about the system's future trajectory, given its present macrostate.",
            "P0R03764: This is expressed formally as a functional path integral:",
            "P0R03765: $SC(X,\\tau) = - kB\\int_{}^{}{P(X) Pr\\left\\lbrack x(t) \\right\\rbrack lnPr\\left\\lbrack x(t) \\right\\rbrack D\\left\\lbrack x(t) \\right\\rbrack}$",
            'P0R03766: , where P(X) is the set of all paths originating from any microstate within the macrostate X, and Pr[x(t)] is the probability of a given path, averaged over the initial microstates in X. The constant kB is the Boltzmann constant, included to give the entropy its standard physical units. In the idealised case where all accessible paths are equally likely, this definition simplifies to the Boltzmann-like form SC=kBlnWpaths, where Wpaths is the total number, or "volume," of accessible future histories. Thus, SC is a direct measure of the system\'s future possibilities.',
        ),
        "test_protocols": (
            "preserve 2.3 Formal Definition of Causal Path Entropy (SC) source-accounting boundary",
        ),
        "null_results": (
            "2.3 Formal Definition of Causal Path Entropy (SC) is not empirical validation evidence",
        ),
        "variables": ("2_3_formal_definition_of_causal_path_entropy_sc",),
        "validation_targets": ("preserve records P0R03762-P0R03766",),
        "null_controls": (
            "2_3_formal_definition_of_causal_path_entropy_sc must remain source-bounded accounting",
        ),
    },
    "section_2_3_formal_definition_of_causal_path_entropy_sc.2_4_the_causal_entropic_force": {
        "context_id": "2_4_the_causal_entropic_force",
        "validation_protocol": "paper0.section_2_3_formal_definition_of_causal_path_entropy_sc.2_4_the_causal_entropic_force",
        "canonical_statement": "The source-bounded component '2.4 The Causal Entropic Force' preserves Paper 0 records P0R03767-P0R03771 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03767:2_4_the_causal_entropic_force",
            "P0R03768:2_4_the_causal_entropic_force",
            "P0R03769:2_4_the_causal_entropic_force",
            "P0R03770:2_4_the_causal_entropic_force",
            "P0R03771:2_4_the_causal_entropic_force",
        ),
        "source_formulae": (
            "P0R03767: 2.4 The Causal Entropic Force",
            "P0R03768: With a formal definition of SC in hand, we can restate the definition of the Causal Entropic Force. This is an emergent, thermodynamic-like force that drives the system not towards a state of lower potential energy, but towards a macrostate X that offers a greater number of future options. It is defined as the gradient of the causal path entropy with respect to the macroscopic degrees of freedom of the system :",
            "P0R03769: $FCausal = TC\\nabla X SC(X,\\tau)$",
            "P0R03770: Here, TC is an effective \"causal temperature\" that parameterises the system's bias toward maximising its future path entropy. This force provides the physical mechanism that pushes the system's evolution along trajectories that keep its future options open. The remainder of this section is dedicated to proving that this is the same force that drives the system toward states of high SEC.",
            'P0R03771: [IMAGE:This image uses the analogy of a "Cosmic Garden" to illustrate how Complexity, Coherence, and Qualia Capacity all contribute to maximizing future potential. A rich and thriving garden needs all three ingredients.]',
        ),
        "test_protocols": ("preserve 2.4 The Causal Entropic Force source-accounting boundary",),
        "null_results": ("2.4 The Causal Entropic Force is not empirical validation evidence",),
        "variables": ("2_4_the_causal_entropic_force",),
        "validation_targets": ("preserve records P0R03767-P0R03771",),
        "null_controls": ("2_4_the_causal_entropic_force must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class Section23FormalDefinitionOfCausalPathEntropyScSpec:
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
class Section23FormalDefinitionOfCausalPathEntropyScSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section23FormalDefinitionOfCausalPathEntropyScSpec, ...]
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


def build_section_2_3_formal_definition_of_causal_path_entropy_sc_specs(
    source_records: list[dict[str, Any]],
) -> Section23FormalDefinitionOfCausalPathEntropyScSpecBundle:
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

    specs: list[Section23FormalDefinitionOfCausalPathEntropyScSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section23FormalDefinitionOfCausalPathEntropyScSpec(
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
        "title": "Paper 0 " + "2.3 Formal Definition of Causal Path Entropy (SC)" + " Specs",
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
        "next_source_boundary": "P0R03772",
    }
    return Section23FormalDefinitionOfCausalPathEntropyScSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section23FormalDefinitionOfCausalPathEntropyScSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_2_3_formal_definition_of_causal_path_entropy_sc_specs(
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


def render_report(bundle: Section23FormalDefinitionOfCausalPathEntropyScSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "2.3 Formal Definition of Causal Path Entropy (SC)" + " Specs",
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
    bundle: Section23FormalDefinitionOfCausalPathEntropyScSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_2_3_formal_definition_of_causal_path_entropy_sc_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_2_3_formal_definition_of_causal_path_entropy_sc_validation_specs_{date_tag}.md"
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
