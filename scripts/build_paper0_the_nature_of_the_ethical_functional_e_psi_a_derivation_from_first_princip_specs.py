#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Nature of the Ethical Functional E[Psi]: A Derivation from First Principles spec builder
"""Promote Paper 0 The Nature of the Ethical Functional E[Psi]: A Derivation from First Principles records."""

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
    "P0R03603",
    "P0R03604",
    "P0R03605",
    "P0R03606",
    "P0R03607",
    "P0R03608",
    "P0R03609",
    "P0R03610",
    "P0R03611",
)
CLAIM_BOUNDARY = "source-bounded the nature of the ethical functional e psi a derivation from first princip source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip.the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princ": {
        "context_id": "the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princ",
        "validation_protocol": "paper0.the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip.the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princ",
        "canonical_statement": "The source-bounded component 'The Nature of the Ethical Functional E[Psi]: A Derivation from First Principles' preserves Paper 0 records P0R03603-P0R03605 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03603:the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princ",
            "P0R03604:the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princ",
            "P0R03605:the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princ",
        ),
        "source_formulae": (
            "P0R03603: The Nature of the Ethical Functional E[Psi]: A Derivation from First Principles",
            "P0R03604: The proposal to treat ethics as a physical function to be optimised is the most radical and philosophically challenging aspect of the SCPN framework. The primary difficulty, as identified in our query, is the risk of committing a category error: conflating a normative, philosophical concept (ethics) with a descriptive, physical quantity.",
            "P0R03605: This section demonstrates that this category error can be avoided by deriving the form and function of the Ethical Functional, E[Psi], from the fundamental symmetries of the Psi-field itself, thereby grounding it in physics rather than postulating it from philosophy.",
        ),
        "test_protocols": (
            "preserve The Nature of the Ethical Functional E[Psi]: A Derivation from First Principles source-accounting boundary",
        ),
        "null_results": (
            "The Nature of the Ethical Functional E[Psi]: A Derivation from First Principles is not empirical validation evidence",
        ),
        "variables": ("the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princ",),
        "validation_targets": ("preserve records P0R03603-P0R03605",),
        "null_controls": (
            "the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princ must remain source-bounded accounting",
        ),
    },
    "the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip.section_1_avoiding_the_category_error_from_philosophy_to_physics": {
        "context_id": "section_1_avoiding_the_category_error_from_philosophy_to_physics",
        "validation_protocol": "paper0.the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip.section_1_avoiding_the_category_error_from_philosophy_to_physics",
        "canonical_statement": "The source-bounded component 'Section 1: Avoiding the Category Error: From Philosophy to Physics' preserves Paper 0 records P0R03606-P0R03611 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03606:section_1_avoiding_the_category_error_from_philosophy_to_physics",
            "P0R03607:section_1_avoiding_the_category_error_from_philosophy_to_physics",
            "P0R03608:section_1_avoiding_the_category_error_from_philosophy_to_physics",
            "P0R03609:section_1_avoiding_the_category_error_from_philosophy_to_physics",
            "P0R03610:section_1_avoiding_the_category_error_from_philosophy_to_physics",
            "P0R03611:section_1_avoiding_the_category_error_from_philosophy_to_physics",
        ),
        "source_formulae": (
            "P0R03606: Section 1: Avoiding the Category Error: From Philosophy to Physics",
            'P0R03607: The traditional separation between the "is" of descriptive science and the "ought" of normative ethics presents a formidable barrier to any physical theory of morality. A naive attempt to derive ethics from physics would be to claim that a specific human moral code is a direct consequence of physical law-a claim that is philosophically and empirically untenable.',
            'P0R03608: The SCPN framework, in its most rigorous formulation, employs a more sophisticated strategy. It does not attempt to derive a specific set of human moral rules. Instead, it identifies a universal physical principle of optimisation inherent in the dynamics of the Psi-field and then labels this principle "ethical." The strategy is to first define the components of "Sustainable Ethical Coherence" (SEC) in purely physical and informational terms:',
            "P0R03609: Coherence (C): A measure of system-wide synchrony and integration. | Complexity (K): A measure of irreducible, integrated information. | Qualia Capacity (Q): A measure of the richness of the state space for subjective experience.",
            "P0R03610: These are not traditional moral virtues but are, in principle, measurable properties of a complex system's organisation. The subsequent task, which this report undertakes, is to demonstrate that the universe's evolution is biased towards maximising these specific physical quantities as a direct consequence of its fundamental symmetries.",
            'P0R03611: This approach avoids the category error by identifying the "ought" (the ethical imperative) with a physical trajectory (the path of least action for the Psi-field). The labelling of this trajectory as "ethical" is a semantic bridge, but the underlying dynamics being described are purely physical. This reframes the problem into a tractable form, moving it from the domain of metaphysics to that of theoretical physics.',
        ),
        "test_protocols": (
            "preserve Section 1: Avoiding the Category Error: From Philosophy to Physics source-accounting boundary",
        ),
        "null_results": (
            "Section 1: Avoiding the Category Error: From Philosophy to Physics is not empirical validation evidence",
        ),
        "variables": ("section_1_avoiding_the_category_error_from_philosophy_to_physics",),
        "validation_targets": ("preserve records P0R03606-P0R03611",),
        "null_controls": (
            "section_1_avoiding_the_category_error_from_philosophy_to_physics must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipSpec:
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
class TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipSpec, ...]
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


def build_the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_specs(
    source_records: list[dict[str, Any]],
) -> TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipSpecBundle:
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

    specs: list[TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipSpec(
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
        + "The Nature of the Ethical Functional E[Psi]: A Derivation from First Principles"
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
        "next_source_boundary": "P0R03612",
    }
    return TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_specs(
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


def render_report(
    bundle: TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "The Nature of the Ethical Functional E[Psi]: A Derivation from First Principles"
        + " Specs",
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
    bundle: TheNatureOfTheEthicalFunctionalEPsiADerivationFromFirstPrincipSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_nature_of_the_ethical_functional_e_psi_a_derivation_from_first_princip_validation_specs_{date_tag}.md"
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
