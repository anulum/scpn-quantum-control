#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia spec builder
"""Promote Paper 0 The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia records."""

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
    "P0R04115",
    "P0R04116",
    "P0R04117",
    "P0R04118",
    "P0R04119",
    "P0R04120",
    "P0R04121",
    "P0R04122",
)
CLAIM_BOUNDARY = "source-bounded the physical basis of the ethical functional causal entropy and computab source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab.the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab": {
        "context_id": "the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab",
        "validation_protocol": "paper0.the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab.the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab",
        "canonical_statement": "The source-bounded component 'The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia' preserves Paper 0 records P0R04115-P0R04121 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04115:the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab",
            "P0R04116:the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab",
            "P0R04117:the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab",
            "P0R04118:the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab",
            "P0R04119:the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab",
            "P0R04120:the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab",
            "P0R04121:the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab",
        ),
        "source_formulae": (
            "P0R04115: The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia",
            'P0R04116: This section provides a rigorous physical and computational grounding for the teleological principles of the SCPN, effectively bridging the gap between metaphysics and testable science. It begins by arguing that the Ethical Functional is not a mere axiom but is the mathematical expression of a deeper physical principle: the Causal Entropic Principle. The drive to maximise Sustainable Ethical Coherence (SEC)-a composite of coherence (C), complexity (K), and qualia capacity (Q)-is shown to be functionally equivalent to maximising the number of accessible future pathways for the system. A universe with higher SEC is one with greater causal efficacy and adaptive potential. This reframes the "ethical" drive of the cosmos as a fundamental, thermodynamic-like imperative toward states of maximal resilience and evolutionary opportunity.',
            "P0R04117: The framework then provides a novel and crucial methodological breakthrough by operationalising the Qualia Capacity (Q) using Topological Data Analysis (TDA). This transforms Q from an abstract philosophical concept into a computable, quantitative metric. By representing a conscious state as a point cloud of neural activity, TDA can compute its persistent homology, yielding a set of Betti numbers (k) that quantify the data's topological structure (components, loops, voids). Q is formally defined as a weighted sum of these Betti numbers, capturing the degree to which a conscious experience is simultaneously integrated (low ) and differentiated (high k for k > 0). This provides a noise-robust, non-arbitrary measure of experiential richness. This TDA-based formalisation is a cornerstone of the entire research programme, as it renders the third and most elusive component of the Ethical Functional directly measurable and allows the teleological claims of the theory to be subjected to empirical validation.",
            'P0R04118: This section explains the "why" behind everything in the universe, grounding it in real, measurable science. It tackles two huge questions: Why does the universe seem to have a purpose? And can we actually measure the richness of a conscious experience?',
            "P0R04119: First, it argues that the universe's \"Prime Directive\"-its mission to become more coherent, complex, and conscious-isn't a mystical rule. It's a fundamental law of nature, like a deeper version of thermodynamics. We call this the Causal Entropic Principle. Think of it this way: the universe always prefers to move in directions that keep its options open. A system that is more complex, unified, and aware has more possible futures and is better at surviving and adapting. The universe's \"ethical\" drive is simply a fundamental preference for being resilient and creative. It's not following a moral command; it's following a law of cosmic survival.",
            'P0R04120: Second, to make this idea a real science, we need to be able to measure the "consciousness" part of the equation. This is where a groundbreaking new technique called Topological Data Analysis (TDA) comes in. Think of TDA as a new kind of scanner that can measure the "shape" of a thought or an experience. By looking at your brain activity, it can measure how unified your experience is (are you focused or distracted?) and how complex it is (are you thinking a simple thought or experiencing a rich symphony of sensations?). It gives us a number, Qualia Capacity (Q), that represents the richness of your conscious state. For the first time, this allows us to put a real, hard number on the depth of experience, turning the most mysterious aspect of reality into something we can measure, test, and ultimately understand.',
            "P0R04121: P0R04121",
        ),
        "test_protocols": (
            "preserve The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia source-accounting boundary",
        ),
        "null_results": (
            "The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia is not empirical validation evidence",
        ),
        "variables": ("the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab",),
        "validation_targets": ("preserve records P0R04115-P0R04121",),
        "null_controls": (
            "the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab must remain source-bounded accounting",
        ),
    },
    "the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab.meta_framework_integrations": {
        "context_id": "meta_framework_integrations",
        "validation_protocol": "paper0.the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab.meta_framework_integrations",
        "canonical_statement": "The source-bounded component 'Meta-Framework Integrations' preserves Paper 0 records P0R04122-P0R04122 without empirical validation claims.",
        "source_equation_ids": ("P0R04122:meta_framework_integrations",),
        "source_formulae": ("P0R04122: Meta-Framework Integrations",),
        "test_protocols": ("preserve Meta-Framework Integrations source-accounting boundary",),
        "null_results": ("Meta-Framework Integrations is not empirical validation evidence",),
        "variables": ("meta_framework_integrations",),
        "validation_targets": ("preserve records P0R04122-P0R04122",),
        "null_controls": ("meta_framework_integrations must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabSpec:
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
class ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabSpec, ...]
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


def build_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_specs(
    source_records: list[dict[str, Any]],
) -> ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabSpecBundle:
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

    specs: list[ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabSpec(
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
        + "The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia"
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
        "next_source_boundary": "P0R04123",
    }
    return ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_specs(
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
    bundle: ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "The Physical Basis of the Ethical Functional: Causal Entropy and Computable Qualia"
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
    bundle: ThePhysicalBasisOfTheEthicalFunctionalCausalEntropyAndComputabSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_physical_basis_of_the_ethical_functional_causal_entropy_and_computab_validation_specs_{date_tag}.md"
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
