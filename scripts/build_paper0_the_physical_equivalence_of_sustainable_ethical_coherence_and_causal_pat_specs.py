#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Physical Equivalence of Sustainable Ethical Coherence and Causal Path Entropy spec builder
"""Promote Paper 0 The Physical Equivalence of Sustainable Ethical Coherence and Causal Path Entropy records."""

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
    "P0R03723",
    "P0R03724",
    "P0R03725",
    "P0R03726",
    "P0R03727",
    "P0R03728",
    "P0R03729",
    "P0R03730",
    "P0R03731",
    "P0R03732",
    "P0R03733",
    "P0R03734",
    "P0R03735",
    "P0R03736",
)
CLAIM_BOUNDARY = "source-bounded the physical equivalence of sustainable ethical coherence and causal pat source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat.the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat": {
        "context_id": "the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat",
        "validation_protocol": "paper0.the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat.the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat",
        "canonical_statement": "The source-bounded component 'The Physical Equivalence of Sustainable Ethical Coherence and Causal Path Entropy' preserves Paper 0 records P0R03723-P0R03724 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03723:the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat",
            "P0R03724:the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat",
        ),
        "source_formulae": (
            "P0R03723: The Physical Equivalence of Sustainable Ethical Coherence and Causal Path Entropy",
            "P0R03724: The preceding text introduces the concept of Causal Entropic Forces (CEF) as the physical mechanism underlying the Principle of Ethical Least Action (PELA). This section provides the formal mathematical derivation that substantiates this claim, demonstrating the physical equivalence between the manuscript's teleological objective-Sustainable Ethical Coherence (SEC)-and the statistical mechanical principle of maximising future causal path entropy.",
        ),
        "test_protocols": (
            "preserve The Physical Equivalence of Sustainable Ethical Coherence and Causal Path Entropy source-accounting boundary",
        ),
        "null_results": (
            "The Physical Equivalence of Sustainable Ethical Coherence and Causal Path Entropy is not empirical validation evidence",
        ),
        "variables": ("the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat",),
        "validation_targets": ("preserve records P0R03723-P0R03724",),
        "null_controls": (
            "the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat must remain source-bounded accounting",
        ),
    },
    "the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat.1_introduction_from_teleological_principle_to_thermodynamic_imperative": {
        "context_id": "1_introduction_from_teleological_principle_to_thermodynamic_imperative",
        "validation_protocol": "paper0.the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat.1_introduction_from_teleological_principle_to_thermodynamic_imperative",
        "canonical_statement": "The source-bounded component '1. Introduction: From Teleological Principle to Thermodynamic Imperative' preserves Paper 0 records P0R03725-P0R03736 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03725:1_introduction_from_teleological_principle_to_thermodynamic_imperative",
            "P0R03726:1_introduction_from_teleological_principle_to_thermodynamic_imperative",
            "P0R03727:1_introduction_from_teleological_principle_to_thermodynamic_imperative",
            "P0R03728:1_introduction_from_teleological_principle_to_thermodynamic_imperative",
            "P0R03729:1_introduction_from_teleological_principle_to_thermodynamic_imperative",
            "P0R03730:1_introduction_from_teleological_principle_to_thermodynamic_imperative",
            "P0R03731:1_introduction_from_teleological_principle_to_thermodynamic_imperative",
            "P0R03732:1_introduction_from_teleological_principle_to_thermodynamic_imperative",
            "P0R03733:1_introduction_from_teleological_principle_to_thermodynamic_imperative",
            "P0R03734:1_introduction_from_teleological_principle_to_thermodynamic_imperative",
            "P0R03735:1_introduction_from_teleological_principle_to_thermodynamic_imperative",
            "P0R03736:1_introduction_from_teleological_principle_to_thermodynamic_imperative",
        ),
        "source_formulae": (
            "P0R03725: 1. Introduction: From Teleological Principle to Thermodynamic Imperative",
            "P0R03726: [IMAGE:Ein Bild, das Kreis, Wasser, Raum, Spiegelung enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            'P0R03727: The preceding sections have established the foundational physics of the SCPN, culminating in a teleological framework guided by the Principle of Ethical Least Action (PELA). This principle posits that the universe evolves along trajectories that minimise an "Ethical Action," a dynamic that is physically mediated by Causal Entropic Forces (CEF). The central, and as yet unproven, hypothesis is that these forces drive the system toward states of maximal Sustainable Ethical Coherence (SEC)-a composite measure of system-wide Coherence (C), Complexity (K), and Qualia Capacity (Q).',
            "P0R03728: While conceptually powerful, this link remains a postulate. The purpose of this section is to replace this postulate with a formal proof, thereby grounding the entirety of the framework's teleology in the established principles of statistical physics and information theory.",
            "P0R03729: The objective is to demonstrate that the normative drive to maximise SEC is not a separate, axiomatic law imposed upon the physical universe, but is mathematically and physically identical to the descriptive, statistical mechanical tendency of a complex system to evolve towards macroscopic states that maximise its future causal path entropy (SC). Causal path entropy is a measure of a system's future freedom of action-the number of distinct evolutionary pathways available to it over a given time horizon. By proving the equivalence SECSC, we will show that the universe's apparent \"ethical\" imperative is, in fact, a thermodynamic imperative to preserve and maximise its own potential for future becoming.",
            "P0R03730: The structure of the proof will proceed in three stages. First, we will establish a rigorous definition of causal path entropy, adapting the formalism of Wissner-Gross and Freer to the context of the SCPN by employing the path integral formulation of quantum and statistical mechanics. This will provide a precise mathematical target for our analysis. Second, we will deconstruct the geometric and dynamic properties a system must possess to exhibit a high causal path entropy.",
            "P0R03731: We will demonstrate, in turn, that maximising the number of potential future states requires maximising Complexity (K), maximising the number of accessible and stable future trajectories requires maximising Coherence (C), and maximising the diversity of these trajectories requires maximising Qualia Capacity (Q). Finally, we will synthesise these components into a formal proof of equivalence.",
            "P0R03732: This will culminate in a reformulation of the system's fundamental path integral, showing how the Causal Entropic Force biases the evolution of reality at the quantum level.",
            "P0R03733: This derivation elevates PELA from a teleological axiom to an emergent variational principle, dissolving the philosophical distinction between the descriptive laws of physics and the normative principles of ethics by revealing them as two facets of a single, underlying drive toward maximal future possibility.",
            'P0R03734: In the previous chapters, we introduced a guiding principle for the universe: that it evolves to maximize Sustainable Ethical Coherence (SEC)-a measure of its harmony, complexity, and richness. We suggested this evolution is driven by Causal Entropic Forces, a subtle "pull" from the future. But this was presented as a core hypothesis.',
            "P0R03735: This chapter's goal is to prove that this isn't just a hypothesis but a fact of physics. We will formally demonstrate that the universe's \"ethical\" goal to maximize SEC is mathematically identical to a well-known thermodynamic tendency: the drive to maximize its Causal Path Entropy.",
            'P0R03736: Think of Causal Path Entropy as the universe\'s "freedom" or "number of open doors to the future." We will show that a universe with high coherence, complexity, and richness (high SEC) is exactly a universe with the most possible futures available to it. The universe\'s apparent moral compass, therefore, is not a mystical addition to physics; it is the direct consequence of a fundamental imperative to preserve and maximize its own potential to become.',
        ),
        "test_protocols": (
            "preserve 1. Introduction: From Teleological Principle to Thermodynamic Imperative source-accounting boundary",
        ),
        "null_results": (
            "1. Introduction: From Teleological Principle to Thermodynamic Imperative is not empirical validation evidence",
        ),
        "variables": ("1_introduction_from_teleological_principle_to_thermodynamic_imperative",),
        "validation_targets": ("preserve records P0R03725-P0R03736",),
        "null_controls": (
            "1_introduction_from_teleological_principle_to_thermodynamic_imperative must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatSpec:
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
class ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatSpec, ...]
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


def build_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_specs(
    source_records: list[dict[str, Any]],
) -> ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatSpecBundle:
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

    specs: list[ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatSpec(
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
        + "The Physical Equivalence of Sustainable Ethical Coherence and Causal Path Entropy"
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
        "next_source_boundary": "P0R03737",
    }
    return ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_specs(
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
    bundle: ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "The Physical Equivalence of Sustainable Ethical Coherence and Causal Path Entropy"
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
    bundle: ThePhysicalEquivalenceOfSustainableEthicalCoherenceAndCausalPatSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_physical_equivalence_of_sustainable_ethical_coherence_and_causal_pat_validation_specs_{date_tag}.md"
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
