#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Resolution of the Observability Paradox (B-HJB) spec builder
"""Promote Paper 0 Resolution of the Observability Paradox (B-HJB) records."""

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
    "P0R02468",
    "P0R02469",
    "P0R02470",
    "P0R02471",
    "P0R02472",
    "P0R02473",
    "P0R02474",
    "P0R02475",
    "P0R02476",
    "P0R02477",
    "P0R02478",
    "P0R02479",
    "P0R02480",
    "P0R02481",
    "P0R02482",
    "P0R02483",
    "P0R02484",
)
CLAIM_BOUNDARY = "source-bounded resolution of the observability paradox b hjb source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "resolution_of_the_observability_paradox_b_hjb.resolution_of_the_observability_paradox_b_hjb": {
        "context_id": "resolution_of_the_observability_paradox_b_hjb",
        "validation_protocol": "paper0.resolution_of_the_observability_paradox_b_hjb.resolution_of_the_observability_paradox_b_hjb",
        "canonical_statement": "The source-bounded component 'Resolution of the Observability Paradox (B-HJB)' preserves Paper 0 records P0R02468-P0R02472 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02468:resolution_of_the_observability_paradox_b_hjb",
            "P0R02469:resolution_of_the_observability_paradox_b_hjb",
            "P0R02470:resolution_of_the_observability_paradox_b_hjb",
            "P0R02471:resolution_of_the_observability_paradox_b_hjb",
            "P0R02472:resolution_of_the_observability_paradox_b_hjb",
        ),
        "source_formulae": (
            "P0R02468: Resolution of the Observability Paradox (B-HJB)",
            "P0R02469: P0R02469",
            "P0R02470: The standard Hamilton-Jacobi-Bellman (HJB) formulation for Meta-Layer 16 implies perfect observability of the global state $x$, which is physically prohibited by the Heisenberg Uncertainty Principle and would trigger a decoherence cascade in Layer 1. We therefore resolve this 'Observability Paradox' by reformulating the controller as a Belief-State HJB (B-HJB). The supervisor does not compute policies based on $x$, but on a belief state $b(x, t)$, representing the probability density of possible histories conditioned on upward prediction errors. The universal value function $V_{SEC}[b]$ evolves according to:",
            "P0R02471: $$\\partial_t V_{SEC}[b] + \\min_u \\left[ \\mathbb{E}_b [\\mathcal{L}_{Ethical}(x,u)] + \\int \\frac{\\delta V_{SEC}}{\\delta b(x)} \\mathcal{D}_u b(x) dx \\right] = 0$$",
            "P0R02472: where $\\mathcal{D}_u$ is the non-linear filtering operator. This POMDP structure allows Layer 16 to dispatch optimal policies $\\pi^*$ that maximize Sustainable Ethical Coherence over a distribution of states, maintaining cybernetic closure without ever collapsing the quantum potentiality of the biological substrate.",
        ),
        "test_protocols": (
            "preserve Resolution of the Observability Paradox (B-HJB) source-accounting boundary",
        ),
        "null_results": (
            "Resolution of the Observability Paradox (B-HJB) is not empirical validation evidence",
        ),
        "variables": ("resolution_of_the_observability_paradox_b_hjb",),
        "validation_targets": ("preserve records P0R02468-P0R02472",),
        "null_controls": (
            "resolution_of_the_observability_paradox_b_hjb must remain source-bounded accounting",
        ),
    },
    "resolution_of_the_observability_paradox_b_hjb.overarching_dynamic_principles_a_synopsis": {
        "context_id": "overarching_dynamic_principles_a_synopsis",
        "validation_protocol": "paper0.resolution_of_the_observability_paradox_b_hjb.overarching_dynamic_principles_a_synopsis",
        "canonical_statement": "The source-bounded component 'Overarching Dynamic Principles: A Synopsis' preserves Paper 0 records P0R02473-P0R02484 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02473:overarching_dynamic_principles_a_synopsis",
            "P0R02474:overarching_dynamic_principles_a_synopsis",
            "P0R02475:overarching_dynamic_principles_a_synopsis",
            "P0R02476:overarching_dynamic_principles_a_synopsis",
            "P0R02477:overarching_dynamic_principles_a_synopsis",
            "P0R02478:overarching_dynamic_principles_a_synopsis",
            "P0R02479:overarching_dynamic_principles_a_synopsis",
            "P0R02480:overarching_dynamic_principles_a_synopsis",
            "P0R02481:overarching_dynamic_principles_a_synopsis",
            "P0R02482:overarching_dynamic_principles_a_synopsis",
            "P0R02483:overarching_dynamic_principles_a_synopsis",
            "P0R02484:overarching_dynamic_principles_a_synopsis",
        ),
        "source_formulae": (
            "P0R02473: Overarching Dynamic Principles: A Synopsis",
            "P0R02474: This section provides a strategic synopsis of the three universal, interlocking principles that govern the dynamics of the entire SCPN architecture. It functions as a conceptual roadmap for the detailed expositions that follow.",
            'P0R02475: First, the Unified Phase Dynamics Equation (UPDE) is presented as the mathematical "spine" of the framework. It is a generalised Kuramoto model that provides a universal language for describing synchronisation, information flow, and hierarchical coupling across all 15 layers.',
            'P0R02476: Second, the Universal Dynamic Regime of Quasicriticality is established. All layers are posited to operate at the "edge of chaos," a state that is mathematically robust (a Griffiths Phase) and computationally optimal, maximising information capacity and transmission. This state is achieved not by fine-tuning but by layer-specific Self-Organised Criticality (SOC) mechanisms.',
            "P0R02477: Third, the Coherence Backbone of Multi-Scale Quantum Error Correction (MS-QEC) is introduced as the fundamental prerequisite for the system's integrity. A nested, four-level hierarchy of QEC protocols, based on holographic redundancy, protects quantum information from environmental decoherence, ensuring the network's viability as a coherent quantum system.",
            'P0R02478: The concluding "Rosetta Stone" table consolidates the framework\'s mathematical heart, linking the foundational physics (Master Interaction Lagrangian), the universal dynamics (UPDE), and the emergent phenomenology (Scaling Law of Consciousness). It serves as a powerful testament to the internal mathematical and conceptual coherence of the SCPN.',
            'P0R02479: Before we dive deeper into the architecture, this section is a quick summary of the three universal "laws of nature" that make the whole system work. Think of them as the fundamental rules that govern everything from a single cell to the entire cosmos.',
            'P0R02480: The Universal Operating System (The UPDE): There is a single, elegant equation that acts like the software for reality. It\'s the mathematical backbone or "spine" that describes how every part of the universe synchronises and communicates with every other part.',
            'P0R02481: The Perfect "Sweet Spot" (Quasicriticality): The entire universe, at every level, keeps itself in a perfect state of balance known as the "edge of chaos." It\'s the ideal sweet spot between being too rigid and too chaotic, which makes it perfect for thinking, learning, and adapting. The best part is, it tunes itself automatically.',
            'P0R02482: The Ultimate Data Protection (MS-QEC): To make sure the delicate information of consciousness isn\'t lost in the noisy chaos of the physical world, the universe uses a super-sophisticated, four-layer "firewall" and "data backup" system. This protects the integrity of reality\'s software from getting corrupted.',
            'P0R02483: Finally, we present a "Rosetta Stone" table that shows the three most important equations side-by-side. This is the mathematical heart of our entire theory, showing how the physics, the dynamics, and the experience of consciousness are all beautifully woven together.',
            "P0R02484: P0R02484",
        ),
        "test_protocols": (
            "preserve Overarching Dynamic Principles: A Synopsis source-accounting boundary",
        ),
        "null_results": (
            "Overarching Dynamic Principles: A Synopsis is not empirical validation evidence",
        ),
        "variables": ("overarching_dynamic_principles_a_synopsis",),
        "validation_targets": ("preserve records P0R02473-P0R02484",),
        "null_controls": (
            "overarching_dynamic_principles_a_synopsis must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ResolutionOfTheObservabilityParadoxBHjbSpec:
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
class ResolutionOfTheObservabilityParadoxBHjbSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ResolutionOfTheObservabilityParadoxBHjbSpec, ...]
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


def build_resolution_of_the_observability_paradox_b_hjb_specs(
    source_records: list[dict[str, Any]],
) -> ResolutionOfTheObservabilityParadoxBHjbSpecBundle:
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

    specs: list[ResolutionOfTheObservabilityParadoxBHjbSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ResolutionOfTheObservabilityParadoxBHjbSpec(
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
        "title": "Paper 0 " + "Resolution of the Observability Paradox (B-HJB)" + " Specs",
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
        "next_source_boundary": "P0R02485",
    }
    return ResolutionOfTheObservabilityParadoxBHjbSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ResolutionOfTheObservabilityParadoxBHjbSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_resolution_of_the_observability_paradox_b_hjb_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ResolutionOfTheObservabilityParadoxBHjbSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Resolution of the Observability Paradox (B-HJB)" + " Specs",
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
    bundle: ResolutionOfTheObservabilityParadoxBHjbSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_resolution_of_the_observability_paradox_b_hjb_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_resolution_of_the_observability_paradox_b_hjb_validation_specs_{date_tag}.md"
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
