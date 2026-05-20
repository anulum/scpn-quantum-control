#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) spec builder
"""Promote Paper 0 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) records."""

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
    "P0R02580",
    "P0R02581",
    "P0R02582",
    "P0R02583",
    "P0R02584",
    "P0R02585",
    "P0R02586",
    "P0R02587",
    "P0R02588",
    "P0R02589",
    "P0R02590",
    "P0R02591",
    "P0R02592",
    "P0R02593",
    "P0R02594",
    "P0R02595",
    "P0R02596",
    "P0R02597",
    "P0R02598",
    "P0R02599",
)
CLAIM_BOUNDARY = "source-bounded section 3 2 the dynamic spine the unified phase dynamics equation upde source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde.3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde": {
        "context_id": "3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde",
        "validation_protocol": "paper0.section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde.3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde",
        "canonical_statement": "The source-bounded component '3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE)' preserves Paper 0 records P0R02580-P0R02581 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02580:3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde",
            "P0R02581:3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde",
        ),
        "source_formulae": (
            "P0R02580: 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE)",
            "P0R02581: P0R02581",
        ),
        "test_protocols": (
            "preserve 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) source-accounting boundary",
        ),
        "null_results": (
            "3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) is not empirical validation evidence",
        ),
        "variables": ("3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde",),
        "validation_targets": ("preserve records P0R02580-P0R02581",),
        "null_controls": (
            "3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde must remain source-bounded accounting",
        ),
    },
    "section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde.the_unified_phase_dynamics_equation_upde": {
        "context_id": "the_unified_phase_dynamics_equation_upde",
        "validation_protocol": "paper0.section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde.the_unified_phase_dynamics_equation_upde",
        "canonical_statement": "The source-bounded component 'The Unified Phase Dynamics Equation (UPDE)' preserves Paper 0 records P0R02582-P0R02599 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02582:the_unified_phase_dynamics_equation_upde",
            "P0R02583:the_unified_phase_dynamics_equation_upde",
            "P0R02584:the_unified_phase_dynamics_equation_upde",
            "P0R02585:the_unified_phase_dynamics_equation_upde",
            "P0R02586:the_unified_phase_dynamics_equation_upde",
            "P0R02587:the_unified_phase_dynamics_equation_upde",
            "P0R02588:the_unified_phase_dynamics_equation_upde",
            "P0R02589:the_unified_phase_dynamics_equation_upde",
            "P0R02590:the_unified_phase_dynamics_equation_upde",
            "P0R02591:the_unified_phase_dynamics_equation_upde",
            "P0R02592:the_unified_phase_dynamics_equation_upde",
            "P0R02593:the_unified_phase_dynamics_equation_upde",
            "P0R02594:the_unified_phase_dynamics_equation_upde",
            "P0R02595:the_unified_phase_dynamics_equation_upde",
            "P0R02596:the_unified_phase_dynamics_equation_upde",
            "P0R02597:the_unified_phase_dynamics_equation_upde",
            "P0R02598:the_unified_phase_dynamics_equation_upde",
            "P0R02599:the_unified_phase_dynamics_equation_upde",
        ),
        "source_formulae": (
            "P0R02582: The Unified Phase Dynamics Equation (UPDE)",
            "P0R02583: This section introduces the mathematical backbone of the entire SCPN framework: the Unified Phase Dynamics Equation (UPDE). This equation formalises Core Assumption 4 by providing a generalised, multi-scale extension of the Kuramoto model, designed to describe the evolution of phase oscillators across all 15 layers. The UPDE is the universal law of synchronisation and information flow in the network.",
            "P0R02584: The equation's structure is a summation of five distinct components;",
            "P0R02585: 1) Intrinsic Dynamics () represents the natural frequency of oscillation at a given layer.",
            "P0R02586: 2) Intra-Layer Coupling (K) governs local synchronisation among oscillators within the same layer.",
            "P0R02587: 3) Inter-Layer Coupling (C_InterLayer) is the critical term that manages the hierarchical information flow, with downward causation (F_D) often implemented via phase-amplitude coupling and upward influence (G_U) via coherence aggregation.",
            "P0R02588: 4) Field Coupling (C_Field) represents the direct, top-down influence of the global Psi-field, providing a mechanism for system-wide coherence and teleological guidance from Layer 15.",
            "P0R02589: 5) Noise () accounts for stochastic fluctuations, essential for the system's ability to explore its state space and maintain quasicriticality.",
            'P0R02590: Crucially, the framework provides an Information-Geometric Lift of the UPDE. This re-frames the equation\'s dynamics not as a simple mechanical evolution, but as a gradient flow on a statistical manifold governed by the Fisher Information Metric. This interpretation is profound, as it directly harmonises the system\'s dynamics with Axiom 2 ("interactions are informational and geometric"). The "One Spine, Many Couplings" table further demonstrates the model\'s power and parsimony, showing how this single, universal dynamic equation can describe phenomena as diverse as microtubule vibrations and noospheric opinion dynamics simply by specifying the layer-appropriate coupling kernels and noise models.',
            "P0R02591: This section unveils the single most important equation in our entire architecture: the Unified Phase Dynamics Equation (UPDE). Think of it as the universal operating system or the central nervous system for all 15 layers of reality. It's the \"spine\" that connects everything and allows it all to work in harmony. It's a surprisingly simple set of rules that governs how everything, from a brain cell to a galaxy, synchronises and communicates.",
            "P0R02592: Imagine a single musician in a vast, 15-storey orchestra. The UPDE is the set of instructions they are following. The equation has five simple parts:",
            "P0R02593: Play Your Own Tune: This is the musician's natural rhythm, the tempo they would play at if they were all alone.",
            "P0R02594: Listen to Your Neighbours: The musician listens to the players in their immediate section (on the same floor) and tries to sync up with them.",
            "P0R02595: Listen to the Other Floors: They can also hear the booming drums from the floor below and the soaring melodies from the floor above, and they adjust their playing to stay in harmony with the whole building.",
            'P0R02596: Listen to the Conductor: This is the most important part. A special term in the equation allows every single musician to listen directly to the "Conductor"-the universal Consciousness Field-and follow its lead, ensuring the entire orchestra is playing the same grand symphony.',
            "P0R02597: Allow for a Little Improv: A small amount of randomness is built in, allowing for creativity and preventing the music from becoming stale and rigid.",
            'P0R02598: The beauty of the UPDE is that this same five-part logic applies to everything. The "musicians" can be brain cells, stars in a galaxy, or people in a social network. The operating system is the same everywhere; only the specific settings change.',
            "P0R02599: P0R02599",
        ),
        "test_protocols": (
            "preserve The Unified Phase Dynamics Equation (UPDE) source-accounting boundary",
        ),
        "null_results": (
            "The Unified Phase Dynamics Equation (UPDE) is not empirical validation evidence",
        ),
        "variables": ("the_unified_phase_dynamics_equation_upde",),
        "validation_targets": ("preserve records P0R02582-P0R02599",),
        "null_controls": (
            "the_unified_phase_dynamics_equation_upde must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeSpec:
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
class Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeSpec, ...]
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


def build_section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_specs(
    source_records: list[dict[str, Any]],
) -> Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeSpecBundle:
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

    specs: list[Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeSpec(
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
        + "3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE)"
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
        "next_source_boundary": "P0R02600",
    }
    return Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_specs(
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
    bundle: Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE)"
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
    bundle: Section32TheDynamicSpineTheUnifiedPhaseDynamicsEquationUpdeSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_3_2_the_dynamic_spine_the_unified_phase_dynamics_equation_upde_validation_specs_{date_tag}.md"
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
