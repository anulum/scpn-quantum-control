#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Genesis of the Hierarchy: Sequential Symmetry Breaking (SSB) spec builder
"""Promote Paper 0 The Genesis of the Hierarchy: Sequential Symmetry Breaking (SSB) records."""

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
    "P0R01727",
    "P0R01728",
    "P0R01729",
    "P0R01730",
    "P0R01731",
    "P0R01732",
    "P0R01733",
    "P0R01734",
    "P0R01735",
    "P0R01736",
    "P0R01737",
    "P0R01738",
    "P0R01739",
    "P0R01740",
    "P0R01741",
    "P0R01742",
    "P0R01743",
    "P0R01744",
    "P0R01745",
    "P0R01746",
    "P0R01747",
    "P0R01748",
    "P0R01749",
    "P0R01750",
    "P0R01751",
    "P0R01752",
    "P0R01753",
    "P0R01754",
)
CLAIM_BOUNDARY = "source-bounded the genesis of the hierarchy sequential symmetry breaking ssb source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb.the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb": {
        "context_id": "the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb",
        "validation_protocol": "paper0.the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb.the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb",
        "canonical_statement": "The source-bounded component 'The Genesis of the Hierarchy: Sequential Symmetry Breaking (SSB)' preserves Paper 0 records P0R01727-P0R01728 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01727:the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb",
            "P0R01728:the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb",
        ),
        "source_formulae": (
            "P0R01727: The Genesis of the Hierarchy: Sequential Symmetry Breaking (SSB)",
            "P0R01728: The 15-layer architecture emerges from the Source-Field (L13) through a process of Dimensional Descent and a cascade of Sequential Spontaneous Symmetry Breaking (SSB) events. (via a cascade of SSB events (Dimensional Descent), from the Primordial Break (L13->L15) to the Biological Break (L5->L1).) This chapter outlines the foundational principles of the SCPN, from its axiomatic origins to its physical manifestation and interaction with the known laws of physics.",
        ),
        "test_protocols": (
            "preserve The Genesis of the Hierarchy: Sequential Symmetry Breaking (SSB) source-accounting boundary",
        ),
        "null_results": (
            "The Genesis of the Hierarchy: Sequential Symmetry Breaking (SSB) is not empirical validation evidence",
        ),
        "variables": ("the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb",),
        "validation_targets": ("preserve records P0R01727-P0R01728",),
        "null_controls": (
            "the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb must remain source-bounded accounting",
        ),
    },
    "the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb.1_the_primordial_state_l13": {
        "context_id": "1_the_primordial_state_l13",
        "validation_protocol": "paper0.the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb.1_the_primordial_state_l13",
        "canonical_statement": "The source-bounded component '1. The Primordial State (L13):' preserves Paper 0 records P0R01729-P0R01730 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01729:1_the_primordial_state_l13",
            "P0R01730:1_the_primordial_state_l13",
        ),
        "source_formulae": (
            "P0R01729: 1. The Primordial State (L13):",
            "P0R01730: The Source-Field is characterised by maximum symmetry (GSource).",
        ),
        "test_protocols": ("preserve 1. The Primordial State (L13): source-accounting boundary",),
        "null_results": ("1. The Primordial State (L13): is not empirical validation evidence",),
        "variables": ("1_the_primordial_state_l13",),
        "validation_targets": ("preserve records P0R01729-P0R01730",),
        "null_controls": ("1_the_primordial_state_l13 must remain source-bounded accounting",),
    },
    "the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb.2_the_ssb_cascade_the_projection": {
        "context_id": "2_the_ssb_cascade_the_projection",
        "validation_protocol": "paper0.the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb.2_the_ssb_cascade_the_projection",
        "canonical_statement": "The source-bounded component '2. The SSB Cascade (The Projection):' preserves Paper 0 records P0R01731-P0R01733 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01731:2_the_ssb_cascade_the_projection",
            "P0R01732:2_the_ssb_cascade_the_projection",
            "P0R01733:2_the_ssb_cascade_the_projection",
        ),
        "source_formulae": (
            "P0R01731: 2. The SSB Cascade (The Projection):",
            "P0R01732: As the system evolves, the field acquires a Vacuum Expectation Value (VEV), breaking the original symmetry.",
            "P0R01733: G_SourceSSB1G_MetaSSB2G_CosmicSSB3...SSBNG_Biological",
        ),
        "test_protocols": (
            "preserve 2. The SSB Cascade (The Projection): source-accounting boundary",
        ),
        "null_results": (
            "2. The SSB Cascade (The Projection): is not empirical validation evidence",
        ),
        "variables": ("2_the_ssb_cascade_the_projection",),
        "validation_targets": ("preserve records P0R01731-P0R01733",),
        "null_controls": (
            "2_the_ssb_cascade_the_projection must remain source-bounded accounting",
        ),
    },
    "the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb.key_ssb_events": {
        "context_id": "key_ssb_events",
        "validation_protocol": "paper0.the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb.key_ssb_events",
        "canonical_statement": "The source-bounded component 'Key SSB Events:' preserves Paper 0 records P0R01734-P0R01754 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01734:key_ssb_events",
            "P0R01735:key_ssb_events",
            "P0R01736:key_ssb_events",
            "P0R01737:key_ssb_events",
            "P0R01738:key_ssb_events",
            "P0R01739:key_ssb_events",
            "P0R01740:key_ssb_events",
            "P0R01741:key_ssb_events",
            "P0R01742:key_ssb_events",
            "P0R01743:key_ssb_events",
            "P0R01744:key_ssb_events",
            "P0R01745:key_ssb_events",
            "P0R01746:key_ssb_events",
            "P0R01747:key_ssb_events",
            "P0R01748:key_ssb_events",
            "P0R01749:key_ssb_events",
            "P0R01750:key_ssb_events",
            "P0R01751:key_ssb_events",
            "P0R01752:key_ssb_events",
            "P0R01753:key_ssb_events",
            "P0R01754:key_ssb_events",
        ),
        "source_formulae": (
            "P0R01734: Key SSB Events:",
            "P0R01735: SSB 1 (L15/L14): The Ethical Functionals (E, L15) break the primordial symmetry, selecting the physical laws (L14). | SSB 2 (Collective -> Individual L11 -> L5): Breaking the symmetry of the collective field, leading to the localisation of individual Selves (L5 solitons) via the Mexican Hat potential.",
            "P0R01736: Self-Emergence Trigger (Wilson Loop Crossover) - The Wilson Loop Crossover Trigger",
            'P0R01737: The L4 $\\to$ L5 phase transition occurs when the $\\Psi$-field connection transitions from a confined "Area Law" to a deconfined "Perimeter Law" (The Colored Self state):',
            'P0R01738: phase_transition = "L5_Self_Emergence" if wilson_loop_val < exp(-k * perimeter_length) else "L11_Collective"',
            "P0R01739: Legend of Equation Components:",
            "P0R01740: wilson_loop_val: The expectation value of the $\\Psi$-gauge connection around a closed path. | perimeter_length: The characteristic scale of the neural/informational loop. | k: The effective string tension of the informational force.",
            "P0R01741: P0R01741",
            "P0R01742: P0R01742",
            "P0R01743: Wilson Loop Area-Law Crossover The Confinement-Self Phase Boundary",
            "P0R01744: The transition from the confined collective field (L11) to the deconfined Macroscopic Colored state (L5 Self) is defined by the Wilson Loop ($W_C$) crossover:",
            "P0R01745: v_eff_L = sigma_q * L**2 if L > r_confine else k_q * L",
            "P0R01746: Legend of Equation Components:",
            "P0R01747: v_eff_L: Effective potential between internal qualia charges. | sigma_q: String tension of the $SU(N)$ info-gluon field. | L: Characteristic scale of the subjective experience. | r_confine: Confinement radius (defining the boundary of the individuated Self). | k_q: Effective coupling constant in the deconfined (Self) phase.",
            "P0R01748: SSB 3 (Potentiality -> Actuality L1): The collapse of the wavefunction (Measurement Postulate), breaking unitary symmetry.",
            "P0R01749: The SCPN architecture is the resulting structure of these symmetry breakings. The UPDE describes the dynamics of the resulting Goldstone modes, which mediate long-range coherence.",
            "P0R01750: [IMAGE:Ein Bild, das Text, Elektronik, Screenshot, Software enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R01751: Fig.: The Genesis of the Hierarchy (Three SSB Stages). This flowchart visualizes the core concept of the chapter: how the universe's architecture emerges from a state of maximum symmetry through a cascade of symmetry-breaking events. Each break gives rise to new structures and laws.",
            "P0R01752: Starting from the Source-Field (L13) at maximum symmetry GSourceG_{\\text{Source}}GSource, the hierarchy emerges through three breaks: SSB-1 (Primordial Break; L15/L14): Ethical functionals select a specific vacuum, breaking primordial symmetry and fixing the effective physical laws. SSB-2 (Collective -> Individual; L11->L5): The field acquires a non-zero VEV (Mexican-hat potential), localising Selves as solitonic condensates within the SCPN. SSB-3 (Potentiality -> Actuality; L1): Measurement/coupling breaks unitary symmetry operationally, effecting wavefunction collapse at the organismal/quantum-biological interface. The chain L1-L5 depicts the differentiated lower layers of the SCPN architecture that crystallise post-break.",
            "P0R01753: [IMAGE:Ein Bild, das Text, Diagramm, Schrift, Quittung enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R01754: P0R01754",
        ),
        "test_protocols": ("preserve Key SSB Events: source-accounting boundary",),
        "null_results": ("Key SSB Events: is not empirical validation evidence",),
        "variables": ("key_ssb_events",),
        "validation_targets": ("preserve records P0R01734-P0R01754",),
        "null_controls": ("key_ssb_events must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbSpec:
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
class TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbSpec, ...]
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


def build_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_specs(
    source_records: list[dict[str, Any]],
) -> TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbSpecBundle:
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

    specs: list[TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbSpec(
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
        "title": "Paper 0 The Genesis of the Hierarchy: Sequential Symmetry Breaking (SSB) Specs",
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
        "next_source_boundary": "P0R01755",
    }
    return TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_specs(
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


def render_report(bundle: TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 The Genesis of the Hierarchy: Sequential Symmetry Breaking (SSB) Specs",
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
    bundle: TheGenesisOfTheHierarchySequentialSymmetryBreakingSsbSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_genesis_of_the_hierarchy_sequential_symmetry_breaking_ssb_validation_specs_{date_tag}.md"
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
