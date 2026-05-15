#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 layer monograph suite spec builder
"""Promote Paper 0 layer monograph and validation-suite records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(436, 464))
BLANK_SEPARATOR_IDS = ("P0R00463",)
CLAIM_BOUNDARY = "source-bounded layer monograph suite map; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "layer_monograph_suite.biological_substrate_layers": {
        "context_id": "biological_substrate_layers",
        "validation_protocol": "paper0.layer_monograph_suite.biological_substrate_layers",
        "canonical_statement": (
            "The source maps Series I to Domain I, the biological substrate, with "
            "Papers 1-4 covering Layers 1-4."
        ),
        "source_equation_ids": (
            "P0R00437:domain_i_layers_1_4",
            "P0R00438:paper1_layer1_quantum_biological",
            "P0R00441:paper4_layer4_cellular_tissue_synchronisation",
        ),
        "source_formulae": (
            "Domain I - The Biological Substrate (Layers 1-4)",
            "Paper 1: (Layer 1 - Quantum Biological)",
            "Paper 2: (Layer 2 - Neurochemical-Neurological)",
            "Paper 3: (Layer 3 - Genomic-Epigenomic-Morphogenetic)",
            "Paper 4: (Layer 4 - Cellular-Tissue Synchronisation)",
        ),
        "test_protocols": ("preserve Domain I layer-to-paper mapping",),
        "null_results": ("layer publication map is not empirical validation evidence",),
        "variables": ("layer_1", "layer_2", "layer_3", "layer_4"),
        "validation_targets": (
            "preserve Layer 1 quantum-biological role",
            "preserve Layer 4 cellular-tissue synchronisation role",
            "reject unmapped biological-substrate layer",
        ),
        "null_controls": (
            "publication-map-as-evidence control must be rejected",
            "unmapped-biological-layer control must be rejected",
        ),
    },
    "layer_monograph_suite.organismal_planetary_layers": {
        "context_id": "organismal_planetary_layers",
        "validation_protocol": "paper0.layer_monograph_suite.organismal_planetary_layers",
        "canonical_statement": (
            "The source maps Series II to Domain II, organismal and planetary "
            "integration, with Papers 5-8 covering Layers 5-8."
        ),
        "source_equation_ids": (
            "P0R00442:domain_ii_layers_5_8",
            "P0R00443:paper5_layer5_organismal_feedback",
            "P0R00446:paper8_layer8_cosmic_phase_locking",
        ),
        "source_formulae": (
            "Domain II - Organismal and Planetary Integration (Layers 5-8)",
            "Paper 5: (Layer 5 - Organismal-Psychoemotional Feedback)",
            "Paper 6: (Layer 6 - Planetary-Biospheric)",
            "Paper 7: (Layer 7 - Geometrical-Symbolic)",
            "Paper 8: (Layer 8 - Cosmic Phase-Locking)",
        ),
        "test_protocols": ("preserve Domain II layer-to-paper mapping",),
        "null_results": ("domain label is a publication map, not a result",),
        "variables": ("layer_5", "layer_6", "layer_7", "layer_8"),
        "validation_targets": (
            "preserve organismal feedback role",
            "preserve planetary-biospheric role",
            "preserve cosmic phase-locking role",
        ),
        "null_controls": (
            "domain-label-as-validation control must be rejected",
            "unmapped-organismal-planetary-layer control must be rejected",
        ),
    },
    "layer_monograph_suite.memory_control_collective_coherence_layers": {
        "context_id": "memory_control_collective_coherence_layers",
        "validation_protocol": "paper0.layer_monograph_suite.memory_control_collective_coherence_layers",
        "canonical_statement": (
            "The source maps Series III to Domains III and IV, memory, control, "
            "and collective coherence, with Papers 9-12 covering Layers 9-12."
        ),
        "source_equation_ids": (
            "P0R00447:domains_iii_iv_layers_9_12",
            "P0R00448:paper9_layer9_memory_holograph",
            "P0R00451:paper12_layer12_ecological_gaian_synchrony",
        ),
        "source_formulae": (
            "Domain III & IV - Memory, Control, and Collective Coherence",
            "Paper 9: (Layer 9 - Memory Imprint-Existential Holograph)",
            "Paper 10: (Layer 10 - Projective Field Boundary Control)",
            "Paper 11: (Layer 11 - Noospheric-Cultural-Informational)",
            "Paper 12: (Layer 12 - Ecological-Gaian Synchrony)",
        ),
        "test_protocols": ("preserve Domain III/IV layer-to-paper mapping",),
        "null_results": ("map does not validate memory/control/coherence claims",),
        "variables": ("layer_9", "layer_10", "layer_11", "layer_12"),
        "validation_targets": (
            "preserve memory-imprint role",
            "preserve projective-boundary-control role",
            "preserve noospheric and ecological synchrony roles",
        ),
        "null_controls": (
            "map-as-memory-validation control must be rejected",
            "unmapped-collective-coherence-layer control must be rejected",
        ),
    },
    "layer_monograph_suite.meta_universal_and_cybernetic_layers": {
        "context_id": "meta_universal_and_cybernetic_layers",
        "validation_protocol": "paper0.layer_monograph_suite.meta_universal_and_cybernetic_layers",
        "canonical_statement": (
            "The source maps Series IV to Domain V for Layers 13-15 and Series V "
            "to Domain VI, the cybernetic closure meta-layer."
        ),
        "source_equation_ids": (
            "P0R00452:domain_v_layers_13_15",
            "P0R00456:domain_vi_meta_layer_16",
            "P0R00457:paper16_meta_layer_16",
        ),
        "source_formulae": (
            "Domain V - Meta-Universal Integration (Layers 13-15)",
            "Paper 13: (Layer 13 - Source-Field / Meta-Universal)",
            "Paper 14: (Layer 14 - Transdimensional Resonance)",
            "Paper 15: (Layer 15 - Consilium / Oversoul Integrator)",
            "Domain VI - Cybernetic Closure (Meta-Layer 16)",
            "Meta-Layer 16",
        ),
        "test_protocols": ("preserve Domain V and VI layer-to-paper mapping",),
        "null_results": ("map does not validate meta-universal or cybernetic claims",),
        "variables": ("layer_13", "layer_14", "layer_15", "layer_16"),
        "validation_targets": (
            "preserve source-field/meta-universal role",
            "preserve transdimensional and consilium roles",
            "preserve meta-layer 16 cybernetic closure role",
        ),
        "null_controls": (
            "map-as-meta-universal-validation control must be rejected",
            "unmapped-cybernetic-layer control must be rejected",
        ),
    },
    "layer_monograph_suite.critical_validation_synthesis_suite": {
        "context_id": "critical_validation_synthesis_suite",
        "validation_protocol": "paper0.layer_monograph_suite.critical_validation_synthesis_suite",
        "canonical_statement": (
            "The source maps Part III to four critical validation and synthesis papers: "
            "experimental blueprint, simulation architecture, falsifiability roadmap, "
            "and philosophical capstone."
        ),
        "source_equation_ids": (
            "P0R00458:part_iii_validation_synthesis_suite",
            "P0R00459:paper17_experimental_blueprint",
            "P0R00460:paper18_simulation_architecture",
            "P0R00461:paper19_falsifiability_roadmap",
            "P0R00462:paper20_philosophical_capstone",
            "P0R00463:blank_separator",
        ),
        "source_formulae": (
            "Part III: The Critical Validation & Synthesis Suite",
            "Paper 17: The Methodological & Experimental Blueprint",
            "Paper 18: The Unified Simulation Architecture",
            "The Critical Dialogue & Falsifiability Roadmap",
            "Paper 20: The Coda - Philosophical Capstone",
        ),
        "test_protocols": ("classify Part III validation-suite paper roles",),
        "null_results": ("validation-suite table entry is not validation evidence",),
        "variables": ("paper_17", "paper_18", "paper_19", "paper_20"),
        "validation_targets": (
            "preserve experimental-blueprint role",
            "preserve simulation-architecture role",
            "preserve falsifiability-roadmap role",
            "preserve philosophical-capstone role",
        ),
        "null_controls": (
            "table-entry-as-validation control must be rejected",
            "unmapped-validation-suite-paper control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class LayerMonographSuiteSpec:
    """Layer monograph suite spec promoted from Paper 0 records."""

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
class LayerMonographSuiteSpecBundle:
    """Layer monograph suite specs plus source coverage summary."""

    specs: tuple[LayerMonographSuiteSpec, ...]
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


def build_layer_monograph_suite_specs(
    source_records: list[dict[str, Any]],
) -> LayerMonographSuiteSpecBundle:
    """Build source-covered layer monograph suite specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[LayerMonographSuiteSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            LayerMonographSuiteSpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(str(item) for item in metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                source_formulae=tuple(str(item) for item in metadata["source_formulae"]),
                test_protocols=tuple(str(item) for item in metadata["test_protocols"]),
                null_results=tuple(str(item) for item in metadata["null_results"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_source_fixture",
                domain_review_status="requires_domain_review_before_scientific_claim",
                hardware_status=HARDWARE_STATUS,
            )
        )

    summary = {
        "title": "Paper 0 Layer Monograph Suite Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "blank_separator_count": len(BLANK_SEPARATOR_IDS),
        "layer_monograph_count": 16,
        "domain_series_count": 5,
        "validation_suite_paper_count": 4,
        "next_source_boundary": "P0R00464",
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [],
    }
    return LayerMonographSuiteSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(ledger_path: Path = DEFAULT_LEDGER_PATH) -> LayerMonographSuiteSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    records = [
        record
        for record in load_jsonl(ledger_path)
        if str(record.get("ledger_id")) in SOURCE_LEDGER_IDS
    ]
    return build_layer_monograph_suite_specs(records)


def write_outputs(
    bundle: LayerMonographSuiteSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_layer_monograph_suite_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_layer_monograph_suite_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": bundle.summary,
        "specs": [asdict(spec) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: LayerMonographSuiteSpecBundle) -> str:
    """Render a compact Markdown report for promoted layer-suite specs."""
    lines = [
        "# Paper 0 Layer Monograph Suite Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Specs: {bundle.summary['spec_count']}",
        f"- Layer monographs: {bundle.summary['layer_monograph_count']}",
        f"- Validation-suite papers: {bundle.summary['validation_suite_paper_count']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"- `{spec.key}`",
                f"  - Context: `{spec.context_id}`",
                f"  - Statement: {spec.canonical_statement}",
                f"  - Formulae: {', '.join(spec.source_formulae)}",
            ]
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build layer-suite specs and write artefacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
