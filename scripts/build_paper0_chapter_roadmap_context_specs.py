#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 chapter roadmap context spec builder
"""Promote Paper 0 chapter roadmap context records."""

from __future__ import annotations

import argparse
import json
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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(105, 218))
BLANK_MARKER_IDS = ("P0R00133", "P0R00153", "P0R00184", "P0R00200", "P0R00217")
CLAIM_BOUNDARY = "source-bounded chapter roadmap context; not validation evidence"
HARDWARE_STATUS = "source_context_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "chapter_roadmap_context.axiomatic_foundations": {
        "roadmap_id": "axiomatic_foundations",
        "validation_protocol": "paper0.chapter_roadmap_context.axiomatic_foundations",
        "canonical_statement": (
            "Part I maps field architecture, the SCPN mandate, the three axioms, "
            "category grammar, and tripartite information ontology."
        ),
        "source_equation_ids": (
            "P0R00105-P0R00111:chapter1_foundational_paradigm",
            "P0R00112-P0R00121:three_axioms_roadmap",
            "P0R00122-P0R00133:category_and_information_ontology_roadmap",
        ),
        "source_formulae": (
            "Primacy of Consciousness",
            "Fisher Information Metric",
            "teleological optimisation",
            "category-theoretic framework",
            "Experiential Information, Geometric/Semantic Information, Syntactic Information",
        ),
        "test_protocols": (
            "preserve Part I targets as roadmap entries requiring later source claims",
        ),
        "null_results": ("Part I ToC entries are not equations or empirical evidence",),
        "variables": ("Psi", "FIM", "SEC", "Phi", "G", "H"),
        "validation_targets": (
            "preserve axiomatic-foundation roadmap",
            "preserve FIM and teleological optimisation roadmap entries",
            "reject ToC headings as validation evidence",
        ),
        "null_controls": (
            "toc-heading-as-equation control must be rejected",
            "missing-axiom-roadmap control must be rejected",
        ),
    },
    "chapter_roadmap_context.psi_field_physics": {
        "roadmap_id": "psi_field_physics",
        "validation_protocol": "paper0.chapter_roadmap_context.psi_field_physics",
        "canonical_statement": (
            "Part II maps the Psi-field gauge theory, FIM bridge, ALP interface, "
            "Mexican-hat potential, symmetry breaking, and solitonic self topics."
        ),
        "source_equation_ids": (
            "P0R00134-P0R00143:gauge_fim_roadmap",
            "P0R00144-P0R00153:alp_symmetry_soliton_roadmap",
        ),
        "source_formulae": (
            "U(1) Gauge Principle",
            "Master Interaction Lagrangian",
            "Fisher Information Metric",
            "Mexican Hat Potential",
            "Spontaneous Symmetry Breaking",
            "Self as a Soliton",
        ),
        "test_protocols": ("preserve physics roadmap entries without promoting derived formulae",),
        "null_results": ("physics ToC entries are not derivations",),
        "variables": ("Psi", "U1", "FIM", "ALP", "SSB"),
        "validation_targets": (
            "preserve gauge-theory roadmap",
            "preserve FIM bridge roadmap",
            "preserve symmetry-breaking and soliton roadmap",
        ),
        "null_controls": (
            "toc-as-lagrangian-derivation control must be rejected",
            "missing-FIM-bridge control must be rejected",
        ),
    },
    "chapter_roadmap_context.dynamic_spine": {
        "roadmap_id": "dynamic_spine",
        "validation_protocol": "paper0.chapter_roadmap_context.dynamic_spine",
        "canonical_statement": (
            "Part III maps the 15+1 architecture, UPDE spine, quasicritical controller, "
            "SOC signatures, and MS-QEC coherence backbone."
        ),
        "source_equation_ids": (
            "P0R00154-P0R00160:architecture_roadmap",
            "P0R00161-P0R00166:upde_roadmap",
            "P0R00167-P0R00175:quasicriticality_roadmap",
            "P0R00176-P0R00183:ms_qec_roadmap",
        ),
        "source_formulae": (
            "15 Layers and 6 Domains",
            "UPDE",
            "Information-Geometric Lift",
            "Quasicriticality",
            "Self-Organised Criticality",
            "Multi-Scale Quantum Error Correction",
        ),
        "test_protocols": (
            "preserve architecture, UPDE, quasicriticality, and MS-QEC as later validation targets",
        ),
        "null_results": ("dynamic-spine ToC entries are not runtime evidence",),
        "variables": ("UPDE", "SOC", "MS_QEC", "L1_L16"),
        "validation_targets": (
            "preserve UPDE roadmap",
            "preserve quasicritical and SOC roadmap",
            "preserve MS-QEC roadmap",
        ),
        "null_controls": (
            "toc-as-UPDE-implementation control must be rejected",
            "missing-MS-QEC-roadmap control must be rejected",
        ),
    },
    "chapter_roadmap_context.experience_engine": {
        "roadmap_id": "experience_engine",
        "validation_protocol": "paper0.chapter_roadmap_context.experience_engine",
        "canonical_statement": (
            "Part IV maps predictive coding, free energy, active inference, geometric qualia, "
            "IIT integration, TDA, and qualia-capacity topics."
        ),
        "source_equation_ids": (
            "P0R00185-P0R00190:hpc_free_energy_roadmap",
            "P0R00191-P0R00199:qualia_geometry_roadmap",
        ),
        "source_formulae": (
            "Hierarchical Predictive Coding",
            "Free Energy Principle",
            "Active Inference",
            "Geometric Qualia Hypothesis",
            "Integrated Information Theory",
            "Topological Data Analysis",
        ),
        "test_protocols": (
            "preserve experience-engine roadmap without asserting solved hard problem",
        ),
        "null_results": ("experience ToC entries are not qualia measurements",),
        "variables": ("HPC", "FEP", "IIT", "TDA", "Q"),
        "validation_targets": (
            "preserve HPC/FEP roadmap",
            "preserve geometric qualia roadmap",
            "preserve TDA and qualia-capacity roadmap",
        ),
        "null_controls": (
            "toc-as-qualia-measurement control must be rejected",
            "missing-active-inference control must be rejected",
        ),
    },
    "chapter_roadmap_context.teleology_and_closure": {
        "roadmap_id": "teleology_and_closure",
        "validation_protocol": "paper0.chapter_roadmap_context.teleology_and_closure",
        "canonical_statement": (
            "Part V maps causal entropy, ethical coherence, Consilium, PELA, "
            "Meta-Layer 16, recursive optimisation Hamiltonian, and Anulum closure."
        ),
        "source_equation_ids": (
            "P0R00201-P0R00209:teleology_ethical_functional_roadmap",
            "P0R00210-P0R00215:recursive_closure_roadmap",
        ),
        "source_formulae": (
            "Causal Entropic Forces",
            "Principle of Ethical Least Action",
            "Meta-Layer 16",
            "Recursive Optimisation Hamiltonian",
            "self-observing loop",
        ),
        "test_protocols": ("preserve teleology and closure roadmap as later validation targets",),
        "null_results": ("teleology ToC entries are not equivalence proofs",),
        "variables": ("CEF", "PELA", "Meta_Layer_16", "H_rec"),
        "validation_targets": (
            "preserve causal entropy roadmap",
            "preserve ethical functional roadmap",
            "preserve recursive closure roadmap",
        ),
        "null_controls": (
            "toc-as-equivalence-proof control must be rejected",
            "missing-H-rec-roadmap control must be rejected",
        ),
    },
    "chapter_roadmap_context.empirical_trajectory_boundary": {
        "roadmap_id": "empirical_trajectory_boundary",
        "validation_protocol": "paper0.chapter_roadmap_context.empirical_trajectory_boundary",
        "canonical_statement": (
            "The roadmap ends this slice at Chapter 18, which names falsifiable predictions "
            "and empirical trajectories but does not supply the protocols yet."
        ),
        "source_equation_ids": (
            "P0R00216:chapter18_falsifiable_predictions",
            "P0R00217:blank_section_break",
        ),
        "source_formulae": ("Falsifiable Predictions and Empirical Trajectories",),
        "test_protocols": ("preserve Chapter 18 as the next empirical-trajectory boundary",),
        "null_results": ("Chapter 18 title alone is not an empirical protocol",),
        "variables": ("falsifiable_predictions", "empirical_trajectories"),
        "validation_targets": (
            "preserve empirical trajectory boundary",
            "prevent title-only protocol promotion",
        ),
        "null_controls": (
            "chapter-title-as-protocol control must be rejected",
            "missing-empirical-boundary control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ChapterRoadmapContextSpec:
    """Chapter roadmap context spec promoted from Paper 0 records."""

    key: str
    roadmap_id: str
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
class ChapterRoadmapContextSpecBundle:
    """Chapter roadmap context specs plus source coverage summary."""

    specs: tuple[ChapterRoadmapContextSpec, ...]
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


def build_chapter_roadmap_context_specs(
    source_records: list[dict[str, Any]],
) -> ChapterRoadmapContextSpecBundle:
    """Build source-covered chapter roadmap context specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    blank_marker_count = sum(
        1 for ledger_id in BLANK_MARKER_IDS if not str(records_by_ledger[ledger_id]["text"])
    )
    specs: list[ChapterRoadmapContextSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ChapterRoadmapContextSpec(
                key=key,
                roadmap_id=str(metadata["roadmap_id"]),
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
                implementation_status="implemented_executable_fixture",
                domain_review_status="source_context_preserved",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed_ids = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 Chapter Roadmap Context Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed_ids),
        "coverage_match": tuple(consumed_ids) == SOURCE_LEDGER_IDS,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "part_count": 5,
        "chapter_count": 18,
        "blank_marker_count": blank_marker_count,
        "numbering_inconsistency_present": True,
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed_ids
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
    }
    return ChapterRoadmapContextSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(path: Path = DEFAULT_LEDGER_PATH) -> ChapterRoadmapContextSpecBundle:
    """Build chapter roadmap context specs from the canonical ledger."""
    return build_chapter_roadmap_context_specs(load_jsonl(path))


def write_outputs(
    bundle: ChapterRoadmapContextSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown chapter roadmap context spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_chapter_roadmap_context_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_chapter_roadmap_context_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "specs": [asdict(spec) for spec in bundle.specs],
        "summary": bundle.summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: ChapterRoadmapContextSpecBundle) -> str:
    """Render a compact Markdown report for human review."""
    lines = [
        "# Paper 0 Chapter Roadmap Context Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Parts: {bundle.summary['part_count']}",
        f"- Chapters: {bundle.summary['chapter_count']}",
        f"- Blank markers: {bundle.summary['blank_marker_count']}",
        f"- Numbering inconsistency present: {bundle.summary['numbering_inconsistency_present']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.append(f"- `{spec.key}`: {spec.canonical_statement}")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build and write Paper 0 chapter roadmap context validation specs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0 if bundle.summary["coverage_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
