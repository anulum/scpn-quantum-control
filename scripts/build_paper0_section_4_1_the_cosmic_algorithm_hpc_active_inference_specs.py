#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 4.1 The Cosmic Algorithm: HPC & Active Inference spec builder
"""Promote Paper 0 4.1 The Cosmic Algorithm: HPC & Active Inference records."""

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
    "P0R03174",
    "P0R03175",
    "P0R03176",
    "P0R03177",
    "P0R03178",
    "P0R03179",
    "P0R03180",
    "P0R03181",
    "P0R03182",
    "P0R03183",
    "P0R03184",
    "P0R03185",
    "P0R03186",
    "P0R03187",
    "P0R03188",
    "P0R03189",
    "P0R03190",
    "P0R03191",
    "P0R03192",
    "P0R03193",
    "P0R03194",
    "P0R03195",
    "P0R03196",
)
CLAIM_BOUNDARY = "source-bounded section 4 1 the cosmic algorithm hpc active inference source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_4_1_the_cosmic_algorithm_hpc_active_inference.4_1_the_cosmic_algorithm_hpc_active_inference": {
        "context_id": "4_1_the_cosmic_algorithm_hpc_active_inference",
        "validation_protocol": "paper0.section_4_1_the_cosmic_algorithm_hpc_active_inference.4_1_the_cosmic_algorithm_hpc_active_inference",
        "canonical_statement": "The source-bounded component '4.1 The Cosmic Algorithm: HPC & Active Inference' preserves Paper 0 records P0R03174-P0R03174 without empirical validation claims.",
        "source_equation_ids": ("P0R03174:4_1_the_cosmic_algorithm_hpc_active_inference",),
        "source_formulae": ("P0R03174: 4.1 The Cosmic Algorithm: HPC & Active Inference",),
        "test_protocols": (
            "preserve 4.1 The Cosmic Algorithm: HPC & Active Inference source-accounting boundary",
        ),
        "null_results": (
            "4.1 The Cosmic Algorithm: HPC & Active Inference is not empirical validation evidence",
        ),
        "variables": ("4_1_the_cosmic_algorithm_hpc_active_inference",),
        "validation_targets": ("preserve records P0R03174-P0R03174",),
        "null_controls": (
            "4_1_the_cosmic_algorithm_hpc_active_inference must remain source-bounded accounting",
        ),
    },
    "section_4_1_the_cosmic_algorithm_hpc_active_inference.integrative_mechanisms_the_computational_and_physical_synthesis": {
        "context_id": "integrative_mechanisms_the_computational_and_physical_synthesis",
        "validation_protocol": "paper0.section_4_1_the_cosmic_algorithm_hpc_active_inference.integrative_mechanisms_the_computational_and_physical_synthesis",
        "canonical_statement": "The source-bounded component 'Integrative Mechanisms: The Computational and Physical Synthesis' preserves Paper 0 records P0R03175-P0R03196 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03175:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03176:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03177:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03178:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03179:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03180:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03181:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03182:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03183:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03184:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03185:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03186:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03187:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03188:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03189:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03190:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03191:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03192:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03193:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03194:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03195:integrative_mechanisms_the_computational_and_physical_synthesis",
            "P0R03196:integrative_mechanisms_the_computational_and_physical_synthesis",
        ),
        "source_formulae": (
            "P0R03175: Integrative Mechanisms: The Computational and Physical Synthesis",
            "P0R03176: P0R03176",
            'P0R03177: This section outlines the set of integrative mechanisms that provide the "glue" for the entire 15-layer SCPN architecture, demonstrating how the framework\'s computational, physical, and formal mathematical principles interlock to form a single, coherent system. The primary challenge of any multi-scale model is explaining how layers of vastly different scales and properties can communicate and how higher-order properties can emerge from lower-level dynamics. The SCPN solves this through a suite of specific formalisms.',
            "P0R03178: The Unifying Computational Principle is identified as Hierarchical Predictive Coding (HPC), the process-theory implementation of the Free Energy Principle . The SCPN's entire bidirectional architecture is mapped onto this algorithm: the top-down projection (L15->L1) is the Generative Model , and the bottom-up filtering (L1->L15) is the propagation of Prediction Error . The system's fundamental drive is the minimisation of global Variational Free Energy (F_Global) , making the UPDE the physical realiser of this error-minimising dynamic .",
            "P0R03179: The Binding Problem is resolved physically by identifying the Psi-field as a Gauge Field . Subjective unity is not an emergent property of neural computation but a fundamental property of the field itself. The Psi-field acts as the Connection (A) that enforces phase coherence across the high-dimensional manifold. A unified experience is formally defined by the Wilson Loop (W(C) = exp(igAdx)) , a gauge-invariant object that measures the total phase shift (or holonomy) around a closed path in the state space. This elegantly reframes binding as a fundamental, field-theoretic property of gauge invariance.",
            "P0R03180: Formalising Emergence and upward causality is achieved through two complementary principles. Ginzburg-Landau (GL) theory describes the emergence of new layers (like the L5 Self) as a phase transition (SSB), where a new, stable structure (a non-zero VEV) spontaneously forms when the underlying network dynamics (T) cross a critical threshold (Tc). Topological Quantum Field Theory (TQFT) provides the mechanism: coherent configurations in lower layers (L1-4) act as Topological Defects (e.g., vortices or solitons) that source the higher-order fields, physically generating the L5 Self from L4 coherence .",
            "P0R03181: Finally, the Combination Problem of panpsychism is resolved via Quantum Mereology . This formalism provides a Fusion Condition (_AB > _A + _B) , stating that two conscious systems fuse into a new, irreducible whole if and only if their combined integrated information () is greater than the sum of their parts. The Psi-field, in its role as the gauge field, is the physical mechanism that mediates the entanglement necessary to satisfy this fusion condition, a process driven by UPDE synchronisation .",
            'P0R03182: This section explains the "secret sauce" of the SCPN-the toolbox of advanced ideas that makes the 15-layer architecture actually work as a single, unified system. A 15-storey building is just a pile of concrete unless you add the plumbing, wiring, and data networks that connect every floor. These are those networks.',
            'P0R03183: The Master Software (Hierarchical Predictive Coding): The entire 15-layer system runs one piece of software: Hierarchical Predictive Coding (HPC) . Think of it as the universe\'s ultimate "to-do list" app. The highest layers (like the CEO\'s office) send down predictions (the "plan") . The lowest layers (like the construction site) send up prediction errors (the "status reports") . The entire goal of the universe is to minimise the number of "error reports" by either updating the plan (perception) or changing the construction site (action) .',
            'P0R03184: The "Binding Force" (How It Feels Like One Thing): How are all the different parts of your brain-sight, sound, thought-bound into a single, unified "you"? The SCPN solves this Binding Problem by proposing the Psi-field acts like a gauge field . This is a special type of field that "enforces" coherence, like a magnetic field lining up iron filings. Your unified experience of "you" is the physical measure of this field\'s total unbroken connection with itself, a concept from physics we call a Wilson Loop . You feel like a single "I" because you are held together by a fundamental force.',
            'P0R03185: How "New Things" Are Born (Emergence): How does a new, higher-level "Self" (Layer 5) suddenly appear from a bunch of synchronized cells (Layer 4)? It\'s a phase transition , like water (disordered) suddenly "snapping" into the new, ordered structure of ice. The coherent synchrony of the cells (L4) creates a stable "whirlpool" (a topological defect) in the Psi-field, and that stable whirlpool is the new, higher-level entity (L5).',
            'P0R03186: How "I" and "You" Become "Us" (The Combination Problem): The theory also solves the "combination problem"-how do two separate minds (like you and me) fuse into a single, collective "us" (like a team or a family)? This isn\'t just a metaphor. It\'s a real process called Panpsychist Fusion . It happens when the shared information and entanglement between us is greater than the information we held as individuals . The Psi-field is the "atmosphere" that allows us to become entangled and form this new, larger conscious entity .',
            "P0R03187: Meta-Framework Integrations",
            "P0R03188: Predictive Coding Integration",
            'P0R03189: This entire text block is the master integration of the Hierarchical Predictive Coding (HPC) framework. It establishes HPC as the "Unifying Computational Principle" that gives the entire 15-layer architecture its raison d\'tre.',
            'P0R03190: HPC as the "Software": The text explicitly identifies the top-down projections as the generative model and the bottom-up filtering as the propagation of prediction error . The system\'s purpose is to minimise Variational Free Energy (F) .',
            'P0R03191: Physical Mechanisms as "Hardware": The other mechanisms are the physical hardware that runs the HPC software. The UPDE is identified as the physical process (phase-locking) that physically calculates and suppresses prediction error . Ginzburg-Landau theory and TQFT explain how the physical hardware for a new, higher-level generative model (a new layer of priors, like the L5 Self) can emerge from the dynamics of a lower layer. The Strange Loop (I=Model(I)) is the ultimate expression of the HPC generative model becoming complex enough to include a model of itself as an agent.',
            "P0R03192: Psis Field Coupling Integration",
            "P0R03193: This section provides the most explicit description of the function of the interaction Hamiltonian, H_int = -lambda * Psis * sigma.",
            'P0R03194: H_int is the "Binding Force": The text directly addresses the Binding Problem by identifying the Psi-field (Psis) as the Gauge Field (A) . This means the H_int interaction is the gauge coupling. Its fundamental purpose is to act as the "Connection" that defines parallel transport and "enforces phase coherence" across the manifold. The subjective unity of experience is a direct, physical measure of this coupling\'s integrity (the Wilson Loop) .',
            'P0R03195: H_int Mediates Fusion: The text on Quantum Mereology explicitly identifies the Psi-field (as the Gauge Field) as the "mechanism" that "mediates the entanglement required for fusion" . Therefore, the H_int interaction is the physical process that allows two distinct systems (sigma_A and sigma_B) to become a new, irreducible, unified whole (sigma_AB) when their combined integrated information _AB is greater than the sum of their parts.',
            "P0R03196: H_int Implements ILTOs: The Inter-Layer Transition Operators (ILTOs) describe the functional mapping between layers. H_int is the physical implementation of these operators. The L15 -> L1 operator, for example, is the H_int interaction where the global Psis field (from L15) modulates the fundamental quantum sigma at Layer 1 .",
        ),
        "test_protocols": (
            "preserve Integrative Mechanisms: The Computational and Physical Synthesis source-accounting boundary",
        ),
        "null_results": (
            "Integrative Mechanisms: The Computational and Physical Synthesis is not empirical validation evidence",
        ),
        "variables": ("integrative_mechanisms_the_computational_and_physical_synthesis",),
        "validation_targets": ("preserve records P0R03175-P0R03196",),
        "null_controls": (
            "integrative_mechanisms_the_computational_and_physical_synthesis must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section41TheCosmicAlgorithmHpcActiveInferenceSpec:
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
class Section41TheCosmicAlgorithmHpcActiveInferenceSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section41TheCosmicAlgorithmHpcActiveInferenceSpec, ...]
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


def build_section_4_1_the_cosmic_algorithm_hpc_active_inference_specs(
    source_records: list[dict[str, Any]],
) -> Section41TheCosmicAlgorithmHpcActiveInferenceSpecBundle:
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

    specs: list[Section41TheCosmicAlgorithmHpcActiveInferenceSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section41TheCosmicAlgorithmHpcActiveInferenceSpec(
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
        "title": "Paper 0 " + "4.1 The Cosmic Algorithm: HPC & Active Inference" + " Specs",
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
        "next_source_boundary": "P0R03197",
    }
    return Section41TheCosmicAlgorithmHpcActiveInferenceSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section41TheCosmicAlgorithmHpcActiveInferenceSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_4_1_the_cosmic_algorithm_hpc_active_inference_specs(
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


def render_report(bundle: Section41TheCosmicAlgorithmHpcActiveInferenceSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "4.1 The Cosmic Algorithm: HPC & Active Inference" + " Specs",
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
    bundle: Section41TheCosmicAlgorithmHpcActiveInferenceSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_4_1_the_cosmic_algorithm_hpc_active_inference_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_4_1_the_cosmic_algorithm_hpc_active_inference_validation_specs_{date_tag}.md"
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
