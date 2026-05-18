#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Pseudoscalar Coupling spec builder
"""Promote Paper 0 The Pseudoscalar Coupling records."""

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
    "P0R04291",
    "P0R04292",
    "P0R04293",
    "P0R04294",
    "P0R04295",
    "P0R04296",
    "P0R04297",
    "P0R04298",
    "P0R04299",
    "P0R04300",
    "P0R04301",
    "P0R04302",
    "P0R04303",
    "P0R04304",
    "P0R04305",
    "P0R04306",
    "P0R04307",
    "P0R04308",
    "P0R04309",
)
CLAIM_BOUNDARY = (
    "source-bounded the pseudoscalar coupling source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_pseudoscalar_coupling.the_pseudoscalar_coupling": {
        "context_id": "the_pseudoscalar_coupling",
        "validation_protocol": "paper0.the_pseudoscalar_coupling.the_pseudoscalar_coupling",
        "canonical_statement": "The source-bounded component 'The Pseudoscalar Coupling' preserves Paper 0 records P0R04291-P0R04309 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04291:the_pseudoscalar_coupling",
            "P0R04292:the_pseudoscalar_coupling",
            "P0R04293:the_pseudoscalar_coupling",
            "P0R04294:the_pseudoscalar_coupling",
            "P0R04295:the_pseudoscalar_coupling",
            "P0R04296:the_pseudoscalar_coupling",
            "P0R04297:the_pseudoscalar_coupling",
            "P0R04298:the_pseudoscalar_coupling",
            "P0R04299:the_pseudoscalar_coupling",
            "P0R04300:the_pseudoscalar_coupling",
            "P0R04301:the_pseudoscalar_coupling",
            "P0R04302:the_pseudoscalar_coupling",
            "P0R04303:the_pseudoscalar_coupling",
            "P0R04304:the_pseudoscalar_coupling",
            "P0R04305:the_pseudoscalar_coupling",
            "P0R04306:the_pseudoscalar_coupling",
            "P0R04307:the_pseudoscalar_coupling",
            "P0R04308:the_pseudoscalar_coupling",
            "P0R04309:the_pseudoscalar_coupling",
        ),
        "source_formulae": (
            "P0R04291: The Pseudoscalar Coupling",
            "P0R04292: The complex scalar Psi-field can be decomposed into a radial component (magnitude, related to the Psi-Higgs) and an angular component (phase).",
            "P0R04293: The angular component behaves as a pseudoscalar field (a), analogous to the axion. Pseudoscalar fields naturally couple to the EM field via the interaction Lagrangian:",
            "P0R04294: $\\mathbf{L}_{\\mathbf{a}}\\mathbf{gammagamma =}\\mathbf{g}_{\\mathbf{a}}\\mathbf{gammagammaa}\\mathbf{F}_{\\mathbf{m}}\\mathbf{unutildeFmunu =}\\mathbf{g}_{\\mathbf{a}}\\mathbf{gammagammaa}\\left( \\mathbf{mathbfEcdotmathbfB} \\right)$",
            "P0R04295: where g_agammagamma is the coupling constant, and tildeFmunu is the dual electromagnetic tensor.",
            'P0R04296: We cannot have a single phase degree of freedom act as both the longitudinal polarization of a massive vector boson (the "eaten" Goldstone) and a free, propagating Axion-Like Particle (ALP). In Quantum Field Theory, once a phase is eaten by the Higgs mechanism, it disappears from the physical spectrum.',
            "P0R04297: [IMAGE:Ein Bild, das Text, Screenshot, Display, Schrift enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R04298: Fig.: Pseudoscalar Decomposition & Primakoff Interconversion. This diagram formally illustrates the decomposition of the complex Psi-field and presents the Primakoff effect as a Feynman-like diagram, explicitly linking it to the interaction Lagrangian.",
            "P0R04299: A. Complex Field Decomposition: The consciousness field is written as Psi=rho eitheta\\Psi=\\rho\\,e^{i\\theta}Psi=rhoeitheta. The radial mode rho\\rhorho yields a Psi-Higgs (massive scalar), while the angular mode theta\\thetatheta defines the pseudoscalar aaa (ALP). B. Primakoff Effect: In a background magnetic field BBB, ALPs and photons interconvert at a vertex: forward a -> a\\!\\to\\!\\gammaa-> (with BBB) and inverse -> a\\gamma\\!\\to\\!a->a (with BBB). The effective interaction is Lint = 14 ga a FmuF~mu ga a E B,\\mathcal L_{\\text{int}} \\;=\\; \\tfrac{1}{4}\\,g_{a\\gamma\\gamma}\\,a\\,F_{\\mu\\nu}\\tilde F^{\\mu\\nu} \\;\\propto\\; g_{a\\gamma\\gamma}\\,a\\,\\mathbf E\\!\\cdot\\!\\mathbf B,Lint=41gaaFmuF~mugaaEB, providing the gauge coupling bridge between the Psi-phase sector and electromagnetism.",
            "P0R04300: [IMAGE:Ein Bild, das Text, Screenshot, Schrift enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]Fig: Psi-Field Decomposition & Electromagnetic Coupling (Primakoff) This diagram establishes the core theoretical components. It shows how the fundamental Psi-field is decomposed into its magnitude and phase, and how the phase component (the Axion-Like Particle or ALP) provides the essential bridge to the world of electromagnetism via the Primakoff effect.",
            "P0R04301: The complex scalar Psi-field decomposes into magnitude rho\\rhorho (radial mode) and phase theta\\thetatheta (angular mode). The radial excitation yields a Psi-Higgs (massive spin-0), while the angular excitation manifests as a pseudoscalar aaa (ALP). The ALP couples to electromagnetism via the Primakoff effect, governed by the pseudoscalar interaction",
            "P0R04302: La=14 ga a FmuF~mu ga a E B,\\mathcal{L}_{a\\gamma\\gamma}=\\tfrac{1}{4}\\,g_{a\\gamma\\gamma}\\,a\\,F_{\\mu\\nu}\\tilde F^{\\mu\\nu}\\;\\;\\Longleftrightarrow\\;\\; g_{a\\gamma\\gamma}\\,a\\,\\mathbf{E}\\!\\cdot\\!\\mathbf{B},La=41gaaFmuF~mugaaEB,",
            "P0R04303: enabling aa \\leftrightarrow \\gammaa conversion in external fields and providing a concrete bridge between the Psi-phase sector and the EM field.",
            "P0R04304: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Zahl enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R04305: Fig.: Bidirectional PsiEM Coupling in Neural Tissue. This schematic details the two causal pathways within the brain. It formally separates the downward causation from the Psi-field to the brain's EM activity and the upward causation from EM activity back to the Psi-field.",
            "P0R04306: A. Downward causation (Psi->EM): Psi-field phase dynamics (theta)(\\theta)(theta) generate a pseudoscalar field aaa which undergoes Primakoff conversion in the presence of endogenous magnetic fields BendoB_{\\text{endo}}Bendo, yielding photons \\gamma that modulate endogenous EM activity (EEG, bioelectric fields). B. Upward causation (EM->Psi): Source EM fields ()(\\gamma)() (e.g., from neural synchrony) drive inverse Primakoff conversion (+Bendo->a)(\\gamma + B_{\\text{endo}}\\to a)(+Bendo->a); the resulting ALP field modulates the Psi-field phase (theta)(\\theta)(theta), closing the loop. Dotted links highlight the role of BendoB_{\\text{endo}}Bendo and the bidirectional bridge between the panels.",
            "P0R04307: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R04308: Fig.: The Brain as a Transducer (Bidirectional PsiEM) This is the central illustration for the chapter. It places the abstract physics from the first diagram into the concrete context of the brain. It visually separates the two causal pathways: how the mind (Psi-Field) influences the brain's electrical activity, and how the brain's electrical activity can, in turn, influence the mind.",
            "P0R04309: Downward path (Psi->EM): Coherent Psi-field dynamics seed an ALP field aaa that couples via the Primakoff effect to produce observable photons \\gamma, aligning with EEG rhythms and bioelectric fields. Endogenous magnetic fields BendoB_{\\text{endo}}Bendo from neural currents (L4) interact with the Primakoff vertex, enhancing conversion locally. Upward path (EM->Psi): External/internal EM fields drive the Primakoff interaction to generate an ALP field, which modulates the phase of the Psi-field, closing the bidirectional transduction loop.",
        ),
        "test_protocols": ("preserve The Pseudoscalar Coupling source-accounting boundary",),
        "null_results": ("The Pseudoscalar Coupling is not empirical validation evidence",),
        "variables": ("the_pseudoscalar_coupling",),
        "validation_targets": ("preserve records P0R04291-P0R04309",),
        "null_controls": ("the_pseudoscalar_coupling must remain source-bounded accounting",),
    }
}


@dataclass(frozen=True, slots=True)
class ThePseudoscalarCouplingSpec:
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
class ThePseudoscalarCouplingSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ThePseudoscalarCouplingSpec, ...]
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


def build_the_pseudoscalar_coupling_specs(
    source_records: list[dict[str, Any]],
) -> ThePseudoscalarCouplingSpecBundle:
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

    specs: list[ThePseudoscalarCouplingSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ThePseudoscalarCouplingSpec(
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
        "title": "Paper 0 " + "The Pseudoscalar Coupling" + " Specs",
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
        "next_source_boundary": "P0R04310",
    }
    return ThePseudoscalarCouplingSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ThePseudoscalarCouplingSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_pseudoscalar_coupling_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ThePseudoscalarCouplingSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Pseudoscalar Coupling" + " Specs",
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
    bundle: ThePseudoscalarCouplingSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_the_pseudoscalar_coupling_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_the_pseudoscalar_coupling_validation_specs_{date_tag}.md"
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
