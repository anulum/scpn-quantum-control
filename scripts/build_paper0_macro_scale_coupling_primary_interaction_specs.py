#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Macro-Scale Coupling (Primary Interaction): spec builder
"""Promote Paper 0 Macro-Scale Coupling (Primary Interaction): records."""

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
    "P0R02107",
    "P0R02108",
    "P0R02109",
    "P0R02110",
    "P0R02111",
    "P0R02112",
    "P0R02113",
    "P0R02114",
    "P0R02115",
    "P0R02116",
    "P0R02117",
    "P0R02118",
    "P0R02119",
    "P0R02120",
    "P0R02121",
    "P0R02122",
    "P0R02123",
    "P0R02124",
    "P0R02125",
    "P0R02126",
    "P0R02127",
)
CLAIM_BOUNDARY = "source-bounded macro scale coupling primary interaction source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "macro_scale_coupling_primary_interaction.macro_scale_coupling_primary_interaction": {
        "context_id": "macro_scale_coupling_primary_interaction",
        "validation_protocol": "paper0.macro_scale_coupling_primary_interaction.macro_scale_coupling_primary_interaction",
        "canonical_statement": "The source-bounded component 'Macro-Scale Coupling (Primary Interaction):' preserves Paper 0 records P0R02107-P0R02108 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02107:macro_scale_coupling_primary_interaction",
            "P0R02108:macro_scale_coupling_primary_interaction",
        ),
        "source_formulae": (
            "P0R02107: Macro-Scale Coupling (Primary Interaction):",
            "P0R02108: At the highest level for this layer, the Organismal Field (Psis = PsiO) couples directly to the large-scale bioelectric field. The collective state variable is the vector field of membrane potentials across the entire tissue: sigma_electric = V_mem(x, y, z). The H_int term acts here, allowing the informational content of the Psi-field to impress a specific geometric pattern onto the bioelectric field.",
        ),
        "test_protocols": (
            "preserve Macro-Scale Coupling (Primary Interaction): source-accounting boundary",
        ),
        "null_results": (
            "Macro-Scale Coupling (Primary Interaction): is not empirical validation evidence",
        ),
        "variables": ("macro_scale_coupling_primary_interaction",),
        "validation_targets": ("preserve records P0R02107-P0R02108",),
        "null_controls": (
            "macro_scale_coupling_primary_interaction must remain source-bounded accounting",
        ),
    },
    "macro_scale_coupling_primary_interaction.meso_scale_transduction": {
        "context_id": "meso_scale_transduction",
        "validation_protocol": "paper0.macro_scale_coupling_primary_interaction.meso_scale_transduction",
        "canonical_statement": "The source-bounded component 'Meso-Scale Transduction:' preserves Paper 0 records P0R02109-P0R02110 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02109:meso_scale_transduction",
            "P0R02110:meso_scale_transduction",
        ),
        "source_formulae": (
            "P0R02109: Meso-Scale Transduction:",
            "P0R02110: The bioelectric field sigma_electric is not the final step. As detailed in the text, this classical field generates a local magnetic field (B_local) within the cell nucleus.",
        ),
        "test_protocols": ("preserve Meso-Scale Transduction: source-accounting boundary",),
        "null_results": ("Meso-Scale Transduction: is not empirical validation evidence",),
        "variables": ("meso_scale_transduction",),
        "validation_targets": ("preserve records P0R02109-P0R02110",),
        "null_controls": ("meso_scale_transduction must remain source-bounded accounting",),
    },
    "macro_scale_coupling_primary_interaction.quantum_scale_coupling_secondary_interaction": {
        "context_id": "quantum_scale_coupling_secondary_interaction",
        "validation_protocol": "paper0.macro_scale_coupling_primary_interaction.quantum_scale_coupling_secondary_interaction",
        "canonical_statement": "The source-bounded component 'Quantum-Scale Coupling (Secondary Interaction):' preserves Paper 0 records P0R02111-P0R02113 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02111:quantum_scale_coupling_secondary_interaction",
            "P0R02112:quantum_scale_coupling_secondary_interaction",
            "P0R02113:quantum_scale_coupling_secondary_interaction",
        ),
        "source_formulae": (
            "P0R02111: Quantum-Scale Coupling (Secondary Interaction):",
            "P0R02112: This B_local field then enters the radical-pair Hamiltonian. The crucial insight is that this Hamiltonian governs the dynamics of a quantum collective variable: the total spin state of the radical pair, sigma_spin.",
            "P0R02113: The framework thus describes a complete transduction cascade: The fundamental Psi-field interaction sets the pattern of a classical collective variable (sigma_electric), which in turn modulates the dynamics of a quantum collective variable (sigma_spin), which finally controls the discrete, digital information stored in the epigenetic code. This provides a physically plausible and mathematically formal pathway from the continuous, holistic information of the consciousness field to the discrete, specific instructions that build an organism.",
        ),
        "test_protocols": (
            "preserve Quantum-Scale Coupling (Secondary Interaction): source-accounting boundary",
        ),
        "null_results": (
            "Quantum-Scale Coupling (Secondary Interaction): is not empirical validation evidence",
        ),
        "variables": ("quantum_scale_coupling_secondary_interaction",),
        "validation_targets": ("preserve records P0R02111-P0R02113",),
        "null_controls": (
            "quantum_scale_coupling_secondary_interaction must remain source-bounded accounting",
        ),
    },
    "macro_scale_coupling_primary_interaction.domain_i_biological_substrate_layers_1_4": {
        "context_id": "domain_i_biological_substrate_layers_1_4",
        "validation_protocol": "paper0.macro_scale_coupling_primary_interaction.domain_i_biological_substrate_layers_1_4",
        "canonical_statement": "The source-bounded component 'Domain I: Biological Substrate (Layers 1-4):' preserves Paper 0 records P0R02114-P0R02127 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02114:domain_i_biological_substrate_layers_1_4",
            "P0R02115:domain_i_biological_substrate_layers_1_4",
            "P0R02116:domain_i_biological_substrate_layers_1_4",
            "P0R02117:domain_i_biological_substrate_layers_1_4",
            "P0R02118:domain_i_biological_substrate_layers_1_4",
            "P0R02119:domain_i_biological_substrate_layers_1_4",
            "P0R02120:domain_i_biological_substrate_layers_1_4",
            "P0R02121:domain_i_biological_substrate_layers_1_4",
            "P0R02122:domain_i_biological_substrate_layers_1_4",
            "P0R02123:domain_i_biological_substrate_layers_1_4",
            "P0R02124:domain_i_biological_substrate_layers_1_4",
            "P0R02125:domain_i_biological_substrate_layers_1_4",
            "P0R02126:domain_i_biological_substrate_layers_1_4",
            "P0R02127:domain_i_biological_substrate_layers_1_4",
        ),
        "source_formulae": (
            "P0R02114: Domain I: Biological Substrate (Layers 1-4):",
            "P0R02115: Quantum encoding (L1), Neurochemical filtering (L2), Genomic logic and Morphogenesis (L3),",
            "P0R02116: Layer 3: Genomic Logic and Morphogenesis: A CISS-Bioelectric Framework",
            "P0R02117: The genome is not a static blueprint but a dynamic computational substrate that interfaces with organismal fields to guide morphogenesis. This coupling is not mediated by speculative forces but by two well-established biophysical mechanisms: (1) Chiral-Induced Spin Selectivity (CISS) in DNA, which translates field information into epigenetic changes, and (2) large-scale bioelectric fields, which orchestrate gene expression to create anatomical structure.",
            "P0R02118: Mechanism 1: Chiral-Induced Spin Selectivity (CISS) and Epigenetic Modulation",
            "P0R02119: The helical, chiral structure of DNA makes it an efficient spin filter. As electrons are transferred along the DNA strand, their spins become polarised, with selectivity reaching 60-90%. This has a direct impact on radical pair reactions, which are fundamental to many biochemical processes, including the DNA methylation and demethylation that form the basis of the epigenetic code.",
            "P0R02120: The spin-dependent recombination of a radical pair is governed by a Hamiltonian that includes spin-orbit coupling (lambda) within the DNA helix. This coupling generates a powerful, effective magnetic field (Beff1 T) that influences the probability of singlet-to-triplet conversion.",
            "P0R02121: $$ H_{total} = \\epsilon_0 + \\frac{\\Delta}{2}\\sigma_z + \\frac{\\lambda}{L^2}(\\sigma \\cdot L) + gS \\cdot \\sigma $$ The recombination probability, S, is highly sensitive to this effective field. By influencing the ambient electromagnetic environment, the organismal field (O) can bias this probability, thereby modulating the rates of epigenetic modifications. This provides a concrete physical mechanism for the top-down guidance of gene expression.",
            "P0R02122: The CISS-TET/DNMT Transduction Pathway",
            "P0R02123: The concrete mechanism linking CISS-induced spin polarisation to epigenetic changes centres on the activity of key enzymes: DNA Methyltransferases (DNMTs) and Ten-Eleven Translocation (TET) enzymes. These enzymes utilise radical reaction pathways that are inherently sensitive to the spin states of their intermediates.",
            "P0R02124: TET enzymes, which initiate active DNA demethylation, are iron-dependent dioxygenases. Their catalytic cycle involves the formation of radical pairs. The recombination probability of a radical pair is governed by the interplay between the exchange interaction (J) and the local magnetic environment (B_eff), as described by the radical pair Hamiltonian:",
            "P0R02125: $\\mathbf{HRP =}\\sum_{}^{}\\mathbf{i}\\mathbf{= 12}\\left( \\mathbf{\\omega i Siz +}\\sum_{}^{}\\mathbf{k Aik Si}\\mathbf{\\cdot}\\mathbf{Ik} \\right)\\mathbf{+ J}\\left( \\mathbf{1}\\textbf{/}\\mathbf{2 + 2S1}\\mathbf{\\cdot}\\mathbf{S2} \\right)$",
            "P0R02126: CISS in the DNA helix generates a powerful, effective magnetic field (B_eff) that directly influences the singlet-triplet interconversion rate of these radical pairs. By biasing the initial spin state of the electrons transferred to the enzyme's active site, CISS can act as a \"spin-valve,\" modulating the enzyme's catalytic efficiency and thereby the rate of DNA methylation or demethylation at specific genomic loci.",
            "P0R02127: P0R02127",
        ),
        "test_protocols": (
            "preserve Domain I: Biological Substrate (Layers 1-4): source-accounting boundary",
        ),
        "null_results": (
            "Domain I: Biological Substrate (Layers 1-4): is not empirical validation evidence",
        ),
        "variables": ("domain_i_biological_substrate_layers_1_4",),
        "validation_targets": ("preserve records P0R02114-P0R02127",),
        "null_controls": (
            "domain_i_biological_substrate_layers_1_4 must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class MacroScaleCouplingPrimaryInteractionSpec:
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
class MacroScaleCouplingPrimaryInteractionSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[MacroScaleCouplingPrimaryInteractionSpec, ...]
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


def build_macro_scale_coupling_primary_interaction_specs(
    source_records: list[dict[str, Any]],
) -> MacroScaleCouplingPrimaryInteractionSpecBundle:
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

    specs: list[MacroScaleCouplingPrimaryInteractionSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            MacroScaleCouplingPrimaryInteractionSpec(
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
        "title": "Paper 0 " + "Macro-Scale Coupling (Primary Interaction):" + " Specs",
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
        "next_source_boundary": "P0R02128",
    }
    return MacroScaleCouplingPrimaryInteractionSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> MacroScaleCouplingPrimaryInteractionSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_macro_scale_coupling_primary_interaction_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: MacroScaleCouplingPrimaryInteractionSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Macro-Scale Coupling (Primary Interaction):" + " Specs",
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
    bundle: MacroScaleCouplingPrimaryInteractionSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_macro_scale_coupling_primary_interaction_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_macro_scale_coupling_primary_interaction_validation_specs_{date_tag}.md"
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
