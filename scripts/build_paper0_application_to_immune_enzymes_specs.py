#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Application to Immune Enzymes spec builder
"""Promote Paper 0 Application to Immune Enzymes records."""

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
    "P0R05517",
    "P0R05518",
    "P0R05519",
    "P0R05520",
    "P0R05521",
    "P0R05522",
    "P0R05523",
    "P0R05524",
    "P0R05525",
    "P0R05526",
    "P0R05527",
)
CLAIM_BOUNDARY = "source-bounded application to immune enzymes source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "application_to_immune_enzymes.application_to_immune_enzymes": {
        "context_id": "application_to_immune_enzymes",
        "validation_protocol": "paper0.application_to_immune_enzymes.application_to_immune_enzymes",
        "canonical_statement": "The source-bounded component 'Application to Immune Enzymes' preserves Paper 0 records P0R05517-P0R05519 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05517:application_to_immune_enzymes",
            "P0R05518:application_to_immune_enzymes",
            "P0R05519:application_to_immune_enzymes",
        ),
        "source_formulae": (
            "P0R05517: Application to Immune Enzymes",
            "P0R05518: The immune system's capacity for rapid, specific, and robust responses relies on a suite of enzymatic reactions. Extending the principle of quantum enzymology to this domain provides a deeper mechanistic understanding of its efficiency.",
            'P0R05519: Lysozyme: This enzyme, a key component of the innate immune system, functions by hydrolysing the peptidoglycan cell walls of bacteria. While direct experimental evidence for tunnelling in lysozyme is not yet established, its mechanism, which involves proton transfer in the active site to cleave a glycosidic bond, is a prime candidate. We can model the interaction with a Hamiltonian that includes a Psi-field coupling term modulating the barrier width for the key proton transfer step, suggesting that the efficiency of this fundamental immune defence is quantum-enhanced. | Immunoglobulins (Antibodies): The antigen-binding function of immunoglobulins relies on achieving high affinity and specificity through conformational adjustments, a process known as induced fit. While much research has focused on using quantum dots as fluorescent labels for antibodies, the intrinsic quantum properties of the binding process itself are unexplored. A novel mechanism of "quantum-assisted induced fit" is proposed, where the initial binding event triggers a rapid conformational search for the optimal, high-affinity state. This search is not a classical random walk through the conformational energy landscape but is accelerated by quantum tunnelling between local energy minima. This allows the antibody\'s Fab region to lock onto its target antigen with a speed and precision that would be classically improbable, providing a quantum basis for the immune system\'s remarkable specificity. | Cytokine Signalling and T-Cell Activation: This provides the most robust, evidence-based example. The "quantum model of T-cell activation" posits that the energy transfer driving T-cell activation is not a continuous flow but occurs in discrete "quanta" of phosphates. This process is mediated by oscillating cycles of receptor phosphorylation and dephosphorylation, initiated by dynamic "catch-slip" pulses in the peptide-major histocompatibility complex-T cell receptor (pMHC-TcR) interaction. This quantised energy transfer is here explicitly linked to quantum tunnelling events within the protein kinases of the T-cell signalling cascade (e.g., Lck, ZAP-70). The manuscript\'s Neuro-Immune Hamiltonian ( HNI) can be expanded to include these specific kinase dynamics, with a term describing Psi-field-mediated enhancement of proton/electron tunnelling during the phosphotransfer reaction, thereby directly linking the universal consciousness field to the activation threshold of an adaptive immune response.',
        ),
        "test_protocols": ("preserve Application to Immune Enzymes source-accounting boundary",),
        "null_results": ("Application to Immune Enzymes is not empirical validation evidence",),
        "variables": ("application_to_immune_enzymes",),
        "validation_targets": ("preserve records P0R05517-P0R05519",),
        "null_controls": ("application_to_immune_enzymes must remain source-bounded accounting",),
    },
    "application_to_immune_enzymes.the_biophysics_of_coherence_a_scale_invariant_cybernetic_thread": {
        "context_id": "the_biophysics_of_coherence_a_scale_invariant_cybernetic_thread",
        "validation_protocol": "paper0.application_to_immune_enzymes.the_biophysics_of_coherence_a_scale_invariant_cybernetic_thread",
        "canonical_statement": "The source-bounded component 'The Biophysics of Coherence: A Scale-Invariant Cybernetic Thread' preserves Paper 0 records P0R05520-P0R05521 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05520:the_biophysics_of_coherence_a_scale_invariant_cybernetic_thread",
            "P0R05521:the_biophysics_of_coherence_a_scale_invariant_cybernetic_thread",
        ),
        "source_formulae": (
            "P0R05520: The Biophysics of Coherence: A Scale-Invariant Cybernetic Thread",
            "P0R05521: A central requirement of the SCPN framework is the maintenance of coherence across vast biological and ecological scales. This section constructs a continuous narrative of homeostatic regulation, arguing that the same fundamental cybernetic principles that ensure stability at the neuronal level are scaled up to maintain the coherence of the entire planetary biosphere. This reveals a scale-invariant logic of active environmental engineering, or niche construction, as a core dynamic of life.",
        ),
        "test_protocols": (
            "preserve The Biophysics of Coherence: A Scale-Invariant Cybernetic Thread source-accounting boundary",
        ),
        "null_results": (
            "The Biophysics of Coherence: A Scale-Invariant Cybernetic Thread is not empirical validation evidence",
        ),
        "variables": ("the_biophysics_of_coherence_a_scale_invariant_cybernetic_thread",),
        "validation_targets": ("preserve records P0R05520-P0R05521",),
        "null_controls": (
            "the_biophysics_of_coherence_a_scale_invariant_cybernetic_thread must remain source-bounded accounting",
        ),
    },
    "application_to_immune_enzymes.micro_scale_homeostasis_glial_control_of_neuronal_criticality": {
        "context_id": "micro_scale_homeostasis_glial_control_of_neuronal_criticality",
        "validation_protocol": "paper0.application_to_immune_enzymes.micro_scale_homeostasis_glial_control_of_neuronal_criticality",
        "canonical_statement": "The source-bounded component 'Micro-Scale Homeostasis: Glial Control of Neuronal Criticality' preserves Paper 0 records P0R05522-P0R05527 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05522:micro_scale_homeostasis_glial_control_of_neuronal_criticality",
            "P0R05523:micro_scale_homeostasis_glial_control_of_neuronal_criticality",
            "P0R05524:micro_scale_homeostasis_glial_control_of_neuronal_criticality",
            "P0R05525:micro_scale_homeostasis_glial_control_of_neuronal_criticality",
            "P0R05526:micro_scale_homeostasis_glial_control_of_neuronal_criticality",
            "P0R05527:micro_scale_homeostasis_glial_control_of_neuronal_criticality",
        ),
        "source_formulae": (
            "P0R05522: Micro-Scale Homeostasis: Glial Control of Neuronal Criticality",
            'P0R05523: The SCPN posits in Layer 4 that neural tissues operate in a "quasicritical regime," a state balanced between order and chaos (sigma1) that is computationally optimal but inherently unstable and requires constant fine-tuning. The manuscript proposes a "slow control layer" composed of glial cells, particularly astrocytes, as the homeostatic mechanism responsible for maintaining this delicate state.',
            "P0R05524: Research into Self-Organised Criticality (SoC) in Spiking Neural Networks (SNNs) provides a concrete biophysical basis for this proposal. These studies demonstrate that SoC can emerge from the interplay of local, activity-dependent learning rules-specifically, Spike-Timing-Dependent Plasticity (STDP)-and the presence of ambient noise. STDP strengthens or weakens synaptic connections based on the precise timing of pre- and postsynaptic spikes.",
            "P0R05525: Simulations show that a network can self-organise to the critical point through a dynamic balance: excitatory STDP can push the network into a supercritical (chaotic) state, which in turn activates inhibitory STDP that pushes it back toward the critical point. However, this process is highly sensitive to parameters like the level of stochastic noise and the STDP time windows.",
            "P0R05526: This is where the SCPN's concept of glial slow control becomes critical. The glial network, through its slow (seconds to minutes) intercellular calcium waves, integrates neuronal activity over long timescales. This allows it to act as the biological system that actively modulates the very parameters-local noise levels, neurotransmitter concentrations, ionic balances-that govern the efficacy of STDP.",
            "P0R05527: The glial network is thus the homeostatic regulator that solves the fine-tuning problem of criticality. While fast, local STDP rules drive the system towards the critical point, the slow, integrative feedback from the glial network robustly stabilises the system at the critical point. This can be understood as a form of internal niche construction, where one part of the biological system (the glia) actively engineers the local environment of another part (the neurons) to ensure optimal computational function.",
        ),
        "test_protocols": (
            "preserve Micro-Scale Homeostasis: Glial Control of Neuronal Criticality source-accounting boundary",
        ),
        "null_results": (
            "Micro-Scale Homeostasis: Glial Control of Neuronal Criticality is not empirical validation evidence",
        ),
        "variables": ("micro_scale_homeostasis_glial_control_of_neuronal_criticality",),
        "validation_targets": ("preserve records P0R05522-P0R05527",),
        "null_controls": (
            "micro_scale_homeostasis_glial_control_of_neuronal_criticality must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ApplicationToImmuneEnzymesSpec:
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
class ApplicationToImmuneEnzymesSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ApplicationToImmuneEnzymesSpec, ...]
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


def build_application_to_immune_enzymes_specs(
    source_records: list[dict[str, Any]],
) -> ApplicationToImmuneEnzymesSpecBundle:
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

    specs: list[ApplicationToImmuneEnzymesSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ApplicationToImmuneEnzymesSpec(
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
        "title": "Paper 0 " + "Application to Immune Enzymes" + " Specs",
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
        "next_source_boundary": "P0R05528",
    }
    return ApplicationToImmuneEnzymesSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ApplicationToImmuneEnzymesSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_application_to_immune_enzymes_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: ApplicationToImmuneEnzymesSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Application to Immune Enzymes" + " Specs",
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
    bundle: ApplicationToImmuneEnzymesSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_application_to_immune_enzymes_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_application_to_immune_enzymes_validation_specs_{date_tag}.md"
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
