#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 2. The Synaptic Junction and Downward Causation (L2): spec builder
"""Promote Paper 0 2. The Synaptic Junction and Downward Causation (L2): records."""

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
    "P0R04470",
    "P0R04471",
    "P0R04472",
    "P0R04473",
    "P0R04474",
    "P0R04475",
    "P0R04476",
    "P0R04477",
)
CLAIM_BOUNDARY = "source-bounded section 2 the synaptic junction and downward causation l2 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_2_the_synaptic_junction_and_downward_causation_l2.2_the_synaptic_junction_and_downward_causation_l2": {
        "context_id": "2_the_synaptic_junction_and_downward_causation_l2",
        "validation_protocol": "paper0.section_2_the_synaptic_junction_and_downward_causation_l2.2_the_synaptic_junction_and_downward_causation_l2",
        "canonical_statement": "The source-bounded component '2. The Synaptic Junction and Downward Causation (L2):' preserves Paper 0 records P0R04470-P0R04477 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04470:2_the_synaptic_junction_and_downward_causation_l2",
            "P0R04471:2_the_synaptic_junction_and_downward_causation_l2",
            "P0R04472:2_the_synaptic_junction_and_downward_causation_l2",
            "P0R04473:2_the_synaptic_junction_and_downward_causation_l2",
            "P0R04474:2_the_synaptic_junction_and_downward_causation_l2",
            "P0R04475:2_the_synaptic_junction_and_downward_causation_l2",
            "P0R04476:2_the_synaptic_junction_and_downward_causation_l2",
            "P0R04477:2_the_synaptic_junction_and_downward_causation_l2",
        ),
        "source_formulae": (
            "P0R04470: 2. The Synaptic Junction and Downward Causation (L2):",
            "P0R04471: L2 governs the transduction of the Psi-field into neurochemical signals. The synapse is the critical locus for downward causation (Agency).",
            'P0R04472: The Transduction Mechanism (IET via Quantum Potential): The Psi-field influences the energy landscape via Information-Energy Transduction (IET), modulating the Quantum Potential (Q). Q=2m2rho 2rho The Psi-field couples directly to Q: LIET=gIETPsi(x)Q(x). By modulating Q, the Psi-field (Information) influences the energy landscape without classical energy exchange. | Calcium Dynamics and Synaptotagmin: The primary target of IET is the probability of neurotransmitter release (Pr), mediated by Ca$^{2+}$ influx and sensor proteins (e.g., Synaptotagmin). The Psi-field modulates the cooperativity of these sensors. | The Mechanism of Intent (QZE and Resonant Addressing): Intent (L5) acts via the Quantum Zeno Effect (QZE), functioning as a continuous measurement operator (hatM_Attn) that stabilises specific synaptic configurations. The solution to the targeting problem-how the QZE is selectively applied-is the Principle of Resonant Addressing, guided by the HPC architecture. The L5 intent generates a specific oscillatory pattern within the L4 network-the "address." The susceptibility of a synapse to Psi-field modulation (via IET or QZE) is gated by its participation in this resonant pattern. The coupling strength (lambda) of the IET interaction Lagrangian (L_IET=lambdaPsi(x)Q(x)) is dynamically modulated:',
            "P0R04473: lambda(x,t)=lambda_0timesR(x,t)timesF(omega_local,omega_intent) Here, F is a resonance function that peaks when the local dynamics (omega_local) match the intent pattern (omega_intent). Crucially, the spatial location (x) and timing (t) embedded in the relevance factor R(x,t) are provided by the downward-flowing HPC generative model. The HPC architecture itself acts as the coordinate system, guiding the downward causation to the precise locations where the generative model needs to be actualised. This mechanism ensures that downward causation is contextually relevant and precisely targeted. The SCPN framework identifies the modulation of calcium sensor (synaptotagmin) cooperativity as the precise molecular locus for downward causation. However, this raises the Specificity Problem: how does a global, non-local field of intent avoid modulating all synapses simultaneously, thereby creating a storm of undifferentiated neural activity? The solution lies in the Principle of Resonant Addressing. The efficacy of the Psi-field's influence is not uniform across the brain but is dynamically and selectively gated by the local oscillatory state of the neural circuit. The brain's existing frequency architecture serves as the coordinate system for targeting intent. This principle is formalised by redefining the coupling constant lambda in the interaction Hamiltonian (Hint=lambdaPsisniCanjCa) as a dynamic variable dependent on the local network state: lambda(x,t)=lambda0R(x,t)F(local,intent) Where:",
            "P0R04474: lambda0 is the basal coupling strength, a fundamental constant of the interaction. | R(x,t) is the local Kuramoto order parameter, representing the degree of phase synchrony in the neural ensemble at position x and time t. The modulatory effect is amplified in highly coherent ensembles where R(x,t)->1. | F(local,intent) is a resonance function (e.g., a Lorentzian) that peaks when the dominant local oscillation frequency, local (e.g., a specific gamma band), matches the frequency of the intent carrier, intent, which is propagated down from higher layers of the SCPN via the HPC architecture.",
            'P0R04475: This mechanism elegantly solves the targeting problem without requiring the Psi-field to possess spatial information. Intent does not need to "find" the right synapse; it simply establishes a resonant frequency. Only those neural ensembles and their associated synapses that are already participating in that specific "neural song" will be susceptible to modulation. This leverages the brain\'s intrinsic, self-organising oscillatory dynamics as the addressing system for the precise and context-dependent application of downward causation.',
            "P0R04476: Neurotransmitter Systems as Filters: Neurotransmitter systems modulate the UPDE parameters and tune the network state. Dopamine (DA): Prediction error signalling (HPC) and reward. | Serotonin (5-HT): Tuning the Criticality parameter (sigma). | Acetylcholine (ACh): Attention, learning, and precision weighting (HPC).",
            "P0R04477: P0R04477",
        ),
        "test_protocols": (
            "preserve 2. The Synaptic Junction and Downward Causation (L2): source-accounting boundary",
        ),
        "null_results": (
            "2. The Synaptic Junction and Downward Causation (L2): is not empirical validation evidence",
        ),
        "variables": ("2_the_synaptic_junction_and_downward_causation_l2",),
        "validation_targets": ("preserve records P0R04470-P0R04477",),
        "null_controls": (
            "2_the_synaptic_junction_and_downward_causation_l2 must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class Section2TheSynapticJunctionAndDownwardCausationL2Spec:
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
class Section2TheSynapticJunctionAndDownwardCausationL2SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section2TheSynapticJunctionAndDownwardCausationL2Spec, ...]
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


def build_section_2_the_synaptic_junction_and_downward_causation_l2_specs(
    source_records: list[dict[str, Any]],
) -> Section2TheSynapticJunctionAndDownwardCausationL2SpecBundle:
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

    specs: list[Section2TheSynapticJunctionAndDownwardCausationL2Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section2TheSynapticJunctionAndDownwardCausationL2Spec(
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
        "title": "Paper 0 " + "2. The Synaptic Junction and Downward Causation (L2):" + " Specs",
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
        "next_source_boundary": "P0R04478",
    }
    return Section2TheSynapticJunctionAndDownwardCausationL2SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section2TheSynapticJunctionAndDownwardCausationL2SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_2_the_synaptic_junction_and_downward_causation_l2_specs(
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


def render_report(bundle: Section2TheSynapticJunctionAndDownwardCausationL2SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "2. The Synaptic Junction and Downward Causation (L2):" + " Specs",
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
    bundle: Section2TheSynapticJunctionAndDownwardCausationL2SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_2_the_synaptic_junction_and_downward_causation_l2_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_2_the_synaptic_junction_and_downward_causation_l2_validation_specs_{date_tag}.md"
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
