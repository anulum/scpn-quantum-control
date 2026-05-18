#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Mechanism 2: Quantum Stochastic Resonance (QSR) spec builder
"""Promote Paper 0 Mechanism 2: Quantum Stochastic Resonance (QSR) records."""

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
    "P0R03368",
    "P0R03369",
    "P0R03370",
    "P0R03371",
    "P0R03372",
    "P0R03373",
    "P0R03374",
    "P0R03375",
    "P0R03376",
    "P0R03377",
    "P0R03378",
    "P0R03379",
    "P0R03380",
    "P0R03381",
    "P0R03382",
    "P0R03383",
    "P0R03384",
    "P0R03385",
)
CLAIM_BOUNDARY = "source-bounded mechanism 2 quantum stochastic resonance qsr p0r03368 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "mechanism_2_quantum_stochastic_resonance_qsr_p0r03368.mechanism_2_quantum_stochastic_resonance_qsr": {
        "context_id": "mechanism_2_quantum_stochastic_resonance_qsr",
        "validation_protocol": "paper0.mechanism_2_quantum_stochastic_resonance_qsr_p0r03368.mechanism_2_quantum_stochastic_resonance_qsr",
        "canonical_statement": "The source-bounded component 'Mechanism 2: Quantum Stochastic Resonance (QSR)' preserves Paper 0 records P0R03368-P0R03385 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03368:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03369:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03370:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03371:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03372:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03373:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03374:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03375:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03376:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03377:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03378:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03379:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03380:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03381:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03382:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03383:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03384:mechanism_2_quantum_stochastic_resonance_qsr",
            "P0R03385:mechanism_2_quantum_stochastic_resonance_qsr",
        ),
        "source_formulae": (
            "P0R03368: Mechanism 2: Quantum Stochastic Resonance (QSR)",
            "P0R03369: Once a coherent quantum state is selected, its weak signal must be amplified to a macroscopic level. This is achieved through Quantum Stochastic Resonance (QSR), a phenomenon where a noisy, non-linear system can amplify a weak, sub-threshold signal. The brain, operating at the quasicritical state described in Layer 4, is precisely such a system: it is inherently noisy and poised at a threshold for large-scale activation.",
            'P0R03370: This process can be formalised by modelling a key molecular component, such as a voltage-gated ion channel, as a particle in a bistable potential V(x) with two wells (e.g., "closed" and "open") separated by an energy barrier DeltaV. The weak, periodic signal from the selected quantum state, with amplitude A and frequency , is insufficient on its own to cause the channel to cross the barrier. However, the brain\'s intrinsic biological noise (thermal and synaptic), with intensity D, can occasionally provide the necessary energy boost.',
            "P0R03371: The rate of noise-induced barrier crossing is given by the Kramers rate, rK. In the presence of the weak signal, the rates of hopping between the two wells become asymmetric. Stochastic resonance occurs when the characteristic timescale of the noise-induced hopping aligns with the period of the weak signal, leading to a massive amplification of the system's response. The signal-to-noise ratio (SNR) of the output is maximized at an optimal, non-zero noise level, Dopt, and can be approximated by:",
            "P0R03372: SNR(D2ADeltaV)2exp(D2DeltaV)",
            "P0R03373: The Psi-field, via Guided Einselection, provides the coherent, information-rich signal (A), and the brain's own thermal and synaptic noise (D) provides the energy that amplifies it. This can trigger a single action potential that then propagates as a neuronal avalanche through the critical network of Layer 4.",
            "P0R03374: This mechanism reframes the role of biological noise from a mere nuisance to a necessary and functional component of consciousness.",
            "P0R03375: The brain does not work despite noise; it works because of it.",
            "P0R03376: Noise is the energy source that the quasicritical brain harnesses to amplify the subtle, information-rich signals of the Psi-field, providing a deeply physical and elegant solution to the amplification problem of downward causation.",
            "P0R03377: The Free Energy Principle (FEP) as a Universal Mandate",
            "P0R03378: Source Material: The introduction of Karl Friston's FEP as the fundamental variational principle governing the dynamics of any self-organising system, positing it as the computational expression of the third axiom (Teleological Optimisation).",
            "P0R03379: P0R03379",
            "P0R03380: Mapping the SCPN to Hierarchical Predictive Coding (HPC)",
            "P0R03381: Source Material: The core of the chapter, explicitly mapping the SCPN's architecture to the HPC algorithm. It will define the downward projections as the system's Generative Model and the upward projections as the propagation of Prediction Error.",
            "P0R03382: P0R03382",
            "P0R03383: Active Inference: Action as the Fulfilment of Prophecy",
            'P0R03384: Source Material: The explanation of how "action" (or any physical process) is recast under the FEP as a means of changing the world to make it conform to the system\'s predictions, thereby reducing surprise. This unifies perception and action.',
            "P0R03385: P0R03385",
        ),
        "test_protocols": (
            "preserve Mechanism 2: Quantum Stochastic Resonance (QSR) source-accounting boundary",
        ),
        "null_results": (
            "Mechanism 2: Quantum Stochastic Resonance (QSR) is not empirical validation evidence",
        ),
        "variables": ("mechanism_2_quantum_stochastic_resonance_qsr",),
        "validation_targets": ("preserve records P0R03368-P0R03385",),
        "null_controls": (
            "mechanism_2_quantum_stochastic_resonance_qsr must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class Mechanism2QuantumStochasticResonanceQsrP0r03368Spec:
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
class Mechanism2QuantumStochasticResonanceQsrP0r03368SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Mechanism2QuantumStochasticResonanceQsrP0r03368Spec, ...]
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


def build_mechanism_2_quantum_stochastic_resonance_qsr_p0r03368_specs(
    source_records: list[dict[str, Any]],
) -> Mechanism2QuantumStochasticResonanceQsrP0r03368SpecBundle:
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

    specs: list[Mechanism2QuantumStochasticResonanceQsrP0r03368Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Mechanism2QuantumStochasticResonanceQsrP0r03368Spec(
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
        "title": "Paper 0 " + "Mechanism 2: Quantum Stochastic Resonance (QSR)" + " Specs",
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
        "next_source_boundary": "P0R03386",
    }
    return Mechanism2QuantumStochasticResonanceQsrP0r03368SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Mechanism2QuantumStochasticResonanceQsrP0r03368SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_mechanism_2_quantum_stochastic_resonance_qsr_p0r03368_specs(
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


def render_report(bundle: Mechanism2QuantumStochasticResonanceQsrP0r03368SpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Mechanism 2: Quantum Stochastic Resonance (QSR)" + " Specs",
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
    bundle: Mechanism2QuantumStochasticResonanceQsrP0r03368SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_mechanism_2_quantum_stochastic_resonance_qsr_p0r03368_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_mechanism_2_quantum_stochastic_resonance_qsr_p0r03368_validation_specs_{date_tag}.md"
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
