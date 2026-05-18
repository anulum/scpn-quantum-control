#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Quantum-to-Classical Transition: Amplification of Intent spec builder
"""Promote Paper 0 The Quantum-to-Classical Transition: Amplification of Intent records."""

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
    "P0R03360",
    "P0R03361",
    "P0R03362",
    "P0R03363",
    "P0R03364",
    "P0R03365",
    "P0R03366",
    "P0R03367",
)
CLAIM_BOUNDARY = "source-bounded the quantum to classical transition amplification of intent p0r03360 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_quantum_to_classical_transition_amplification_of_intent_p0r03360.the_quantum_to_classical_transition_amplification_of_intent": {
        "context_id": "the_quantum_to_classical_transition_amplification_of_intent",
        "validation_protocol": "paper0.the_quantum_to_classical_transition_amplification_of_intent_p0r03360.the_quantum_to_classical_transition_amplification_of_intent",
        "canonical_statement": "The source-bounded component 'The Quantum-to-Classical Transition: Amplification of Intent' preserves Paper 0 records P0R03360-P0R03361 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03360:the_quantum_to_classical_transition_amplification_of_intent",
            "P0R03361:the_quantum_to_classical_transition_amplification_of_intent",
        ),
        "source_formulae": (
            "P0R03360: The Quantum-to-Classical Transition: Amplification of Intent",
            'P0R03361: A central challenge for any theory involving downward causation is the amplification problem: how can a weak, non-local, and information-rich signal from the Psi-field produce a macroscopic, energy-dependent effect like the firing of a neuron? The SCPN proposes a two-stage mechanism that leverages the unique properties of biological systems operating at quasicriticality, reframing the "problem" of decoherence and noise into the solution for selection and amplification.',
        ),
        "test_protocols": (
            "preserve The Quantum-to-Classical Transition: Amplification of Intent source-accounting boundary",
        ),
        "null_results": (
            "The Quantum-to-Classical Transition: Amplification of Intent is not empirical validation evidence",
        ),
        "variables": ("the_quantum_to_classical_transition_amplification_of_intent",),
        "validation_targets": ("preserve records P0R03360-P0R03361",),
        "null_controls": (
            "the_quantum_to_classical_transition_amplification_of_intent must remain source-bounded accounting",
        ),
    },
    "the_quantum_to_classical_transition_amplification_of_intent_p0r03360.mechanism_1_guided_einselection": {
        "context_id": "mechanism_1_guided_einselection",
        "validation_protocol": "paper0.the_quantum_to_classical_transition_amplification_of_intent_p0r03360.mechanism_1_guided_einselection",
        "canonical_statement": "The source-bounded component 'Mechanism 1: Guided Einselection' preserves Paper 0 records P0R03362-P0R03367 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03362:mechanism_1_guided_einselection",
            "P0R03363:mechanism_1_guided_einselection",
            "P0R03364:mechanism_1_guided_einselection",
            "P0R03365:mechanism_1_guided_einselection",
            "P0R03366:mechanism_1_guided_einselection",
            "P0R03367:mechanism_1_guided_einselection",
        ),
        "source_formulae": (
            "P0R03362: Mechanism 1: Guided Einselection",
            'P0R03363: The transition from a quantum superposition to a definite classical state is governed by a process known as environment-induced superselection, or einselection. In this process, the environment continuously "monitors" a quantum system, causing a rapid collapse of most possible states while selecting for a small, stable set of "pointer states" that are robust to this interaction.',
            "P0R03364: The SCPN posits that the Psi-field is not merely a passive background but acts as the primary selective environment for biological quantum systems (Layer 1). The evolution of the quantum substrate's density matrix, rho, is described by a Lindblad master equation:",
            "P0R03365: $$ \\frac{d\\rho}{dt} = -\\frac{i}{\\hbar}[H, \\rho] + \\sum_k \\left( L_k \\rho L_k^\\dagger - \\frac{1}{2}{L_k^\\dagger L_k, \\rho} \\right) $$",
            "P0R03366: The interaction term in the system's Hamiltonian, Hint, couples the quantum substrate to the Psi-field. This term dominates the environmental interaction, meaning the Lindblad operators, Lk, are primarily determined by the state of the Psi-field.",
            'P0R03367: The Psi-field effectively "measures" the quantum substrate, causing rapid decoherence of quantum states that are dissonant with the top-down generative model propagated by the Hierarchical Predictive Coding architecture. Simultaneously, it stabilizes the pointer states that are aligned with the model\'s predictions. In this view, decoherence is not a destructive bug but a crucial feature: it is the physical mechanism that selects for functionally relevant, coherent states, effectively "carving" classical reality out of quantum potentiality according to the dictates of higher-order meaning.',
        ),
        "test_protocols": (
            "preserve Mechanism 1: Guided Einselection source-accounting boundary",
        ),
        "null_results": ("Mechanism 1: Guided Einselection is not empirical validation evidence",),
        "variables": ("mechanism_1_guided_einselection",),
        "validation_targets": ("preserve records P0R03362-P0R03367",),
        "null_controls": (
            "mechanism_1_guided_einselection must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheQuantumToClassicalTransitionAmplificationOfIntentP0r03360Spec:
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
class TheQuantumToClassicalTransitionAmplificationOfIntentP0r03360SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheQuantumToClassicalTransitionAmplificationOfIntentP0r03360Spec, ...]
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


def build_the_quantum_to_classical_transition_amplification_of_intent_p0r03360_specs(
    source_records: list[dict[str, Any]],
) -> TheQuantumToClassicalTransitionAmplificationOfIntentP0r03360SpecBundle:
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

    specs: list[TheQuantumToClassicalTransitionAmplificationOfIntentP0r03360Spec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheQuantumToClassicalTransitionAmplificationOfIntentP0r03360Spec(
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
        + "The Quantum-to-Classical Transition: Amplification of Intent"
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
        "next_source_boundary": "P0R03368",
    }
    return TheQuantumToClassicalTransitionAmplificationOfIntentP0r03360SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheQuantumToClassicalTransitionAmplificationOfIntentP0r03360SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_quantum_to_classical_transition_amplification_of_intent_p0r03360_specs(
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
    bundle: TheQuantumToClassicalTransitionAmplificationOfIntentP0r03360SpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Quantum-to-Classical Transition: Amplification of Intent" + " Specs",
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
    bundle: TheQuantumToClassicalTransitionAmplificationOfIntentP0r03360SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_quantum_to_classical_transition_amplification_of_intent_p0r03360_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_quantum_to_classical_transition_amplification_of_intent_p0r03360_validation_specs_{date_tag}.md"
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
