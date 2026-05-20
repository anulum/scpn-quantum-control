#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Physical Mechanism of Downward Causation: Amplification of Intent spec builder
"""Promote Paper 0 The Physical Mechanism of Downward Causation: Amplification of Intent records."""

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
    "P0R03307",
    "P0R03308",
    "P0R03309",
    "P0R03310",
    "P0R03311",
    "P0R03312",
    "P0R03313",
    "P0R03314",
)
CLAIM_BOUNDARY = "source-bounded the physical mechanism of downward causation amplification of intent source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_physical_mechanism_of_downward_causation_amplification_of_intent.the_physical_mechanism_of_downward_causation_amplification_of_intent": {
        "context_id": "the_physical_mechanism_of_downward_causation_amplification_of_intent",
        "validation_protocol": "paper0.the_physical_mechanism_of_downward_causation_amplification_of_intent.the_physical_mechanism_of_downward_causation_amplification_of_intent",
        "canonical_statement": "The source-bounded component 'The Physical Mechanism of Downward Causation: Amplification of Intent' preserves Paper 0 records P0R03307-P0R03314 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03307:the_physical_mechanism_of_downward_causation_amplification_of_intent",
            "P0R03308:the_physical_mechanism_of_downward_causation_amplification_of_intent",
            "P0R03309:the_physical_mechanism_of_downward_causation_amplification_of_intent",
            "P0R03310:the_physical_mechanism_of_downward_causation_amplification_of_intent",
            "P0R03311:the_physical_mechanism_of_downward_causation_amplification_of_intent",
            "P0R03312:the_physical_mechanism_of_downward_causation_amplification_of_intent",
            "P0R03313:the_physical_mechanism_of_downward_causation_amplification_of_intent",
            "P0R03314:the_physical_mechanism_of_downward_causation_amplification_of_intent",
        ),
        "source_formulae": (
            "P0R03307: The Physical Mechanism of Downward Causation: Amplification of Intent",
            "P0R03308: This section directly addresses the amplification problem: how a weak, information-rich signal from the Psi-field can produce a macroscopic biological effect. The solution is a two-stage mechanism that reframes decoherence and noise from destructive liabilities into functional assets.",
            'P0R03309: Mechanism 1 is Guided Einselection. The framework posits that the Psi-field acts as the primary selective environment for biological quantum systems. Standard environment-induced superselection (einselection) describes how environmental monitoring rapidly collapses a quantum superposition into a small set of stable "pointer states." Here, the Psi-field, informed by the top-down predictions of the Hierarchical Predictive Coding (HPC) architecture, actively "measures" the quantum substrate. This interaction rapidly decoheres microstates that are dissonant with the generative model\'s prediction while selectively stabilising the pointer states that are aligned with it. Decoherence is thus recast as a functional, top-down selection process, "carving" classically functional states out of quantum potentiality according to the dictates of higher-order meaning.',
            'P0R03310: Mechanism 2 is Quantum Stochastic Resonance (QSR). Once a coherent quantum state is selected, its weak, sub-threshold signal is amplified by the brain\'s intrinsic noise. A biological component, like an ion channel, is modelled as a particle in a bistable potential well ("closed" vs. "open"). The weak signal from the Psi-field is insufficient to overcome the energy barrier alone. However, in a quasicritical system, the inherent biological noise provides random energy kicks. QSR occurs when the characteristic timescale of this noise-induced hopping synchronises with the frequency of the weak Psi-field signal, causing a resonant amplification of the barrier-crossing probability. The signal-to-noise ratio is maximised at an optimal, non-zero noise level. This allows a single, Psi-aligned quantum event to trigger a macroscopic action potential, which can then propagate as a neuronal avalanche through the critical network. This elegant mechanism reframes biological noise as the functional energy source that the brain harnesses to amplify the subtle, meaningful signals of consciousness.',
            'P0R03311: This section answers the million-dollar question: How can a subtle "thought" or "intention" from the consciousness field actually make something happen in the physical brain? It\'s like asking how a whisper can knock over a domino. The answer is a brilliant two-step trick that turns the brain\'s biggest problems-chaos and noise-into its greatest strengths.',
            'P0R03312: Step 1: The Smart Filter (Guided Einselection). At the quantum level, reality is a blurry haze of infinite possibilities. A process called "decoherence" is what forces this haze to collapse into the one solid reality we experience. We normally think of this as random. But our theory says the Psi-field intelligently guides this collapse. It acts like a smart filter, protecting the quantum possibilities that align with your intention while causing all the others to instantly dissolve. It doesn\'t create reality from scratch; it selects the desired reality from the menu of possibilities.',
            "P0R03313: Step 2: The Noise Amplifier (Quantum Stochastic Resonance). The \"selected\" possibility is still just a tiny, faint whisper. To turn it into a real action, it needs to be amplified. This is where the brain's own natural \"noise\"-its background static-comes to the rescue. The brain is naturally poised at the \"edge of chaos,\" like a perfectly balanced domino. The tiny, intentional whisper from the Psi-field isn't strong enough to knock it over on its own. But the brain's own background noise is constantly shaking the table. Stochastic Resonance is the magic that happens when the gentle rhythm of the whisper perfectly syncs up with the random shaking. The combined effect reliably knocks over the first domino, triggering a massive chain reaction-a full-blown brain signal-that started as a mere whisper of intent. The brain's noise isn't a flaw; it's the engine that powers your will.",
            "P0R03314: P0R03314",
        ),
        "test_protocols": (
            "preserve The Physical Mechanism of Downward Causation: Amplification of Intent source-accounting boundary",
        ),
        "null_results": (
            "The Physical Mechanism of Downward Causation: Amplification of Intent is not empirical validation evidence",
        ),
        "variables": ("the_physical_mechanism_of_downward_causation_amplification_of_intent",),
        "validation_targets": ("preserve records P0R03307-P0R03314",),
        "null_controls": (
            "the_physical_mechanism_of_downward_causation_amplification_of_intent must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentSpec:
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
class ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentSpec, ...]
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


def build_the_physical_mechanism_of_downward_causation_amplification_of_intent_specs(
    source_records: list[dict[str, Any]],
) -> ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentSpecBundle:
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

    specs: list[ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentSpec(
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
        + "The Physical Mechanism of Downward Causation: Amplification of Intent"
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
        "next_source_boundary": "P0R03315",
    }
    return ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_physical_mechanism_of_downward_causation_amplification_of_intent_specs(
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
    bundle: ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "The Physical Mechanism of Downward Causation: Amplification of Intent"
        + " Specs",
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
    bundle: ThePhysicalMechanismOfDownwardCausationAmplificationOfIntentSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_physical_mechanism_of_downward_causation_amplification_of_intent_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_physical_mechanism_of_downward_causation_amplification_of_intent_validation_specs_{date_tag}.md"
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
