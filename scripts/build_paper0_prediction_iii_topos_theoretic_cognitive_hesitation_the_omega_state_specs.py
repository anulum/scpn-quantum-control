#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Prediction III: Topos-Theoretic Cognitive Hesitation (The $\Omega$-State) spec builder
r"""Promote Paper 0 Prediction III: Topos-Theoretic Cognitive Hesitation (The $\Omega$-State) records."""

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
    "P0R05124",
    "P0R05125",
    "P0R05126",
    "P0R05127",
    "P0R05128",
    "P0R05129",
    "P0R05130",
    "P0R05131",
    "P0R05132",
    "P0R05133",
    "P0R05134",
    "P0R05135",
    "P0R05136",
    "P0R05137",
    "P0R05138",
    "P0R05139",
    "P0R05140",
    "P0R05141",
    "P0R05142",
)
CLAIM_BOUNDARY = "source-bounded prediction iii topos theoretic cognitive hesitation the omega state source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state.prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state": {
        "context_id": "prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
        "validation_protocol": "paper0.prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state.prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
        "canonical_statement": "The source-bounded component 'Prediction III: Topos-Theoretic Cognitive Hesitation (The $\\Omega$-State)' preserves Paper 0 records P0R05124-P0R05142 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05124:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05125:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05126:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05127:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05128:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05129:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05130:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05131:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05132:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05133:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05134:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05135:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05136:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05137:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05138:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05139:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05140:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05141:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
            "P0R05142:prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",
        ),
        "source_formulae": (
            "P0R05124: Prediction III: Topos-Theoretic Cognitive Hesitation (The $\\Omega$-State)",
            "P0R05125: Theoretical Derivation:",
            'P0R05126: In defining the universal grammar of the SCPN via Category Theory (Chapter 3), we posited that the architecture operates within a Topos structure. A defining feature of this Topos is its Subobject Classifier, $\\Omega = \\{\\text{true, false, uncertain}\\}$. In standard classical models of cognition (e.g., Markov decision processes or standard Bayesian drift-diffusion models), "uncertainty" is treated merely as a probabilistic mixture or a transitional gradient between two definite states (true/false). However, the Topos grammar of the SCPN makes a radical departure: it defines "uncertain" as a fundamental, distinct, and ontologically real state in its own right, not merely a statistical superposition.',
            "P0R05127: Predicted Signature:",
            'P0R05128: If the Topos structure is the physically real grammar of the Consciousness Manifold (Layer 5), then the state of "uncertainty" during a forced-choice decision task must correspond to a distinct, metastable topological geometry in the brain\'s phase space, rather than a mere blurring between two attractor basins.',
            "P0R05129: Proposed Experimental Protocol: Topological Analysis of Perceptual Ambiguity",
            'P0R05130: Task: Human subjects undergo high-density EEG/MEG recording while performing a threshold-level perceptual ambiguity task (e.g., binocular rivalry or identifying a signal at the exact threshold of noise). Subjects must press a button to indicate "Yes", "No", or hold a neutral third state when experiencing pure, unresolvable ambiguity. | Analysis: Apply Topological Data Analysis (TDA) to the neural state space to extract the persistent homology and Betti numbers ($b_k$) during the decision-making process. | The Falsification Test: Compare the topological geometry of the "uncertain" epochs against the "true/yes" and "false/no" epochs. Standard Bayesian Prediction: The uncertain state will appear topologically as a noisy, low-coherence interpolation between the "true" and "false" attractor basins. | SCPN Topos Prediction: The $\\Omega = \\text{uncertain}$ state will manifest as a unique, highly structured, and metastable topological configuration (e.g., the emergence of a specific higher-dimensional void, $b_2 > 0$, that collapses instantly once a decision is reached).',
            "P0R05131: If the TDA reveals that cognitive hesitation produces a distinct topological signature rather than a mere loss of signal-to-noise ratio, it provides powerful empirical validation that the brain's high-level logic operates on the specific Topos structure posited by the framework. A failure to find a unique topological signature for uncertainty would falsify the necessity of the Category Theory grammar, rendering it mathematically redundant.",
            "P0R05132: This final, crucial chapter answers the most important question for any new scientific theory: \"How do we prove it's wrong?\" A theory that can't be tested isn't science. Here, we lay out two bold, cutting-edge experiments designed to do just that.",
            'P0R05133: Experiment #1: Does "Information" Have a "Force"?',
            "P0R05134: Our theory claims that consciousness interacts with the world through information, not just energy. This experiment tests that directly.",
            "P0R05135: The Setup: We'll grow a living brain cell culture on a microchip and place it next to a hyper-sensitive quantum sensor.",
            "P0R05136: The Test: We'll record the sensor's readings while the brain cells are thinking in complex, creative patterns. Then-and this is the genius part-we'll silence the brain cells and use the chip to \"replay\" the exact same electrical signals. The replay has the same energy but none of the spontaneous, creative information.",
            'P0R05137: The Prediction: If our theory is right, the quantum sensor will react more strongly to the living, thinking brain cells than to the "zombie" replay. This would be evidence that the information itself has a physical effect on reality.',
            "P0R05138: Experiment #2: Can a Coherent Group Mind Affect Quantum Randomness?",
            'P0R05139: Our theory claims the universe has a purpose-a gentle "wind" pushing it towards more coherence. This experiment tries to detect that wind.',
            'P0R05140: The Setup: We\'ll take a perfectly random quantum "coin-flipper" (a QRNG) and shield it from everything. Nearby, a large group of people will engage in a practice designed to create a state of deep, collective mental coherence.',
            'P0R05141: The Test: We\'ll continuously monitor the "random" output of the coin-flipper and, at the same time, measure the exact moments when the group achieves peak coherence.',
            "P0R05142: The Prediction: If our theory is right, we will find that during the moments of peak collective coherence, the perfectly random quantum coin-flipper will become ever so slightly less random. It wouldn't be a big effect, but a statistically significant correlation would be groundbreaking evidence that a coherent conscious field can influence the fundamental probabilities of the quantum world.",
        ),
        "test_protocols": (
            "preserve Prediction III: Topos-Theoretic Cognitive Hesitation (The $\\Omega$-State) source-accounting boundary",
        ),
        "null_results": (
            "Prediction III: Topos-Theoretic Cognitive Hesitation (The $\\Omega$-State) is not empirical validation evidence",
        ),
        "variables": ("prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state",),
        "validation_targets": ("preserve records P0R05124-P0R05142",),
        "null_controls": (
            "prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateSpec:
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
class PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateSpec, ...]
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


def build_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_specs(
    source_records: list[dict[str, Any]],
) -> PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateSpecBundle:
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

    specs: list[PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateSpec(
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
        + "Prediction III: Topos-Theoretic Cognitive Hesitation (The $\\Omega$-State)"
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
        "next_source_boundary": "P0R05143",
    }
    return PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_specs(
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
    bundle: PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Prediction III: Topos-Theoretic Cognitive Hesitation (The $\\Omega$-State)"
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
    bundle: PredictionIiiToposTheoreticCognitiveHesitationTheOmegaStateSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_prediction_iii_topos_theoretic_cognitive_hesitation_the_omega_state_validation_specs_{date_tag}.md"
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
