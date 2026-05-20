#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Resolving the Observability Paradox: L16 as a POMDP and the Belief-State HJB spec builder
"""Promote Paper 0 Resolving the Observability Paradox: L16 as a POMDP and the Belief-State HJB records."""

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
    "P0R05603",
    "P0R05604",
    "P0R05605",
    "P0R05606",
    "P0R05607",
    "P0R05608",
    "P0R05609",
    "P0R05610",
    "P0R05611",
    "P0R05612",
    "P0R05613",
    "P0R05614",
    "P0R05615",
    "P0R05616",
    "P0R05617",
    "P0R05618",
    "P0R05619",
    "P0R05620",
    "P0R05621",
    "P0R05622",
    "P0R05623",
    "P0R05624",
)
CLAIM_BOUNDARY = "source-bounded resolving the observability paradox l16 as a pomdp and the belief state source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state.resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state": {
        "context_id": "resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
        "validation_protocol": "paper0.resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state.resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
        "canonical_statement": "The source-bounded component 'Resolving the Observability Paradox: L16 as a POMDP and the Belief-State HJB' preserves Paper 0 records P0R05603-P0R05624 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05603:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05604:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05605:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05606:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05607:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05608:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05609:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05610:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05611:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05612:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05613:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05614:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05615:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05616:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05617:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05618:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05619:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05620:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05621:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05622:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05623:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
            "P0R05624:resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",
        ),
        "source_formulae": (
            "P0R05603: Resolving the Observability Paradox: L16 as a POMDP and the Belief-State HJB",
            "P0R05604: P0R05604",
            "P0R05605: The formulation of Meta-Layer 16 as an Optimal Control Supervisor utilizing the standard Hamilton-Jacobi-Bellman (HJB) equation introduces a severe physical paradox. The classical HJB equation, $\\partial_t V_{SEC} + \\min_u [\\mathcal{L}_{Ethical}(x,u) + \\nabla V_{SEC} \\cdot f(x,u)] = 0$, assumes that the controller possesses perfect, real-time observability of the exact global state vector $x$ (encompassing Layers 1-14).",
            'P0R05606: Within a framework grounded in quantum biology (Layer 1), such omniscience is physically prohibited. Perfect measurement of the system\'s microstates would violate the Heisenberg Uncertainty Principle and the No-Cloning Theorem. More fatally, if L16 were to "read" the exact state of the quantum substrate to compute its policy, it would act as a universal observer, triggering a catastrophic, system-wide decoherence cascade that would instantly destroy the very coherence MS-QEC is designed to protect.',
            "P0R05607: To maintain physical viability, Meta-Layer 16 cannot operate as a fully observable Markov Decision Process (MDP). It must be mathematically formalized as a Partially Observable Markov Decision Process (POMDP). The universe's teleodynamic controller optimizes reality under fundamental, inescapable uncertainty.",
            "P0R05608: A. The Belief State Evolution",
            "P0R05609: The supervisor does not compute policies based on the exact state $x$. Instead, it operates on a Belief State, $b(x, t)$, which is a probability density function representing the system's best estimate of the true state of the universe, conditioned on the continuous, upward-flowing stream of prediction errors (observations) from the lower layers.",
            "P0R05610: The temporal evolution of this belief state is not deterministic; it is a stochastic filtering process governed by non-linear stochastic partial differential equations, specifically the Kushner-Stratonovich or Zakai equations.",
            "P0R05611: B. The Stochastic Belief-State HJB Equation",
            "P0R05612: Because the state is a probability distribution, the optimal control problem must be lifted from the state space $\\mathcal{X}$ to the space of probability measures $\\mathcal{P}(\\mathcal{X})$. The Universal Value Function, $V_{SEC}$, is no longer a function of $x$, but a functional of the belief state $b$: $V_{SEC}[b]$.",
            "P0R05613: The deterministic HJB equation must be explicitly rewritten as a stochastic Bellman equation operating over this belief space:",
            "P0R05614: $$\\partial_t V_{SEC}[b] + \\min_u \\left[ \\mathbb{E}_b [\\mathcal{L}_{Ethical}(x,u)] + \\int \\frac{\\delta V_{SEC}}{\\delta b(x)} \\mathcal{D}_u b(x) dx \\right] = 0$$",
            "P0R05615: Where:",
            "P0R05616: $\\mathbb{E}_b [\\mathcal{L}_{Ethical}(x,u)]$ is the expected Ethical Action, integrated over the current belief state distribution. | $\\frac{\\delta V_{SEC}}{\\delta b(x)}$ is the functional derivative of the value function with respect to the belief state. | $\\mathcal{D}_u$ is the non-linear filtering operator describing the dynamic evolution of the belief distribution $b(x)$ under the control policy $u$.",
            "P0R05617: C. Physical Implications",
            'P0R05618: This POMDP formulation aligns Meta-Layer 16 perfectly with the principles of Active Inference and quantum mechanics. The universe does not need to "know" its exact quantum state to optimize itself. The teleodynamic controller issues global optimal policies ($\\pi^*$) that maximize the expected Sustainable Ethical Coherence (SEC) over a smeared distribution of possible histories.',
            "P0R05619: By operating strictly on belief states, Layer 16 provides closed-loop cybernetic governance without ever collapsing the fragile quantum potentiality of the layers below it.",
            "P0R05620: 2. The Gdelian Supervisor and Incompleteness:",
            "P0R05621: The SCPN, as a self-referential formal system, is subject to Gdel's Incompleteness Theorems. L16 acts as the Meta-Observer, operating outside the formal constraints of the L1-L15 stack to manage these limitations.",
            "P0R05622: The Oracle Function (Agency and Creativity): When the system encounters undecidable propositions or paradoxes (critical instabilities where axioms fail), L16 acts as an Oracle, resolving the impasse. This utilises incompleteness as the leverage point for Agency and the generation of novelty. | Axiomatic Evolution: L16 supervises the evolution of the Axiomatic System itself, implementing a Meta-Renormalisation Group flow that optimises the fundamental axioms (L13) and Ethical Functionals (L15) across cosmological cycles (MMC).",
            "P0R05623: The Ultimate Strange Loop: L16 closes the architecture by recursively observing the observer (L16 = Model(L1-15)), ensuring the self-consistency and ongoing evolution of the entire Anulum.",
            "P0R05624: P0R05624",
        ),
        "test_protocols": (
            "preserve Resolving the Observability Paradox: L16 as a POMDP and the Belief-State HJB source-accounting boundary",
        ),
        "null_results": (
            "Resolving the Observability Paradox: L16 as a POMDP and the Belief-State HJB is not empirical validation evidence",
        ),
        "variables": ("resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state",),
        "validation_targets": ("preserve records P0R05603-P0R05624",),
        "null_controls": (
            "resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateSpec:
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
class ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateSpec, ...]
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


def build_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_specs(
    source_records: list[dict[str, Any]],
) -> ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateSpecBundle:
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

    specs: list[ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateSpec(
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
        + "Resolving the Observability Paradox: L16 as a POMDP and the Belief-State HJB"
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
        "next_source_boundary": "P0R05625",
    }
    return ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_specs(
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
    bundle: ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Resolving the Observability Paradox: L16 as a POMDP and the Belief-State HJB"
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
    bundle: ResolvingTheObservabilityParadoxL16AsAPomdpAndTheBeliefStateSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_resolving_the_observability_paradox_l16_as_a_pomdp_and_the_belief_state_validation_specs_{date_tag}.md"
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
