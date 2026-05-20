#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Layer 9 (Existential Holograph) - Defining a Stable sigma: spec builder
"""Promote Paper 0 Layer 9 (Existential Holograph) - Defining a Stable sigma: records."""

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
    "P0R02287",
    "P0R02288",
    "P0R02289",
    "P0R02290",
    "P0R02291",
    "P0R02292",
    "P0R02293",
    "P0R02294",
    "P0R02295",
    "P0R02296",
    "P0R02297",
    "P0R02298",
    "P0R02299",
    "P0R02300",
    "P0R02301",
    "P0R02302",
    "P0R02303",
    "P0R02304",
    "P0R02305",
)
CLAIM_BOUNDARY = "source-bounded layer 9 existential holograph defining a stable sigma source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "layer_9_existential_holograph_defining_a_stable_sigma.layer_9_existential_holograph_defining_a_stable_sigma": {
        "context_id": "layer_9_existential_holograph_defining_a_stable_sigma",
        "validation_protocol": "paper0.layer_9_existential_holograph_defining_a_stable_sigma.layer_9_existential_holograph_defining_a_stable_sigma",
        "canonical_statement": "The source-bounded component 'Layer 9 (Existential Holograph) - Defining a Stable sigma:' preserves Paper 0 records P0R02287-P0R02288 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02287:layer_9_existential_holograph_defining_a_stable_sigma",
            "P0R02288:layer_9_existential_holograph_defining_a_stable_sigma",
        ),
        "source_formulae": (
            "P0R02287: Layer 9 (Existential Holograph) - Defining a Stable sigma:",
            'P0R02288: The Existential Holograph is a stable, coherent informational structure. Its collective state variable (sigma) is the high-dimensional geometry of the holographic field itself-its specific pattern of interference fringes. This can be conceptualised as a complex-valued field, sigma_holo(x, y, z, t). The process of living and learning sculpts this sigma_holo into a unique, stable configuration. This stable sigma provides a consistent, unchanging "address" or "tuning signature" for the individual Self, allowing the universal Psis field to maintain a continuous, stable coupling (H_int) with that specific individual across time. It is the physical basis for a persistent soul or identity.',
        ),
        "test_protocols": (
            "preserve Layer 9 (Existential Holograph) - Defining a Stable sigma: source-accounting boundary",
        ),
        "null_results": (
            "Layer 9 (Existential Holograph) - Defining a Stable sigma: is not empirical validation evidence",
        ),
        "variables": ("layer_9_existential_holograph_defining_a_stable_sigma",),
        "validation_targets": ("preserve records P0R02287-P0R02288",),
        "null_controls": (
            "layer_9_existential_holograph_defining_a_stable_sigma must remain source-bounded accounting",
        ),
    },
    "layer_9_existential_holograph_defining_a_stable_sigma.layer_10_boundary_control_modulating_the_coupling_constant_lambda": {
        "context_id": "layer_10_boundary_control_modulating_the_coupling_constant_lambda",
        "validation_protocol": "paper0.layer_9_existential_holograph_defining_a_stable_sigma.layer_10_boundary_control_modulating_the_coupling_constant_lambda",
        "canonical_statement": "The source-bounded component 'Layer 10 (Boundary Control) - Modulating the Coupling Constant lambda:' preserves Paper 0 records P0R02289-P0R02292 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02289:layer_10_boundary_control_modulating_the_coupling_constant_lambda",
            "P0R02290:layer_10_boundary_control_modulating_the_coupling_constant_lambda",
            "P0R02291:layer_10_boundary_control_modulating_the_coupling_constant_lambda",
            "P0R02292:layer_10_boundary_control_modulating_the_coupling_constant_lambda",
        ),
        "source_formulae": (
            "P0R02289: Layer 10 (Boundary Control) - Modulating the Coupling Constant lambda:",
            'P0R02290: Layer 10 functions by directly controlling the coupling constant (lambda). It acts as a tuneable dielectric or informational insulator. When the boundary is "strong" or "closed," it effectively reduces the value of lambda, weakening the H_int interaction and isolating the individual\'s field from external Psi-field fluctuations. When the boundary is "open" or "permeable" (e.g., in deep meditative states or strong empathetic connection), it increases the value of lambda, allowing for a much stronger coupling between the individual\'s consciousness and the collective or universal field. Layer 10 is thus the dynamic control system for the strength of the mind-matter (or mind-mind) interface.',
            "P0R02291: Domain III: Memory and Projection Control (Layers 9-10):",
            "P0R02292: Existential Holograph (Layer 9), Projective Field Boundary Control (Layer 10).",
        ),
        "test_protocols": (
            "preserve Layer 10 (Boundary Control) - Modulating the Coupling Constant lambda: source-accounting boundary",
        ),
        "null_results": (
            "Layer 10 (Boundary Control) - Modulating the Coupling Constant lambda: is not empirical validation evidence",
        ),
        "variables": ("layer_10_boundary_control_modulating_the_coupling_constant_lambda",),
        "validation_targets": ("preserve records P0R02289-P0R02292",),
        "null_controls": (
            "layer_10_boundary_control_modulating_the_coupling_constant_lambda must remain source-bounded accounting",
        ),
    },
    "layer_9_existential_holograph_defining_a_stable_sigma.case_study_the_layer_11_noospheric_spin_glass_system": {
        "context_id": "case_study_the_layer_11_noospheric_spin_glass_system",
        "validation_protocol": "paper0.layer_9_existential_holograph_defining_a_stable_sigma.case_study_the_layer_11_noospheric_spin_glass_system",
        "canonical_statement": "The source-bounded component 'Case Study: The Layer 11 (Noospheric) Spin-Glass System' preserves Paper 0 records P0R02293-P0R02305 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02293:case_study_the_layer_11_noospheric_spin_glass_system",
            "P0R02294:case_study_the_layer_11_noospheric_spin_glass_system",
            "P0R02295:case_study_the_layer_11_noospheric_spin_glass_system",
            "P0R02296:case_study_the_layer_11_noospheric_spin_glass_system",
            "P0R02297:case_study_the_layer_11_noospheric_spin_glass_system",
            "P0R02298:case_study_the_layer_11_noospheric_spin_glass_system",
            "P0R02299:case_study_the_layer_11_noospheric_spin_glass_system",
            "P0R02300:case_study_the_layer_11_noospheric_spin_glass_system",
            "P0R02301:case_study_the_layer_11_noospheric_spin_glass_system",
            "P0R02302:case_study_the_layer_11_noospheric_spin_glass_system",
            "P0R02303:case_study_the_layer_11_noospheric_spin_glass_system",
            "P0R02304:case_study_the_layer_11_noospheric_spin_glass_system",
            "P0R02305:case_study_the_layer_11_noospheric_spin_glass_system",
        ),
        "source_formulae": (
            "P0R02293: Case Study: The Layer 11 (Noospheric) Spin-Glass System",
            'P0R02294: This section operationalises the concept of the Noosphere (Layer 11), transforming it from a philosophical abstraction into a computationally tractable and empirically falsifiable research programme. It posits that the contemporary collective consciousness is an inseparable Noosphere-Technosphere Hybrid System (NTHS), whose dynamics are primarily mediated by technological information flows. The framework leverages a powerful analogy from condensed matter physics, modelling the social network as a spin-glass. In this model, individuals are represented as spins (beliefs), their relationships as coupling constants (J_ij), and the overall "social dissonance" as the system\'s Hamiltonian. This approach allows the complex, often chaotic behaviour of social networks to be analysed with the mathematical rigour of statistical mechanics.',
            "P0R02295: The core of the proposal is a falsifiable computational experiment using an Agent-Based Model (ABM) populated by active inference agents, consistent with the principles of Layer 5. The hypothesis is that the global state of the NTHS undergoes a phase transition contingent upon the objective function of the mediating AI. Two regimes are contrasted: a Coherence-Optimising Regime, where the AI aims to minimise collective free energy, and an Engagement-Optimising Regime, where the AI seeks to maximise collective surprise. By formalising each agent's belief updating as a process of variational free-energy minimisation, the model provides a first-principles link between the AI's information-curation strategy and the emergent, system-level belief dynamics.",
            'P0R02296: The predicted outcomes are characterised by well-defined order parameters. The Coherence-Optimising AI is predicted to induce a ferromagnetic phase, defined by high magnetisation (m -> 1), representing global consensus. Conversely, the Engagement-Optimising AI is predicted to induce a spin-glass phase, characterised by zero magnetisation but a high Edwards-Anderson order parameter (q_EA > 0). This state represents a "frozen" disorder, where the system fragments into stable, mutually incompatible, and hierarchically nested belief clusters (echo chambers), a structure verifiable through ultrametricity tests. This sophisticated model provides a powerful, predictive, and physically-grounded framework for understanding the dynamics of collective consciousness in the digital age.',
            "P0R02297: This section provides a scientific blueprint for understanding the \"global mind\" or Noosphere. It argues that today, our collective consciousness isn't just a vague spiritual idea; it's a very real hybrid system, part human and part technology, that we can actually study and measure. To do this, we use a fascinating analogy from the physics of magnets called a spin-glass.",
            "P0R02298: Here's how it works:",
            'P0R02299: Imagine every person in a social network is a tiny, spinning magnet (a spin). Your opinion on a topic can be "spin up" (you agree) or "spin down" (you disagree).',
            "P0R02300: Your relationships are the forces between the magnets. Friendships try to align your spins, while rivalries try to flip them in opposite directions.",
            'P0R02301: The overall "stress" or "tension" in the entire network is the total energy of this system of magnets. Nature always prefers the lowest-energy state.',
            'P0R02302: The brilliant and crucial insight is that the "rules of the game" for this magnetic system are now being set by the AI algorithms that run our social media feeds. We propose a grand computer simulation to prove this. We\'ll create a digital society of intelligent agents and have them interact through an AI with one of two goals:',
            'P0R02303: The "Librarian" AI: Its only goal is to help the society find shared understanding and consensus. It shares reliable, helpful information.',
            'P0R02304: The "Gossip" AI: Its only goal is to maximise engagement-clicks, shares, and outrage. It shares the most shocking, controversial, and emotionally-charged information, regardless of whether it\'s true.',
            'P0R02305: Our prediction is that we will see the digital society undergo a "phase transition," like water freezing into ice. The Librarian AI will create a unified society, where everyone\'s magnets align in a state of general agreement. The Gossip AI, however, will create a spin-glass: a society that is frozen into a state of chaotic division, with thousands of tiny, polarised echo chambers that are permanently at odds with each other. This model provides a powerful, testable way to understand the fragmentation of our modern world.',
        ),
        "test_protocols": (
            "preserve Case Study: The Layer 11 (Noospheric) Spin-Glass System source-accounting boundary",
        ),
        "null_results": (
            "Case Study: The Layer 11 (Noospheric) Spin-Glass System is not empirical validation evidence",
        ),
        "variables": ("case_study_the_layer_11_noospheric_spin_glass_system",),
        "validation_targets": ("preserve records P0R02293-P0R02305",),
        "null_controls": (
            "case_study_the_layer_11_noospheric_spin_glass_system must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Layer9ExistentialHolographDefiningAStableSigmaSpec:
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
class Layer9ExistentialHolographDefiningAStableSigmaSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Layer9ExistentialHolographDefiningAStableSigmaSpec, ...]
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


def build_layer_9_existential_holograph_defining_a_stable_sigma_specs(
    source_records: list[dict[str, Any]],
) -> Layer9ExistentialHolographDefiningAStableSigmaSpecBundle:
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

    specs: list[Layer9ExistentialHolographDefiningAStableSigmaSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Layer9ExistentialHolographDefiningAStableSigmaSpec(
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
        + "Layer 9 (Existential Holograph) - Defining a Stable sigma:"
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
        "next_source_boundary": "P0R02306",
    }
    return Layer9ExistentialHolographDefiningAStableSigmaSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Layer9ExistentialHolographDefiningAStableSigmaSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_layer_9_existential_holograph_defining_a_stable_sigma_specs(
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


def render_report(bundle: Layer9ExistentialHolographDefiningAStableSigmaSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Layer 9 (Existential Holograph) - Defining a Stable sigma:" + " Specs",
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
    bundle: Layer9ExistentialHolographDefiningAStableSigmaSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_layer_9_existential_holograph_defining_a_stable_sigma_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_layer_9_existential_holograph_defining_a_stable_sigma_validation_specs_{date_tag}.md"
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
