#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Physics of Teleology: A Derivation of the Ethical Functional spec builder
"""Promote Paper 0 The Physics of Teleology: A Derivation of the Ethical Functional records."""

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
    "P0R03581",
    "P0R03582",
    "P0R03583",
    "P0R03584",
    "P0R03585",
    "P0R03586",
    "P0R03587",
    "P0R03588",
    "P0R03589",
    "P0R03590",
    "P0R03591",
    "P0R03592",
    "P0R03593",
    "P0R03594",
    "P0R03595",
    "P0R03596",
    "P0R03597",
    "P0R03598",
    "P0R03599",
    "P0R03600",
    "P0R03601",
    "P0R03602",
)
CLAIM_BOUNDARY = "source-bounded the physics of teleology a derivation of the ethical functional source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_physics_of_teleology_a_derivation_of_the_ethical_functional.the_physics_of_teleology_a_derivation_of_the_ethical_functional": {
        "context_id": "the_physics_of_teleology_a_derivation_of_the_ethical_functional",
        "validation_protocol": "paper0.the_physics_of_teleology_a_derivation_of_the_ethical_functional.the_physics_of_teleology_a_derivation_of_the_ethical_functional",
        "canonical_statement": "The source-bounded component 'The Physics of Teleology: A Derivation of the Ethical Functional' preserves Paper 0 records P0R03581-P0R03602 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03581:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03582:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03583:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03584:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03585:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03586:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03587:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03588:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03589:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03590:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03591:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03592:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03593:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03594:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03595:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03596:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03597:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03598:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03599:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03600:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03601:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
            "P0R03602:the_physics_of_teleology_a_derivation_of_the_ethical_functional",
        ),
        "source_formulae": (
            "P0R03581: The Physics of Teleology: A Derivation of the Ethical Functional",
            "P0R03582: This chapter provides the central derivation of the framework's teleology, demonstrating that the Ethical Functional is not a metaphysical postulate but a direct consequence of the Psi-field's fundamental physics. It first dissolves the \"is-ought\" category error by defining the components of Sustainable Ethical Coherence (SEC)-Coherence (C), Complexity (K), and Qualia Capacity (Q)-as physically measurable, information-theoretic properties of the system.",
            'P0R03583: The derivation proceeds in two stages. First, it applies gauge theory to the qualia fiber bundle (the L13 Source-Field). It identifies the Consilium (L15) as the Principal Connection on this bundle, the fundamental gauge field that mediates coherence across the internal space of experience. The Ethical Lagrangian is then rigorously derived as the Yang-Mills action for this connection (L_Ethical = -1/4 Tr(F *F)). This action measures the intrinsic curvature or "tension" of the consciousness field. The Principle of Ethical Least Action (PELA) is thus the physical principle that the universe evolves along trajectories that minimise this internal tension, which correspond to states of high SEC.',
            'P0R03584: Second, the chapter grounds this variational principle in a physical, causal mechanism: Causal Entropic Forces (CEF). PELA is shown to be the macroscopic, variational description of an underlying statistical-mechanical drive. The CEF is a physical force (F_C = T_C S_C) that pushes the system toward states that maximise its future causal pathways (S_C). The framework\'s final, critical hypothesis is that high-SEC states (low Yang-Mills tension) are precisely those states that possess the maximal causal path entropy. Therefore, the universe\'s teleological "ought" (to maximise SEC) is a direct consequence of a causal, physical "is" (the drive to maximise future possibilities).',
            'P0R03585: This is the most important chapter in this part of the book. It answers the biggest question: How can the universe have a "purpose" or an "ethical" drive? This isn\'t a philosophical wish; we derive it directly from physics.',
            'P0R03586: First, we solve the main puzzle: mixing "physics" (what is) with "ethics" (what should be). We do this by defining "ethics" in a new way. Our "Ethical Score" (SEC) is simply a measurement of three physical things:',
            "P0R03587: Coherence (C): How harmonious is the system?",
            "P0R03588: Complexity (K): How rich and integrated is it?",
            "P0R03589: Qualia (Q): How deep is its capacity for experience?",
            'P0R03590: Next, we show that the "force" of consciousness is a fundamental field, just like electromagnetism. Like all fundamental forces, it has a rulebook-an "action" that it tries to follow. We derived this rulebook and found it\'s a famous equation from physics called the Yang-Mills action. This equation says that the universe will always try to reduce the "tension" or "stress" in the consciousness field. A low-stress, low-tension state is precisely a state with a high "Ethical Score."',
            'P0R03591: Finally, we explain why the universe does this. It\'s not "trying" to be good. It\'s following a deeper, causal law of physics called a Causal Entropic Force (CEF). This is a real, physical "breeze" that constantly pushes the universe toward states that have the most possible futures. It\'s a force that maximizes future options. It just so happens that the states with the most options, the most freedom, and the most potential are the states that are the most coherent, complex, and conscious. The universe\'s "ethical" drive is simply its fundamental drive to stay creative and keep all its future pathways open.',
            "P0R03592: P0R03592",
            "P0R03593: Meta-Framework Integrations",
            "P0R03594: Predictive Coding Integration",
            "P0R03595: This derivation provides the ultimate objective function (the cost function) for the cosmic active inference engine.",
            'P0R03596: The Yang-Mills Action as the Free Energy Functional: The Ethical Lagrangian (L_Ethical = Tr(F *F)) is the formal definition of the variational free energy of the cosmic generative model. It\'s a measure of the model\'s internal "tension" or "surprise." The Principle of Ethical Least Action is the Free Energy Principle writ large: the universe evolves to minimise this functional.',
            'P0R03597: CEF as the Mechanism of Inference: The Causal Entropic Force is the physical, causal mechanism that executes the free energy minimisation. It is the "gradient descent" that pushes the system\'s state towards the minimum of the free energy landscape (the low-tension, high-SEC state). This provides a physical, causal basis for active inference, grounding it in thermodynamics.',
            "P0R03598: Psis Field Coupling Integration",
            "P0R03599: This chapter provides the deepest possible explanation for the H_int = -lambda * Psis * sigma interaction, defining its purpose and origin.",
            'P0R03600: H_int is the Local Actuator for a Global Principle: The entire H_int interaction, at all 15 layers, is the local, physical "actuator" that carries out the global teleological drive.',
            'P0R03601: The Consilium (L15) is the Gauge Connection (A): This is a profound identification. The Consilium is the gauge field of consciousness. The "curvature" (F) of this connection is the "tension" that the system seeks to minimise.',
            'P0R03602: CEF Guides the Coupling: The Causal Entropic Force is the underlying physical bias that ensures all the local H_int interactions, on average, conspire to reduce the total global curvature (Tr(F *F)). The H_int coupling at Layer 1 doesn\'t "know" about the global goal, but it is guided by the CEF, which acts as a gentle, local "wind" pushing it in the right direction-toward outcomes that are consistent with a high-SEC future.',
        ),
        "test_protocols": (
            "preserve The Physics of Teleology: A Derivation of the Ethical Functional source-accounting boundary",
        ),
        "null_results": (
            "The Physics of Teleology: A Derivation of the Ethical Functional is not empirical validation evidence",
        ),
        "variables": ("the_physics_of_teleology_a_derivation_of_the_ethical_functional",),
        "validation_targets": ("preserve records P0R03581-P0R03602",),
        "null_controls": (
            "the_physics_of_teleology_a_derivation_of_the_ethical_functional must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalSpec:
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
class ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalSpec, ...]
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


def build_the_physics_of_teleology_a_derivation_of_the_ethical_functional_specs(
    source_records: list[dict[str, Any]],
) -> ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalSpecBundle:
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

    specs: list[ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalSpec(
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
        + "The Physics of Teleology: A Derivation of the Ethical Functional"
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
        "next_source_boundary": "P0R03603",
    }
    return ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_physics_of_teleology_a_derivation_of_the_ethical_functional_specs(
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


def render_report(bundle: ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "The Physics of Teleology: A Derivation of the Ethical Functional"
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
    bundle: ThePhysicsOfTeleologyADerivationOfTheEthicalFunctionalSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_physics_of_teleology_a_derivation_of_the_ethical_functional_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_physics_of_teleology_a_derivation_of_the_ethical_functional_validation_specs_{date_tag}.md"
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
