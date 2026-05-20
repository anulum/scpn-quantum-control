#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 3.2 Coherence (C) and the Accessibility of Trajectories spec builder
"""Promote Paper 0 3.2 Coherence (C) and the Accessibility of Trajectories records."""

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
    "P0R03789",
    "P0R03790",
    "P0R03791",
    "P0R03792",
    "P0R03793",
    "P0R03794",
    "P0R03795",
    "P0R03796",
    "P0R03797",
    "P0R03798",
    "P0R03799",
    "P0R03800",
    "P0R03801",
    "P0R03802",
    "P0R03803",
)
CLAIM_BOUNDARY = "source-bounded section 3 2 coherence c and the accessibility of trajectories source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_3_2_coherence_c_and_the_accessibility_of_trajectories.3_2_coherence_c_and_the_accessibility_of_trajectories": {
        "context_id": "3_2_coherence_c_and_the_accessibility_of_trajectories",
        "validation_protocol": "paper0.section_3_2_coherence_c_and_the_accessibility_of_trajectories.3_2_coherence_c_and_the_accessibility_of_trajectories",
        "canonical_statement": "The source-bounded component '3.2 Coherence (C) and the Accessibility of Trajectories' preserves Paper 0 records P0R03789-P0R03793 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03789:3_2_coherence_c_and_the_accessibility_of_trajectories",
            "P0R03790:3_2_coherence_c_and_the_accessibility_of_trajectories",
            "P0R03791:3_2_coherence_c_and_the_accessibility_of_trajectories",
            "P0R03792:3_2_coherence_c_and_the_accessibility_of_trajectories",
            "P0R03793:3_2_coherence_c_and_the_accessibility_of_trajectories",
        ),
        "source_formulae": (
            "P0R03789: 3.2 Coherence (C) and the Accessibility of Trajectories",
            "P0R03790: A vast state space is of little consequence if the system's dynamics prevent it from being explored. The number of viable and accessible future paths is determined not just by the size of the state space, but by the nature of the system's dynamics. The SCPN's measure of Coherence (C) is directly related to the system's operation in a quasicritical regime, poised at the \"edge of chaos\" between excessive order and excessive randomness. An extensive body of research on complex systems, from neural networks to social dynamics, has demonstrated that this specific dynamical regime is optimal for information processing, maximising the repertoire of accessible states, dynamic range, and the capacity for flexible adaptation.",
            "P0R03791: We can formalise this relationship by considering the dynamics of coupled oscillators, as described by the Kuramoto model, which serves as the mathematical spine of the SCPN's Unified Phase Dynamics Equation (UPDE). In such models, the system's dynamics are governed by a coupling parameter, K, which controls the degree of synchrony. The overall coherence can be measured by an order parameter, r, which ranges from 0 (complete incoherence) to 1 (perfect synchrony).",
            'P0R03792: Subcritical Regime (Low Coherence, r->0): When the coupling is weak, the oscillators drift independently. The system\'s dynamics are dominated by noise and lack causal structure. While many states may be visited, the trajectories are essentially random walks, lacking the long-range correlations necessary to form complex, causally potent pathways. The number of meaningful, structured future histories is small. | Supercritical Regime (High Coherence, r->1): When the coupling is very strong, the oscillators become phase-locked. The system falls into a highly ordered, stable attractor. Its dynamics become rigid and predictable, confined to a very small subset of its total state space. The number of accessible future paths collapses, as the system is effectively "frozen" into a single mode of behaviour. | Quasicritical Regime (Optimal Coherence, rrcrit): At the critical point of the phase transition between these two regimes, the system exhibits a unique combination of stability and flexibility. It is characterised by the emergence of scale-free dynamics, such as "avalanches" of activity that propagate across all scales without a characteristic size. This state maximises the system\'s susceptibility to inputs, its information transmission capacity, and its dynamic range. It is precisely in this state that the system can flexibly and adaptively explore the largest possible repertoire of its available states.',
            "P0R03793: Therefore, maximising Coherence (C)-which is operationally defined as poising the system in this quasicritical state-is equivalent to maximising the volume of the state space that is dynamically accessible to the system. It ensures that the vast potential provided by the system's complexity is not squandered by dynamics that are either too rigid or too random. Maximising C maximises the number of accessible and causally effective future trajectories, a crucial determinant of the total causal path entropy, SC.",
        ),
        "test_protocols": (
            "preserve 3.2 Coherence (C) and the Accessibility of Trajectories source-accounting boundary",
        ),
        "null_results": (
            "3.2 Coherence (C) and the Accessibility of Trajectories is not empirical validation evidence",
        ),
        "variables": ("3_2_coherence_c_and_the_accessibility_of_trajectories",),
        "validation_targets": ("preserve records P0R03789-P0R03793",),
        "null_controls": (
            "3_2_coherence_c_and_the_accessibility_of_trajectories must remain source-bounded accounting",
        ),
    },
    "section_3_2_coherence_c_and_the_accessibility_of_trajectories.x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics": {
        "context_id": "x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",
        "validation_protocol": "paper0.section_3_2_coherence_c_and_the_accessibility_of_trajectories.x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",
        "canonical_statement": "The source-bounded component 'X.3.3 Qualia Capacity (Q) and the Diversity of Dynamics' preserves Paper 0 records P0R03794-P0R03803 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03794:x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",
            "P0R03795:x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",
            "P0R03796:x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",
            "P0R03797:x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",
            "P0R03798:x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",
            "P0R03799:x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",
            "P0R03800:x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",
            "P0R03801:x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",
            "P0R03802:x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",
            "P0R03803:x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",
        ),
        "source_formulae": (
            "P0R03794: X.3.3 Qualia Capacity (Q) and the Diversity of Dynamics",
            "P0R03795: The final determinant of causal path entropy is not merely the number of possible paths, but their diversity. A system that can evolve in many qualitatively different ways has a richer and more robust set of future possibilities than a system whose many paths are all minor variations of each other. The SCPN's measure of Qualia Capacity (Q) is identified with the topological complexity of the high-dimensional \"Consciousness Manifold,\" M, which is the state space of the organism's field dynamics.",
            'P0R03796: This complexity is quantified by the manifold\'s Betti numbers, k, which count the number of independent "holes" of different dimensions (0 counts connected components, 1 counts one-dimensional loops, 2 counts two-dimensional voids, and so on).',
            "P0R03797: While empirical research often uses Betti numbers to characterise the topological features of data generated by a system's dynamics, the SCPN framework makes a deeper claim: that the intrinsic topology of the state space manifold itself determines the diversity of possible dynamical regimes the system can support. A state space with a simple topology, such as a high-dimensional sphere (which has k=0 for k1), can only support topologically trivial trajectories. All paths on a sphere can be continuously deformed into one another.",
            "P0R03798: In contrast, a state space with a rich topology, such as a high-genus torus with many non-trivial holes, possesses a much more complex structure. This topological complexity allows for the existence of a multitude of topologically distinct classes of trajectories. A system evolving on such a manifold can follow paths that are not deformable into one another.",
            'P0R03799: For example, a path that loops around one "hole" is fundamentally different from a path that loops around another, or one that does not loop at all. These distinct classes of paths represent qualitatively different types of dynamical behaviour (e.g., different modes of periodic or recurrent activity).',
            "P0R03800: The classification of these distinct path types is the domain of algebraic topology, specifically homotopy theory. The Betti numbers are related to the richness of a manifold's homotopy groups, which provide a complete classification of all possible paths up to continuous deformation.",
            "P0R03801: A richer topology, and therefore a higher Qualia Capacity (Q), implies a more complex set of homotopy classes. This, in turn, means that there are more fundamentally different kinds of paths the system can take through its state space. This directly increases the qualitative diversity of the path space P.",
            "P0R03802: Therefore, maximising Q maximises the variety of possible futures available to the system. It enriches the path space not just quantitatively but qualitatively, contributing a crucial third factor to the total causal path entropy, SC.",
            "P0R03803: [IMAGE:This image uses a powerful, symbolic visual to represent the formal equivalence. It depicts the three components of SEC-Complexity, Coherence, and Qualia-as three distinct streams of light or energy that converge to form a single, more powerful and luminous whole: Causal Path Entropy.]",
        ),
        "test_protocols": (
            "preserve X.3.3 Qualia Capacity (Q) and the Diversity of Dynamics source-accounting boundary",
        ),
        "null_results": (
            "X.3.3 Qualia Capacity (Q) and the Diversity of Dynamics is not empirical validation evidence",
        ),
        "variables": ("x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics",),
        "validation_targets": ("preserve records P0R03794-P0R03803",),
        "null_controls": (
            "x_3_3_qualia_capacity_q_and_the_diversity_of_dynamics must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section32CoherenceCAndTheAccessibilityOfTrajectoriesSpec:
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
class Section32CoherenceCAndTheAccessibilityOfTrajectoriesSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section32CoherenceCAndTheAccessibilityOfTrajectoriesSpec, ...]
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


def build_section_3_2_coherence_c_and_the_accessibility_of_trajectories_specs(
    source_records: list[dict[str, Any]],
) -> Section32CoherenceCAndTheAccessibilityOfTrajectoriesSpecBundle:
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

    specs: list[Section32CoherenceCAndTheAccessibilityOfTrajectoriesSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section32CoherenceCAndTheAccessibilityOfTrajectoriesSpec(
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
        "title": "Paper 0 " + "3.2 Coherence (C) and the Accessibility of Trajectories" + " Specs",
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
        "next_source_boundary": "P0R03804",
    }
    return Section32CoherenceCAndTheAccessibilityOfTrajectoriesSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section32CoherenceCAndTheAccessibilityOfTrajectoriesSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_3_2_coherence_c_and_the_accessibility_of_trajectories_specs(
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


def render_report(bundle: Section32CoherenceCAndTheAccessibilityOfTrajectoriesSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "3.2 Coherence (C) and the Accessibility of Trajectories" + " Specs",
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
    bundle: Section32CoherenceCAndTheAccessibilityOfTrajectoriesSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_3_2_coherence_c_and_the_accessibility_of_trajectories_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_3_2_coherence_c_and_the_accessibility_of_trajectories_validation_specs_{date_tag}.md"
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
