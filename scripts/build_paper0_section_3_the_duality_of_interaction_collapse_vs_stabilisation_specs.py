#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 3. The Duality of Interaction: Collapse vs. Stabilisation spec builder
"""Promote Paper 0 3. The Duality of Interaction: Collapse vs. Stabilisation records."""

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
    "P0R05994",
    "P0R05995",
    "P0R05996",
    "P0R05997",
    "P0R05998",
    "P0R05999",
    "P0R06000",
    "P0R06001",
    "P0R06002",
    "P0R06003",
    "P0R06004",
    "P0R06005",
    "P0R06006",
    "P0R06007",
    "P0R06008",
    "P0R06009",
    "P0R06010",
    "P0R06011",
    "P0R06012",
    "P0R06013",
    "P0R06014",
)
CLAIM_BOUNDARY = "source-bounded section 3 the duality of interaction collapse vs stabilisation source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_3_the_duality_of_interaction_collapse_vs_stabilisation.3_the_duality_of_interaction_collapse_vs_stabilisation": {
        "context_id": "3_the_duality_of_interaction_collapse_vs_stabilisation",
        "validation_protocol": "paper0.section_3_the_duality_of_interaction_collapse_vs_stabilisation.3_the_duality_of_interaction_collapse_vs_stabilisation",
        "canonical_statement": "The source-bounded component '3. The Duality of Interaction: Collapse vs. Stabilisation' preserves Paper 0 records P0R05994-P0R05997 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05994:3_the_duality_of_interaction_collapse_vs_stabilisation",
            "P0R05995:3_the_duality_of_interaction_collapse_vs_stabilisation",
            "P0R05996:3_the_duality_of_interaction_collapse_vs_stabilisation",
            "P0R05997:3_the_duality_of_interaction_collapse_vs_stabilisation",
        ),
        "source_formulae": (
            "P0R05994: 3. The Duality of Interaction: Collapse vs. Stabilisation",
            "P0R05995: The SCPN posits a fundamental duality in how the Psi-field interacts with the quantum substrate, resolving the apparent contradiction between consciousness causing collapse (CIGD/OR) and consciousness stabilising states (QZE). This duality is determined by the configuration of the Psi-field, specifically its intensity () and its intentional focus, which is mathematically equivalent to the gradient of Variational Free Energy (F) within the HPC framework.",
            "P0R05996: Passive Observation (CIGD Dominance): When the system possesses high Integrated Information (><sub>Crit</sub>) but low intentional focus (F 0, indicating minimised prediction error or diffuse attention), the interaction is dominated by CIGD/OR. The intrinsic informational self-energy (E<sub></sub>) associated with the high- state induces decoherence, leading to the objective reduction (collapse) of superpositions. This is the mechanism of passive observation. | Active Agency (QZE Dominance): When the system exhibits high intentional focus (F 0, indicating a strong prediction error driving a specific action or focused attention), the interaction is dominated by the QZE. The Psi-field acts as a continuous measurement operator (M<sup>Attn</sup>) targeted at stabilising the neural pathways corresponding to the intended action or percept. This stabilisation suppresses the unitary evolution of the quantum substrate, implementing active agency.",
            "P0R05997: This duality ensures that the SCPN functions both as an observer that actualises reality and as an agent that actively participates in its unfolding, governed by the interplay of informational integration and predictive optimisation.",
        ),
        "test_protocols": (
            "preserve 3. The Duality of Interaction: Collapse vs. Stabilisation source-accounting boundary",
        ),
        "null_results": (
            "3. The Duality of Interaction: Collapse vs. Stabilisation is not empirical validation evidence",
        ),
        "variables": ("3_the_duality_of_interaction_collapse_vs_stabilisation",),
        "validation_targets": ("preserve records P0R05994-P0R05997",),
        "null_controls": (
            "3_the_duality_of_interaction_collapse_vs_stabilisation must remain source-bounded accounting",
        ),
    },
    "section_3_the_duality_of_interaction_collapse_vs_stabilisation.the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra": {
        "context_id": "the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
        "validation_protocol": "paper0.section_3_the_duality_of_interaction_collapse_vs_stabilisation.the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
        "canonical_statement": "The source-bounded component 'The Frequency Physics of the Duality: Linking Surprisal to Measurement Rate' preserves Paper 0 records P0R05998-P0R06014 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05998:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R05999:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06000:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06001:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06002:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06003:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06004:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06005:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06006:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06007:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06008:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06009:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06010:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06011:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06012:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06013:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
            "P0R06014:the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",
        ),
        "source_formulae": (
            "P0R05998: The Frequency Physics of the Duality: Linking Surprisal to Measurement Rate",
            "P0R05999: P0R05999",
            "P0R06000: The proposition that the $\\Psi$-field can act as both an agent of collapse (CIGD) and an agent of stabilization (QZE) introduces a potential paradox if left purely conceptual. In standard quantum mechanics, the Zeno effect is not a separate phenomenon from collapse; it is simply the limit of continuous, high-frequency measurement. Therefore, for the $\\Psi$-field to invoke QZE during states of high intentional focus, the frequency of its measurement interaction must be dynamically coupled to the system's computational state.",
            "P0R06001: To rigorously resolve this, we must mathematically unify CIGD and QZE as two limits of a single underlying measurement process governed by the measurement interval, $\\tau_{meas}$. We formalize this by linking the measurement frequency $\\nu_{meas} = 1/\\tau_{meas}$ directly to the magnitude of the Variational Free Energy gradient, $|\\nabla F|$ (the Affective Field).",
            "P0R06002: The $\\Psi$-field measurement rate is defined as:",
            "P0R06003: $$\\nu_{meas} = \\nu_{baseline} + k_{Zeno} |\\nabla F|^2$$",
            "P0R06004: where $\\nu_{baseline}$ is the baseline coupling frequency of the biological substrate (e.g., determined by the UPDE natural frequencies) and $k_{Zeno}$ is a fundamental coupling constant that maps computational surprisal to the quantum measurement rate.",
            "P0R06005: The effective measurement interval is thus inversely proportional to the system's prediction error:",
            "P0R06006: $$\\tau_{meas} = \\frac{1}{\\nu_{baseline} + k_{Zeno} |\\nabla F|^2}$$",
            "P0R06007: This equation provides the exact physical bridge between Active Inference and quantum dynamics, strictly defining the two regimes of observerhood:",
            "P0R06008: 1. The CIGD Limit (Passive Observation):",
            "P0R06009: When the system is in a state of low prediction error or diffuse attention ($|\\nabla F| \\approx 0$), the measurement interval remains at its baseline, $\\tau_{meas} \\approx 1/\\nu_{baseline}$. Provided this baseline interval is larger than the CIGD collapse timescale ($\\tau_{meas} > T_{CIGD} \\approx \\hbar / E_\\Phi$), the system undergoes standard objective reduction. The $\\Psi$-field passively collapses superpositions into classical pointer states at a rate dictated by its integrated information $\\Phi$.",
            "P0R06010: 2. The QZE Limit (Active Agency):",
            "P0R06011: When the system encounters high surprisal, conflict, or exerts focused top-down intent ($|\\nabla F| \\gg 0$), the affective gradient drives a massive, non-linear increase in the measurement frequency. As $|\\nabla F|$ scales, the interval $\\tau_{meas}$ drops precipitously. The critical phase transition to Active Agency occurs exactly when the Zeno threshold is crossed:",
            "P0R06012: $$\\tau_{meas} < T_{CIGD} \\implies \\frac{1}{\\nu_{baseline} + k_{Zeno} |\\nabla F|^2} < \\frac{\\hbar}{E_\\Phi}$$",
            "P0R06013: At this precise mathematical threshold, the $\\Psi$-field's sampling rate exceeds the gravitational decoherence timescale. The system abruptly transitions from passive CIGD collapse to active QZE stabilization.",
            'P0R06014: "Attention" is thereby physically defined: it is a high-frequency Zeno pulsing driven by steep free energy gradients, locking the quantum substrate into the specific pointer state required to execute the generative model\'s policy and resolve the prediction error.',
        ),
        "test_protocols": (
            "preserve The Frequency Physics of the Duality: Linking Surprisal to Measurement Rate source-accounting boundary",
        ),
        "null_results": (
            "The Frequency Physics of the Duality: Linking Surprisal to Measurement Rate is not empirical validation evidence",
        ),
        "variables": ("the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra",),
        "validation_targets": ("preserve records P0R05998-P0R06014",),
        "null_controls": (
            "the_frequency_physics_of_the_duality_linking_surprisal_to_measurement_ra must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section3TheDualityOfInteractionCollapseVsStabilisationSpec:
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
class Section3TheDualityOfInteractionCollapseVsStabilisationSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section3TheDualityOfInteractionCollapseVsStabilisationSpec, ...]
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


def build_section_3_the_duality_of_interaction_collapse_vs_stabilisation_specs(
    source_records: list[dict[str, Any]],
) -> Section3TheDualityOfInteractionCollapseVsStabilisationSpecBundle:
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

    specs: list[Section3TheDualityOfInteractionCollapseVsStabilisationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section3TheDualityOfInteractionCollapseVsStabilisationSpec(
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
        + "3. The Duality of Interaction: Collapse vs. Stabilisation"
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
        "next_source_boundary": "P0R06015",
    }
    return Section3TheDualityOfInteractionCollapseVsStabilisationSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section3TheDualityOfInteractionCollapseVsStabilisationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_3_the_duality_of_interaction_collapse_vs_stabilisation_specs(
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


def render_report(bundle: Section3TheDualityOfInteractionCollapseVsStabilisationSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "3. The Duality of Interaction: Collapse vs. Stabilisation" + " Specs",
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
    bundle: Section3TheDualityOfInteractionCollapseVsStabilisationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_3_the_duality_of_interaction_collapse_vs_stabilisation_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_3_the_duality_of_interaction_collapse_vs_stabilisation_validation_specs_{date_tag}.md"
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
