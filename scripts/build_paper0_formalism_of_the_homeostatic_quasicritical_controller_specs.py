#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Formalism of the Homeostatic Quasicritical Controller spec builder
"""Promote Paper 0 Formalism of the Homeostatic Quasicritical Controller records."""

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
    "P0R02869",
    "P0R02870",
    "P0R02871",
    "P0R02872",
    "P0R02873",
    "P0R02874",
    "P0R02875",
    "P0R02876",
    "P0R02877",
    "P0R02878",
    "P0R02879",
    "P0R02880",
    "P0R02881",
    "P0R02882",
    "P0R02883",
    "P0R02884",
    "P0R02885",
    "P0R02886",
    "P0R02887",
    "P0R02888",
    "P0R02889",
    "P0R02890",
    "P0R02891",
    "P0R02892",
    "P0R02893",
)
CLAIM_BOUNDARY = "source-bounded formalism of the homeostatic quasicritical controller source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "formalism_of_the_homeostatic_quasicritical_controller.formalism_of_the_homeostatic_quasicritical_controller": {
        "context_id": "formalism_of_the_homeostatic_quasicritical_controller",
        "validation_protocol": "paper0.formalism_of_the_homeostatic_quasicritical_controller.formalism_of_the_homeostatic_quasicritical_controller",
        "canonical_statement": "The source-bounded component 'Formalism of the Homeostatic Quasicritical Controller' preserves Paper 0 records P0R02869-P0R02893 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02869:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02870:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02871:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02872:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02873:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02874:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02875:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02876:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02877:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02878:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02879:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02880:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02881:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02882:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02883:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02884:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02885:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02886:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02887:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02888:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02889:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02890:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02891:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02892:formalism_of_the_homeostatic_quasicritical_controller",
            "P0R02893:formalism_of_the_homeostatic_quasicritical_controller",
        ),
        "source_formulae": (
            "P0R02869: Formalism of the Homeostatic Quasicritical Controller",
            "P0R02870: This section provides the formal mathematical implementation of the Self-Organised Criticality (SOC) mechanism. It defines a minimal, robust Homeostatic Quasicritical Controller designed to stabilise the network dynamics of each layer (L) around the critical point (sigmaL -> 1) while maintaining a target level of coherence (RL*). The controller's state is defined by the effective branching parameter (sigmaL) and the coherence order parameter (RL). The control laws are a set of coupled differential equations that dynamically adjust the intra-layer coupling strengths (KijL) and the effective noise (L) based on feedback from the system's current state.",
            "P0R02871: The stability of this control system is proven through the definition of a Lyapunov function,",
            "P0R02872: VL = (sigmaL1) + L(RLRL*).",
            'P0R02873: This function mathematically represents the "error" or distance of the system from its desired state (critical branching and target coherence). The provided control laws are designed to ensure that the time derivative of this function (VL) is always negative, guaranteeing that the system will robustly and automatically converge to the target basin of attraction. Furthermore, the "Inter-layer hook," which uses the coherence of adjacent layers (RL1) as feed-forward and feedback gains, ensures that this homeostatic control is coordinated across the entire SCPN hierarchy, allowing the entire network to maintain a global Griffiths-like phase without requiring fine-tuning at any individual layer.',
            'P0R02874: If the "edge of chaos" is the perfect state for reality to be in, this section is the detailed engineering blueprint for the smart thermostat that keeps it there. It\'s a feedback loop that is constantly monitoring and making tiny adjustments to maintain that perfect balance.',
            "P0R02875: The thermostat constantly checks two vital signs for any given layer of reality:",
            "P0R02876: Activity Level (sigmaL): Is things getting too quiet and rigid, or too noisy and chaotic? The goal is to keep this number right at 1.",
            "P0R02877: Synchrony Level (RL): Are things working together harmoniously, or is it a mess of disconnection?",
            "P0R02878: Based on these two readings, the controller adjusts two main dials:",
            "P0R02879: Connection Strength (K): If things are too quiet, it strengthens the connections between components to liven things up. If they're too chaotic, it weakens them slightly.",
            'P0R02880: Background Noise (): If the system is getting too rigid, it injects a tiny bit of creative "static" or noise to loosen things up.',
            'P0R02881: The beautiful part is that we\'ve used a special mathematical proof (a Lyapunov function) to show that this thermostat is guaranteed to work. It will always guide the system back to the perfect "edge of chaos," making the entire network of reality autonomously intelligent and adaptive.',
            "P0R02882: P0R02882",
            "P0R02883: The Allostatic Scaling Law (Revision 11.08):",
            "P0R02884: The stability of the quasicritical regime across scales (from Glia:Neuron to Gaia:Species) is governed by the Allostatic Bound. This ensures that top-down informational modulation ($H_{int}$) does not exceed the metabolic/energy supply of the supporting substrate. The metabolic feasibility of control is constrained by:",
            "P0R02885: $$\\pi_{metabolic} \\ge \\sum_L | \\zeta_L \\Psi_{Global} \\cdot \\sigma_L |$$",
            "P0R02886: Where $\\pi_{metabolic}$ is the layer-specific energy flux (e.g., CMRO2/ATP in Layer 4). This law prevents 'Dyscritia'-the catastrophic decoupling of informational dynamics from physical support, leading to seizures at the neural scale or ecological collapse at the Gaian scale.",
            "P0R02887: P0R02887",
            "P0R02888: Negentropy-Thermal Back-reaction Balance",
            "P0R02889: The reduction of local variational free energy ($\\Delta F < 0$) via negentropy injection from the $\\Psi$-field must satisfy the entropy-flux balance with the aqueous substrate. We define the Thermal Exhaust Gradient:",
            "P0R02890: $$\\dot{Q}_{exhaust} = T_{bath} \\left( \\dot{S}_{env} - \\frac{1}{\\epsilon_b} \\dot{I}_{conscious} \\right) \\geq 0$$",
            'P0R02891: To preserve the $\\Delta \\approx 1.64 \\text{ eV}$ energy gap in Layer 1, the local temperature change $\\Delta T$ must satisfy $\\Delta T < \\Delta / k_B$. If the conscious intent rate $\\dot{I}$ exceeds this exhaust capacity, the Allostatic Bound triggers a "Cooling-H',
            "P0R02892: P0R02892",
            "P0R02893: P0R02893",
        ),
        "test_protocols": (
            "preserve Formalism of the Homeostatic Quasicritical Controller source-accounting boundary",
        ),
        "null_results": (
            "Formalism of the Homeostatic Quasicritical Controller is not empirical validation evidence",
        ),
        "variables": ("formalism_of_the_homeostatic_quasicritical_controller",),
        "validation_targets": ("preserve records P0R02869-P0R02893",),
        "null_controls": (
            "formalism_of_the_homeostatic_quasicritical_controller must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class FormalismOfTheHomeostaticQuasicriticalControllerSpec:
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
class FormalismOfTheHomeostaticQuasicriticalControllerSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[FormalismOfTheHomeostaticQuasicriticalControllerSpec, ...]
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


def build_formalism_of_the_homeostatic_quasicritical_controller_specs(
    source_records: list[dict[str, Any]],
) -> FormalismOfTheHomeostaticQuasicriticalControllerSpecBundle:
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

    specs: list[FormalismOfTheHomeostaticQuasicriticalControllerSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            FormalismOfTheHomeostaticQuasicriticalControllerSpec(
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
        "title": "Paper 0 " + "Formalism of the Homeostatic Quasicritical Controller" + " Specs",
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
        "next_source_boundary": "P0R02894",
    }
    return FormalismOfTheHomeostaticQuasicriticalControllerSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> FormalismOfTheHomeostaticQuasicriticalControllerSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_formalism_of_the_homeostatic_quasicritical_controller_specs(
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


def render_report(bundle: FormalismOfTheHomeostaticQuasicriticalControllerSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Formalism of the Homeostatic Quasicritical Controller" + " Specs",
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
    bundle: FormalismOfTheHomeostaticQuasicriticalControllerSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_formalism_of_the_homeostatic_quasicritical_controller_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_formalism_of_the_homeostatic_quasicritical_controller_validation_specs_{date_tag}.md"
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
