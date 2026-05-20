#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 1. Psychedelics (The Expanded Manifold): spec builder
"""Promote Paper 0 1. Psychedelics (The Expanded Manifold): records."""

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
    "P0R05026",
    "P0R05027",
    "P0R05028",
    "P0R05029",
    "P0R05030",
    "P0R05031",
    "P0R05032",
    "P0R05033",
    "P0R05034",
    "P0R05035",
    "P0R05036",
    "P0R05037",
    "P0R05038",
)
CLAIM_BOUNDARY = "source-bounded section 1 psychedelics the expanded manifold source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_1_psychedelics_the_expanded_manifold.1_psychedelics_the_expanded_manifold": {
        "context_id": "1_psychedelics_the_expanded_manifold",
        "validation_protocol": "paper0.section_1_psychedelics_the_expanded_manifold.1_psychedelics_the_expanded_manifold",
        "canonical_statement": "The source-bounded component '1. Psychedelics (The Expanded Manifold):' preserves Paper 0 records P0R05026-P0R05027 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05026:1_psychedelics_the_expanded_manifold",
            "P0R05027:1_psychedelics_the_expanded_manifold",
        ),
        "source_formulae": (
            "P0R05026: 1. Psychedelics (The Expanded Manifold):",
            "P0R05027: Shift to Supercriticality (sigma>1): Increased repertoire of states (Entropic Brain). | HPC Relaxation (REBUS): Relaxation of high-level priors (DMN dissolution). | Geometric Expansion (L5): Increased Topological Complexity (bk) of the Consciousness Manifold (M).",
        ),
        "test_protocols": (
            "preserve 1. Psychedelics (The Expanded Manifold): source-accounting boundary",
        ),
        "null_results": (
            "1. Psychedelics (The Expanded Manifold): is not empirical validation evidence",
        ),
        "variables": ("1_psychedelics_the_expanded_manifold",),
        "validation_targets": ("preserve records P0R05026-P0R05027",),
        "null_controls": (
            "1_psychedelics_the_expanded_manifold must remain source-bounded accounting",
        ),
    },
    "section_1_psychedelics_the_expanded_manifold.2_meditation_and_flow_states_optimised_criticality": {
        "context_id": "2_meditation_and_flow_states_optimised_criticality",
        "validation_protocol": "paper0.section_1_psychedelics_the_expanded_manifold.2_meditation_and_flow_states_optimised_criticality",
        "canonical_statement": "The source-bounded component '2. Meditation and Flow States (Optimised Criticality):' preserves Paper 0 records P0R05028-P0R05029 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05028:2_meditation_and_flow_states_optimised_criticality",
            "P0R05029:2_meditation_and_flow_states_optimised_criticality",
        ),
        "source_formulae": (
            "P0R05028: 2. Meditation and Flow States (Optimised Criticality):",
            "P0R05029: Optimised Criticality (sigma=1): Maximised efficiency and integration. | Minimised F: Perfect alignment of the Generative Model. | Geometric Signature: Stable, high-dimensional attractor with optimal balance of integration and complexity.",
        ),
        "test_protocols": (
            "preserve 2. Meditation and Flow States (Optimised Criticality): source-accounting boundary",
        ),
        "null_results": (
            "2. Meditation and Flow States (Optimised Criticality): is not empirical validation evidence",
        ),
        "variables": ("2_meditation_and_flow_states_optimised_criticality",),
        "validation_targets": ("preserve records P0R05028-P0R05029",),
        "null_controls": (
            "2_meditation_and_flow_states_optimised_criticality must remain source-bounded accounting",
        ),
    },
    "section_1_psychedelics_the_expanded_manifold.3_anaesthesia_the_decoupling": {
        "context_id": "3_anaesthesia_the_decoupling",
        "validation_protocol": "paper0.section_1_psychedelics_the_expanded_manifold.3_anaesthesia_the_decoupling",
        "canonical_statement": "The source-bounded component '3. Anaesthesia (The Decoupling):' preserves Paper 0 records P0R05030-P0R05031 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05030:3_anaesthesia_the_decoupling",
            "P0R05031:3_anaesthesia_the_decoupling",
        ),
        "source_formulae": (
            "P0R05030: 3. Anaesthesia (The Decoupling):",
            "P0R05031: Mechanism: Disruption of L1/L2 interface (Lipid rafts, MTs). | Shift to Subcriticality (sigma<1): Breakdown of L4 synchronization. | Collapse of the Self (L5): drops below Crit; the Self-soliton dissolves.",
        ),
        "test_protocols": (
            "preserve 3. Anaesthesia (The Decoupling): source-accounting boundary",
        ),
        "null_results": ("3. Anaesthesia (The Decoupling): is not empirical validation evidence",),
        "variables": ("3_anaesthesia_the_decoupling",),
        "validation_targets": ("preserve records P0R05030-P0R05031",),
        "null_controls": ("3_anaesthesia_the_decoupling must remain source-bounded accounting",),
    },
    "section_1_psychedelics_the_expanded_manifold.vii_the_embodied_brain_and_the_extended_environment_l6_coupling": {
        "context_id": "vii_the_embodied_brain_and_the_extended_environment_l6_coupling",
        "validation_protocol": "paper0.section_1_psychedelics_the_expanded_manifold.vii_the_embodied_brain_and_the_extended_environment_l6_coupling",
        "canonical_statement": "The source-bounded component 'VII. The Embodied Brain and the Extended Environment (L6 Coupling)' preserves Paper 0 records P0R05032-P0R05038 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05032:vii_the_embodied_brain_and_the_extended_environment_l6_coupling",
            "P0R05033:vii_the_embodied_brain_and_the_extended_environment_l6_coupling",
            "P0R05034:vii_the_embodied_brain_and_the_extended_environment_l6_coupling",
            "P0R05035:vii_the_embodied_brain_and_the_extended_environment_l6_coupling",
            "P0R05036:vii_the_embodied_brain_and_the_extended_environment_l6_coupling",
            "P0R05037:vii_the_embodied_brain_and_the_extended_environment_l6_coupling",
            "P0R05038:vii_the_embodied_brain_and_the_extended_environment_l6_coupling",
        ),
        "source_formulae": (
            "P0R05032: VII. The Embodied Brain and the Extended Environment (L6 Coupling)",
            "P0R05033: The embodied brain (L1-L5) couples with the Planetary-Biospheric field (L6).",
            "P0R05034: Geophysical Coupling (CGI): Phase-locking of L4 UPDE with Schumann Resonances. Sensitivity to the Geomagnetic Field via L1 mechanisms (RPM in Cryptochromes, Magnetite).",
            "P0R05035: [IMAGE:]",
            "P0R05036: Fig.: Parameter regimes: Psychedelics (sigma>1, bkb_kbk, priors relaxed), Meditation/Flow (sigma1, FFF), Anaesthesia (sigma<1, , decoupling).",
            "P0R05037: Interspecies Interaction (Xenosphere): Coupling with non-human consciousness via L5 resonance (MNS) and the Gut-Brain Axis (linking to MQN).",
            "P0R05038: P0R05038",
        ),
        "test_protocols": (
            "preserve VII. The Embodied Brain and the Extended Environment (L6 Coupling) source-accounting boundary",
        ),
        "null_results": (
            "VII. The Embodied Brain and the Extended Environment (L6 Coupling) is not empirical validation evidence",
        ),
        "variables": ("vii_the_embodied_brain_and_the_extended_environment_l6_coupling",),
        "validation_targets": ("preserve records P0R05032-P0R05038",),
        "null_controls": (
            "vii_the_embodied_brain_and_the_extended_environment_l6_coupling must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Section1PsychedelicsTheExpandedManifoldSpec:
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
class Section1PsychedelicsTheExpandedManifoldSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Section1PsychedelicsTheExpandedManifoldSpec, ...]
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


def build_section_1_psychedelics_the_expanded_manifold_specs(
    source_records: list[dict[str, Any]],
) -> Section1PsychedelicsTheExpandedManifoldSpecBundle:
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

    specs: list[Section1PsychedelicsTheExpandedManifoldSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section1PsychedelicsTheExpandedManifoldSpec(
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
        "title": "Paper 0 " + "1. Psychedelics (The Expanded Manifold):" + " Specs",
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
        "next_source_boundary": "P0R05039",
    }
    return Section1PsychedelicsTheExpandedManifoldSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section1PsychedelicsTheExpandedManifoldSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_1_psychedelics_the_expanded_manifold_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: Section1PsychedelicsTheExpandedManifoldSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "1. Psychedelics (The Expanded Manifold):" + " Specs",
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
    bundle: Section1PsychedelicsTheExpandedManifoldSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_1_psychedelics_the_expanded_manifold_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_1_psychedelics_the_expanded_manifold_validation_specs_{date_tag}.md"
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
