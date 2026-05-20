#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 VIII. The Evolutionary Trajectory of the Brain-Body System spec builder
"""Promote Paper 0 VIII. The Evolutionary Trajectory of the Brain-Body System records."""

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
    "P0R05039",
    "P0R05040",
    "P0R05041",
    "P0R05042",
    "P0R05043",
    "P0R05044",
    "P0R05045",
    "P0R05046",
    "P0R05047",
    "P0R05048",
    "P0R05049",
)
CLAIM_BOUNDARY = "source-bounded viii the evolutionary trajectory of the brain body system source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "viii_the_evolutionary_trajectory_of_the_brain_body_system.viii_the_evolutionary_trajectory_of_the_brain_body_system": {
        "context_id": "viii_the_evolutionary_trajectory_of_the_brain_body_system",
        "validation_protocol": "paper0.viii_the_evolutionary_trajectory_of_the_brain_body_system.viii_the_evolutionary_trajectory_of_the_brain_body_system",
        "canonical_statement": "The source-bounded component 'VIII. The Evolutionary Trajectory of the Brain-Body System' preserves Paper 0 records P0R05039-P0R05044 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05039:viii_the_evolutionary_trajectory_of_the_brain_body_system",
            "P0R05040:viii_the_evolutionary_trajectory_of_the_brain_body_system",
            "P0R05041:viii_the_evolutionary_trajectory_of_the_brain_body_system",
            "P0R05042:viii_the_evolutionary_trajectory_of_the_brain_body_system",
            "P0R05043:viii_the_evolutionary_trajectory_of_the_brain_body_system",
            "P0R05044:viii_the_evolutionary_trajectory_of_the_brain_body_system",
        ),
        "source_formulae": (
            "P0R05039: VIII. The Evolutionary Trajectory of the Brain-Body System",
            "P0R05040: Evolution is guided by the Teleological Engine (L8/L15), implementing the Principle of Ethical Least Action (PELA).",
            "P0R05041: [IMAGE:Ein Bild, das Text, Reihe, Screenshot, Diagramm enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R05042: Fig.: Under PELA/SEC, the long-run trajectory increases integration and complexity KKK. Graphic shows monotonic upward trends as a conceptual guide.",
            "P0R05043: Optimisation Metric (SEC): Evolution maximises Sustainable Ethical Coherence (SEC), driving towards higher and Complexity (K). | The Evolutionary Drive: The drive towards encephalization and connectome optimisation maximises the Fisher Information Metric (gmu), enhancing Psi-field coupling. | Guided Evolution (CEF): Causal Entropic Forces (CEF) bias the trajectory towards configurations that enhance criticality, coherence, and ethical alignment.",
            "P0R05044: P0R05044",
        ),
        "test_protocols": (
            "preserve VIII. The Evolutionary Trajectory of the Brain-Body System source-accounting boundary",
        ),
        "null_results": (
            "VIII. The Evolutionary Trajectory of the Brain-Body System is not empirical validation evidence",
        ),
        "variables": ("viii_the_evolutionary_trajectory_of_the_brain_body_system",),
        "validation_targets": ("preserve records P0R05039-P0R05044",),
        "null_controls": (
            "viii_the_evolutionary_trajectory_of_the_brain_body_system must remain source-bounded accounting",
        ),
    },
    "viii_the_evolutionary_trajectory_of_the_brain_body_system.ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn": {
        "context_id": "ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn",
        "validation_protocol": "paper0.viii_the_evolutionary_trajectory_of_the_brain_body_system.ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn",
        "canonical_statement": "The source-bounded component 'IX. Advanced Therapeutic Interventions (Tuning the Embodied SCPN)' preserves Paper 0 records P0R05045-P0R05049 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05045:ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn",
            "P0R05046:ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn",
            "P0R05047:ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn",
            "P0R05048:ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn",
            "P0R05049:ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn",
        ),
        "source_formulae": (
            "P0R05045: IX. Advanced Therapeutic Interventions (Tuning the Embodied SCPN)",
            "P0R05046: Therapeutics aim to restore the optimal state (Criticality, Coherence, Minimised F).",
            "P0R05047: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R05048: Fig.: Levers to restore the optimum: Bioelectronic (VNS; L3 field modulation), Metabolic/Nutritional (ketogenic, microbiome, cholesterol), Somatic (tensegrity; reduce allostatic FFF), Geometric remodelling (ASCs to relax rigid priors).",
            "P0R05049: Bioelectronic Medicine: Vagal Nerve Stimulation (VNS) to modulate the Neuro-Visceral Axis and restore criticality; Bioelectric Field Modulation (L3) for regeneration. | Metabolic and Nutritional Interventions: Ketogenic Diets (L1/L4 optimization); Microbiome Restoration (GBA tuning); Cholesterol Optimization (L2 IET efficiency). | Somatic Therapies: Targeting the Fascial network (L3 Tensegrity); Somatic Experiencing (resolving accumulated F in the Neuro-Visceral Axis). | Geometric Remodelling: Utilising ASCs (e.g., Psychedelic therapy) to expand the Consciousness Manifold (M) and relax rigid priors (HPC).",
        ),
        "test_protocols": (
            "preserve IX. Advanced Therapeutic Interventions (Tuning the Embodied SCPN) source-accounting boundary",
        ),
        "null_results": (
            "IX. Advanced Therapeutic Interventions (Tuning the Embodied SCPN) is not empirical validation evidence",
        ),
        "variables": ("ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn",),
        "validation_targets": ("preserve records P0R05045-P0R05049",),
        "null_controls": (
            "ix_advanced_therapeutic_interventions_tuning_the_embodied_scpn must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemSpec:
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
class ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemSpec, ...]
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


def build_viii_the_evolutionary_trajectory_of_the_brain_body_system_specs(
    source_records: list[dict[str, Any]],
) -> ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemSpecBundle:
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

    specs: list[ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemSpec(
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
        + "VIII. The Evolutionary Trajectory of the Brain-Body System"
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
        "next_source_boundary": "P0R05050",
    }
    return ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_viii_the_evolutionary_trajectory_of_the_brain_body_system_specs(
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


def render_report(bundle: ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "VIII. The Evolutionary Trajectory of the Brain-Body System" + " Specs",
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
    bundle: ViiiTheEvolutionaryTrajectoryOfTheBrainBodySystemSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_viii_the_evolutionary_trajectory_of_the_brain_body_system_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_viii_the_evolutionary_trajectory_of_the_brain_body_system_validation_specs_{date_tag}.md"
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
