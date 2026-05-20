#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Dynamics and Evolution of the Embodied SCPN spec builder
"""Promote Paper 0 The Dynamics and Evolution of the Embodied SCPN records."""

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
    "P0R04956",
    "P0R04957",
    "P0R04958",
    "P0R04959",
    "P0R04960",
    "P0R04961",
    "P0R04962",
    "P0R04963",
    "P0R04964",
    "P0R04965",
    "P0R04966",
)
CLAIM_BOUNDARY = "source-bounded the dynamics and evolution of the embodied scpn source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_dynamics_and_evolution_of_the_embodied_scpn.the_dynamics_and_evolution_of_the_embodied_scpn": {
        "context_id": "the_dynamics_and_evolution_of_the_embodied_scpn",
        "validation_protocol": "paper0.the_dynamics_and_evolution_of_the_embodied_scpn.the_dynamics_and_evolution_of_the_embodied_scpn",
        "canonical_statement": "The source-bounded component 'The Dynamics and Evolution of the Embodied SCPN' preserves Paper 0 records P0R04956-P0R04956 without empirical validation claims.",
        "source_equation_ids": ("P0R04956:the_dynamics_and_evolution_of_the_embodied_scpn",),
        "source_formulae": ("P0R04956: The Dynamics and Evolution of the Embodied SCPN",),
        "test_protocols": (
            "preserve The Dynamics and Evolution of the Embodied SCPN source-accounting boundary",
        ),
        "null_results": (
            "The Dynamics and Evolution of the Embodied SCPN is not empirical validation evidence",
        ),
        "variables": ("the_dynamics_and_evolution_of_the_embodied_scpn",),
        "validation_targets": ("preserve records P0R04956-P0R04956",),
        "null_controls": (
            "the_dynamics_and_evolution_of_the_embodied_scpn must remain source-bounded accounting",
        ),
    },
    "the_dynamics_and_evolution_of_the_embodied_scpn.i_the_dynamics_of_consciousness_in_the_brain_body_system": {
        "context_id": "i_the_dynamics_of_consciousness_in_the_brain_body_system",
        "validation_protocol": "paper0.the_dynamics_and_evolution_of_the_embodied_scpn.i_the_dynamics_of_consciousness_in_the_brain_body_system",
        "canonical_statement": "The source-bounded component 'I. The Dynamics of Consciousness in the Brain-Body System' preserves Paper 0 records P0R04957-P0R04958 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04957:i_the_dynamics_of_consciousness_in_the_brain_body_system",
            "P0R04958:i_the_dynamics_of_consciousness_in_the_brain_body_system",
        ),
        "source_formulae": (
            "P0R04957: I. The Dynamics of Consciousness in the Brain-Body System",
            "P0R04958: The conscious state (L5) is an emergent property of the entire embodied system, extending beyond the cranium. Its dynamics are governed by the core principles of the SCPN: UPDE, HPC, and Quasicriticality.",
        ),
        "test_protocols": (
            "preserve I. The Dynamics of Consciousness in the Brain-Body System source-accounting boundary",
        ),
        "null_results": (
            "I. The Dynamics of Consciousness in the Brain-Body System is not empirical validation evidence",
        ),
        "variables": ("i_the_dynamics_of_consciousness_in_the_brain_body_system",),
        "validation_targets": ("preserve records P0R04957-P0R04958",),
        "null_controls": (
            "i_the_dynamics_of_consciousness_in_the_brain_body_system must remain source-bounded accounting",
        ),
    },
    "the_dynamics_and_evolution_of_the_embodied_scpn.1_the_embodied_upde_the_symphony_of_the_self": {
        "context_id": "1_the_embodied_upde_the_symphony_of_the_self",
        "validation_protocol": "paper0.the_dynamics_and_evolution_of_the_embodied_scpn.1_the_embodied_upde_the_symphony_of_the_self",
        "canonical_statement": "The source-bounded component '1. The Embodied UPDE (The Symphony of the Self)' preserves Paper 0 records P0R04959-P0R04966 without empirical validation claims.",
        "source_equation_ids": (
            "P0R04959:1_the_embodied_upde_the_symphony_of_the_self",
            "P0R04960:1_the_embodied_upde_the_symphony_of_the_self",
            "P0R04961:1_the_embodied_upde_the_symphony_of_the_self",
            "P0R04962:1_the_embodied_upde_the_symphony_of_the_self",
            "P0R04963:1_the_embodied_upde_the_symphony_of_the_self",
            "P0R04964:1_the_embodied_upde_the_symphony_of_the_self",
            "P0R04965:1_the_embodied_upde_the_symphony_of_the_self",
            "P0R04966:1_the_embodied_upde_the_symphony_of_the_self",
        ),
        "source_formulae": (
            "P0R04959: 1. The Embodied UPDE (The Symphony of the Self)",
            "P0R04960: The Unified Phase Dynamics Equation (UPDE) governs the synchronisation of the entire organism.",
            "P0R04961: The Tri-Axial UPDE (Brain-Heart-Gut): This is the primary oscillatory network of the Embodied Self (L5). dtdthetaBrain=B+KBHsin(thetaHthetaB)+KBGsin(thetaGthetaB)+... Coherence (R) within this Tri-Axial system is the fundamental measure of organismal integration, with Heart Rate Variability (HRV) as its primary readout.",
            "P0R04962: [IMAGE:]",
            "P0R04963: Fig.: Tri-axial coupling of phase oscillators (Brain-Heart-Gut). Interaction terms KBH,KBG,KHGK_{BH},K_{BG},K_{HG}KBH,KBG,KHG drive coherence RRR; HRV serves as a primary readout of organismal integration.",
            "P0R04964: Multi-Scale Synchronisation: The UPDE operates hierarchically, nesting ultra-fast quantum dynamics (L1) within cellular oscillations (L4), which are further nested within slower visceral and endocrine rhythms. Cross-Frequency Coupling (CFC/PAC) implements this nesting.",
            "P0R04965: [IMAGE:]",
            "P0R04966: Fig.: Interoceptive HPC: Insula/ACC generate predictions about bodily state; mismatches yield prediction errors that are precision-weighted, shaping the Affective Field A=FA=-\\nabla FA=F.",
        ),
        "test_protocols": (
            "preserve 1. The Embodied UPDE (The Symphony of the Self) source-accounting boundary",
        ),
        "null_results": (
            "1. The Embodied UPDE (The Symphony of the Self) is not empirical validation evidence",
        ),
        "variables": ("1_the_embodied_upde_the_symphony_of_the_self",),
        "validation_targets": ("preserve records P0R04959-P0R04966",),
        "null_controls": (
            "1_the_embodied_upde_the_symphony_of_the_self must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class TheDynamicsAndEvolutionOfTheEmbodiedScpnSpec:
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
class TheDynamicsAndEvolutionOfTheEmbodiedScpnSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheDynamicsAndEvolutionOfTheEmbodiedScpnSpec, ...]
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


def build_the_dynamics_and_evolution_of_the_embodied_scpn_specs(
    source_records: list[dict[str, Any]],
) -> TheDynamicsAndEvolutionOfTheEmbodiedScpnSpecBundle:
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

    specs: list[TheDynamicsAndEvolutionOfTheEmbodiedScpnSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheDynamicsAndEvolutionOfTheEmbodiedScpnSpec(
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
        "title": "Paper 0 " + "The Dynamics and Evolution of the Embodied SCPN" + " Specs",
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
        "next_source_boundary": "P0R04967",
    }
    return TheDynamicsAndEvolutionOfTheEmbodiedScpnSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheDynamicsAndEvolutionOfTheEmbodiedScpnSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_dynamics_and_evolution_of_the_embodied_scpn_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheDynamicsAndEvolutionOfTheEmbodiedScpnSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Dynamics and Evolution of the Embodied SCPN" + " Specs",
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
    bundle: TheDynamicsAndEvolutionOfTheEmbodiedScpnSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_dynamics_and_evolution_of_the_embodied_scpn_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_dynamics_and_evolution_of_the_embodied_scpn_validation_specs_{date_tag}.md"
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
