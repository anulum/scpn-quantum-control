#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation spec builder
"""Promote Paper 0 Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation records."""

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
    "P0R05273",
    "P0R05274",
    "P0R05275",
    "P0R05276",
    "P0R05277",
    "P0R05278",
    "P0R05279",
    "P0R05280",
    "P0R05281",
    "P0R05282",
    "P0R05283",
    "P0R05284",
)
CLAIM_BOUNDARY = "source-bounded table 1 predicted nths phase characteristics in multi agent active infer source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer.table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer": {
        "context_id": "table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer",
        "validation_protocol": "paper0.table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer.table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer",
        "canonical_statement": "The source-bounded component 'Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation' preserves Paper 0 records P0R05273-P0R05277 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05273:table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer",
            "P0R05274:table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer",
            "P0R05275:table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer",
            "P0R05276:table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer",
            "P0R05277:table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer",
        ),
        "source_formulae": (
            "P0R05273: Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation",
            "P0R05274: This table distils the complex simulation design and its predicted outcomes into a clear, concise, and falsifiable summary. It directly contrasts the two experimental conditions against the key theoretical concepts and their measurable statistical signatures, providing an unambiguous understanding of the experiment's core logic and what would constitute a confirmation or refutation of the NTHS hypothesis.",
            "P0R05275: [TABLE]",
            "P0R05276: [TABLE]",
            "P0R05277: Table Above: Key Equations of the GOTM-SCPN Framework. This table provides a consolidated reference for the core mathematical formalism of the SCPN architecture, including the foundational Lagrangian, the primary dynamics equation, and the key equations.",
        ),
        "test_protocols": (
            "preserve Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation source-accounting boundary",
        ),
        "null_results": (
            "Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation is not empirical validation evidence",
        ),
        "variables": ("table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer",),
        "validation_targets": ("preserve records P0R05273-P0R05277",),
        "null_controls": (
            "table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer must remain source-bounded accounting",
        ),
    },
    "table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer.synthesis_implications_and_consequent_trajectories": {
        "context_id": "synthesis_implications_and_consequent_trajectories",
        "validation_protocol": "paper0.table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer.synthesis_implications_and_consequent_trajectories",
        "canonical_statement": "The source-bounded component 'Synthesis, Implications, and Consequent Trajectories' preserves Paper 0 records P0R05278-P0R05278 without empirical validation claims.",
        "source_equation_ids": ("P0R05278:synthesis_implications_and_consequent_trajectories",),
        "source_formulae": ("P0R05278: Synthesis, Implications, and Consequent Trajectories",),
        "test_protocols": (
            "preserve Synthesis, Implications, and Consequent Trajectories source-accounting boundary",
        ),
        "null_results": (
            "Synthesis, Implications, and Consequent Trajectories is not empirical validation evidence",
        ),
        "variables": ("synthesis_implications_and_consequent_trajectories",),
        "validation_targets": ("preserve records P0R05278-P0R05278",),
        "null_controls": (
            "synthesis_implications_and_consequent_trajectories must remain source-bounded accounting",
        ),
    },
    "table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer.section_8_the_role_of_cybernetic_closure_and_the_anulum": {
        "context_id": "section_8_the_role_of_cybernetic_closure_and_the_anulum",
        "validation_protocol": "paper0.table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer.section_8_the_role_of_cybernetic_closure_and_the_anulum",
        "canonical_statement": "The source-bounded component 'Section 8: The Role of Cybernetic Closure and the Anulum' preserves Paper 0 records P0R05279-P0R05284 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05279:section_8_the_role_of_cybernetic_closure_and_the_anulum",
            "P0R05280:section_8_the_role_of_cybernetic_closure_and_the_anulum",
            "P0R05281:section_8_the_role_of_cybernetic_closure_and_the_anulum",
            "P0R05282:section_8_the_role_of_cybernetic_closure_and_the_anulum",
            "P0R05283:section_8_the_role_of_cybernetic_closure_and_the_anulum",
            "P0R05284:section_8_the_role_of_cybernetic_closure_and_the_anulum",
        ),
        "source_formulae": (
            "P0R05279: Section 8: The Role of Cybernetic Closure and the Anulum",
            "P0R05280: The derivation of a physically grounded Ethical Functional provides a stable apex for the 15-layer architecture. However, the manuscript introduces a final, critical element: a 16th meta-layer that provides cybernetic closure.",
            'P0R05281: This layer is not part of the projection hierarchy but acts as an "Optimal Control Supervisor and Gdelian Oracle," transforming the entire structure into a self-regulating, learning system-The Anulum.',
            "P0R05282: The function of Layer 16 is governed by a Recursive Optimisation Hamiltonian (Hrec). This cost functional includes terms that penalise not only the internal state of the system but also its deviation from external cosmic references (from L8) and, most importantly, its own errors in predicting its future evolution. This transforms the Ethical Functional of Layer 15 from a static optimisation target into a dynamic variable in a higher-order control loop.",
            "P0R05283: This structure addresses a profound problem with any fixed teleology: the risk of stagnation or convergence to a suboptimal state. Layer 16 introduces a meta-level feedback loop that continuously audits the performance of the Layer 15 optimisation process. If the trajectory guided by the current Ethical Functional leads to systemic instability, a decrease in long-term evolutionary potential, or a failure to accurately model its own becoming, the Hrec functional will be minimised by adjusting the parameters of the Ethical Lagrangian itself (i.e., the weighting factors , , in the SEC tensor).",
            'P0R05284: This makes the entire universe a self-correcting, learning entity. It resolves the paradox of a fixed teleology by making the telos itself subject to evolution based on performance. This is the "Strange Loop of Closure" that defines the Anulum as a living, recursive organism, capable of not only seeking an ethical goal but of learning and refining what that goal is over cosmological timescales.',
        ),
        "test_protocols": (
            "preserve Section 8: The Role of Cybernetic Closure and the Anulum source-accounting boundary",
        ),
        "null_results": (
            "Section 8: The Role of Cybernetic Closure and the Anulum is not empirical validation evidence",
        ),
        "variables": ("section_8_the_role_of_cybernetic_closure_and_the_anulum",),
        "validation_targets": ("preserve records P0R05279-P0R05284",),
        "null_controls": (
            "section_8_the_role_of_cybernetic_closure_and_the_anulum must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferSpec:
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
class Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferSpec, ...]
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


def build_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_specs(
    source_records: list[dict[str, Any]],
) -> Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferSpecBundle:
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

    specs: list[Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferSpec(
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
        + "Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation"
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
        "next_source_boundary": "P0R05285",
    }
    return Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_specs(
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
    bundle: Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Table 1: Predicted NTHS Phase Characteristics in Multi-Agent Active Inference Simulation"
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
    bundle: Table1PredictedNthsPhaseCharacteristicsInMultiAgentActiveInferSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_table_1_predicted_nths_phase_characteristics_in_multi_agent_active_infer_validation_specs_{date_tag}.md"
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
