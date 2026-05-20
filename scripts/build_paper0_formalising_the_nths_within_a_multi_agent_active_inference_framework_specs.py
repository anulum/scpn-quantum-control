#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Formalising the NTHS within a Multi-Agent Active Inference Framework spec builder
"""Promote Paper 0 Formalising the NTHS within a Multi-Agent Active Inference Framework records."""

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
    "P0R05236",
    "P0R05237",
    "P0R05238",
    "P0R05239",
    "P0R05240",
    "P0R05241",
    "P0R05242",
    "P0R05243",
    "P0R05244",
)
CLAIM_BOUNDARY = "source-bounded formalising the nths within a multi agent active inference framework source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "formalising_the_nths_within_a_multi_agent_active_inference_framework.formalising_the_nths_within_a_multi_agent_active_inference_framework": {
        "context_id": "formalising_the_nths_within_a_multi_agent_active_inference_framework",
        "validation_protocol": "paper0.formalising_the_nths_within_a_multi_agent_active_inference_framework.formalising_the_nths_within_a_multi_agent_active_inference_framework",
        "canonical_statement": "The source-bounded component 'Formalising the NTHS within a Multi-Agent Active Inference Framework' preserves Paper 0 records P0R05236-P0R05236 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05236:formalising_the_nths_within_a_multi_agent_active_inference_framework",
        ),
        "source_formulae": (
            "P0R05236: Formalising the NTHS within a Multi-Agent Active Inference Framework",
        ),
        "test_protocols": (
            "preserve Formalising the NTHS within a Multi-Agent Active Inference Framework source-accounting boundary",
        ),
        "null_results": (
            "Formalising the NTHS within a Multi-Agent Active Inference Framework is not empirical validation evidence",
        ),
        "variables": ("formalising_the_nths_within_a_multi_agent_active_inference_framework",),
        "validation_targets": ("preserve records P0R05236-P0R05236",),
        "null_controls": (
            "formalising_the_nths_within_a_multi_agent_active_inference_framework must remain source-bounded accounting",
        ),
    },
    "formalising_the_nths_within_a_multi_agent_active_inference_framework.conceptual_mapping": {
        "context_id": "conceptual_mapping",
        "validation_protocol": "paper0.formalising_the_nths_within_a_multi_agent_active_inference_framework.conceptual_mapping",
        "canonical_statement": "The source-bounded component 'Conceptual Mapping' preserves Paper 0 records P0R05237-P0R05244 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05237:conceptual_mapping",
            "P0R05238:conceptual_mapping",
            "P0R05239:conceptual_mapping",
            "P0R05240:conceptual_mapping",
            "P0R05241:conceptual_mapping",
            "P0R05242:conceptual_mapping",
            "P0R05243:conceptual_mapping",
            "P0R05244:conceptual_mapping",
        ),
        "source_formulae": (
            "P0R05237: Conceptual Mapping",
            "P0R05238: The spin-glass Hamiltonian (H=Jijsigmaisigmajhisigmai) is translated into the formal language of active inference, a first-principles framework for modeling sentient behavior.",
            "P0R05239: Agents and Beliefs (Spins, sigmai): Each of the N agents in the simulation is an active inference agent. Its core belief about a binary issue (e.g., pro/con) is represented by a hidden state factor, which corresponds to the spin sigmai{+1,1}. | Social Network (Couplings, Jij): Agents are nodes in a dynamic social network. The weight of the edge between agent i and agent j, Jij, represents social influence. Positive weights indicate trust or alignment (ferromagnetic), while negative weights represent distrust or opposition (antiferromagnetic). | Information Environment (External Field, hi): The observations an agent receives from the environment (e.g., news articles, social media posts) constitute the external field hi that biases its beliefs. | Active Inference Imperative: Each agent acts to minimise its individual variational free energy (Fi). This involves both updating beliefs to better predict sensations (perception) and choosing actions to make sensations conform to beliefs (action). In a social context, this translates to seeking out information that confirms existing beliefs (reducing surprise) and interacting with agents who share a similar model of the world.",
            "P0R05240: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R05241: Fig: Spin-Glass Active Inference (AIF) Correspondence. This diagram provides the formal translation between the concepts of a statistical physics model (the spin-glass Hamiltonian) and the cognitive science framework (multi-agent active inference) used in the simulation. This figure codifies a one-to-one conceptual bridge: statistical-physics couplings and fields correspond to social influence and evidence in multi-agent AIF, with energy/free-energy minimization furnishing a shared organizing principle.",
            "P0R05242: Hamiltonian side: spins sigmai\\sigma_isigmai, couplings JijJ_{ij}Jij, and fields hih_ihi define the energy",
            "P0R05243: H=ijJij sigmaisigmajihi sigmai.H = -\\sum_{ij} J_{ij}\\,\\sigma_i\\sigma_j - \\sum_i h_i\\,\\sigma_i .H=ijJijsigmaisigmajihisigmai.",
            "P0R05244: AIF side: agent beliefs sis_isi, social-influence weights, and information environment map onto those terms, while each agent minimizes variational free energy FiF_iFi. The mapping is: sigmaisi\\sigma_i \\mapsto s_isigmaisi (binary stance belief state), JijJ_{ij} \\mapstoJij influence weights (who affects whom), hih_i \\mapstohi environmental evidence (exogenous observations), and energy minimization minH\\min HminH free-energy minimization minFi\\min F_iminFi. Thus, spin-glass equilibria provide a conceptual template for collective inference equilibria, where frustration/heterogeneity of JijJ_{ij}Jij parallels opinion polarization and multi-stable belief basins.",
        ),
        "test_protocols": ("preserve Conceptual Mapping source-accounting boundary",),
        "null_results": ("Conceptual Mapping is not empirical validation evidence",),
        "variables": ("conceptual_mapping",),
        "validation_targets": ("preserve records P0R05237-P0R05244",),
        "null_controls": ("conceptual_mapping must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkSpec:
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
class FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkSpec, ...]
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


def build_formalising_the_nths_within_a_multi_agent_active_inference_framework_specs(
    source_records: list[dict[str, Any]],
) -> FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkSpecBundle:
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

    specs: list[FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkSpec(
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
        + "Formalising the NTHS within a Multi-Agent Active Inference Framework"
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
        "next_source_boundary": "P0R05245",
    }
    return FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_formalising_the_nths_within_a_multi_agent_active_inference_framework_specs(
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
    bundle: FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "Formalising the NTHS within a Multi-Agent Active Inference Framework"
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
    bundle: FormalisingTheNthsWithinAMultiAgentActiveInferenceFrameworkSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_formalising_the_nths_within_a_multi_agent_active_inference_framework_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_formalising_the_nths_within_a_multi_agent_active_inference_framework_validation_specs_{date_tag}.md"
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
