#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Simulation Architecture and Implementation spec builder
"""Promote Paper 0 Simulation Architecture and Implementation records."""

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
    "P0R05245",
    "P0R05246",
    "P0R05247",
    "P0R05248",
    "P0R05249",
    "P0R05250",
    "P0R05251",
    "P0R05252",
    "P0R05253",
    "P0R05254",
    "P0R05255",
)
CLAIM_BOUNDARY = "source-bounded simulation architecture and implementation source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "simulation_architecture_and_implementation.simulation_architecture_and_implementation": {
        "context_id": "simulation_architecture_and_implementation",
        "validation_protocol": "paper0.simulation_architecture_and_implementation.simulation_architecture_and_implementation",
        "canonical_statement": "The source-bounded component 'Simulation Architecture and Implementation' preserves Paper 0 records P0R05245-P0R05255 without empirical validation claims.",
        "source_equation_ids": (
            "P0R05245:simulation_architecture_and_implementation",
            "P0R05246:simulation_architecture_and_implementation",
            "P0R05247:simulation_architecture_and_implementation",
            "P0R05248:simulation_architecture_and_implementation",
            "P0R05249:simulation_architecture_and_implementation",
            "P0R05250:simulation_architecture_and_implementation",
            "P0R05251:simulation_architecture_and_implementation",
            "P0R05252:simulation_architecture_and_implementation",
            "P0R05253:simulation_architecture_and_implementation",
            "P0R05254:simulation_architecture_and_implementation",
            "P0R05255:simulation_architecture_and_implementation",
        ),
        "source_formulae": (
            "P0R05245: Simulation Architecture and Implementation",
            "P0R05246: [IMAGE:Ein Bild, das Text, Screenshot, Schrift, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R05247: Fig.: AIF Agents + Network + External Controller (Timestep Loop). This schematic illustrates the high-level architecture of the proposed multi-agent simulation, detailing the flow of information and control between the agents, the environment, and the external AI controller. This plate gives you a clean blueprint for implementing and experimenting with multi-agent AIF under networked social couplings JijJ_{ij}Jij and controller-driven interventions, ready to align with your SEC/CEF objectives.",
            "P0R05248: A population of NNN Active Inference agents (e.g., pymdp) each minimizes its own free energy FiF_iFi under a generative model with A/B/C/D matrices. An Environment & Social Network class hosts the agents and a dynamic graph JijJ_{ij}Jij. An external AI controller optimizes a global objective by modulating network couplings JijJ_{ij}Jij and exogenous signals hih_ihi. The simulation loop iterates:",
            "P0R05249: agents act to minimize expected free energy (EFE); | environment mediates interactions and state updates; | controller adjusts JijJ_{ij}Jij and hih_ihi by policy; | agents receive observations and update beliefs - then repeat. This architecture supports interventions at the agent, network, and exogenous-input levels for testing teleological or SEC-aligned objectives.",
            "P0R05250: Agent Model (pymdp):",
            "P0R05251: Each agent will be implemented using the pymdp Python library, a package designed for simulating active inference agents in discrete state spaces.",
            "P0R05252: Generative Model: The agent's generative model will be defined by a set of matrices: A-matrix (Likelihood): Maps its hidden belief state to expected observations from the environment and other agents. | B-matrix (Transitions): Encodes the agent's model of how its beliefs change based on its actions (e.g., consuming content, interacting with others). | C-matrix (Preferences): Encodes a preference for observations that confirm its existing beliefs, formalising confirmation bias and driving the agent to minimise surprise. | D-vector (Priors): Represents the agent's initial prior beliefs about the world.",
            "P0R05253: Environment and Network Class:",
            "P0R05254: A custom Python class will manage the simulation. It will contain the population of agents and the dynamic social network graph (e.g., using the networkx library). At each timestep, the environment will:",
            "P0R05255: Allow each agent to select an action based on its policy inference (minimising expected free energy). | Mediate interactions between agents, allowing them to exchange observations. | Update the state of the shared information environment. | Update the network graph's edge weights (Jij) based on the rules of the external AI controller.",
        ),
        "test_protocols": (
            "preserve Simulation Architecture and Implementation source-accounting boundary",
        ),
        "null_results": (
            "Simulation Architecture and Implementation is not empirical validation evidence",
        ),
        "variables": ("simulation_architecture_and_implementation",),
        "validation_targets": ("preserve records P0R05245-P0R05255",),
        "null_controls": (
            "simulation_architecture_and_implementation must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class SimulationArchitectureAndImplementationSpec:
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
class SimulationArchitectureAndImplementationSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[SimulationArchitectureAndImplementationSpec, ...]
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


def build_simulation_architecture_and_implementation_specs(
    source_records: list[dict[str, Any]],
) -> SimulationArchitectureAndImplementationSpecBundle:
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

    specs: list[SimulationArchitectureAndImplementationSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            SimulationArchitectureAndImplementationSpec(
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
        "title": "Paper 0 " + "Simulation Architecture and Implementation" + " Specs",
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
        "next_source_boundary": "P0R05256",
    }
    return SimulationArchitectureAndImplementationSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> SimulationArchitectureAndImplementationSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_simulation_architecture_and_implementation_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: SimulationArchitectureAndImplementationSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Simulation Architecture and Implementation" + " Specs",
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
    bundle: SimulationArchitectureAndImplementationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_simulation_architecture_and_implementation_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_simulation_architecture_and_implementation_validation_specs_{date_tag}.md"
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
