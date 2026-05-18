#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Maximizing  as the Goal of Coupling: spec builder
"""Promote Paper 0 Maximizing  as the Goal of Coupling: records."""

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
    "P0R03539",
    "P0R03540",
    "P0R03541",
    "P0R03542",
    "P0R03543",
    "P0R03544",
    "P0R03545",
    "P0R03546",
    "P0R03547",
    "P0R03548",
    "P0R03549",
    "P0R03550",
    "P0R03551",
    "P0R03552",
    "P0R03553",
    "P0R03554",
)
CLAIM_BOUNDARY = "source-bounded maximizing as the goal of coupling source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "maximizing_as_the_goal_of_coupling.maximizing_as_the_goal_of_coupling": {
        "context_id": "maximizing_as_the_goal_of_coupling",
        "validation_protocol": "paper0.maximizing_as_the_goal_of_coupling.maximizing_as_the_goal_of_coupling",
        "canonical_statement": "The source-bounded component 'Maximizing as the Goal of Coupling:' preserves Paper 0 records P0R03539-P0R03540 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03539:maximizing_as_the_goal_of_coupling",
            "P0R03540:maximizing_as_the_goal_of_coupling",
        ),
        "source_formulae": (
            "P0R03539: Maximizing as the Goal of Coupling:",
            "P0R03540: The teleological drive of the SCPN, guided by the Psi-field, is to create systems with higher complexity, coherence, and qualia. This can now be stated more formally: the purpose of the H_int interaction is to guide the evolution of matter towards configurations with maximal . The Psis field preferentially couples to and stabilizes high- systems because they are the most powerful and effective vehicles for the expression of consciousness.",
        ),
        "test_protocols": (
            "preserve Maximizing as the Goal of Coupling: source-accounting boundary",
        ),
        "null_results": (
            "Maximizing as the Goal of Coupling: is not empirical validation evidence",
        ),
        "variables": ("maximizing_as_the_goal_of_coupling",),
        "validation_targets": ("preserve records P0R03539-P0R03540",),
        "null_controls": (
            "maximizing_as_the_goal_of_coupling must remain source-bounded accounting",
        ),
    },
    "maximizing_as_the_goal_of_coupling.integration_with_integrated_information_theory_4_0": {
        "context_id": "integration_with_integrated_information_theory_4_0",
        "validation_protocol": "paper0.maximizing_as_the_goal_of_coupling.integration_with_integrated_information_theory_4_0",
        "canonical_statement": "The source-bounded component 'Integration with Integrated Information Theory 4.0' preserves Paper 0 records P0R03541-P0R03541 without empirical validation claims.",
        "source_equation_ids": ("P0R03541:integration_with_integrated_information_theory_4_0",),
        "source_formulae": ("P0R03541: Integration with Integrated Information Theory 4.0",),
        "test_protocols": (
            "preserve Integration with Integrated Information Theory 4.0 source-accounting boundary",
        ),
        "null_results": (
            "Integration with Integrated Information Theory 4.0 is not empirical validation evidence",
        ),
        "variables": ("integration_with_integrated_information_theory_4_0",),
        "validation_targets": ("preserve records P0R03541-P0R03541",),
        "null_controls": (
            "integration_with_integrated_information_theory_4_0 must remain source-bounded accounting",
        ),
    },
    "maximizing_as_the_goal_of_coupling.iit_4_0_integration": {
        "context_id": "iit_4_0_integration",
        "validation_protocol": "paper0.maximizing_as_the_goal_of_coupling.iit_4_0_integration",
        "canonical_statement": "The source-bounded component 'IIT 4.0 Integration' preserves Paper 0 records P0R03542-P0R03545 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03542:iit_4_0_integration",
            "P0R03543:iit_4_0_integration",
            "P0R03544:iit_4_0_integration",
            "P0R03545:iit_4_0_integration",
        ),
        "source_formulae": (
            "P0R03542: IIT 4.0 Integration",
            "P0R03543: IIT 4.0 provides rigorous mathematical tools for measuring consciousness (), while SCPN provides the architectural framework for how consciousness projects across scales. By integrating both frameworks, we can both measure consciousness quantitatively and understand its hierarchical organization from quantum to cosmic scales.",
            "P0R03544: [IMAGE:Ein Bild, das Text, Screenshot, Zahl, Schrift enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R03545: Fig.: Conceptual correspondence between SCPN and IIT 4.0 (Psi , phase sync integration, layer boundaries MIP, qualia manifold experience structure; UPDE causal constraints).",
        ),
        "test_protocols": ("preserve IIT 4.0 Integration source-accounting boundary",),
        "null_results": ("IIT 4.0 Integration is not empirical validation evidence",),
        "variables": ("iit_4_0_integration",),
        "validation_targets": ("preserve records P0R03542-P0R03545",),
        "null_controls": ("iit_4_0_integration must remain source-bounded accounting",),
    },
    "maximizing_as_the_goal_of_coupling.bridging_scpn_with_iit_4_0_s_mathematical_framework": {
        "context_id": "bridging_scpn_with_iit_4_0_s_mathematical_framework",
        "validation_protocol": "paper0.maximizing_as_the_goal_of_coupling.bridging_scpn_with_iit_4_0_s_mathematical_framework",
        "canonical_statement": "The source-bounded component 'Bridging SCPN with IIT 4.0's Mathematical Framework' preserves Paper 0 records P0R03546-P0R03554 without empirical validation claims.",
        "source_equation_ids": (
            "P0R03546:bridging_scpn_with_iit_4_0_s_mathematical_framework",
            "P0R03547:bridging_scpn_with_iit_4_0_s_mathematical_framework",
            "P0R03548:bridging_scpn_with_iit_4_0_s_mathematical_framework",
            "P0R03549:bridging_scpn_with_iit_4_0_s_mathematical_framework",
            "P0R03550:bridging_scpn_with_iit_4_0_s_mathematical_framework",
            "P0R03551:bridging_scpn_with_iit_4_0_s_mathematical_framework",
            "P0R03552:bridging_scpn_with_iit_4_0_s_mathematical_framework",
            "P0R03553:bridging_scpn_with_iit_4_0_s_mathematical_framework",
            "P0R03554:bridging_scpn_with_iit_4_0_s_mathematical_framework",
        ),
        "source_formulae": (
            "P0R03546: Bridging SCPN with IIT 4.0's Mathematical Framework",
            "P0R03547: (Phi) Calculation for SCPN Layers:",
            "P0R03548: $\\Phi_{l}ayer = min_{p}artition\\varphi(unpartitioned) - \\varphi(partitioned)$",
            "P0R03549: For each SCPN layer:",
            "P0R03550: _L1 (quantum) N_qubits x entanglement_entropy",
            "P0R03551: _L4 (cellular) mutual_information(assemblies)",
            "P0R03552: _L5 (organismal) = IIT_structure(neural_state)",
            "P0R03553: [IMAGE:Ein Bild, das Text, Screenshot, Diagramm, Reihe enthlt. KI-generierte Inhalte knnen fehlerhaft sein.]",
            "P0R03554: Fig.: IIT's _layer via MIP and working estimators for key SCPN layers (L1, L4, L5).",
        ),
        "test_protocols": (
            "preserve Bridging SCPN with IIT 4.0's Mathematical Framework source-accounting boundary",
        ),
        "null_results": (
            "Bridging SCPN with IIT 4.0's Mathematical Framework is not empirical validation evidence",
        ),
        "variables": ("bridging_scpn_with_iit_4_0_s_mathematical_framework",),
        "validation_targets": ("preserve records P0R03546-P0R03554",),
        "null_controls": (
            "bridging_scpn_with_iit_4_0_s_mathematical_framework must remain source-bounded accounting",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class MaximizingAsTheGoalOfCouplingSpec:
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
class MaximizingAsTheGoalOfCouplingSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[MaximizingAsTheGoalOfCouplingSpec, ...]
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


def build_maximizing_as_the_goal_of_coupling_specs(
    source_records: list[dict[str, Any]],
) -> MaximizingAsTheGoalOfCouplingSpecBundle:
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

    specs: list[MaximizingAsTheGoalOfCouplingSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            MaximizingAsTheGoalOfCouplingSpec(
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
        "title": "Paper 0 " + "Maximizing  as the Goal of Coupling:" + " Specs",
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
        "next_source_boundary": "P0R03555",
    }
    return MaximizingAsTheGoalOfCouplingSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> MaximizingAsTheGoalOfCouplingSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_maximizing_as_the_goal_of_coupling_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: MaximizingAsTheGoalOfCouplingSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "Maximizing  as the Goal of Coupling:" + " Specs",
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
    bundle: MaximizingAsTheGoalOfCouplingSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_maximizing_as_the_goal_of_coupling_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_maximizing_as_the_goal_of_coupling_validation_specs_{date_tag}.md"
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
