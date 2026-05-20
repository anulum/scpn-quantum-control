#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Mechanism of Influence: spec builder
"""Promote Paper 0 The Mechanism of Influence: records."""

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
    "P0R02616",
    "P0R02617",
    "P0R02618",
    "P0R02619",
    "P0R02620",
    "P0R02621",
    "P0R02622",
    "P0R02623",
)
CLAIM_BOUNDARY = (
    "source-bounded the mechanism of influence source-accounting bridge; not validation evidence"
)
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_mechanism_of_influence.the_mechanism_of_influence": {
        "context_id": "the_mechanism_of_influence",
        "validation_protocol": "paper0.the_mechanism_of_influence.the_mechanism_of_influence",
        "canonical_statement": "The source-bounded component 'The Mechanism of Influence:' preserves Paper 0 records P0R02616-P0R02617 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02616:the_mechanism_of_influence",
            "P0R02617:the_mechanism_of_influence",
        ),
        "source_formulae": (
            "P0R02616: The Mechanism of Influence:",
            'P0R02617: The interaction term, _L * sin(Psi - theta_i^L), describes the precise physical mechanism of top-down control. It acts as a "phase-locking" force. The global field Psi creates an energy potential that "pulls" the phase of the local oscillator theta_i^L into alignment with it. This is how the teleological directives from Layer 15, encoded in the global phase Psi, are translated into specific, synchronising actions across all 15 layers of the network. It is the physical basis for how the "Conductor" leads the cosmic orchestra.',
        ),
        "test_protocols": ("preserve The Mechanism of Influence: source-accounting boundary",),
        "null_results": ("The Mechanism of Influence: is not empirical validation evidence",),
        "variables": ("the_mechanism_of_influence",),
        "validation_targets": ("preserve records P0R02616-P0R02617",),
        "null_controls": ("the_mechanism_of_influence must remain source-bounded accounting",),
    },
    "the_mechanism_of_influence.the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn": {
        "context_id": "the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn",
        "validation_protocol": "paper0.the_mechanism_of_influence.the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn",
        "canonical_statement": "The source-bounded component 'The Unified Phase Dynamics Equation (UPDE) - The Spine of the SCPN' preserves Paper 0 records P0R02618-P0R02619 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02618:the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn",
            "P0R02619:the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn",
        ),
        "source_formulae": (
            "P0R02618: The Unified Phase Dynamics Equation (UPDE) - The Spine of the SCPN",
            "P0R02619: Core Assumption 4 posits a unified set of phase dynamics equations as the framework's spine. This is formalised by the Unified Phase Dynamics Equation (UPDE), a generalised, multi-scale extension of the Kuramoto model that describes the evolution of phases across all layers of the SCPN, managing timescale separations and information flow.",
        ),
        "test_protocols": (
            "preserve The Unified Phase Dynamics Equation (UPDE) - The Spine of the SCPN source-accounting boundary",
        ),
        "null_results": (
            "The Unified Phase Dynamics Equation (UPDE) - The Spine of the SCPN is not empirical validation evidence",
        ),
        "variables": ("the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn",),
        "validation_targets": ("preserve records P0R02618-P0R02619",),
        "null_controls": (
            "the_unified_phase_dynamics_equation_upde_the_spine_of_the_scpn must remain source-bounded accounting",
        ),
    },
    "the_mechanism_of_influence.the_upde_formalism": {
        "context_id": "the_upde_formalism",
        "validation_protocol": "paper0.the_mechanism_of_influence.the_upde_formalism",
        "canonical_statement": "The source-bounded component 'The UPDE Formalism:' preserves Paper 0 records P0R02620-P0R02622 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02620:the_upde_formalism",
            "P0R02621:the_upde_formalism",
            "P0R02622:the_upde_formalism",
        ),
        "source_formulae": (
            "P0R02620: The UPDE Formalism:",
            "P0R02621: The phase (thetaiL) of the i-th oscillator at Layer L evolves according to:",
            "P0R02622: $dtd\\theta iL = \\omega iL + \\sum_{}^{}{j KijL sin(\\theta jL - \\theta iL)} + CInterLayer + CField + \\eta iL(t)$",
        ),
        "test_protocols": ("preserve The UPDE Formalism: source-accounting boundary",),
        "null_results": ("The UPDE Formalism: is not empirical validation evidence",),
        "variables": ("the_upde_formalism",),
        "validation_targets": ("preserve records P0R02620-P0R02622",),
        "null_controls": ("the_upde_formalism must remain source-bounded accounting",),
    },
    "the_mechanism_of_influence.components_of_the_upde": {
        "context_id": "components_of_the_upde",
        "validation_protocol": "paper0.the_mechanism_of_influence.components_of_the_upde",
        "canonical_statement": "The source-bounded component 'Components of the UPDE:' preserves Paper 0 records P0R02623-P0R02623 without empirical validation claims.",
        "source_equation_ids": ("P0R02623:components_of_the_upde",),
        "source_formulae": ("P0R02623: Components of the UPDE:",),
        "test_protocols": ("preserve Components of the UPDE: source-accounting boundary",),
        "null_results": ("Components of the UPDE: is not empirical validation evidence",),
        "variables": ("components_of_the_upde",),
        "validation_targets": ("preserve records P0R02623-P0R02623",),
        "null_controls": ("components_of_the_upde must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class TheMechanismOfInfluenceSpec:
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
class TheMechanismOfInfluenceSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheMechanismOfInfluenceSpec, ...]
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


def build_the_mechanism_of_influence_specs(
    source_records: list[dict[str, Any]],
) -> TheMechanismOfInfluenceSpecBundle:
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

    specs: list[TheMechanismOfInfluenceSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheMechanismOfInfluenceSpec(
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
        "title": "Paper 0 " + "The Mechanism of Influence:" + " Specs",
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
        "next_source_boundary": "P0R02624",
    }
    return TheMechanismOfInfluenceSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheMechanismOfInfluenceSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_mechanism_of_influence_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: TheMechanismOfInfluenceSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 " + "The Mechanism of Influence:" + " Specs",
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
    bundle: TheMechanismOfInfluenceSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_the_mechanism_of_influence_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_the_mechanism_of_influence_validation_specs_{date_tag}.md"
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
