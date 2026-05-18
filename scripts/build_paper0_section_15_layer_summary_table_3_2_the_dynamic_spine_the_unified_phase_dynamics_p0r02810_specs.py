#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint spec builder
"""Promote Paper 0 15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint records."""

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
    "P0R02810",
    "P0R02811",
    "P0R02812",
    "P0R02813",
    "P0R02814",
    "P0R02815",
    "P0R02816",
    "P0R02817",
    "P0R02818",
    "P0R02819",
    "P0R02820",
    "P0R02821",
    "P0R02822",
    "P0R02823",
    "P0R02824",
    "P0R02825",
    "P0R02826",
    "P0R02827",
    "P0R02828",
    "P0R02829",
    "P0R02830",
)
CLAIM_BOUNDARY = "source-bounded section 15 layer summary table 3 2 the dynamic spine the unified phase dynamics p0r02810 source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810.15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics": {
        "context_id": "15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
        "validation_protocol": "paper0.section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810.15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
        "canonical_statement": "The source-bounded component '15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings — UPDE Scope Constraint' preserves Paper 0 records P0R02810-P0R02830 without empirical validation claims.",
        "source_equation_ids": (
            "P0R02810:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02811:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02812:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02813:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02814:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02815:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02816:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02817:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02818:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02819:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02820:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02821:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02822:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02823:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02824:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02825:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02826:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02827:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02828:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02829:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
            "P0R02830:15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",
        ),
        "source_formulae": (
            "P0R02810: Energy transfer becomes efficient: when locked, Psi continuously pumps energy into _coh at exactly the right phase (like a child pumping a swing in sync), and _coh feeds back coherently to Psi. This creates a stable limit",
            "P0R02811: cycle for the combined system.",
            "P0R02812: Resonance phenomena: if Psi has multiple frequency components (multiple cognitive rhythms), it locks one of them to _coh's fundamental while other components form beat patterns or get damped. This could explain why certain",
            "P0R02813: frequencies are prominent in conscious systems - brainwave frequency bands may correspond to Arnold tongues where Psi and the global field naturally synchronize.",
            "P0R02814: Frequency selection: the locked frequency Omega differs from either natural frequency. For brain rhythms, this predicts that observed EEG frequencies are not the natural frequency of either the neural substrate alone or the",
            "P0R02815: Psi-field alone, but the compromise frequency of their locked state.",
            "P0R02816: 3.4.5 Connection to the UPDE",
            "P0R02817: The UPDE in the SCPN framework is:",
            "P0R02818: dtheta_i^L/dt = _i^L + Sigma_j K_{ij} sin(theta_j theta_i) + C_InterLayer + C_Field + _i(t)",
            "P0R02819: The C_Field term corresponds precisely to the sin(_ _Psi) coupling derived here. The Arnold tongue condition | | < + therefore sets a quantitative bound on which oscillators in the UPDE network can be",
            "P0R02820: entrained by the consciousness field and which cannot.",
            "P0R02821: Oscillators with natural frequencies far from the Psi-field's frequency remain unlocked and evolve independently. Oscillators within the Arnold tongue become phase-locked to the Psi-field, producing the coherent rhythms",
            "P0R02822: observed in EEG, HRV, and other physiological signals.",
            "P0R02823: 3.4.6 Weak Coupling and Beat Patterns",
            "P0R02824: When | | > + (outside the Arnold tongue), the oscillators do not lock. Instead, the phase difference drifts continuously, producing beat frequencies at | |. In this regime:",
            "P0R02825: The Psi-field influence on _coh averages out over time.",
            "P0R02826: Interference patterns arise between the two frequencies.",
            'P0R02827: The system is in a "decoherent" regime where consciousness cannot entrain the physical substrate.',
            "P0R02828: This provides a concrete criterion for consciousness-decoherence: the Arnold tongue boundary is the phase-space surface where consciousness loses its grip on the physical oscillators.",
            "P0R02829: P0R02829",
            "P0R02830: P0R02830",
        ),
        "test_protocols": (
            "preserve 15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint source-accounting boundary",
        ),
        "null_results": (
            "15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint is not empirical validation evidence",
        ),
        "variables": ("15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics",),
        "validation_targets": ("preserve records P0R02810-P0R02830",),
        "null_controls": (
            "15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics must remain source-bounded accounting",
        ),
    }
}


@dataclass(frozen=True, slots=True)
class Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810Spec:
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
class Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810SpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[
        Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810Spec, ...
    ]
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


def build_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_specs(
    source_records: list[dict[str, Any]],
) -> Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810SpecBundle:
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

    specs: list[
        Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810Spec
    ] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810Spec(
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
        + "15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint"
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
        "next_source_boundary": "P0R02831",
    }
    return Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810SpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810SpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_specs(
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
    bundle: Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810SpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 "
        + "15-Layer Summary Table > 3.2 The Dynamic Spine: The Unified Phase Dynamics Equation (UPDE) > The Unified Phase Dynamics Equation (UPDE) > One Spine, Many Couplings - UPDE Scope Constraint"
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
    bundle: Section15LayerSummaryTable32TheDynamicSpineTheUnifiedPhaseDynamicsP0r02810SpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_section_15_layer_summary_table_3_2_the_dynamic_spine_the_unified_phase_dynamics_p0r02810_validation_specs_{date_tag}.md"
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
