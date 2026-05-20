#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 advanced mechanisms spec builder
"""Promote Paper 0 advanced mechanisms records into validation specs."""

from __future__ import annotations

import argparse
import json
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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(6382, 6402))
STRUCTURAL_SOURCE_LEDGER_IDS = (
    "P0R06382",
    "P0R06383",
    "P0R06385",
    "P0R06388",
    "P0R06390",
    "P0R06393",
    "P0R06396",
    "P0R06398",
    "P0R06401",
)
MECHANISMS_BY_SPEC = {
    "advanced_mechanisms.geometric_physical_transduction": (
        ("P0R06384:O_superscript_S", "P0R06384:U_S"),
        (
            "symbol S acts as symmetry operator O^S",
            "resonance induces local gauge transformation U_S",
        ),
        (
            "L7 symbol operator couples to L8 physical dynamics",
            "Psi-field resonance modifies field connection",
            "abstract meaning remains a source-bounded gauge-transduction claim",
        ),
    ),
    "advanced_mechanisms.holographic_memory_encoding": (
        (),
        (),
        (
            "L4-UPDE coherent synchronisation modulates L1 substrate",
            "L1 quantum-state bias via hyperfine or Infoton-CISS channels",
            "MERA isometries/disentanglers map boundary state into bulk entanglement",
        ),
    ),
    "advanced_mechanisms.holographic_memory_retrieval": (
        ("P0R06395:hatR_QEC",),
        (
            "boundary cue is treated analogously to a QEC syndrome",
            "hatR_QEC recovery traces ER=EPR geodesic flow",
        ),
        (
            "L5 cue induces syndrome decoding",
            "QEC recovery operator traces bulk-to-boundary geodesic flow",
            "reconstructed L1 state biases L4 dynamics via IET/QZE",
        ),
    ),
    "advanced_mechanisms.consilium_multiobjective_optimisation": (
        ("P0R06399:L_Ethical", "P0R06400:w_i", "P0R06400:1_over_E"),
        (
            "L_Ethical = f(Coherence C, Complexity K, Qualia Q)",
            "optimise on Pareto front with dynamic weights w_i",
            "geodesic flow minimises ethical dissonance 1/E",
        ),
        (
            "L13 source field is formalised as a fibre bundle E",
            "L15 Consilium is formalised as principal connection",
            "multi-objective optimisation uses Pareto, dynamic weighting, and geodesic flow",
        ),
    ),
}

SPEC_METADATA: dict[str, dict[str, tuple[str, ...] | str]] = {
    "advanced_mechanisms.geometric_physical_transduction": {
        "validation_protocol": "paper0.advanced_mechanisms.geometric_physical_transduction",
        "canonical_statement": (
            "L7-to-L8 transduction wording is bounded to symbol operators, Psi-field "
            "resonance, local gauge transformation, and field-connection modification."
        ),
        "variables": ("S", "O^S", "U_S", "Psi_field", "field_connection"),
        "validation_targets": (
            "preserve symbol-operator and local-gauge labels",
            "require field-connection modification channel",
            "reject treating symbolic transduction as observed gauge physics",
        ),
        "null_controls": (
            "missing-field-connection control must be rejected",
            "missing-local-gauge control must be rejected",
            "observed-gauge-physics control must be rejected",
        ),
    },
    "advanced_mechanisms.holographic_memory_encoding": {
        "validation_protocol": "paper0.advanced_mechanisms.holographic_memory_encoding",
        "canonical_statement": (
            "L9 encoding wording is bounded to L4 synchronisation, L1 quantum memory "
            "bias, and MERA boundary-to-bulk entanglement renormalisation."
        ),
        "variables": ("L4", "UPDE", "L1", "MERA", "entanglement_renormalisation"),
        "validation_targets": (
            "preserve the L4-to-L1-to-MERA encoding path",
            "require all source-listed encoding stages",
            "reject memory evidence without empirical substrate data",
        ),
        "null_controls": (
            "missing-L1-bias control must be rejected",
            "missing-MERA-boundary control must be rejected",
            "memory-evidence control must be rejected",
        ),
    },
    "advanced_mechanisms.holographic_memory_retrieval": {
        "validation_protocol": "paper0.advanced_mechanisms.holographic_memory_retrieval",
        "canonical_statement": (
            "L9 retrieval wording is bounded to cue syndrome decoding, hatR_QEC "
            "recovery, ER=EPR geodesic flow, and L1/L4 reconstruction bias."
        ),
        "variables": ("L5", "syndrome", "hatR_QEC", "ER_EPR", "IET", "QZE"),
        "validation_targets": (
            "preserve QEC recovery and geodesic-flow labels",
            "require cue, syndrome, recovery, and reconstruction stages",
            "reject retrieval wording as evidence of holographic memory",
        ),
        "null_controls": (
            "missing-recovery-operator control must be rejected",
            "missing-reconstruction-bias control must be rejected",
            "holographic-memory-evidence control must be rejected",
        ),
    },
    "advanced_mechanisms.consilium_multiobjective_optimisation": {
        "validation_protocol": "paper0.advanced_mechanisms.consilium_multiobjective_optimisation",
        "canonical_statement": (
            "L13-L15 Consilium wording is bounded to fibre-bundle/source-field context, "
            "principal connection, and multi-objective Pareto/geodesic optimisation."
        ),
        "variables": ("L_Ethical", "C", "K", "Q", "w_i", "g_star", "E"),
        "validation_targets": (
            "preserve C/K/Q multi-objective optimisation",
            "require Pareto, dynamic-weighting, and geodesic-flow channels",
            "reject meta-universal geometry as empirical evidence",
        ),
        "null_controls": (
            "missing-Pareto-front control must be rejected",
            "missing-dynamic-weighting control must be rejected",
            "meta-universal-evidence control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class AdvancedMechanismsValidationSpec:
    """Validation spec promoted from Paper 0 advanced mechanisms records."""

    key: str
    validation_protocol: str
    manuscript: str
    section_path: str
    canonical_statement: str
    source_equation_ids: tuple[str, ...]
    source_formulae: tuple[str, ...]
    source_mechanisms: tuple[str, ...]
    source_ledger_ids: tuple[str, ...]
    source_record_ids: tuple[str, ...]
    source_block_indices: tuple[int, ...]
    anchor_math_ids: tuple[str, ...]
    variables: tuple[str, ...]
    validation_targets: tuple[str, ...]
    executable_validation_targets: tuple[str, ...]
    null_controls: tuple[str, ...]
    claim_boundary: str
    implementation_status: str
    domain_review_status: str
    hardware_status: str


@dataclass(frozen=True, slots=True)
class AdvancedMechanismsValidationSpecBundle:
    """Advanced mechanisms validation specs plus coverage summary."""

    specs: tuple[AdvancedMechanismsValidationSpec, ...]
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


def build_advanced_mechanisms_specs(
    source_records: list[dict[str, Any]],
) -> AdvancedMechanismsValidationSpecBundle:
    """Build source-covered validation specs for advanced mechanisms records."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    specs: list[AdvancedMechanismsValidationSpec] = []
    for key in (
        "advanced_mechanisms.geometric_physical_transduction",
        "advanced_mechanisms.holographic_memory_encoding",
        "advanced_mechanisms.holographic_memory_retrieval",
        "advanced_mechanisms.consilium_multiobjective_optimisation",
    ):
        metadata = SPEC_METADATA[key]
        equation_ids, formulae, mechanisms = MECHANISMS_BY_SPEC[key]
        specs.append(
            AdvancedMechanismsValidationSpec(
                key=key,
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=equation_ids,
                source_formulae=formulae,
                source_mechanisms=mechanisms,
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                anchor_math_ids=(),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=(
                    "source-bounded advanced-mechanisms simulator contract; not empirical evidence"
                ),
                implementation_status="implemented_executable_fixture",
                domain_review_status="promoted_to_validation_spec",
                hardware_status="simulator_only_no_provider_submission",
            )
        )

    summary = {
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(SOURCE_LEDGER_IDS),
        "coverage_match": True,
        "unconsumed_source_ledger_ids": [],
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "structural_source_ledger_ids": list(STRUCTURAL_SOURCE_LEDGER_IDS),
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "source_formula_ids": [
            formula_id
            for ids, _formulae, _mechanisms in MECHANISMS_BY_SPEC.values()
            for formula_id in ids
        ],
        "source_mechanism_count": sum(
            len(mechanisms) for _ids, _formulae, mechanisms in MECHANISMS_BY_SPEC.values()
        ),
        "all_specs_have_null_controls": all(bool(spec.null_controls) for spec in specs),
        "all_specs_have_executable_targets": all(
            bool(spec.executable_validation_targets) for spec in specs
        ),
        "all_specs_are_source_anchored": all(bool(spec.source_ledger_ids) for spec in specs),
        "all_specs_are_claim_bounded": all(
            "not empirical evidence" in spec.claim_boundary for spec in specs
        ),
        "hardware_status": "simulator_only_no_provider_submission",
        "policy": (
            "P0R06382-P0R06401 are promoted as source-covered advanced-mechanisms "
            "specifications only. Passing fixtures is not empirical evidence."
        ),
    }
    return AdvancedMechanismsValidationSpecBundle(specs=tuple(specs), summary=summary)


def build_validation_report(bundle: AdvancedMechanismsValidationSpecBundle) -> str:
    """Render a concise Markdown report for advanced mechanisms specs."""
    lines = [
        "# Paper 0 Advanced Mechanisms Specs",
        "",
        f"- Source records: `{bundle.summary['source_record_count']}`",
        f"- Consumed source records: `{bundle.summary['consumed_source_record_count']}`",
        "- Coverage status: `match`",
        f"- Source span: `{', '.join(bundle.summary['source_ledger_span'])}`",
        f"- Spec count: `{bundle.summary['spec_count']}`",
        f"- Hardware status: `{bundle.summary['hardware_status']}`",
        "",
        "## Specs",
        "",
    ]
    for spec in bundle.specs:
        lines.extend(
            [
                f"### {spec.key}",
                "",
                f"- Protocol: `{spec.validation_protocol}`",
                f"- Source ledgers: `{', '.join(spec.source_ledger_ids)}`",
                f"- Source formulae: `{', '.join(spec.source_formulae)}`",
                f"- Source mechanisms: `{len(spec.source_mechanisms)}`",
                f"- Null controls: `{len(spec.null_controls)}`",
                f"- Claim boundary: `{spec.claim_boundary}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Policy",
            "",
            "These records are source-anchored advanced-mechanisms specifications only. "
            "Passing any fixture is not empirical evidence and does not validate gauge "
            "transduction, holographic memory, QEC retrieval, or meta-universal optimisation.",
            "",
        ]
    )
    return "\n".join(lines)


def write_outputs(
    bundle: AdvancedMechanismsValidationSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown outputs for the advanced mechanisms bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_advanced_mechanisms_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_advanced_mechanisms_validation_specs_report_{date_tag}.md"
    payload = {
        "summary": bundle.summary,
        "specs": [asdict(spec) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(build_validation_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def _select_required_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    required = set(SOURCE_LEDGER_IDS)
    return [record for record in records if str(record.get("ledger_id")) in required]


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args(argv)

    records = _select_required_records(load_jsonl(args.ledger))
    bundle = build_advanced_mechanisms_specs(records)
    paths = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    for label, path in paths.items():
        print(f"{label}: {path}")
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
