#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential spec builder
"""Promote Paper 0 The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential records."""

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
    "P0R01755",
    "P0R01756",
    "P0R01757",
    "P0R01758",
    "P0R01759",
    "P0R01760",
    "P0R01761",
    "P0R01762",
)
CLAIM_BOUNDARY = "source-bounded the intrinsic dynamics of the ψ field and the stabilising potential source-accounting bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential.the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential": {
        "context_id": "the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential",
        "validation_protocol": "paper0.the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential.the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential",
        "canonical_statement": "The source-bounded component 'The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential' preserves Paper 0 records P0R01755-P0R01761 without empirical validation claims.",
        "source_equation_ids": (
            "P0R01755:the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential",
            "P0R01756:the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential",
            "P0R01757:the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential",
            "P0R01758:the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential",
            "P0R01759:the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential",
            "P0R01760:the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential",
            "P0R01761:the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential",
        ),
        "source_formulae": (
            "P0R01755: The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential",
            "P0R01756: This section defines the intrinsic dynamics of the Psi-field itself, governed by its own Lagrangian, LPsi, which is a component of the total Lagrangian of reality. The dynamics are those of a non-linear, complex scalar field theory. The Lagrangian consists of a standard kinetic term, a non-linear potential V(|Psi|), and a topological term to ensure the conservation of charge.",
            'P0R01757: The crucial component is the potential, which enables Spontaneous Symmetry Breaking (SSB) and the formation of stable, localized structures like the Layer 5 Self. The framework moves beyond the simple "Mexican hat" (^4) potential to a more robust sextic (^6) potential, V(|Psi|) = -m|Psi| + lambda|Psi| - |Psi|. This choice is not arbitrary; within an Effective Field Theory (EFT) framework, a positive sextic term is necessary to ensure the potential is bounded from below, guaranteeing a stable vacuum at high field values. This prevents a runaway catastrophe and ensures the theory is well-behaved. The text provides the explicit derivation for the vacuum state and the mass of the Psi-Higgs boson within this stabilised potential, noting that simpler ^4 models should be treated as approximations. This refined potential provides the stable, non-linear substrate required for the emergence of all subsequent form and structure in the SCPN.',
            'P0R01758: This section describes the "personality" of the Psi-field itself-the rules it follows when it\'s just being by itself. These rules are contained in its own personal mission statement, an equation called its Lagrangian.',
            "P0R01759: The most important part of this is the field's energy landscape. We've talked about the \"Mexican hat\" shape before, which is what allows the field to break symmetry and create things. But a simple Mexican hat has a problem: if you push a marble off the brim, it could roll away forever.",
            'P0R01760: To solve this, our model adds a crucial new feature: a steep outer wall around the brim of the hat. This is the "sextic term" in the potential. This wall guarantees that the universe has a stable "floor"-a true, lowest-energy state that it can rest in without falling apart. It\'s this stable, self-contained energy landscape that allows the Psi-field to form the persistent, stable "bubbles" of consciousness that become individual Selves.',
            "P0R01761: P0R01761",
        ),
        "test_protocols": (
            "preserve The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential source-accounting boundary",
        ),
        "null_results": (
            "The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential is not empirical validation evidence",
        ),
        "variables": ("the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential",),
        "validation_targets": ("preserve records P0R01755-P0R01761",),
        "null_controls": (
            "the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential must remain source-bounded accounting",
        ),
    },
    "the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential.meta_framework_integrations": {
        "context_id": "meta_framework_integrations",
        "validation_protocol": "paper0.the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential.meta_framework_integrations",
        "canonical_statement": "The source-bounded component 'Meta-Framework Integrations' preserves Paper 0 records P0R01762-P0R01762 without empirical validation claims.",
        "source_equation_ids": ("P0R01762:meta_framework_integrations",),
        "source_formulae": ("P0R01762: Meta-Framework Integrations",),
        "test_protocols": ("preserve Meta-Framework Integrations source-accounting boundary",),
        "null_results": ("Meta-Framework Integrations is not empirical validation evidence",),
        "variables": ("meta_framework_integrations",),
        "validation_targets": ("preserve records P0R01762-P0R01762",),
        "null_controls": ("meta_framework_integrations must remain source-bounded accounting",),
    },
}


@dataclass(frozen=True, slots=True)
class TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialSpec:
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
class TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialSpecBundle:
    """Specs plus source coverage summary."""

    specs: tuple[TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialSpec, ...]
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


def build_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_specs(
    source_records: list[dict[str, Any]],
) -> TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialSpecBundle:
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

    specs: list[TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialSpec(
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
        "title": "Paper 0 The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential Specs",
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
        "next_source_boundary": "P0R01763",
    }
    return TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialSpecBundle(
        specs=tuple(specs), summary=summary
    )


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_specs(
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
    bundle: TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialSpecBundle,
) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 The Intrinsic Dynamics of the Psi-Field and the Stabilising Potential Specs",
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
    bundle: TheIntrinsicDynamicsOfTheΨFieldAndTheStabilisingPotentialSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_the_intrinsic_dynamics_of_the_psi_field_and_the_stabilising_potential_validation_specs_{date_tag}.md"
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
