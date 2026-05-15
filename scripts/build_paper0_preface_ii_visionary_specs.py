#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Preface II visionary spec builder
"""Promote Paper 0 Preface II visionary-register records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(333, 358))
BLANK_SEPARATOR_IDS = ("P0R00341",)
CLAIM_BOUNDARY = "source-bounded Preface II visionary register; not validation evidence"
HARDWARE_STATUS = "source_visionary_register_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "preface_ii_visionary.manifesto_register_boundary": {
        "context_id": "manifesto_register_boundary",
        "validation_protocol": "paper0.preface_ii_visionary.manifesto_register_boundary",
        "canonical_statement": (
            "Preface II is promoted as a visionary manifesto register that frames "
            "consciousness as an active structuring principle while preserving the "
            "boundary that rhetoric is not validation evidence."
        ),
        "source_equation_ids": (
            "P0R00333:preface_ii_architecture_of_being",
            "P0R00334:manifesto_structuring_principle",
            "P0R00337-P0R00340:plain_language_manifesto_and_manual",
        ),
        "source_formulae": (
            "manifesto",
            "active, structuring principle of reality",
            "living architecture",
            "manual",
            "testable plans",
        ),
        "test_protocols": ("preserve visionary-register boundary without evidence promotion",),
        "null_results": ("manifesto register is not empirical validation evidence",),
        "variables": ("visionary_register", "living_architecture", "manual_boundary"),
        "validation_targets": (
            "preserve manifesto register",
            "preserve active-structuring-principle claim as source claim",
            "preserve manual-not-commentary boundary",
        ),
        "null_controls": (
            "manifesto-as-evidence control must be rejected",
            "manual-as-measurement control must be rejected",
        ),
    },
    "preface_ii_visionary.creation_discipline_boundary": {
        "context_id": "creation_discipline_boundary",
        "validation_protocol": "paper0.preface_ii_visionary.creation_discipline_boundary",
        "canonical_statement": (
            "The visionary preface distinguishes Field Architecture as design language "
            "from Consciousness Engineering as intentional creation/modulation."
        ),
        "source_equation_ids": (
            "P0R00335:description_to_creation",
            "P0R00338:field_architecture_design_language",
            "P0R00339:consciousness_engineering_tuning_toolkit",
            "P0R00352-P0R00354:manifesto_discipline_formulations",
        ),
        "source_formulae": (
            "Field Architecture",
            "design language of consciousness",
            "Consciousness Engineering",
            "tuning consciousness fields",
            "vibrational codes",
        ),
        "test_protocols": ("classify visionary discipline roles without evidence overclaim",),
        "null_results": ("creation language is not a demonstrated control protocol",),
        "variables": ("Field_Architecture", "Consciousness_Engineering", "vibrational_codes"),
        "validation_targets": (
            "preserve design-language role",
            "preserve creation/modulation role",
            "preserve requirement for later testing",
        ),
        "null_controls": (
            "creation-language-as-control-proof control must be rejected",
            "untestable-design-language control must be rejected",
        ),
    },
    "preface_ii_visionary.active_inference_mapping": {
        "context_id": "active_inference_mapping",
        "validation_protocol": "paper0.preface_ii_visionary.active_inference_mapping",
        "canonical_statement": (
            "The meta-framework integration maps the visionary register to a conscious "
            "Generative Model, prior pathways, and active-inference intervention."
        ),
        "source_equation_ids": (
            "P0R00342:meta_framework_integrations",
            "P0R00343:predictive_coding_integration",
            "P0R00344:conscious_generative_model",
            "P0R00345:projection_lattices_and_resonance_nodes",
            "P0R00346:active_inference_vibrational_codes",
        ),
        "source_formulae": (
            "vast, conscious Generative Model",
            "projection lattices",
            "resonance nodes",
            "prior pathways",
            "generative model intervention",
        ),
        "test_protocols": ("classify visionary operators by active-inference role",),
        "null_results": ("active-inference mapping is not hardware evidence",),
        "variables": ("projection_lattices", "resonance_nodes", "vibrational_codes"),
        "validation_targets": (
            "map projection lattices to prior pathways",
            "map vibrational codes to generative-model intervention",
            "preserve resonance-node coupling language",
        ),
        "null_controls": (
            "unknown-operator control must be rejected",
            "role-swap control must be rejected",
        ),
    },
    "preface_ii_visionary.hamiltonian_mastery_boundary": {
        "context_id": "hamiltonian_mastery_boundary",
        "validation_protocol": "paper0.preface_ii_visionary.hamiltonian_mastery_boundary",
        "canonical_statement": (
            "The declaration of intent to master the universal coupling Hamiltonian "
            "is preserved as an agenda claim, not as evidence of mastery."
        ),
        "source_equation_ids": (
            "P0R00347:psi_field_coupling_integration",
            "P0R00348:H_int_mastery_declaration",
            "P0R00355:testable_instruments_rigour_boundary",
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma",
            "declaration of intent",
            "equations, operators, and diagrams",
            "testable instruments",
        ),
        "test_protocols": ("normalise Hamiltonian and reject mastery-as-validation overclaim",),
        "null_results": ("Hamiltonian mastery declaration is not experimental confirmation",),
        "variables": ("lambda_coupling", "Psi_s", "sigma", "H_int"),
        "validation_targets": (
            "preserve all Hamiltonian parameters",
            "preserve declaration-of-intent boundary",
            "preserve testable-instrument requirement",
        ),
        "null_controls": (
            "omitted-Hamiltonian-parameter control must be rejected",
            "mastery-as-validation control must be rejected",
        ),
    },
    "preface_ii_visionary.sigma_atlas_design_language": {
        "context_id": "sigma_atlas_design_language",
        "validation_protocol": "paper0.preface_ii_visionary.sigma_atlas_design_language",
        "canonical_statement": (
            "Field Architecture is framed as an atlas of sigma variables, while "
            "Consciousness Engineering and VIBRANA are framed as sigma design language."
        ),
        "source_equation_ids": (
            "P0R00349:sigma_atlas",
            "P0R00350:novel_sigma_design_vibrana",
            "P0R00351:visionary_manifesto_preface",
            "P0R00356-P0R00357:manual_tools_test_and_extend",
        ),
        "source_formulae": (
            "atlas of all possible sigma variables",
            "designing novel sigma variables",
            "VIBRANA",
            "test them",
            "Status and Method boundary P0R00358",
        ),
        "test_protocols": ("preserve sigma-atlas and design-language boundary",),
        "null_results": ("sigma design language is not validated without tests",),
        "variables": ("sigma_atlas", "novel_sigma", "VIBRANA", "status_method_boundary"),
        "validation_targets": (
            "preserve sigma-atlas claim",
            "preserve novel-sigma design claim",
            "preserve testing/extension obligation",
            "preserve Status and Method boundary",
        ),
        "null_controls": (
            "sigma-design-without-testability control must be rejected",
            "missing-status-method-boundary control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PrefaceIIVisionarySpec:
    """Preface II visionary spec promoted from Paper 0 records."""

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
class PrefaceIIVisionarySpecBundle:
    """Preface II visionary specs plus source coverage summary."""

    specs: tuple[PrefaceIIVisionarySpec, ...]
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


def build_preface_ii_visionary_specs(
    source_records: list[dict[str, Any]],
) -> PrefaceIIVisionarySpecBundle:
    """Build source-covered Preface II visionary-register specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    blank_separator_count = len(BLANK_SEPARATOR_IDS)
    specs: list[PrefaceIIVisionarySpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PrefaceIIVisionarySpec(
                key=key,
                context_id=str(metadata["context_id"]),
                validation_protocol=str(metadata["validation_protocol"]),
                manuscript="Paper 0 - The Foundational Framework",
                section_path=str(anchors[0]["section_path"]),
                canonical_statement=str(metadata["canonical_statement"]),
                source_equation_ids=tuple(str(item) for item in metadata["source_equation_ids"]),
                source_ledger_ids=SOURCE_LEDGER_IDS,
                source_record_ids=tuple(str(anchor["source_record_id"]) for anchor in anchors),
                source_block_indices=tuple(
                    int(anchor["source_block_index"]) for anchor in anchors
                ),
                source_formulae=tuple(str(item) for item in metadata["source_formulae"]),
                test_protocols=tuple(str(item) for item in metadata["test_protocols"]),
                null_results=tuple(str(item) for item in metadata["null_results"]),
                variables=tuple(str(item) for item in metadata["variables"]),
                validation_targets=tuple(str(item) for item in metadata["validation_targets"]),
                executable_validation_targets=tuple(
                    str(item) for item in metadata["validation_targets"]
                ),
                null_controls=tuple(str(item) for item in metadata["null_controls"]),
                claim_boundary=CLAIM_BOUNDARY,
                implementation_status="implemented_executable_fixture",
                domain_review_status="source_visionary_register_preserved",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed_ids = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 Preface II Visionary Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed_ids),
        "coverage_match": tuple(consumed_ids) == SOURCE_LEDGER_IDS,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "blank_separator_count": blank_separator_count,
        "interaction_formula_count": 1,
        "status_method_boundary": "P0R00358",
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed_ids
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
    }
    return PrefaceIIVisionarySpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(path: Path = DEFAULT_LEDGER_PATH) -> PrefaceIIVisionarySpecBundle:
    """Build Preface II visionary specs from the canonical ledger."""
    return build_preface_ii_visionary_specs(load_jsonl(path))


def write_outputs(
    bundle: PrefaceIIVisionarySpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown Preface II visionary spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_preface_ii_visionary_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_preface_ii_visionary_validation_specs_report_{date_tag}.md"
    payload = {
        "specs": [asdict(spec) for spec in bundle.specs],
        "summary": bundle.summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: PrefaceIIVisionarySpecBundle) -> str:
    """Render a compact Markdown report for human review."""
    lines = [
        "# Paper 0 Preface II Visionary Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Blank separators: {bundle.summary['blank_separator_count']}",
        f"- Interaction formula count: {bundle.summary['interaction_formula_count']}",
        f"- Status and Method boundary: {bundle.summary['status_method_boundary']}",
        f"- Hardware status: {bundle.summary['hardware_status']}",
        f"- Claim boundary: {bundle.summary['claim_boundary']}",
        "",
        "## Specs",
    ]
    for spec in bundle.specs:
        lines.append(f"- `{spec.key}`: {spec.canonical_statement}")
        if "H_int = -lambda * Psi_s * sigma" in spec.source_formulae:
            lines.append("  - Source formula: `H_int = -lambda * Psi_s * sigma`")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build and write Paper 0 Preface II visionary validation specs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-13")
    args = parser.parse_args()

    bundle = build_from_ledger(args.ledger)
    write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    return 0 if bundle.summary["coverage_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
