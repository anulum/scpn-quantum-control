#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Preface I rigour spec builder
"""Promote Paper 0 Preface I methodological-rigour records."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTRACTION_DIR = REPO_ROOT / "docs" / "internal" / "paper0_foundational_extraction"
DEFAULT_LEDGER_PATH = DEFAULT_EXTRACTION_DIR / "paper0_canonical_review_ledger_2026-05-13.jsonl"

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(307, 333))
BLANK_SEPARATOR_IDS = ("P0R00315", "P0R00332")
CLAIM_BOUNDARY = "source-bounded Preface I methodological rigour; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "preface_i_rigour.methodological_third_path": {
        "context_id": "methodological_third_path",
        "validation_protocol": "paper0.preface_i_rigour.methodological_third_path",
        "canonical_statement": (
            "Preface I rejects epiphenomenal and metaphysical sequestration, then "
            "frames consciousness as a physically real field phenomenon requiring "
            "formal empirical inquiry."
        ),
        "source_equation_ids": (
            "P0R00307:preface_i_mandate_for_rigour",
            "P0R00308:third_path_field_phenomenon",
            "P0R00311-P0R00314:plain_language_third_path_and_manual_boundary",
        ),
        "source_formulae": (
            "third path",
            "physically real field phenomenon",
            "mathematical equations",
            "precise diagrams",
            "testable models",
        ),
        "test_protocols": ("preserve third-path rigour boundary",),
        "null_results": ("methodological framing is not empirical validation evidence",),
        "variables": ("third_path", "field_phenomenon", "formalism"),
        "validation_targets": (
            "preserve rejection of epiphenomenal-only interpretation",
            "preserve rejection of metaphysics-only interpretation",
            "preserve formal empirical-inquiry requirement",
        ),
        "null_controls": (
            "metaphysics-without-formalism control must be rejected",
            "third-path-as-measurement control must be rejected",
        ),
    },
    "preface_i_rigour.discipline_distinction": {
        "context_id": "discipline_distinction",
        "validation_protocol": "paper0.preface_i_rigour.discipline_distinction",
        "canonical_statement": (
            "The preface separates Field Architecture as the theoretical discipline "
            "from Consciousness Engineering as the applied discipline."
        ),
        "source_equation_ids": (
            "P0R00309:maxwell_to_engineering_analogy",
            "P0R00312:field_architecture_blueprints",
            "P0R00313:consciousness_engineering_application",
            "P0R00327-P0R00328:academic_discipline_definitions",
        ),
        "source_formulae": (
            "Field Architecture",
            "Consciousness Engineering",
            "theoretical toolkit",
            "experiments, simulations, and devices",
        ),
        "test_protocols": ("classify discipline roles without collapsing theory and application",),
        "null_results": ("discipline distinction is not a measured causal effect",),
        "variables": ("Field_Architecture", "Consciousness_Engineering"),
        "validation_targets": (
            "preserve theoretical/applied distinction",
            "preserve Maxwell-to-engineering analogy",
            "preserve experiment/simulation/device implementation horizon",
        ),
        "null_controls": (
            "collapsed-discipline control must be rejected",
            "application-without-theory control must be rejected",
        ),
    },
    "preface_i_rigour.formalism_noetic_boundary": {
        "context_id": "formalism_noetic_boundary",
        "validation_protocol": "paper0.preface_i_rigour.formalism_noetic_boundary",
        "canonical_statement": (
            "The preface acknowledges noetic-field lineage while requiring explicit "
            "equations, field operators, layered models, critique, extension, and integration."
        ),
        "source_equation_ids": (
            "P0R00310:noetic_field_theory_formalism_boundary",
            "P0R00314:instruction_manual_boundary",
            "P0R00329:academic_noetic_formalism",
            "P0R00330-P0R00331:architecture_manual_and_open_critique",
        ),
        "source_formulae": (
            "Noetic Field Theory",
            "explicit equations",
            "field operators",
            "layered models",
            "open to critique, extension, and integration",
        ),
        "test_protocols": ("preserve formalism and critique boundary",),
        "null_results": ("formalism requirement is not itself evidence for the theory",),
        "variables": ("Noetic_Field_Theory", "field_operators", "layered_models"),
        "validation_targets": (
            "preserve lineage acknowledgement",
            "preserve distinction from intuition-only traditions",
            "preserve critique and extension boundary",
        ),
        "null_controls": (
            "intuition-only lineage control must be rejected",
            "closed-finality control must be rejected",
        ),
    },
    "preface_i_rigour.hpc_structure_application_mapping": {
        "context_id": "hpc_structure_application_mapping",
        "validation_protocol": "paper0.preface_i_rigour.hpc_structure_application_mapping",
        "canonical_statement": (
            "In the HPC integration, Field Architecture maps to generative-model "
            "structure and Consciousness Engineering maps to prediction-error modulation."
        ),
        "source_equation_ids": (
            "P0R00316:meta_framework_integrations",
            "P0R00317:predictive_coding_integration",
            "P0R00318:hpc_congruence",
            "P0R00319:field_architecture_generative_model_structure",
            "P0R00320:consciousness_engineering_prediction_error_modulation",
        ),
        "source_formulae": (
            "Hierarchical Predictive Coding",
            "generative model structure",
            "prediction error modulation",
            "projection networks",
            "resonance nodes",
        ),
        "test_protocols": ("classify HPC discipline-to-role mapping",),
        "null_results": ("HPC role mapping is not a hardware measurement",),
        "variables": ("generative_model_structure", "prediction_error_modulation"),
        "validation_targets": (
            "map Field Architecture to generative-model structure",
            "map Consciousness Engineering to prediction-error modulation",
            "preserve active-lattice/projection-network/resonance-node vocabulary",
        ),
        "null_controls": (
            "unknown-discipline control must be rejected",
            "role-swap control must be rejected",
        ),
    },
    "preface_i_rigour.sigma_programme_bridge": {
        "context_id": "sigma_programme_bridge",
        "validation_protocol": "paper0.preface_i_rigour.sigma_programme_bridge",
        "canonical_statement": (
            "The preface bridges the abstract interaction Hamiltonian to a concrete "
            "sigma research programme: identify, characterise, design, and control "
            "collective state variables."
        ),
        "source_equation_ids": (
            "P0R00321:psi_field_coupling_integration",
            "P0R00322:H_int_research_programme_bridge",
            "P0R00323:field_architecture_identifies_sigma",
            "P0R00324:consciousness_engineering_designs_sigma",
            "P0R00325-P0R00332:academic_preface_closure",
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma",
            "identifying and characterising sigma",
            "designing and controlling sigma",
            "VIBRANA",
            "Preface II boundary P0R00333",
        ),
        "test_protocols": ("normalise source formula and preserve sigma programme bridge",),
        "null_results": ("sigma programme bridge is not experimental confirmation",),
        "variables": ("lambda_coupling", "Psi_s", "sigma", "VIBRANA"),
        "validation_targets": (
            "preserve Hamiltonian bridge",
            "preserve sigma identification role",
            "preserve sigma design/control role",
            "preserve next-section boundary",
        ),
        "null_controls": (
            "omitted-Hamiltonian-parameter control must be rejected",
            "empirical-validation-overclaim control must be rejected",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PrefaceIRigourSpec:
    """Preface I rigour spec promoted from Paper 0 records."""

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
class PrefaceIRigourSpecBundle:
    """Preface I rigour specs plus source coverage summary."""

    specs: tuple[PrefaceIRigourSpec, ...]
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


def build_preface_i_rigour_specs(
    source_records: list[dict[str, Any]],
) -> PrefaceIRigourSpecBundle:
    """Build source-covered Preface I methodological-rigour specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    blank_separator_count = len(BLANK_SEPARATOR_IDS)
    specs: list[PrefaceIRigourSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PrefaceIRigourSpec(
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
                domain_review_status="source_methodology_preserved",
                hardware_status=HARDWARE_STATUS,
            )
        )

    consumed_ids = sorted({ledger_id for spec in specs for ledger_id in spec.source_ledger_ids})
    summary = {
        "title": "Paper 0 Preface I Rigour Specs",
        "source_record_count": len(SOURCE_LEDGER_IDS),
        "consumed_source_record_count": len(consumed_ids),
        "coverage_match": tuple(consumed_ids) == SOURCE_LEDGER_IDS,
        "source_ledger_span": [SOURCE_LEDGER_IDS[0], SOURCE_LEDGER_IDS[-1]],
        "spec_count": len(specs),
        "spec_keys": [spec.key for spec in specs],
        "blank_separator_count": blank_separator_count,
        "interaction_formula_count": 1,
        "preface_ii_boundary": "P0R00333",
        "all_specs_are_source_anchored": all(spec.source_ledger_ids for spec in specs),
        "all_specs_have_null_controls": all(spec.null_controls for spec in specs),
        "unconsumed_source_ledger_ids": [
            ledger_id for ledger_id in SOURCE_LEDGER_IDS if ledger_id not in consumed_ids
        ],
        "claim_boundary": CLAIM_BOUNDARY,
        "hardware_status": HARDWARE_STATUS,
    }
    return PrefaceIRigourSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(path: Path = DEFAULT_LEDGER_PATH) -> PrefaceIRigourSpecBundle:
    """Build Preface I rigour specs from the canonical ledger."""
    return build_preface_i_rigour_specs(load_jsonl(path))


def write_outputs(
    bundle: PrefaceIRigourSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-13",
) -> dict[str, Path]:
    """Write JSON and Markdown Preface I rigour spec artefacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_preface_i_rigour_validation_specs_{date_tag}.json"
    report_path = output_dir / f"paper0_preface_i_rigour_validation_specs_report_{date_tag}.md"
    payload = {
        "specs": [asdict(spec) for spec in bundle.specs],
        "summary": bundle.summary,
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def render_report(bundle: PrefaceIRigourSpecBundle) -> str:
    """Render a compact Markdown report for human review."""
    lines = [
        "# Paper 0 Preface I Rigour Specs",
        "",
        f"- Source span: {bundle.summary['source_ledger_span'][0]} - "
        f"{bundle.summary['source_ledger_span'][1]}",
        f"- Source records: {bundle.summary['source_record_count']}",
        f"- Consumed source records: {bundle.summary['consumed_source_record_count']}",
        f"- Coverage match: {bundle.summary['coverage_match']}",
        f"- Spec count: {bundle.summary['spec_count']}",
        f"- Blank separators: {bundle.summary['blank_separator_count']}",
        f"- Interaction formula count: {bundle.summary['interaction_formula_count']}",
        f"- Preface II boundary: {bundle.summary['preface_ii_boundary']}",
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
    """Build and write Paper 0 Preface I rigour validation specs."""
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
