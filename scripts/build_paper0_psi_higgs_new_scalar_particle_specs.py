#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 Psi-Higgs scalar spec builder
"""Promote Paper 0 Psi-Higgs new-scalar-particle records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1638, 1647))
CLAIM_BOUNDARY = "source-bounded Psi-Higgs new-scalar-particle bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "psi_higgs_new_scalar_particle.scalar_remnant_identity": {
        "context_id": "scalar_remnant_identity",
        "validation_protocol": "paper0.psi_higgs_new_scalar_particle.scalar_remnant_identity",
        "canonical_statement": (
            "The source identifies the radial h(x) fluctuation left by the Higgs mechanism as a new massive scalar called the Psi-Higgs."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:scalar_remnant_identity" for n in range(1638, 1640)
        ),
        "source_formulae": (
            "The Psi-Higgs Boson: A New Scalar Particle",
            "Higgs mechanism leaves a physical remnant of the original scalar field",
            "h(x) magnitude fluctuations correspond to a new massive physical scalar particle",
            "within SCPN this particle is named the Psi-Higgs boson",
        ),
        "test_protocols": ("preserve Psi-Higgs scalar-remnant identity boundary",),
        "null_results": ("naming the scalar remnant is not particle-discovery evidence",),
        "variables": ("Psi", "h_x", "scalar_field", "Psi_Higgs"),
        "validation_targets": (
            "preserve radial-fluctuation identity",
            "preserve Psi-Higgs naming boundary",
        ),
        "null_controls": (
            "scalar identity claim must remain source prediction, not observed scalar detection",
        ),
    },
    "psi_higgs_new_scalar_particle.potential_mass_term": {
        "context_id": "potential_mass_term",
        "validation_protocol": "paper0.psi_higgs_new_scalar_particle.potential_mass_term",
        "canonical_statement": (
            "The source derives the Psi-Higgs mass term by expanding the potential around the vacuum expectation value."
        ),
        "source_equation_ids": tuple(f"P0R{n:05d}:potential_mass_term" for n in range(1640, 1644)),
        "source_formulae": (
            "expanding V(|Psi|) around VEV v reveals a mass term for h(x)",
            "V(v+h) ~= V(v) + 1/2 (2 mu^2) h^2 + ...",
            "substituting mu^2 = lambda v^2 gives the Psi-Higgs mass term",
            "L_mass,h = -lambda v^2 h^2",
        ),
        "test_protocols": ("preserve potential-expansion mass-term boundary",),
        "null_results": ("potential expansion is not a measured scalar spectrum",),
        "variables": ("V", "v", "h", "mu", "lambda", "L_mass_h"),
        "validation_targets": ("preserve potential expansion", "preserve h-field mass term"),
        "null_controls": (
            "derived potential term must not be treated as collider mass reconstruction",
        ),
    },
    "psi_higgs_new_scalar_particle.mass_and_detection_boundary": {
        "context_id": "mass_and_detection_boundary",
        "validation_protocol": "paper0.psi_higgs_new_scalar_particle.mass_and_detection_boundary",
        "canonical_statement": (
            "The source states the Psi-Higgs mass relation and frames discovery as future direct evidence for the underlying scalar field."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:mass_and_detection_boundary" for n in range(1644, 1647)
        ),
        "source_formulae": (
            "physical particle mass is m_h = sqrt(2 lambda) v",
            "SCPN predicts a new massive scalar boson, the Psi-Higgs",
            "Psi-Higgs would couple to the Psi-field and massive infoton",
            "higher-order Standard Model couplings are potential interactions",
            "discovery would provide evidence only if observed; this fixture records the source prediction",
        ),
        "test_protocols": ("preserve Psi-Higgs mass and discovery-boundary source accounting",),
        "null_results": ("future-discovery language is not present validation evidence",),
        "variables": ("m_h", "lambda", "v", "Psi_field", "infoton", "Standard_Model"),
        "validation_targets": (
            "preserve mass relation",
            "preserve coupling and discovery boundary",
        ),
        "null_controls": (
            "discovery-would-provide-evidence clause must not be reported as current evidence",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PsiHiggsNewScalarParticleSpec:
    """Psi-Higgs new-scalar-particle spec promoted from Paper 0 records."""

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
class PsiHiggsNewScalarParticleSpecBundle:
    """Psi-Higgs new-scalar-particle specs plus source coverage summary."""

    specs: tuple[PsiHiggsNewScalarParticleSpec, ...]
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


def build_psi_higgs_new_scalar_particle_specs(
    source_records: list[dict[str, Any]],
) -> PsiHiggsNewScalarParticleSpecBundle:
    """Build source-covered Psi-Higgs new-scalar-particle specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[PsiHiggsNewScalarParticleSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PsiHiggsNewScalarParticleSpec(
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
        "title": "Paper 0 Psi-Higgs New Scalar Particle Specs",
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
        "next_source_boundary": "P0R01647",
    }
    return PsiHiggsNewScalarParticleSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PsiHiggsNewScalarParticleSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_psi_higgs_new_scalar_particle_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: PsiHiggsNewScalarParticleSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 Psi-Higgs New Scalar Particle Specs",
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
    bundle: PsiHiggsNewScalarParticleSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir / f"paper0_psi_higgs_new_scalar_particle_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir / f"paper0_psi_higgs_new_scalar_particle_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_EXTRACTION_DIR)
    parser.add_argument("--date-tag", default="2026-05-17")
    args = parser.parse_args()
    bundle = build_from_ledger(args.ledger)
    outputs = write_outputs(bundle, output_dir=args.output_dir, date_tag=args.date_tag)
    print(json.dumps(bundle.summary, indent=2, sort_keys=True))
    print(outputs["json"])
    print(outputs["report"])


if __name__ == "__main__":
    main()
