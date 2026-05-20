#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 predicted particles spec builder
"""Promote Paper 0 infoton and Psi-Higgs prediction records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1597, 1623))
CLAIM_BOUNDARY = "source-bounded infoton and Psi-Higgs prediction bridge; not validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "predicted_particles_infoton_psi_higgs.particle_prediction_opening": {
        "context_id": "particle_prediction_opening",
        "validation_protocol": "paper0.predicted_particles_infoton_psi_higgs.particle_prediction_opening",
        "canonical_statement": (
            "The source frames spontaneous symmetry breaking of the local U(1) sector as predicting a massive infoton and a Psi-Higgs scalar."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:particle_prediction_opening" for n in range(1597, 1603)
        ),
        "source_formulae": (
            "Predicted Particles: The Infoton and the Psi-Higgs Boson",
            "local U(1) symmetry breaking via Mexican-hat potential predicts two new physical particles",
            "infoton gauge field A_mu receives mass m_A = g v",
            "Psi-Higgs radial excitation mass is m_h = sqrt(2 lambda) v",
            "two-particle prediction is framed as testable high-energy physics, not observed evidence",
        ),
        "test_protocols": ("preserve particle-prediction opening boundary",),
        "null_results": ("particle-prediction wording is not discovery evidence",),
        "variables": ("A_mu", "m_A", "g", "v", "h_Psi", "m_h", "lambda"),
        "validation_targets": (
            "preserve infoton mass relation",
            "preserve Psi-Higgs scalar relation",
        ),
        "null_controls": ("prediction must not be promoted to observed particle detection",),
    },
    "predicted_particles_infoton_psi_higgs.search_strategy_summary": {
        "context_id": "search_strategy_summary",
        "validation_protocol": "paper0.predicted_particles_infoton_psi_higgs.search_strategy_summary",
        "canonical_statement": (
            "The source lists collider and cosmological search channels as proposed frontiers for the predicted particles."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:search_strategy_summary" for n in range(1603, 1609)
        ),
        "source_formulae": (
            "LHC search strategy includes exotic Higgs decays h_SM -> h_Psi h_Psi",
            "LHC search strategy includes heavy resonance pp -> h_Psi -> ZZ",
            "ultralight Psi-Higgs is framed as a dark-matter candidate",
            "bosonic clouds around spinning black holes could yield monochromatic gravitational waves",
            "finding either signature is a source claim, not current proof",
        ),
        "test_protocols": ("preserve collider and cosmological search-channel boundary",),
        "null_results": ("search roadmap is not experimental confirmation",),
        "variables": ("h_SM", "h_Psi", "ZZ", "LHC", "LISA", "gravitational_wave"),
        "validation_targets": (
            "preserve LHC search modes",
            "preserve cosmological signature mode",
        ),
        "null_controls": (
            "roadmap language must not be interpreted as completed LHC or LISA evidence",
        ),
    },
    "predicted_particles_infoton_psi_higgs.active_inference_mapping": {
        "context_id": "active_inference_mapping",
        "validation_protocol": "paper0.predicted_particles_infoton_psi_higgs.active_inference_mapping",
        "canonical_statement": (
            "The source maps the predicted particles onto active-inference roles for local error signalling and generative-model stability."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:active_inference_mapping" for n in range(1609, 1615)
        ),
        "source_formulae": (
            "predicted particles are physical hardware components for embodied active inference",
            "massive infoton carries prediction-error signal",
            "short infoton range defines boundaries of a local inference engine",
            "Psi-Higgs represents stability and confidence of the generative model",
            "Psi-Higgs mass is the energy cost to perturb the VEV-associated self model",
        ),
        "test_protocols": ("preserve active-inference particle-role mapping boundary",),
        "null_results": ("active-inference mapping is not a measured neural implementation",),
        "variables": (
            "prediction_error",
            "local_inference_engine",
            "generative_model",
            "VEV",
            "self_model",
        ),
        "validation_targets": (
            "preserve infoton error-signal role",
            "preserve Psi-Higgs confidence role",
        ),
        "null_controls": (
            "hardware-component metaphor must remain source accounting, not empirical neurophysics",
        ),
    },
    "predicted_particles_infoton_psi_higgs.h_int_falsifiability_bridge": {
        "context_id": "h_int_falsifiability_bridge",
        "validation_protocol": "paper0.predicted_particles_infoton_psi_higgs.h_int_falsifiability_bridge",
        "canonical_statement": (
            "The source links symmetry-breaking parameters to the H_int interaction and frames particle searches as proposed falsifiability routes."
        ),
        "source_equation_ids": tuple(
            f"P0R{n:05d}:h_int_falsifiability_bridge" for n in range(1615, 1623)
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma",
            "coupling constant in H_int is mapped to gauge coupling g",
            "field strength Psi_s is mapped to background VEV v",
            "interaction range is set by infoton mass m_A = g v",
            "searching for the Psi-Higgs is framed as probing the conscious-field excitation",
            "infoton properties would measure strength and range only if the source prediction were physically realised",
        ),
        "test_protocols": ("preserve H_int parameter and falsifiability boundary",),
        "null_results": ("falsifiability route is not particle-discovery evidence",),
        "variables": ("H_int", "lambda", "Psi_s", "sigma", "g", "v", "m_A"),
        "validation_targets": (
            "preserve H_int parameter mapping",
            "preserve proposed falsifiability route",
        ),
        "null_controls": (
            "high-energy-physics falsifiability language must not be promoted to detection evidence",
        ),
    },
}


@dataclass(frozen=True, slots=True)
class PredictedParticlesInfotonPsiHiggsSpec:
    """Predicted-particles spec promoted from Paper 0 records."""

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
class PredictedParticlesInfotonPsiHiggsSpecBundle:
    """Predicted-particles specs plus source coverage summary."""

    specs: tuple[PredictedParticlesInfotonPsiHiggsSpec, ...]
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


def build_predicted_particles_infoton_psi_higgs_specs(
    source_records: list[dict[str, Any]],
) -> PredictedParticlesInfotonPsiHiggsSpecBundle:
    """Build source-covered infoton and Psi-Higgs prediction specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[PredictedParticlesInfotonPsiHiggsSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            PredictedParticlesInfotonPsiHiggsSpec(
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
        "title": "Paper 0 Predicted Particles Infoton Psi-Higgs Specs",
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
        "next_source_boundary": "P0R01623",
    }
    return PredictedParticlesInfotonPsiHiggsSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> PredictedParticlesInfotonPsiHiggsSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_predicted_particles_infoton_psi_higgs_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: PredictedParticlesInfotonPsiHiggsSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 Predicted Particles Infoton Psi-Higgs Specs",
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
    bundle: PredictedParticlesInfotonPsiHiggsSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artefacts for the spec bundle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = (
        output_dir
        / f"paper0_predicted_particles_infoton_psi_higgs_validation_specs_{date_tag}.json"
    )
    report_path = (
        output_dir
        / f"paper0_predicted_particles_infoton_psi_higgs_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
        "summary": _json_ready(bundle.summary),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build this Paper 0 generated spec bundle from the ledger."""

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
