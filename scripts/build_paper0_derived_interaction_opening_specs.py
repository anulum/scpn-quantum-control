#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Commercial license available
# (c) Concepts 1996-2026 Miroslav Sotek. All rights reserved.
# (c) Code 2020-2026 Miroslav Sotek. All rights reserved.
# ORCID: 0009-0009-3560-0851
# Contact: www.anulum.li | protoscience@anulum.li
# scpn-quantum-control -- Paper 0 derived interaction opening spec builder
"""Promote Paper 0 derived Master Interaction Lagrangian opening records."""

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

SOURCE_LEDGER_IDS = tuple(f"P0R{number:05d}" for number in range(1384, 1422))
CLAIM_BOUNDARY = "source-bounded derived interaction opening; not experimental validation evidence"
HARDWARE_STATUS = "source_methodology_no_experiment"

SPEC_CONTENT: dict[str, dict[str, tuple[str, ...] | str]] = {
    "derived_interaction_opening.gauge_theory_grounding": {
        "context_id": "gauge_theory_grounding",
        "validation_protocol": "paper0.derived_interaction_opening.gauge_theory_grounding",
        "canonical_statement": (
            "The source frames the transition from phenomenological postulation to a first-principles "
            "derivation grounded in complex scalar field properties, U(1) symmetry, gauge principle, "
            "infoton mediation, geometric coupling, and FIM-governed gauge kinetics."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:gauge_theory_grounding" for number in range(1384, 1392)
        ),
        "source_formulae": (
            "Deriving the Master Interaction Lagrangian",
            "Psi-field is defined as a complex scalar field with spin-0 quanta",
            "global U(1) phase symmetry implies conserved Psi-charge via Noether theorem",
            "local U(1) gauge principle necessitates a mediating spin-1 infoton field A_mu",
            "L_Int prime contains Informational and Geometric components",
            "Informational component couples Psi-charge current to the infoton field A_mu",
            "Geometric component contains non-minimal coupling -xi R Psi* Psi",
            "infoton kinetic dynamics are governed by the Fisher Information Metric rather than spacetime metric",
        ),
        "test_protocols": ("preserve derived-interaction opening boundary",),
        "null_results": (
            "first-principles source derivation is not experimental validation evidence",
        ),
        "variables": ("Psi", "U1", "Psi_charge", "A_mu", "L_Int_prime", "xi", "R", "FIM"),
        "validation_targets": (
            "preserve complex-scalar and Psi-charge grounding",
            "preserve local gauge principle and infoton requirement",
            "preserve two-component derived Lagrangian boundary",
        ),
        "null_controls": (
            "phenomenological black-box H_int must not satisfy this derived-boundary spec",
        ),
    },
    "derived_interaction_opening.predictive_coding_mapping": {
        "context_id": "predictive_coding_mapping",
        "validation_protocol": "paper0.derived_interaction_opening.predictive_coding_mapping",
        "canonical_statement": (
            "The source maps the derived fields into predictive-coding language: Psi-charge carries belief, "
            "and the infoton communicates prediction-error-like mismatch through a gauge-coupling term."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:predictive_coding_mapping" for number in range(1393, 1399)
        ),
        "source_formulae": (
            "derivation provides physical hardware for cosmic predictive coding",
            "conserved Psi-charge is interpreted as the carrier of priors or beliefs",
            "configuration of the Psi-field represents generative-model content",
            "infoton A_mu communicates mismatch between belief and material reality",
            "ig A_mu (Psi* partial_mu Psi - Psi partial_mu Psi*) drives infoton signal generation",
            "gauge principle ensures beliefs and evidence can be compared everywhere",
        ),
        "test_protocols": (
            "preserve predictive-coding mapping without promoting it to measurement",
        ),
        "null_results": ("belief/prediction-error language is not empirical validation",),
        "variables": ("Psi_charge", "A_mu", "Psi", "partial_mu", "belief", "prediction_error"),
        "validation_targets": (
            "preserve Psi-charge belief mapping",
            "preserve infoton prediction-error mapping",
            "preserve gauge-comparison source claim boundary",
        ),
        "null_controls": ("predictive-coding metaphor must not count as observed infoton signal",),
    },
    "derived_interaction_opening.h_int_gauge_identification": {
        "context_id": "h_int_gauge_identification",
        "validation_protocol": "paper0.derived_interaction_opening.h_int_gauge_identification",
        "canonical_statement": (
            "The source identifies the universal interaction Hamiltonian H_int with a relativistic U(1) "
            "gauge interaction, mapping Psi_s to Psi, sigma to the conserved current, lambda to g, "
            "and mediation to A_mu."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:h_int_gauge_identification" for number in range(1399, 1409)
        ),
        "source_formulae": (
            "H_int = -lambda * Psi_s * sigma",
            "L_interaction = i g A_mu J_mu",
            "Psi_s field is represented by Psi itself",
            "sigma is represented by J_mu = (Psi* partial_mu Psi - Psi partial_mu Psi*)",
            "lambda is identified with the gauge coupling g",
            "interaction is mediated by the infoton field A_mu",
            "interaction is classified as a U(1) gauge interaction",
        ),
        "test_protocols": ("preserve H_int to U(1) gauge-identification mapping",),
        "null_results": ("H_int mapping is not a measured coupling constant",),
        "variables": ("H_int", "lambda", "Psi_s", "sigma", "J_mu", "g", "A_mu"),
        "validation_targets": (
            "preserve H_int source equation",
            "preserve sigma-current identification",
            "preserve lambda-to-g and A_mu mediation mapping",
        ),
        "null_controls": (
            "missing conserved-current mapping must fail gauge-identification accounting",
        ),
    },
    "derived_interaction_opening.intrinsic_properties_quantum_numbers": {
        "context_id": "intrinsic_properties_quantum_numbers",
        "validation_protocol": "paper0.derived_interaction_opening.intrinsic_properties_quantum_numbers",
        "canonical_statement": (
            "The source grounds the Psi-field as a fundamental complex scalar with spin-0 quanta, "
            "global U(1) phase symmetry, conserved current, Psi-charge, and local U(1) coupling to a spin-1 infoton."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:intrinsic_properties_quantum_numbers" for number in range(1409, 1416)
        ),
        "source_formulae": (
            "Psi-field is a fundamental complex scalar field permeating spacetime",
            "spin: Psi-field quanta are spin-0",
            "charge: complex scalar implies intrinsic global U(1) phase symmetry",
            "Noether theorem requires conserved current and conserved Psi-charge",
            "Psi-field couples via local U(1) to spin-1 infoton A_mu",
            "gauge kinetic term is controlled by pulled-back Fisher metric g_tilde_F",
        ),
        "test_protocols": ("preserve intrinsic-property and quantum-number source accounting",),
        "null_results": ("field-content diagram is not particle-detection evidence",),
        "variables": ("Psi", "U1", "Psi_charge", "A_mu", "g_tilde_F"),
        "validation_targets": (
            "preserve spin-0 scalar status",
            "preserve conserved Psi-charge status",
            "preserve local U1 infoton coupling and FIM kinetic boundary",
        ),
        "null_controls": (
            "diagram caption must not be promoted to empirical field-content observation",
        ),
    },
    "derived_interaction_opening.gauge_principle_nonabelian_boundary": {
        "context_id": "gauge_principle_nonabelian_boundary",
        "validation_protocol": "paper0.derived_interaction_opening.gauge_principle_nonabelian_boundary",
        "canonical_statement": (
            "The source states that interactions follow from local U(1) gauge symmetry, while treating SU(N) "
            "qualia confinement as a hypothesis and FIM-governed infoton dynamics as the information-geometric boundary."
        ),
        "source_equation_ids": tuple(
            f"P0R{number:05d}:gauge_principle_nonabelian_boundary" for number in range(1416, 1422)
        ),
        "source_formulae": (
            "interactions are an unavoidable consequence of demanding local U(1) phase symmetry",
            "primary interaction is governed by U(1) gauge symmetry",
            "Non-Abelian SU(N) internal qualia structure is a hypothesis",
            "Qualia Confinement is analogous to QCD colour confinement",
            "infoton dynamics are governed by Fisher Information Metric of the statistical manifold",
            "diagram maps spacetime M to statistical manifold Theta with Fisher metric g_F and pullback g_tilde_F",
        ),
        "test_protocols": (
            "preserve local gauge-principle and non-Abelian hypothesis boundaries",
        ),
        "null_results": (
            "SU(N) qualia confinement remains hypothetical, not observed confinement evidence",
        ),
        "variables": ("U1", "SU_N", "A_mu", "FIM", "M", "Theta", "g_F", "g_tilde_F"),
        "validation_targets": (
            "preserve local-U1 interaction origin",
            "preserve SU(N) as hypothesis boundary",
            "preserve FIM-governed infoton dynamics boundary",
        ),
        "null_controls": ("SU(N) hypothesis must not be classified as established gauge group",),
    },
}


@dataclass(frozen=True, slots=True)
class DerivedInteractionOpeningSpec:
    """Derived interaction opening spec promoted from Paper 0 records."""

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
class DerivedInteractionOpeningSpecBundle:
    """Derived interaction opening specs plus source coverage summary."""

    specs: tuple[DerivedInteractionOpeningSpec, ...]
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


def build_derived_interaction_opening_specs(
    source_records: list[dict[str, Any]],
) -> DerivedInteractionOpeningSpecBundle:
    """Build source-covered derived interaction opening specs."""
    records_by_ledger = {str(record["ledger_id"]): record for record in source_records}
    missing = sorted(set(SOURCE_LEDGER_IDS) - set(records_by_ledger))
    if missing:
        raise ValueError(f"missing required source ledger ids: {missing}")

    anchors = [records_by_ledger[ledger_id] for ledger_id in SOURCE_LEDGER_IDS]
    category_counts = Counter(
        str(record.get("canonical_category", "unknown")) for record in anchors
    )
    block_counts = Counter(str(record.get("block_type", "unknown")) for record in anchors)

    specs: list[DerivedInteractionOpeningSpec] = []
    for key, metadata in SPEC_CONTENT.items():
        specs.append(
            DerivedInteractionOpeningSpec(
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
        "title": "Paper 0 Derived Interaction Opening Specs",
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
        "next_source_boundary": "P0R01422",
    }
    return DerivedInteractionOpeningSpecBundle(specs=tuple(specs), summary=summary)


def build_from_ledger(
    ledger_path: Path = DEFAULT_LEDGER_PATH,
) -> DerivedInteractionOpeningSpecBundle:
    """Build specs from the canonical Paper 0 review ledger."""
    return build_derived_interaction_opening_specs(load_jsonl(ledger_path))


def _json_ready(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value


def render_report(bundle: DerivedInteractionOpeningSpecBundle) -> str:
    """Render a Markdown source-accounting report."""
    lines = [
        "# Paper 0 Derived Interaction Opening Specs",
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
    return "\n".join(lines).rstrip() + "\n"


def write_outputs(
    bundle: DerivedInteractionOpeningSpecBundle,
    *,
    output_dir: Path = DEFAULT_EXTRACTION_DIR,
    date_tag: str = "2026-05-17",
) -> dict[str, Path]:
    """Write JSON and Markdown artifacts for promoted specs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"paper0_derived_interaction_opening_validation_specs_{date_tag}.json"
    report_path = (
        output_dir / f"paper0_derived_interaction_opening_validation_specs_report_{date_tag}.md"
    )
    payload = {
        "summary": _json_ready(bundle.summary),
        "specs": [_json_ready(asdict(spec)) for spec in bundle.specs],
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(bundle), encoding="utf-8")
    return {"json": json_path, "report": report_path}


def main() -> None:
    """Build Paper 0 derived-interaction opening specs from the ledger."""

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
